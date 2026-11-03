# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from zipfile import ZipFile

import numpy as np
import pytest
from lib.mindvision_camera import MindVisionSource, enumerate_devices, select_device
from lib.tag_dataset import PairDataset, PairRecorder


class Device:
    def __init__(self, serial='test-001'):
        self.serial = serial

    def GetProductName(self):
        return 'Fake MVSDK'

    def GetFriendlyName(self):
        return 'Fake camera'

    def GetSn(self):
        return self.serial

    def GetPortType(self):
        return 'USB3'

    def GetSensorType(self):
        return 'unknown'


def resolution(width=100, height=80, index=0):
    return NS(iIndex=index, iWidth=width, iHeight=height, iHOffsetFOV=0, iVOffsetFOV=0,
              iWidthFOV=width, iHeightFOV=height, iWidthZoomSw=0, iHeightZoomSw=0,
              GetDescription=lambda: f'{width}x{height}')


class FakeSDK:
    CAMERA_STATUS_NO_DEVICE_FOUND = -16
    CAMERA_STATUS_NOT_SUPPORTED = -4
    CAMERA_MEDIA_TYPE_MONO8 = 1
    CAMERA_MEDIA_TYPE_BGR8 = 2

    class CameraException(Exception):
        pass

    def __init__(self, mono=True):
        self.error = 0
        self.fail = None
        self.calls = []
        self.devices = [Device()]
        self.exposure = 5000.
        self.frame_tick = 100
        self.current = resolution()
        self.cap = NS(sIspCapacity=NS(bMonoSensor=int(mono)),
                      sResolutionRange=NS(iWidthMax=100, iHeightMax=80),
                      sExposeDesc=NS(uiAnalogGainMin=1, uiAnalogGainMax=32),
                      pImageSizeDesc=[self.current, resolution(50, 40, 1)], iImageSizeDesc=2,
                      pFrameSpeedDesc=[NS(iIndex=0, GetDescription=lambda: 'normal')], iFrameSpeedDesc=1)
        self.mono = mono
        self.freed = False

    def GetLastError(self):
        return self.error

    def __getattr__(self, name):
        if not name.startswith('Camera'):
            raise AttributeError(name)

        def call(*args):
            self.calls.append(name)
            self.error = -7 if name == self.fail else 0
            if name == 'CameraEnumerateDevice':
                return self.devices
            if name == 'CameraInit':
                return 0  # zero can be a valid handle
            if name == 'CameraGetCapability':
                return self.cap
            if name == 'CameraSetImageResolution':
                self.current = args[1]
            if name == 'CameraGetImageResolution':
                return self.current
            if name == 'CameraGetExposureTimeRange':
                return 100., 10000., 10.
            if name == 'CameraGetExposureTime':
                return self.exposure
            if name == 'CameraSetExposureTime':
                self.exposure = args[1]
            if name == 'CameraGetAnalogGain':
                return 1
            if name == 'CameraAlignMalloc':
                self.storage = (ctypes.c_ubyte*args[0])()
                return ctypes.addressof(self.storage)
            if name == 'CameraAlignFree':
                self.freed = True
            if name == 'CameraGetImageBufferPriority':
                self.frame_tick += 333
                return 123, NS(iWidth=self.current.iWidth, iHeight=self.current.iHeight,
                               uiMediaType=0, uBytes=0, uiTimeStamp=self.frame_tick,
                               uiExpTime=int(self.exposure), fAnalogGain=1.)
            if name == 'CameraImageProcess':
                head = args[3]
                channels = 1 if self.mono else 3
                head.uBytes = head.iWidth*head.iHeight*channels
                head.uiMediaType = self.CAMERA_MEDIA_TYPE_MONO8 if self.mono else self.CAMERA_MEDIA_TYPE_BGR8
                ctypes.memset(args[2], self.frame_tick % 256, head.uBytes)
            return 0
        return call


def config(**settings):
    return dict(camera=dict(backend='mindvision', **settings), calibration={})


def test_gray_native_frame_metadata_and_buffer_lifetime(tmp_path):
    sdk = FakeSDK()
    with MindVisionSource(config(), sdk) as source:
        info = source.describe()
        assert info['views'] == ['left'] and info['left_K'] is None
        assert info['shutter'] == 'unknown' and info['frame_number_origin'] == 'software_read_index'
        first = next(source.frames())
        expected = first.images['left'].copy()
        second = next(source.frames())
        assert np.array_equal(first.images['left'], expected)  # SDK reuses buffer; frame must own its data
        assert not np.array_equal(first.images['left'], second.images['left'])
        assert first.metadata['actual_exposure_us'] == 5000
        assert first.metadata['timestamp_ms'] == pytest.approx(43.3)
        assert first.metadata['skipped_frames'] is None
        recorder = PairRecorder(tmp_path/'clip', dict(info, kind='real'))
        recorder.append(first.images['left'], None, first.metadata, wait=True)
        recorder.close()
        assert source.adjust_exposure(100) == 10000.  # key adjustment clamps to camera range
        assert source.describe()['exposure_readback_us'] == 10000.
    dataset = PairDataset(tmp_path/'clip')
    assert dataset.views == ['left'] and len(dataset.records) == 1
    assert sdk.freed and 'CameraUnInit' in sdk.calls
    assert sdk.calls.count('CameraReleaseImageBuffer') == 2


def test_color_camera_is_converted_to_owned_gray():
    sdk = FakeSDK(mono=False)
    with MindVisionSource(config(), sdk) as source:
        frame = next(source.frames())
        assert frame.images['left'].shape == (80, 100)
        assert frame.images['left'].dtype == np.uint8
        assert frame.metadata['pixel_origin']['left'] == 'sdk_bgr8_to_gray'


def test_silent_setter_failure_stops_initialization_and_closes_handle():
    sdk = FakeSDK()
    sdk.fail = 'CameraSetExposureTime'
    with pytest.raises(RuntimeError, match=r'CameraSetExposureTime failed \(-7\)'):
        MindVisionSource(config(), sdk).__enter__()
    assert 'CameraUnInit' in sdk.calls and 'CameraPlay' not in sdk.calls


def test_failed_processing_releases_raw_buffer_and_owned_buffer():
    sdk = FakeSDK()
    sdk.fail = 'CameraImageProcess'
    with pytest.raises(RuntimeError, match='CameraImageProcess'):
        MindVisionSource(config(), sdk).__enter__()
    assert 'CameraReleaseImageBuffer' in sdk.calls
    assert 'CameraUnInit' in sdk.calls and sdk.freed


@pytest.mark.parametrize('settings', [dict(width=640, height=480), dict(width=100),
                                    dict(fps=30), dict(frame_speed=30), dict(exposure_us=0),
                                    dict(views='stereo'), dict(timeout_ms=0)])
def test_invalid_mode_never_silently_falls_back(settings):
    sdk = FakeSDK()
    with pytest.raises(ValueError):
        MindVisionSource(config(**settings), sdk).__enter__()
    assert 'CameraPlay' not in sdk.calls


def test_serial_selection_and_no_device_errors():
    sdk = FakeSDK()
    devices = [Device('a'), Device('b')]
    with pytest.raises(RuntimeError, match='select serial'):
        select_device(devices, {})
    assert select_device(devices, dict(serial='b')).GetSn() == 'b'
    with pytest.raises(RuntimeError, match='not found'):
        select_device(devices, dict(serial='missing'))
    sdk.devices = []
    assert enumerate_devices(sdk) == []
    with pytest.raises(RuntimeError, match='Found 0'):
        MindVisionSource(config(), sdk).__enter__()


def test_frame_timestamp_rollover_is_unwrapped():
    sdk = FakeSDK()
    with MindVisionSource(config(), sdk) as source:
        source.pending = None
        sdk.frame_tick = 2**32-400
        first = next(source.frames())
        sdk.frame_tick = -300
        second = next(source.frames())
        assert second.metadata['timestamp_ms'] > first.metadata['timestamp_ms']
        assert not second.metadata['device_timestamp_reset']


def tool_module():
    path = Path(__file__).parents[1]/'tools/mindvision_sdk.py'
    spec = importlib.util.spec_from_file_location('mindvision_tool_test', path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_sdk_install_is_local_selective_and_does_not_execute_archive(tmp_path):
    tool = tool_module()
    archive = tmp_path/'CameraSDK.zip'
    with ZipFile(archive, 'w') as bundle:
        for name in tool.SDK_FILES:
            bundle.writestr('CameraSDK/'+name, b'vendor bytes')
        bundle.writestr('CameraSDK/install.sh', 'exit 99')
        bundle.writestr('../../escape', 'bad')
    destination = tmp_path/'sdk'
    result = tool.install(archive, destination)
    assert not result['system_changes'] and not (destination/'install.sh').exists()
    assert not (tmp_path/'escape').exists()
    assert (destination/'pen_spin_sdk_manifest.json').exists()
    tool.install(archive, destination)  # repeated identical install is safe
    (destination/tool.SDK_FILES[0]).write_text('different')
    with pytest.raises(ValueError, match='Different SDK'):
        tool.install(archive, destination)


def test_source_runs_through_existing_capture_controller(monkeypatch, tmp_path):
    import lib.recording_session as module
    sdk = FakeSDK()
    monkeypatch.setattr(module, 'load_camera_config', lambda *_args: config())
    monkeypatch.setattr(module, 'open_camera', lambda cfg: MindVisionSource(cfg, sdk))
    args = NS(dataset=tmp_path/'session', frames=3, preview=False, scene='positive',
              sheet='pen', distance_cm=50., camera_config='test')
    module.run_capture(args, dict(test_source=True))
    clip = PairDataset(tmp_path/'session/clip_0001')
    assert len(clip.records) == 3 and clip.metadata['complete']
    assert clip.metadata['backend'] == 'mindvision' and clip.views == ['left']
    assert clip.metadata['left_K'] is None and clip.metadata['shutter'] == 'unknown'
    for record in clip.records:
        assert clip.images(record)[0].shape == (80, 100)
        assert record['actual_exposure_us'] == 5000
        assert record['skipped_frames'] is None
    assert sdk.freed


def test_inspection_closes_camera_without_starting_stream():
    sdk = FakeSDK()
    sdk.cap.sExposeDesc.fAnalogGainStep = .5
    sdk._provenance = {}
    result = tool_module().inspect_camera(sdk, {})
    assert result['model'] == 'Fake MVSDK' and result['sensor_max_size'] == [100, 80]
    assert result['frame_speed_modes'] == [dict(index=0, description='normal')]
    assert 'CameraPlay' not in sdk.calls and 'CameraUnInit' in sdk.calls
