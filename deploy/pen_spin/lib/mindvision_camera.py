# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""MindVision MVSDK Linux source, using the user's original vendor wrapper."""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from lib.camera_sources import CameraFrame, _calibration

DEFAULT_SDK_ROOT = '~/.local/opt/mindvision/CameraSDK'


def load_sdk(root=None):
    root = Path(root or os.environ.get('MINDVISION_SDK_ROOT', DEFAULT_SDK_ROOT)).expanduser().resolve()
    arch = {'x86_64': 'x64', 'aarch64': 'arm64'}.get(platform.machine())
    if platform.system() != 'Linux' or arch is None:
        raise ValueError('MindVision adapter currently supports Linux x86_64/aarch64')
    wrapper = root/'demo/python_demo/mvsdk.py'
    library = root/f'lib/{arch}/libMVSDK.so'
    if not wrapper.is_file() or not library.is_file():
        raise RuntimeError(f'MindVision SDK missing at {root}; run tools/mindvision_sdk.py install ZIP')
    # Its ELF SONAME is libMVSDK.so. Preload the absolute library so the unchanged
    # vendor wrapper can resolve it without root access or LD_LIBRARY_PATH.
    try:
        native = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        spec = importlib.util.spec_from_file_location('_pen_spin_mindvision_sdk', wrapper)
        sdk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sdk)
    except (OSError, ImportError) as exc:
        raise RuntimeError(f'Cannot load MindVision SDK at {root}: {exc}') from exc
    # Fail rather than silently mixing different SDK installations in one process.
    if sdk._sdk._handle != native._handle:
        raise RuntimeError('Another libMVSDK.so is already loaded; restart with one SDK installation')
    sdk._native_library = native
    sdk._provenance = dict(root=str(root), wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                           library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    return sdk


def sdk_call(sdk, name, *args):
    """Vendor getters/setters often only set GetLastError; check every call."""
    try:
        result = getattr(sdk, name)(*args)
    except sdk.CameraException as exc:
        raise RuntimeError(f'MindVision {name} failed ({exc.error_code}): {exc.message}') from exc
    status = sdk.GetLastError()
    if status != 0:
        raise RuntimeError(f'MindVision {name} failed ({status}); check device access/connection and settings')
    return result


def device_info(device):
    return dict(model=device.GetProductName(), name=device.GetFriendlyName(), serial=device.GetSn(),
                port=device.GetPortType(), sensor=device.GetSensorType())


def enumerate_devices(sdk):
    sdk_call(sdk, 'CameraSdkInit', 1)
    devices = sdk.CameraEnumerateDevice()
    status = sdk.GetLastError()
    if status not in (0, sdk.CAMERA_STATUS_NO_DEVICE_FOUND):
        raise RuntimeError(f'MindVision enumeration failed ({status}); check USB permissions/connection')
    return list(devices)


def select_device(devices, settings):
    serial, index = settings.get('serial'), settings.get('device_index')
    if serial is not None and index is not None:
        raise ValueError('Select camera.serial or camera.device_index, not both')
    if serial is not None:
        matches = [d for d in devices if d.GetSn() == str(serial)]
        if len(matches) != 1:
            raise RuntimeError(f'MindVision serial {serial!r} not found uniquely')
        return matches[0]
    if index is not None:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(devices):
            raise ValueError('camera.device_index is outside the enumerated device list')
        return devices[index]
    if len(devices) != 1:
        raise RuntimeError(f'Found {len(devices)} MindVision cameras; connect camera or select serial/device_index')
    return devices[0]


def capabilities(cap):
    return dict(mono_sensor=bool(cap.sIspCapacity.bMonoSensor),
                sensor_max_size=[cap.sResolutionRange.iWidthMax, cap.sResolutionRange.iHeightMax],
                resolutions=[dict(index=r.iIndex, description=r.GetDescription(), width=r.iWidth,
                                  height=r.iHeight, offset=[r.iHOffsetFOV, r.iVOffsetFOV],
                                  field_size=[r.iWidthFOV, r.iHeightFOV],
                                  software_zoom=[r.iWidthZoomSw, r.iHeightZoomSw])
                             for r in cap.pImageSizeDesc[:cap.iImageSizeDesc]],
                frame_speed_modes=[dict(index=r.iIndex, description=r.GetDescription())
                                   for r in cap.pFrameSpeedDesc[:cap.iFrameSpeedDesc]])


class MindVisionSource:
    """One physical camera; preserve native dimensions and frame-bound timestamps."""

    def __init__(self, cfg, sdk=None):
        self.cfg = cfg
        self.settings = cfg.get('camera', {})
        self.sdk = sdk
        self.handle = None
        self.buffer = None
        self.pending = None
        self.index = 0
        self.previous_tick = None
        self.tick_epoch = 0
        self.info = {}

    def call(self, name, *args):
        return sdk_call(self.sdk, name, *args)

    def __enter__(self):
        try:
            if self.sdk is None:
                self.sdk = load_sdk(self.settings.get('sdk_root'))
            sdk = self.sdk
            if self.settings.get('fps') is not None:
                raise ValueError('MVSDK uses frame_speed mode indices, not camera.fps; query supported modes')
            if self.settings.get('right_device') is not None or self.settings.get('views', 'mono') != 'mono':
                raise ValueError('MindVision source currently supports one physical mono view, without duplicated right images')
            timeout = self.settings.get('timeout_ms', 1000)
            if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 10000:
                raise ValueError('camera.timeout_ms must be an integer in 1..10000')
            self.timeout = timeout
            device = select_device(enumerate_devices(sdk), self.settings)
            self.handle = self.call('CameraInit', device, -1, -1)
            cap = self.call('CameraGetCapability', self.handle)
            self.mono = bool(cap.sIspCapacity.bMonoSensor)
            self.channels = 1 if self.mono else 3
            self.media_type = sdk.CAMERA_MEDIA_TYPE_MONO8 if self.mono else sdk.CAMERA_MEDIA_TYPE_BGR8
            self.call('CameraSetIspOutFormat', self.handle, self.media_type)
            self.call('CameraSetTriggerMode', self.handle, 0)
            options = list(cap.pImageSizeDesc[:cap.iImageSizeDesc])
            # Do not select a software-upscaled preset to manufacture resolution.
            options = [r for r in options if not r.iWidthZoomSw and not r.iHeightZoomSw]
            width, height = self.settings.get('width'), self.settings.get('height')
            if (width is None) != (height is None):
                raise ValueError('Set both camera.width and camera.height, or leave both null')
            if width is not None:
                options = [r for r in options if r.iWidth == width and r.iHeight == height]
            if self.settings.get('resolution_index') is not None:
                options = [r for r in options if r.iIndex == self.settings['resolution_index']]
            if not options:
                raise ValueError('No matching native resolution preset; inspect supported resolutions first')
            resolution = max(options, key=lambda r: r.iWidth*r.iHeight)
            self.size = [resolution.iWidth, resolution.iHeight]
            self.call('CameraSetImageResolution', self.handle, resolution)
            actual = self.call('CameraGetImageResolution', self.handle)
            if [actual.iWidth, actual.iHeight] != self.size:
                raise RuntimeError('MindVision resolution readback differs from requested mode')
            speed = self.settings.get('frame_speed')
            if speed is not None:
                if isinstance(speed, bool) or not isinstance(speed, int):
                    raise ValueError('camera.frame_speed must be an integer mode index')
                modes = [r.iIndex for r in cap.pFrameSpeedDesc[:cap.iFrameSpeedDesc]]
                if speed not in modes:
                    raise ValueError(f'camera.frame_speed must be an enumerated mode index: {modes}')
                self.call('CameraSetFrameSpeed', self.handle, speed)
            auto = self.settings.get('auto_exposure', False)
            if not isinstance(auto, bool):
                raise ValueError('camera.auto_exposure must be true or false')
            self.call('CameraSetAeState', self.handle, int(auto))
            self.exposure_range = self.call('CameraGetExposureTimeRange', self.handle)
            if not auto:
                exposure = float(self.settings.get('exposure_us', 5000))
                if not np.isfinite(exposure) or not self.exposure_range[0] <= exposure <= self.exposure_range[1]:
                    raise ValueError(f'camera.exposure_us outside supported range {self.exposure_range}')
                self.call('CameraSetExposureTime', self.handle, exposure)
            gain = self.settings.get('analog_gain')
            if gain is not None:
                if not isinstance(gain, int) or not cap.sExposeDesc.uiAnalogGainMin <= gain <= cap.sExposeDesc.uiAnalogGainMax:
                    raise ValueError('camera.analog_gain outside SDK integer range')
                self.call('CameraSetAnalogGain', self.handle, gain)
            # Calibration below applies to unrectified frames. Do not inherit an
            # SDK correction profile saved by another application.
            geometry_controls = {}
            try:
                self.call('CameraSetUndistortEnable', self.handle, 0)
                geometry_controls['undistort'] = 'disabled'
            except RuntimeError:
                if sdk.GetLastError() != sdk.CAMERA_STATUS_NOT_SUPPORTED:
                    raise
                geometry_controls['undistort'] = 'not_supported'
            # Normalize inherited SDK image geometry before applying calibration.
            for name, args in [('CameraSetMirror', (0, 0)), ('CameraSetMirror', (1, 0)),
                               ('CameraSetRotate', (0,))]:
                try:
                    self.call(name, self.handle, *args)
                    geometry_controls[f'{name}:{args[0]}'] = 'disabled'
                except RuntimeError:
                    if sdk.GetLastError() != sdk.CAMERA_STATUS_NOT_SUPPORTED:
                        raise
                    geometry_controls[f'{name}:{args[0]}'] = 'not_supported'
            self.buffer_size = int(cap.sResolutionRange.iWidthMax)*int(cap.sResolutionRange.iHeightMax)*self.channels
            self.buffer = sdk.CameraAlignMalloc(self.buffer_size, 16)
            if not self.buffer:
                raise RuntimeError('MindVision image buffer allocation failed')
            self.latest = self.settings.get('buffer_policy', 'newest')
            if self.latest not in ('newest', 'oldest'):
                raise ValueError('camera.buffer_policy must be newest or oldest')
            K, dist = _calibration(self.cfg, 'left', (self.size[1], self.size[0]))
            self.info = dict(backend='mindvision', config=self.cfg, **device_info(device), views=['left'],
                             left_size=self.size, left_K=K, left_dist=dist, shutter='unknown', pairing='single',
                             exposure_units='microseconds', frame_number_origin='software_read_index',
                             buffering=self.latest, intentional_buffer_discard_count=None,
                             sdk=getattr(sdk, '_provenance', {}), capabilities=capabilities(cap),
                             geometry_controls=geometry_controls,
                             resolution_index=actual.iIndex, resolution_offset=[actual.iHOffsetFOV, actual.iVOffsetFOV],
                             resolution_field_size=[actual.iWidthFOV, actual.iHeightFOV],
                             frame_speed_mode=self.call('CameraGetFrameSpeed', self.handle),
                             auto_exposure=bool(self.call('CameraGetAeState', self.handle)),
                             exposure_readback_us=self.call('CameraGetExposureTime', self.handle),
                             analog_gain_readback=self.call('CameraGetAnalogGain', self.handle))
            self.info['isp_readback'] = {}
            for key, name in [('gamma', 'CameraGetGamma'), ('contrast', 'CameraGetContrast'),
                              ('sharpness', 'CameraGetSharpness'), ('lut_mode', 'CameraGetLutMode'),
                              ('rgb_gain', 'CameraGetGain')]:
                try:
                    self.info['isp_readback'][key] = self.call(name, self.handle)
                except RuntimeError:
                    if sdk.GetLastError() != sdk.CAMERA_STATUS_NOT_SUPPORTED:
                        raise
                    self.info['isp_readback'][key] = None
            self.call('CameraPlay', self.handle)
            self.pending = self._read()
            return self
        except Exception:
            self.close()
            raise

    def _read(self):
        raw = None
        try:
            raw, head = self.call('CameraGetImageBufferPriority', self.handle, self.timeout,
                                  1 if self.latest == 'newest' else 0)
            received = time.perf_counter_ns()
            if [head.iWidth, head.iHeight] != self.size:
                raise RuntimeError('MindVision frame dimensions changed; reload source/calibration')
            if head.iWidth*head.iHeight*self.channels > self.buffer_size:
                raise RuntimeError('MindVision frame exceeds allocated image buffer')
            process_start = time.perf_counter_ns()
            self.call('CameraImageProcess', self.handle, raw, self.buffer, head)
            needed = head.iWidth*head.iHeight*self.channels
            if ([head.iWidth, head.iHeight] != self.size or head.uiMediaType != self.media_type
                    or not needed <= head.uBytes <= self.buffer_size):
                raise RuntimeError('MindVision ISP returned unexpected pixel format/byte length')
            data = np.ctypeslib.as_array((ctypes.c_ubyte * needed).from_address(self.buffer))
            image = data.reshape((head.iHeight, head.iWidth) if self.mono else (head.iHeight, head.iWidth, 3))
            gray = image.copy() if self.mono else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            tick = int(head.uiTimeStamp)
            reset = False
            if self.previous_tick is not None and tick < self.previous_tick:
                if self.previous_tick-tick > 2**31:
                    self.tick_epoch += 2**32
                else:
                    reset = True
                    self.tick_epoch = 0
            self.previous_tick = tick
            frame = CameraFrame(dict(left=gray), dict(frame_number=self.index, host_receive_ns=received,
                timestamp_ms=(self.tick_epoch+tick)*.1, timestamp_domain='mindvision_device_0.1ms',
                device_timestamp_ticks=tick, device_timestamp_reset=reset, sync_delta_ms=None,
                actual_exposure_us=int(head.uiExpTime), actual_analog_gain=float(head.fAnalogGain),
                sdk_processing_ms=(time.perf_counter_ns()-process_start)/1e6,
                skipped_frames=None, pixel_origin=dict(left='sdk_mono8' if self.mono else 'sdk_bgr8_to_gray')))
            self.index += 1
            return frame
        finally:
            if raw is not None:
                self.call('CameraReleaseImageBuffer', self.handle, raw)

    def frames(self):
        if self.pending is not None:
            frame, self.pending = self.pending, None
            yield frame
        while self.handle is not None:
            yield self._read()

    def describe(self):
        return self.info

    def adjust_exposure(self, factor):
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError('Exposure factor must be positive and finite')
        current = self.call('CameraGetExposureTime', self.handle)
        target = float(np.clip(current*factor, self.exposure_range[0], self.exposure_range[1]))
        self.call('CameraSetAeState', self.handle, 0)
        self.call('CameraSetExposureTime', self.handle, target)
        self.info['auto_exposure'] = False
        actual = self.call('CameraGetExposureTime', self.handle)
        self.info['exposure_readback_us'] = actual
        self.call('CameraClearBuffer', self.handle)
        self.pending = None
        print(f'MindVision exposure: {actual:.1f} us', flush=True)
        return actual

    def close(self):
        self.pending = None
        try:
            if self.handle is not None:
                handle, self.handle = self.handle, None
                self.call('CameraUnInit', handle)
        finally:
            if self.buffer is not None:
                buffer, self.buffer = self.buffer, None
                # CameraAlignFree is void and does not update GetLastError.
                self.sdk.CameraAlignFree(buffer)

    def __exit__(self, *_exc):
        self.close()
