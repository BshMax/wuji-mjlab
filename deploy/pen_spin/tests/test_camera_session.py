# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from lib.camera_sources import CameraFrame, OpenCVSource, open_camera
from lib.recording_session import RecordingSession, run_capture
from lib.tag_dataset import PairDataset


class FakeCapture:
    def __init__(self, device):
        self.device = device
        self.released = False

    def isOpened(self):
        return True

    def set(self, *_args):
        return True

    def get(self, _prop):
        return 30.

    def read(self):
        return True, np.full((80, 100, 3), [10, 20, 30], np.uint8)

    def release(self):
        self.released = True


def test_opencv_mono_is_real_mono_without_invented_calibration():
    with OpenCVSource(dict(camera=dict(device=0)), FakeCapture) as camera:
        metadata = camera.describe()
        assert metadata['views'] == ['left'] and metadata['left_K'] is None
        assert metadata['shutter'] == 'unknown' and metadata['pairing'] == 'single'
        frame = next(camera.frames())
        assert set(frame.images) == {'left'}
        assert frame.images['left'].shape == (80, 100)
        assert frame.metadata['timestamp_ms'] is None
        assert frame.metadata['pixel_origin']['left'] == 'color_to_gray'


def test_two_uvc_devices_are_not_claimed_to_be_synchronized():
    with OpenCVSource(dict(camera=dict(device=0, right_device=1)), FakeCapture) as camera:
        assert camera.describe()['pairing'] == 'host-sequential-unsynchronized'
        assert next(camera.frames()).metadata['sync_delta_ms'] is None
    with pytest.raises(ValueError, match='different'):
        OpenCVSource(dict(camera=dict(device=0, right_device=0)), FakeCapture).__enter__()


def test_wrong_calibration_is_rejected_and_devices_released():
    captures = []

    def factory(device):
        capture = FakeCapture(device)
        captures.append(capture)
        return capture

    cfg = dict(camera=dict(device=0), calibration=dict(left=dict(width=640, height=480)))
    with pytest.raises(ValueError, match='calibration size'):
        OpenCVSource(cfg, factory).__enter__()
    assert captures[0].released
    with pytest.raises(ValueError, match='Unknown camera'):
        open_camera(dict(camera=dict(backend='not-a-camera')))


class FakeSource:
    def __init__(self, number=0):
        self.number = number

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def describe(self):
        return dict(backend='fake', views=['left'], left_K=None, left_dist=None,
                    pairing='single', source_number=self.number)

    def frames(self):
        for index in range(20):
            yield CameraFrame(dict(left=np.full((80, 100), 255, np.uint8)),
                              dict(frame_number=index, host_receive_ns=(index+1)*33333333))


def fake_gui(monkeypatch, keys):
    import lib.recording_session as module
    stream = iter(keys)
    for name in ('namedWindow', 'imshow', 'resizeWindow', 'destroyAllWindows'):
        monkeypatch.setattr(module.cv2, name, lambda *args: None)
    monkeypatch.setattr(module.cv2, 'pollKey', lambda: next(stream))
    monkeypatch.setattr(module.cv2, 'getWindowProperty', lambda *args: 1)
    monkeypatch.setattr(module, 'load_camera_config', lambda *_args: {})
    return module


def args_for(path, frames=None):
    return SimpleNamespace(dataset=path, preview=False, frames=frames, scene='positive',
                           sheet='pen', distance_cm=50., camera_config='unused')


def test_keyboard_capture_pause_resume_finalizes_separate_clips(monkeypatch, tmp_path):
    module = fake_gui(monkeypatch, [ord('s'), -1, ord('p'), ord('s'), -1, ord('q')])
    monkeypatch.setattr(module, 'open_camera', lambda *_args: FakeSource())
    run_capture(args_for(tmp_path/'session'), {})
    for name in ('clip_0001', 'clip_0002'):
        dataset = PairDataset(tmp_path/'session'/name)
        assert dataset.views == ['left'] and len(dataset.records) == 2
        assert not (dataset.directory/'right').exists()
    events = [json.loads(line)['action'] for line in (tmp_path/'session/events.jsonl').read_text().splitlines()]
    assert events.count('record_start') == events.count('record_stop') == 2


def test_camera_reload_starts_new_source_and_does_not_mix_clips(monkeypatch, tmp_path):
    module = fake_gui(monkeypatch, [ord('s'), ord('k'), ord('s'), -1, ord('q')])
    sources = iter([FakeSource(1), FakeSource(2)])
    monkeypatch.setattr(module, 'open_camera', lambda *_args: next(sources))
    run_capture(args_for(tmp_path/'session'), {})
    first = PairDataset(tmp_path/'session/clip_0001')
    second = PairDataset(tmp_path/'session/clip_0002')
    assert first.metadata['source_number'] == 1 and second.metadata['source_number'] == 2
    assert len(first.records) == 1 and len(second.records) == 2


def test_explicit_fixed_frame_batch_and_mono_evaluation(monkeypatch, tmp_path):
    import lib.recording_session as module
    monkeypatch.setattr(module, 'load_camera_config', lambda *_args: {})
    monkeypatch.setattr(module, 'open_camera', lambda *_args: FakeSource())
    args = args_for(tmp_path/'session', frames=4)
    args.scene = 'negative'
    run_capture(args, {})
    clip = tmp_path/'session/clip_0001'
    dataset = PairDataset(clip)
    assert len(dataset.records) == 4
    path = Path(__file__).parents[1]/'tools/tag_benchmark.py'
    spec = importlib.util.spec_from_file_location('mono_benchmark_test', path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    truth = clip/'truth.json'
    tool.annotate(SimpleNamespace(dataset=clip, output=truth, negative_all=True))
    tool.evaluate(SimpleNamespace(dataset=clip, truth=truth, output=tmp_path/'report', profiles=None, warmup=0))
    report = json.loads((tmp_path/'report/baseline.json').read_text())
    assert report['view_names'] == ['left'] and not report['calibrated']
    assert set(report['views']) == {'left'}


def test_existing_session_resumes_without_overwriting_old_clips(tmp_path):
    folder = tmp_path/'session'
    metadata = dict(backend='fake', kind='real', scene='positive', views=['left'])
    for expected in ('clip_0001', 'clip_0002'):
        session = RecordingSession(folder)
        session.start(metadata)
        session.append(CameraFrame(dict(left=np.zeros((8, 8), np.uint8)), {}))
        session.stop()
        assert session.clip.name == expected
        session.close()


def test_independent_views_keep_different_resolutions_without_faking_stereo(tmp_path):
    from lib.tag_dataset import PairRecorder
    writer = PairRecorder(tmp_path/'clip', dict(views=['left', 'right'], pairing='host-sequential-unsynchronized'))
    writer.append(np.zeros((80, 100), np.uint8), np.zeros((40, 50), np.uint8), {})
    writer.close()
    dataset = PairDataset(tmp_path/'clip')
    assert [image.shape for image in dataset.images(dataset.records[0])] == [(80, 100), (40, 50)]
