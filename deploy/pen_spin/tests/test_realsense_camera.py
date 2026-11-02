# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Calibration compatibility, capture failures, and source-specific ROI checks."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml
from lib.detection_roi import frame_roi, save_selection
from lib.realsense_camera import RealSenseCamera, opencv_intrinsics


def test_distortion_models_are_not_silently_interchanged():
    rs = SimpleNamespace(distortion=SimpleNamespace(none=0, brown_conrady=4))
    intr = SimpleNamespace(fx=423, fy=424, ppx=425, ppy=242, model=4,
                           coeffs=[0.01, 0, 0, 0, 0])
    K, dist = opencv_intrinsics(intr, rs)
    np.testing.assert_allclose(K, [[423, 0, 425], [0, 424, 242], [0, 0, 1]])
    assert dist[0] == 0.01
    intr.model = 0
    assert not opencv_intrinsics(intr, rs)[1].any()
    intr.model = 'inverse_brown_conrady'
    with pytest.raises(ValueError, match='distortion model'):
        opencv_intrinsics(intr, rs)


def test_roi_writes_only_active_config(tmp_path):
    cfg = dict(roi=dict(offset_x=0, offset_y=0), realsense=dict(width=848), fast_roi=None)
    active, other = tmp_path / 'realsense.yaml', tmp_path / 'camera.yaml'
    active.write_text(yaml.safe_dump(cfg))
    other.write_text('do not touch\n')
    saved = save_selection(cfg, (80, 96, 200, 160), (480, 848), config_file=active)
    assert yaml.safe_load(active.read_text())['fast_roi'] == saved
    assert other.read_text() == 'do not touch\n'
    cfg['fast_roi'] = saved
    assert frame_roi(cfg, (480, 848)) == (80, 96, 200, 160)


def test_timeout_is_bounded_and_close_is_idempotent():
    camera = RealSenseCamera({'realsense': {}}, rs_module=SimpleNamespace())
    pipeline = Mock()
    pipeline.wait_for_frames.side_effect = RuntimeError('timeout')
    camera.pipeline = pipeline
    with pytest.raises(RuntimeError, match='no frames for 5 seconds'):
        next(camera.frames())
    assert pipeline.wait_for_frames.call_count == 10
    camera.close()
    camera.close()
    pipeline.stop.assert_called_once()


def test_controls_validate_before_writing():
    sensor = Mock()
    sensor.get_option_range.return_value = SimpleNamespace(min=16, max=248)
    for value in (0, float('nan'), 249):
        with pytest.raises(ValueError, match='outside'):
            RealSenseCamera._set(sensor, 'gain', value)
    sensor.set_option.assert_not_called()


def test_newest_frame_is_used_and_missing_frames_are_counted():
    rs = SimpleNamespace(frame_metadata_value=SimpleNamespace(actual_exposure=1))
    camera = RealSenseCamera({'realsense': {}}, rs_module=rs)
    camera.pipeline = Mock()

    def frameset(number):
        ir = Mock()
        ir.get_frame_number.return_value = number
        ir.get_timestamp.return_value = number * 1000 / 90
        ir.get_frame_metadata.return_value = 1000
        ir.get_data.return_value = np.full((480, 848), number, np.uint8)
        frames = Mock()
        frames.get_infrared_frame.return_value = ir
        return frames

    camera.pipeline.wait_for_frames.side_effect = [frameset(1), frameset(3)]
    camera.pipeline.poll_for_frames.side_effect = [False, frameset(4), False]
    stream = camera.frames()
    first, _ = next(stream)
    latest, _ = next(stream)
    assert first[0, 0] == 1 and latest[0, 0] == 4
    assert camera.skipped_frames == 2
    assert camera.capture_fps == pytest.approx(90)
    assert camera.actual_exposure_us == 1000
    stream.close()


@pytest.fixture
def rig():
    rs = Mock()
    device = Mock()
    rs.context.return_value.query_devices.return_value = [device]
    device.get_info.return_value = 'Intel RealSense D435I'
    pipeline = rs.pipeline.return_value
    active = pipeline.start.return_value.get_stream.return_value.as_video_stream_profile.return_value
    active.width.return_value = 848
    active.height.return_value = 480
    active.fps.return_value = 90
    active.format.return_value = rs.format.y8
    active.get_intrinsics.return_value = SimpleNamespace(
        fx=423, fy=423, ppx=425, ppy=242, model=rs.distortion.brown_conrady, coeffs=[0]*5)
    pipeline.start.return_value.get_device.return_value = device
    sensor = device.first_depth_sensor.return_value
    sensor.get_option_range.return_value = SimpleNamespace(min=0, max=200000)
    sensor.get_option.return_value = 1000
    cfg = dict(realsense=dict(width=848, height=480, fps=90),
               capture=dict(exposure_time=1000, gain=16), fast_roi=None)
    return SimpleNamespace(camera=RealSenseCamera(cfg, rs_module=rs),
                           rs=rs, pipeline=pipeline, active=active, sensor=sensor, cfg=cfg)


def test_active_profile_calibration_and_manual_controls(rig):
    with rig.camera as camera:
        assert camera.K[0, 0] == 423
        assert camera.cfg['roi'] == dict(offset_x=0, offset_y=0, width=848, height=480)
        assert 'roi' not in rig.cfg
        rig.rs.config.return_value.enable_stream.assert_called_once_with(
            rig.rs.stream.infrared, 1, 848, 480, rig.rs.format.y8, 90)
        controls = rig.sensor.set_option.call_args_list
        assert ((rig.rs.option.enable_auto_exposure, 0),) in controls
        assert ((rig.rs.option.exposure, 1000),) in controls
        assert ((rig.rs.option.emitter_enabled, 0),) in controls
    rig.pipeline.stop.assert_called_once()


def test_profile_mismatch_closes_device_without_fallback(rig):
    rig.active.width.return_value = 640
    with pytest.raises(RuntimeError, match='requested IR1 Y8 profile'):
        rig.camera.__enter__()
    rig.pipeline.start.assert_called_once()
    rig.pipeline.stop.assert_called_once()


def test_control_failure_also_closes_device(rig):
    rig.sensor.set_option.side_effect = RuntimeError('control write failed')
    with pytest.raises(RuntimeError, match='control write failed'):
        rig.camera.__enter__()
    rig.pipeline.stop.assert_called_once()


def test_preview_exposure_is_capped_by_frame_period_without_changing_config(rig):
    with rig.camera as camera:
        rig.sensor.get_option_range.return_value = SimpleNamespace(min=1, max=165000, step=1)
        rig.sensor.get_option.side_effect = lambda _option: rig.sensor.set_option.call_args.args[1]
        assert camera.adjust_exposure(2) == 2000
        assert camera.adjust_exposure(.5) == 1000
        value = camera.adjust_exposure(100)
        assert 11110 <= value <= 1e6 / 90
        assert rig.cfg['capture']['exposure_time'] == 1000


def test_stereo_initializes_both_profiles_and_calibrations(rig):
    rig.camera.stereo = True
    rig.active.get_extrinsics_to.return_value = SimpleNamespace(translation=[-.05, 0, 0])
    with rig.camera as camera:
        assert rig.rs.config.return_value.enable_stream.call_count == 2
        np.testing.assert_allclose(camera.right_K, camera.K)
        assert camera.left_to_right.translation == [-.05, 0, 0]
    rig.pipeline.stop.assert_called_once()


def test_stereo_skips_unmatched_frames_instead_of_comparing_different_times():
    rs = SimpleNamespace(frame_metadata_value=SimpleNamespace(actual_exposure=1))
    camera = RealSenseCamera({'realsense': {}}, stereo=True, rs_module=rs)
    camera.pipeline = Mock()

    def frame(number, timestamp, value):
        result = Mock()
        result.get_frame_number.return_value = number
        result.get_timestamp.return_value = timestamp
        result.get_frame_timestamp_domain.return_value = 'hardware_clock'
        result.get_frame_metadata.return_value = 5000
        result.get_data.return_value = np.full((80, 80), value, np.uint8)
        return result

    bad = Mock()
    bad.get_infrared_frame.side_effect = [frame(1, 100, 1), frame(2, 116, 2)]
    good = Mock()
    good.get_infrared_frame.side_effect = [frame(3, 133, 3), frame(3, 133, 4)]
    camera.pipeline.wait_for_frames.side_effect = [bad, good]
    camera.pipeline.poll_for_frames.return_value = False
    stream = camera.frames()
    left, right = next(stream)
    assert camera.sync_skips == 1 and camera.frame_number == 3
    assert camera.sync_delta_ms == 0
    assert left[0, 0] == 3 and right[0, 0] == 4
    stream.close()
