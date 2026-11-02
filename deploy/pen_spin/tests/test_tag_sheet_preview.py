# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Sheet tests keep duplicate IDs distinct and never solve or publish a pen pose."""
import importlib.util
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
from lib.camera_config import load_camera_config
from lib.pen_pose import PenPoseEstimator
from lib.tag_sheet_preview import TagDetection, TagSheetDetector, TagTracker


def estimator():
    return PenPoseEstimator(np.eye(3), np.zeros(5))


def printed_tile(dictionary, tid, inverted=False):
    marker = cv2.aruco.generateImageMarker(dictionary, tid, 120)
    if inverted:
        artwork = np.zeros((160, 160), np.uint8)
        artwork[20:140, 20:140] = 255
        artwork[40:120, 40:120] = 255 - marker[20:100, 20:100]
    else:
        artwork = marker
    return np.pad(artwork, 15, constant_values=255)


def test_both_families_and_repeated_ids_are_detected_without_pnp():
    pen = estimator()
    pen.estimate = Mock(side_effect=AssertionError('Sheet test must not solve PnP'))
    tiles = [printed_tile(pen.layout.dictionary, tid, True) for tid in range(18)]
    tiles += [printed_tile(pen.layout.dictionary, 0, True)] * 2
    april = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tiles += [printed_tile(april, tid) for tid in (0, 0, 7)]
    canvas = np.full((1000, 1200), 255, np.uint8)
    for index, tile in enumerate(tiles):
        x, y = 10 + (index % 6)*195, 10 + (index // 6)*220
        canvas[y:y+len(tile), x:x+len(tile)] = tile
    detections = TagSheetDetector(pen).detect(canvas)
    counts = Counter((d.family, d.marker_id) for d in detections)
    assert counts[('APR36h11', 0)] == 2
    assert counts[('APR36h11', 7)] == 1
    assert counts[('PEN', 0)] == 3
    assert all(counts[('PEN', tid)] == 1 for tid in range(1, 18))
    pen.estimate.assert_not_called()


def detection(x, family='PEN', tid=0):
    return TagDetection(family, tid, np.array([[x, 0], [x+20, 0], [x+20, 20], [x, 20.]]))


def test_tracks_use_location_not_detection_order_or_just_id():
    tracker = TagTracker()
    first = tracker.update([detection(0), detection(100), detection(0, 'APR36h11')], 1)
    assert len({d.track_id for d in first}) == 3
    later = tracker.update([detection(105), detection(5), detection(5, 'APR36h11')], 1.1)
    assert later[0].track_id == first[1].track_id
    assert later[1].track_id == first[0].track_id
    assert later[2].track_id == first[2].track_id
    assert tracker.update([], 1.2) == []  # do not display held detections as current
    newer = tracker.update([detection(5)], 2)
    assert newer[0].track_id not in {d.track_id for d in first}


def test_sheet_mode_bypasses_world_calibration_and_pose_socket(monkeypatch):
    path = Path(__file__).parents[1] / 'scripts/pen_world_observer.py'
    spec = importlib.util.spec_from_file_location('sheet_test_observer', path)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    forbidden = Mock(side_effect=AssertionError('Must not initialize pose publishing or wrist sampling'))
    monkeypatch.setattr(observer.zmq.Context, 'instance', forbidden)
    monkeypatch.setattr(observer, 'WorldFrame', forbidden)
    monkeypatch.setattr(observer, '_source_frames', lambda *args: iter([(np.full((480, 848), 255, np.uint8), None)]))
    args = SimpleNamespace(preview=False, tag_layout=None, detect_only=True,
                           source='video', max_frames=1)
    assert observer._run_observer(args, load_camera_config(), None) == 0
    forbidden.assert_not_called()


def test_detection_rate_limit_paces_fresh_frames_without_duplication(monkeypatch):
    from lib import tag_sheet_preview as module
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: setattr(clock, 'now', clock.now + seconds))
    seen_at = []

    def frames():
        for _ in range(6):
            seen_at.append(clock.now)
            yield np.full((80, 80), 255, np.uint8), None

    args = SimpleNamespace(preview=False, source='video', max_frames=6, detect_hz=50)
    assert module.run_detection_preview(args, frames(), estimator()) == 0
    np.testing.assert_allclose(seen_at, np.arange(6) / 50)
