# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Known-sheet misses, duplicate instances, pacing, and diagnostics isolation."""
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from lib.pen_pose import PenPoseEstimator
from lib.stereo_tag_preview import SheetMissStats, run_stereo_preview
from lib.tag_sheet_preview import TagDetection


def found(tid=0, family='PEN', count=1):
    return [TagDetection(family, tid, np.zeros((4, 2))) for _ in range(count)]


def test_repeated_ids_count_as_instances_and_other_ids_do_not_fill_missing_slots():
    stats = SheetMissStats('pen', measuring=True)
    left = sum((found(tid, count=6) for tid in range(18)), [])
    right = left[:-2] + found(99, count=20)
    result = stats.update(left, right)
    assert result['expected_per_frame'] == 108
    assert result['views']['left']['miss_total'] == 0
    assert result['views']['right']['miss_total'] == 2
    assert result['views']['right']['missing_by_id']['PEN:17'] == 2
    assert result['views']['right']['miss_rate'] == 2/108
    assert result['joint_miss_bounds'] == [0, 0]


def test_pausing_and_reference_changes_do_not_mix_denominators():
    stats = SheetMissStats('wrist')
    paused = stats.update([], [])
    assert paused['frames'] == 0
    assert paused['views']['left']['miss_rate'] is None
    stats.reset(measuring=True)
    result = stats.update(found(0, 'APR36h11', 12), found(0, 'APR36h11', 9))
    assert result['views']['right']['miss_rate'] == .25
    stats.measuring = False
    assert stats.update([], [])['views']['right']['miss_total'] == 3
    stats.reset('pen', measuring=True)
    assert stats.update([], [])['views']['left']['opportunities'] == 108
    stats.reset('unknown')
    assert stats.update([], [])['views']['left']['miss_rate'] is None


def test_joint_missing_counts_are_bounds_not_claimed_stereo_matches():
    stats = SheetMissStats('wrist', measuring=True)
    result = stats.update(found(0, 'APR36h11', 8), found(0, 'APR36h11', 7))
    # With identical repeated IDs, actual overlap is unknown.
    assert result['joint_miss_bounds'] == [0, 4]
    result = stats.update(found(0, 'APR36h11', 2), found(0, 'APR36h11', 3))
    assert result['joint_miss_bounds'] == [7, 9]


def test_stereo_runner_keeps_two_views_and_paces_before_acquiring_next_pair(monkeypatch):
    from lib import stereo_tag_preview as module
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: setattr(clock, 'now', clock.now+seconds))
    camera = SimpleNamespace(right_K=np.eye(3), right_dist=np.zeros(5), frame_number=1,
                             sync_delta_ms=0., capture_fps=60., actual_exposure_us=5000.,
                             sync_skips=0, skipped_frames=0)
    args = SimpleNamespace(preview=False, detect_hz=50., sheet='wrist', measure=True,
                           source='realsense', max_frames=6)
    acquired_at = []

    def frames():
        for _ in range(6):
            acquired_at.append(clock.now)
            yield np.full((80, 80), 255, np.uint8), np.full((80, 80), 255, np.uint8)

    pen = PenPoseEstimator(np.eye(3), np.zeros(5))
    pen.estimate = Mock(side_effect=AssertionError('No rigid-body solve in stereo sheet test'))
    log = io.StringIO()
    assert run_stereo_preview(args, frames(), pen, camera, log) == 0
    np.testing.assert_allclose(acquired_at, np.arange(6)/50)
    records = [json.loads(line) for line in log.getvalue().splitlines()]
    assert len(records) == 6
    assert records[-1]['stats']['views']['left']['miss_total'] == 72
    assert records[-1]['stats']['views']['right']['opportunities'] == 72
    assert records[-1]['left'] == records[-1]['right'] == []
    pen.estimate.assert_not_called()
