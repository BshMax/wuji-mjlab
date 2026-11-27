# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PenPoseEstimator guarded single-tag continuity fallback."""
from __future__ import annotations

import cv2
import numpy as np

from lib.camera_config import get_camera_matrix, get_dist_coeffs, load_camera_config
from lib.math_utils import matrix_from_quat
from lib.pen_pose import PenPoseEstimator, PenTagLayout


def _project_tags(layout: PenTagLayout, K: np.ndarray, dist: np.ndarray,
                  rvec: np.ndarray, tvec: np.ndarray, tag_ids: list[int]):
    quads = []
    for tid in tag_ids:
        uv, _ = cv2.projectPoints(layout.corners[tid], rvec, tvec, K, dist)
        quads.append(uv.reshape(4, 2).astype(np.float64))
    return tag_ids, quads


def _rot_err_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1.0) * 0.5, -1.0, 1.0))))


def _make_estimator(single_tag_fallback: bool = True) -> tuple[PenPoseEstimator, np.ndarray, np.ndarray]:
    cfg = load_camera_config()
    K = get_camera_matrix(cfg)
    dist = get_dist_coeffs(cfg)
    est = PenPoseEstimator(K, dist, layout=PenTagLayout(), min_tags=2,
                           single_tag_fallback=single_tag_fallback)
    return est, K, dist


def _visible_tags_for_pose(layout: PenTagLayout, rvec: np.ndarray, tvec: np.ndarray) -> list[int]:
    from lib.pose_visibility import visibility_failure
    R = cv2.Rodrigues(rvec)[0]
    return [tid for tid in sorted(layout.corners)
            if not visibility_failure(layout, [tid], R, tvec)]


def test_single_tag_fallback_guards_and_recovery(monkeypatch):
    est, K, dist = _make_estimator(single_tag_fallback=True)
    dummy = np.zeros((1024, 1280), dtype=np.uint8)

    rvec0 = np.array([[0.35], [-0.45], [0.20]], dtype=np.float64)
    tvec0 = np.array([[0.01], [-0.02], [0.42]], dtype=np.float64)
    vis_tags = _visible_tags_for_pose(est.layout, rvec0, tvec0)
    assert len(vis_tags) >= 3

    # 1. Cold start with 1 tag MUST be rejected.
    monkeypatch.setattr(est, "detect",
                        lambda gray, roi=None: _project_tags(est.layout, K, dist, rvec0, tvec0, [vis_tags[0]]))
    assert est.estimate(dummy) is None
    assert est.single_tag_solves == 0

    # 2. Warm lock with >= 3 tags.
    monkeypatch.setattr(est, "detect",
                        lambda gray, roi=None: _project_tags(est.layout, K, dist, rvec0, tvec0, vis_tags[:3]))
    pose0 = est.estimate(dummy)
    assert pose0 is not None
    assert pose0.num_tags == 3

    # 3. Single-tag brief occlusion with small motion (8 deg + 5 mm) MUST succeed accurately.
    dR = cv2.Rodrigues(np.array([[np.radians(8.0)], [0.0], [0.0]]))[0]
    R1 = dR @ cv2.Rodrigues(rvec0)[0]
    rvec1 = cv2.Rodrigues(R1)[0]
    tvec1 = tvec0 + np.array([[0.004], [0.003], [0.0]])
    monkeypatch.setattr(est, "detect",
                        lambda gray, roi=None: _project_tags(est.layout, K, dist, rvec1, tvec1, [vis_tags[0]]))
    pose1 = est.estimate(dummy)
    assert pose1 is not None
    assert pose1.num_tags == 1
    assert est.single_tag_solves == 1
    assert _rot_err_deg(matrix_from_quat(pose1.quat), R1) < 2.0
    assert np.linalg.norm(pose1.pos - tvec1.reshape(3)) < 0.005

    # 4. Large jump (60 deg) on single tag MUST be rejected.
    R_jump = cv2.Rodrigues(np.array([[np.radians(60.0)], [0.0], [0.0]]))[0] @ R1
    rvec_jump = cv2.Rodrigues(R_jump)[0]
    monkeypatch.setattr(est, "detect",
                        lambda gray, roi=None: _project_tags(est.layout, K, dist, rvec_jump, tvec1, [vis_tags[0]]))
    assert est.estimate(dummy) is None

    # 5. Streak cap: after 5 total consecutive 1-tag solves, 6th MUST be rejected until multi-tag relock.
    monkeypatch.setattr(est, "detect",
                        lambda gray, roi=None: _project_tags(est.layout, K, dist, rvec1, tvec1, [vis_tags[0]]))
    for _ in range(4):
        assert est.estimate(dummy) is not None
    assert est._single_tag_streak == 5
    assert est.estimate(dummy) is None  # 6th consecutive 1-tag blocked

    # Multi-tag relock resets streak and recovers immediately.
    monkeypatch.setattr(est, "detect",
                        lambda gray, roi=None: _project_tags(est.layout, K, dist, rvec1, tvec1, vis_tags[:2]))
    pose_relock = est.estimate(dummy)
    assert pose_relock is not None
    assert pose_relock.num_tags == 2
    assert est._single_tag_streak == 0


def test_two_tags_one_corrupted_salvaged_by_single_tag(monkeypatch):
    est, K, dist = _make_estimator(single_tag_fallback=True)
    dummy = np.zeros((1024, 1280), dtype=np.uint8)

    rvec0 = np.array([[0.35], [-0.45], [0.20]], dtype=np.float64)
    tvec0 = np.array([[0.01], [-0.02], [0.42]], dtype=np.float64)
    vis_tags = _visible_tags_for_pose(est.layout, rvec0, tvec0)

    # Warm lock
    monkeypatch.setattr(est, "detect",
                        lambda gray, roi=None: _project_tags(est.layout, K, dist, rvec0, tvec0, vis_tags[:3]))
    assert est.estimate(dummy) is not None

    # Now 2 tags detected, but tag #1 has 20 px blur corruption.
    def _corrupted_two(gray, roi=None):
        ids, quads = _project_tags(est.layout, K, dist, rvec0, tvec0, vis_tags[:2])
        quads[1] = quads[1] + np.array([[20.0, -15.0]])
        return ids, quads

    monkeypatch.setattr(est, "detect", _corrupted_two)
    pose = est.estimate(dummy)
    assert pose is not None
    assert pose.tag_ids == (vis_tags[0],)
    assert est.last_ignored_ids == (vis_tags[1],)


def test_single_tag_disabled_rejects_even_when_warm(monkeypatch):
    est, K, dist = _make_estimator(single_tag_fallback=False)
    dummy = np.zeros((1024, 1280), dtype=np.uint8)

    rvec0 = np.array([[0.35], [-0.45], [0.20]], dtype=np.float64)
    tvec0 = np.array([[0.01], [-0.02], [0.42]], dtype=np.float64)
    vis_tags = _visible_tags_for_pose(est.layout, rvec0, tvec0)

    monkeypatch.setattr(est, "detect",
                        lambda gray, roi=None: _project_tags(est.layout, K, dist, rvec0, tvec0, vis_tags[:3]))
    assert est.estimate(dummy) is not None

    monkeypatch.setattr(est, "detect",
                        lambda gray, roi=None: _project_tags(est.layout, K, dist, rvec0, tvec0, [vis_tags[0]]))
    assert est.estimate(dummy) is None
