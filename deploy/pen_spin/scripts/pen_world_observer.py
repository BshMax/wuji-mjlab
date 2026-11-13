#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Publish camera-based pen poses in the calibrated wrist-tag frame over ZMQ.

The pen uses its own 18-pattern codebook. Single-face poses are flagged as
coplanar; --min-faces 2 rejects them. Use --preview to inspect alignment.

Run from the repository root:
    pixi run -e pen-spin-deploy pen-observer --preview"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

import numpy as np
import zmq
from lib.camera_config import (
  get_camera_matrix,
  get_dist_coeffs,
  load_camera_config,
)
from lib.math_utils import matrix_from_quat, quat_from_matrix
from lib.tag_frame import (
  WORLD_TAG_ID,
  WORLD_TAG_SIZE_M,
  WRIST_AXES_IN_OPENCV_TAG,
)
from lib.zmq_bridge import PEN_PORT


class WorldFrame:
  """Camera -> wrist-tag transform, averaged over the first N good frames.

  Held FIXED after that: the tag is on the wrist mount, and re-estimating it
  every frame would inject the tag's own detection noise into every pen pose.
  Press 'w' in the preview window to resample.
  """

  def __init__(self, sample_frames: int = 100):
    if sample_frames < 1:
      raise ValueError("sample_frames must be positive")
    self.sample_frames = sample_frames
    self._R: list[np.ndarray] = []
    self._t: list[np.ndarray] = []
    self.R_cw: np.ndarray | None = None    # camera -> world(tag)
    self.t_cw: np.ndarray | None = None

  @property
  def fixed(self) -> bool:
    return self.R_cw is not None

  @property
  def samples(self) -> int:
    """Good wrist-tag detections collected so far (for the operator's readout)."""
    return len(self._t)

  def reset(self) -> None:
    self._R.clear()
    self._t.clear()
    self.R_cw = self.t_cw = None

  def add(self, R_wc: np.ndarray, t_wc: np.ndarray) -> None:
    if self.fixed:
      return
    self._R.append(R_wc)
    self._t.append(t_wc)
    if len(self._R) >= self.sample_frames:
      self._finalise()

  def _finalise(self) -> None:
    import cv2
    # Average rotations via their quaternions (sign-aligned), not matrices.
    qs = np.array([quat_from_matrix(R) for R in self._R])
    ref = qs[0]
    qs[np.einsum("ij,j->i", qs, ref) < 0] *= -1.0
    q = qs.mean(0)
    q /= np.linalg.norm(q)
    R_wc = matrix_from_quat(q)
    t_wc = np.median(np.array(self._t), axis=0)
    self.R_cw = R_wc.T
    self.t_cw = -R_wc.T @ t_wc
    del cv2

  def to_world(self, R_cam, t_cam):
    """Pose given in the camera frame -> the wrist-tag frame."""
    return self.R_cw @ R_cam, self.R_cw @ t_cam + self.t_cw


@lru_cache(maxsize=1)
def _world_tag_detectors():
  import cv2
  dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
  coarse = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
  params = cv2.aruco.DetectorParameters()
  params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
  return coarse, cv2.aruco.ArucoDetector(dictionary, params)


def _detect_world_tag(gray, K, dist):
  """36h11 pose in the calibrated wrist convention, not raw OpenCV axes."""
  import cv2
  coarse, precise = _world_tag_detectors()
  corners, ids, _ = coarse.detectMarkers(gray)
  if ids is None or WORLD_TAG_ID not in ids.flatten():
    return None
  quad = corners[list(ids.flatten()).index(WORLD_TAG_ID)][0]
  # Integer coarse corners jumped by 1–3 px on a stationary real tag, causing
  # ~5-degree IPPE jumps. Fit AprilTag edges locally: full-image AprilTag
  # processing is expensive on the textured glove/table. The crop is located
  # anew every frame, not tracked or temporally smoothed.
  pad = max(8., .15 * np.max(np.ptp(quad, axis=0)))
  lo = np.maximum(np.floor(quad.min(axis=0) - pad).astype(int), 0)
  hi = np.minimum(np.ceil(quad.max(axis=0) + pad).astype(int), gray.shape[::-1])
  refined, refined_ids, _ = precise.detectMarkers(
      np.ascontiguousarray(gray[lo[1]:hi[1], lo[0]:hi[0]]))
  if refined_ids is None or WORLD_TAG_ID not in refined_ids.flatten():
    return None  # do not reintroduce coarse-corner jumps on failed refinement
  quad = refined[list(refined_ids.flatten()).index(WORLD_TAG_ID)][0].astype(np.float64) + lo
  h = WORLD_TAG_SIZE_M / 2.0
  obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
  ok, rvec, tvec = cv2.solvePnP(obj, quad.astype(np.float64), K, dist,
                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
  if not ok:
    return None
  R, _ = cv2.Rodrigues(rvec)
  return R @ WRIST_AXES_IN_OPENCV_TAG, tvec.reshape(3)


_ARTWORK_CACHE: dict[tuple, np.ndarray] = {}


def _tag_artwork(tid: int, cell: int = 40, layout=None) -> np.ndarray:
  """One tag as it is physically printed: 8x8 cells, only the data inverted.

  Layout, outermost first: 1 black ring (the ArUco border), 1 white ring,
  then the 4x4 data with every bit flipped. Verified cell-by-cell against
  ``textures/outward_A_top_front.png``.
  """
  from lib.pen_pose import PenTagLayout
  layout = layout if layout is not None else PenTagLayout()
  key = (layout.dictionary.bytesList.tobytes(), tid, cell)
  if key in _ARTWORK_CACHE:
    return _ARTWORK_CACHE[key]
  import cv2
  d4 = layout.dictionary
  core = cv2.aruco.generateImageMarker(d4, tid, 6 * cell)   # 6x6: border+data
  img = np.zeros((8 * cell, 8 * cell), np.uint8)            # black quiet ring
  img[cell:7 * cell, cell:7 * cell] = 255                   # white ring
  data = core[cell:5 * cell, cell:5 * cell]                 # the 4x4 data
  img[2 * cell:6 * cell, 2 * cell:6 * cell] = 255 - data    # inverted
  _ARTWORK_CACHE[key] = img
  return img


def _pen_wireframe(layout):
  """Edges of the pen's two heads + the shaft axis, in the pen body frame.

  Drawn from the SAME numbers the solver uses (config/pen_tags.json), so a
  wireframe that does not sit on the real pen means the layout is wrong —
  not that the drawing is decorative.
  """
  pts, edges = [], []
  for box in ("A_top", "A_bottom"):
    half = layout.box_half_sizes[box]
    center = layout.box_centers[box]
    base = len(pts)
    for sx in (-1, 1):
      for sy in (-1, 1):
        for sz in (-1, 1):
          pts.append(center + np.array([sx, sy, sz]) * half)
    # 12 edges of a box over the 8 corners just appended
    for a, b in ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                 (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)):
      edges.append((base + a, base + b))
  base = len(pts)
  pts += [[0, 0, -layout.shaft_half_length], [0, 0, layout.shaft_half_length]]
  edges.append((base, base + 1))
  return np.asarray(pts, dtype=np.float64), edges


def _draw_pose(vis, pose, K, dist, layout, cv2, axis_radius_m: float = 0.0):
  """Project the pen model and a body-frame axis triad through the solved pose."""
  rvec, tvec = pose.rvec.reshape(3, 1), pose.tvec.reshape(3, 1)
  colour = (0, 165, 255) if pose.coplanar else (0, 255, 0)

  pts, edges = _pen_wireframe(layout)
  uv, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
  uv = uv.reshape(-1, 2)
  if np.all(np.isfinite(uv)):
    for a, b in edges:
      cv2.line(vis, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), colour, 1)

  # Axis triad at the pen origin (shaft midpoint): X red, Y green, Z blue.
  # +Z points at A_top (local source-pattern 17 on the default 1x2 pen).
  L = 0.03
  axes = np.float64([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]])
  a, _ = cv2.projectPoints(axes, rvec, tvec, K, dist)
  a = a.reshape(-1, 2)
  if np.all(np.isfinite(a)):
    o = tuple(a[0].astype(int))
    for i, c in ((1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 0, 0))):
      cv2.arrowedLine(vis, o, tuple(a[i].astype(int)), c, 2, tipLength=0.25)
    cv2.circle(vis, o, 4, (255, 255, 255), -1)

  # The policy's axis-point representation: pen origin ±radius along the
  # shaft (+Z source = A_top side). Mark the pen at radius from its centre —
  # the dots should land exactly on the marks if the chain is accurate.
  if axis_radius_m > 0:
    ep = np.float64([[0, 0, axis_radius_m], [0, 0, -axis_radius_m]])
    e, _ = cv2.projectPoints(ep, rvec, tvec, K, dist)
    e = e.reshape(-1, 2)
    if np.all(np.isfinite(e)):
      for pt, text, c in ((0, f"+{axis_radius_m * 1e3:.0f}mm", (255, 255, 0)),
                          (1, f"-{axis_radius_m * 1e3:.0f}mm", (0, 255, 255))):
        cv2.circle(vis, tuple(e[pt].astype(int)), 6, c, -1)
        cv2.circle(vis, tuple(e[pt].astype(int)), 8, (0, 0, 0), 1)
        cv2.putText(vis, text, (int(e[pt][0]) + 10, int(e[pt][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)


def _draw_palm_frame(vis, world, tag_in_palm_pos, tag_in_palm_quat,
                     label, K, dist, cv2, L: float = 0.04):
  """Project the palm_link origin + XYZ triad into the image.

  Chain: camera -> tag (the fixed world calibration) -> palm (inverse of the
  tag's pose in the palm frame). Drawn like the pen axis triad."""
  from lib.math_utils import quat_apply, quat_inv
  qi = quat_inv(np.asarray(tag_in_palm_quat, dtype=np.float64))
  palm_origin_tag = quat_apply(qi, -np.asarray(tag_in_palm_pos, dtype=np.float64))
  R_wc = world.R_cw.T                      # tag axes in the camera frame
  t_wc = -R_wc @ world.t_cw                # tag origin in the camera frame
  origin = R_wc @ palm_origin_tag + t_wc   # palm origin in the camera frame
  R_pc = R_wc @ matrix_from_quat(qi)       # palm axes in the camera frame
  pts = np.vstack([origin,
                   origin + R_pc[:, 0] * L,
                   origin + R_pc[:, 1] * L,
                   origin + R_pc[:, 2] * L]).reshape(-1, 1, 3)
  a, _ = cv2.projectPoints(pts, np.zeros((3, 1)), np.zeros((3, 1)), K, dist)
  a = a.reshape(-1, 2)
  if not np.all(np.isfinite(a)):
    return
  o = tuple(a[0].astype(int))
  for i, c in ((1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 0, 0))):
    cv2.arrowedLine(vis, o, tuple(a[i].astype(int)), c, 2, tipLength=0.25)
  cv2.circle(vis, o, 5, (255, 0, 255), -1)
  cv2.putText(vis, label, (o[0] + 8, o[1] - 8),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)


def _draw_palm_readout(vis, title, rows, cv2):
  """Top-right text panel: point coords (mm) and |d| in the palm frame."""
  pad, lh, w = 8, 22, 336
  x0, y0 = vis.shape[1] - w - 10, 52
  h = pad * 2 + lh * (len(rows) + 1)
  cv2.rectangle(vis, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
  cv2.rectangle(vis, (x0, y0), (x0 + w, y0 + h), (80, 80, 80), 1)
  cv2.putText(vis, title, (x0 + pad, y0 + pad + 12),
              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
  for i, (lbl, p, colour) in enumerate(rows):
    d = np.linalg.norm(p) * 1e3
    txt = (f"{lbl:8s} ({p[0]*1e3:5.0f},{p[1]*1e3:5.0f},{p[2]*1e3:5.0f})"
           f"  |d|={d:5.0f}")
    cv2.putText(vis, txt, (x0 + pad, y0 + pad + 12 + lh * (i + 1)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)


def _draw_world_tag(vis, transform, K, dist, cv2, *, fixed=False):
  """Draw the detected tag or the fixed calibration used by the publisher.

  The cached frame stays visible when the tag is occluded. A separate live
  outline makes camera/mount motion after calibration visible to the operator.
  """
  R, t = transform
  if not np.isfinite(R).all() or not np.isfinite(t).all() or t[2] <= 0:
    return
  rvec = cv2.Rodrigues(R)[0]
  h = WORLD_TAG_SIZE_M / 2
  corners = np.float64([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
  uv = cv2.projectPoints(corners, rvec, t, K, dist)[0].reshape(4, 2)
  if not np.isfinite(uv).all():
    return
  colour = (255, 255, 0) if fixed else (0, 255, 255)
  cv2.polylines(vis, [uv.astype(int)], True, colour, 2)
  # Tag-frame axis triad at the tag centre — read these off the preview to
  # describe the tag's orientation for a mount calibration.
  L = 0.03
  axes = np.float64([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]])
  a, _ = cv2.projectPoints(axes, rvec, t, K, dist)
  a = a.reshape(-1, 2)
  if np.all(np.isfinite(a)):
    o = tuple(a[0].astype(int))
    for i, c in ((1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 0, 0))):
      cv2.arrowedLine(vis, o, tuple(a[i].astype(int)), c, 2, tipLength=0.25)
  label = "world FIXED" if fixed else f"base tag {WORLD_TAG_ID} LIVE"
  anchor = (int(uv[:, 0].min()),
            int(uv[:, 1].max() + 18 if fixed else uv[:, 1].min() - 8))
  cv2.putText(vis, label, anchor,
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
  cv2.putText(vis, label, anchor,
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
  # drawFrameAxes warns every frame if any endpoint leaves the image.
  # Omit the triad in that case; the tag outline/label remain visible.
  axes = np.float64([[0, 0, 0], [.04, 0, 0], [0, .04, 0], [0, 0, .04]])
  pixels = cv2.projectPoints(axes, rvec, t, K, dist)[0].reshape(-1, 2)
  height, width = vis.shape[:2]
  if (np.all((R @ axes.T).T[:, 2] + np.asarray(t).reshape(3)[2] > 0)
      and np.isfinite(pixels).all()
      and np.all(pixels >= 0)
      and np.all(pixels < [width, height])):
    cv2.drawFrameAxes(vis, K, dist, rvec, t, 0.04, 2)


def _synthetic_frames(K, dist, hz, layout=None):
  """Render the layout through a moving pose — no camera, no MuJoCo."""
  import cv2
  from lib.pen_pose import PenTagLayout
  lay = layout if layout is not None else PenTagLayout()
  W, H = 1280, 1024
  t0 = time.monotonic()
  while True:
    t = time.monotonic() - t0
    ax = np.array([math.cos(t * 0.7), math.sin(t * 0.5), 0.4])
    ax /= np.linalg.norm(ax)
    R, _ = cv2.Rodrigues(ax * (0.9 + 0.5 * math.sin(t)))
    tv = np.array([0.02 * math.cos(t), 0.02 * math.sin(t), 0.32])
    canvas = np.full((H, W), 255, np.uint8)
    draw = []
    for tid, C in lay.corners.items():
      Pc = (R @ C.T).T + tv
      n = np.cross(Pc[1] - Pc[0], Pc[2] - Pc[1])
      n /= np.linalg.norm(n)
      c = Pc.mean(0)
      # Canonical clockwise corners have an inward cross-product normal.
      if float(-n @ (-c / np.linalg.norm(c))) < 0.25:
        continue
      px, _ = cv2.projectPoints(C, cv2.Rodrigues(R)[0], tv, K, dist)
      draw.append((c[2], tid, px.reshape(4, 2).astype(np.float32)))
    for _, tid, uv in sorted(draw, reverse=True):
      # The custom codebook inverts data cells, leaving both border rings unchanged.
      src_img = _tag_artwork(tid, layout=lay)
      s_px = src_img.shape[0]
      cell = s_px // 8
      # corners_m describes the WHITE-BORDER OUTER EDGE: cells 1..7 of 8.
      lo, hi = cell, s_px - cell
      src = np.float32([[lo, lo], [hi, lo], [hi, hi], [lo, hi]])
      Hm = cv2.getPerspectiveTransform(src, uv)
      warp = cv2.warpPerspective(src_img, Hm, (W, H), borderValue=255)
      mask = cv2.warpPerspective(np.full(src_img.shape, 255, np.uint8), Hm,
                                 (W, H), borderValue=0)
      canvas[mask > 128] = warp[mask > 128]
    yield canvas, (R, tv)
    time.sleep(max(0.0, 1.0 / hz))


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--source", choices=("camera", "realsense", "video", "synthetic"),
                  default="camera")
  ap.add_argument("--camera-config", help="source-specific camera YAML")
  ap.add_argument("--camera-serial", help="RealSense device serial number")
  ap.add_argument("--max-frames", type=int, help="stop after N frames (bench checks)")
  ap.add_argument("--diagnostics", help="write per-frame measurements to a new JSONL file")
  ap.add_argument("--record-dir",
                  help="segment recording root: creates run_<timestamp>/ with "
                       "key-marked static_NN/dynamic_NN/other JSONL files and "
                       "auto-runs summarize+plot per segment on exit "
                       "(preview keys: 1=static 2=dynamic, press again to stop, "
                       "0=unmarked)")
  ap.add_argument("--palm-frame", choices=["z5", "hand2"], default="z5",
                  help="palm frame shown in the preview readout panel")
  ap.add_argument("--axis-points-mm", type=float, default=100.0,
                  help="draw the policy's ±N mm shaft axis points on the "
                  "preview (0 disables); matches axis_point_radius_m=0.10")
  ap.add_argument("--segment-seconds", type=float, default=30.0,
                  help="auto-end the active segment after this many seconds "
                       "(0 disables; default 30)")
  ap.add_argument("--video", help="path/device for --source video")
  ap.add_argument("--port", type=int, default=PEN_PORT)
  ap.add_argument("--rate-hz", type=float, default=90.0)
  ap.add_argument("--preview", action="store_true")
  ap.add_argument("--detect-only", action="store_true",
                  help="whole-sheet detection and 2D tracks; no wrist calibration or pose publication")
  ap.add_argument("--detect-hz", type=float,
                  help="maximum detection rate in --detect-only mode; capture FPS is configured separately")
  ap.add_argument("--stereo", action="store_true",
                  help="paired IR1+IR2 sheet detection, side-by-side display and miss statistics")
  ap.add_argument("--detector-profile", choices=("baseline", "corners", "ecc1", "tracked"),
                  default="baseline", help="opt-in sheet algorithm ablation; never changes real-pen PnP")
  ap.add_argument("--sheet", choices=('pen', 'wrist', 'unknown'), default='pen',
                  help="stereo reference: pen=108 codes, wrist=12 ID0 tags, unknown=no miss denominator")
  ap.add_argument("--exposure-us", type=float,
                  help="Camera exposure time in microseconds for this run only; defaults to the source camera YAML")
  ap.add_argument("--tag-layout", help="pen tag JSON; default: config/pen_tags.json (1x2, 250 mm)")
  ap.add_argument("--min-faces", type=int, default=1,
                  help="refuse poses spanning fewer cube faces than this "
                       "(2 rejects the ill-conditioned single-face solve)")
  ap.add_argument("--world-frames", type=int, default=100,
                  help="frames averaged to fix the wrist-tag world frame")
  ap.add_argument("--no-world-tag", action="store_true",
                  help="publish in the CAMERA frame; the policy side expects "
                       "the wrist-tag frame, so this is for bench work only")
  ap.add_argument("--measure", action="store_true",
                  help="start stereo miss statistics immediately; otherwise press s when the sheet is ready")
  args = ap.parse_args()
  if args.detector_profile != 'baseline' and not args.stereo:
    ap.error('--detector-profile experiments require --stereo --detect-only')
  if args.stereo and (args.source != 'realsense' or not args.detect_only):
    ap.error('--stereo requires --source realsense --detect-only')
  if args.record_dir and args.detect_only:
    ap.error('--record-dir records pen poses; it does not apply to --detect-only')
  if args.detect_hz is not None:
    if not np.isfinite(args.detect_hz) or args.detect_hz <= 0:
      ap.error("--detect-hz must be finite and positive")
    if not args.detect_only:
      ap.error("--detect-hz requires --detect-only")
  if args.exposure_us is not None and (not np.isfinite(args.exposure_us) or args.exposure_us <= 0):
    ap.error("--exposure-us must be finite and positive")
  if args.world_frames < 1:
    ap.error("--world-frames must be positive")
  if args.segment_seconds < 0 or not np.isfinite(args.segment_seconds):
    ap.error("--segment-seconds must be finite and nonnegative")

  if args.max_frames is not None and args.max_frames < 1:
    ap.error("--max-frames must be positive")
  config_file = args.camera_config
  if config_file is None and args.source == "realsense":
    config_file = os.path.join(os.path.dirname(__file__), "..", "config", "realsense.yaml")
  cfg = load_camera_config(config_file)
  if args.exposure_us is not None:
    cfg['capture']['exposure_time'] = args.exposure_us
  if args.camera_serial and args.source != "realsense":
    ap.error("--camera-serial requires --source realsense")
  recorder = (SegmentRecorder(args.record_dir, limit_s=args.segment_seconds)
              if args.record_dir else None)
  with ExitStack() as resources:
    diagnostics = (resources.enter_context(open(args.diagnostics, 'x', encoding='utf-8'))
                   if args.diagnostics else None)
    camera = None
    if args.source == "realsense":
      from lib.realsense_camera import RealSenseCamera
      camera = resources.enter_context(RealSenseCamera(cfg, serial=args.camera_serial, stereo=args.stereo))
      cfg = camera.cfg
    try:
      rc = _run_observer(args, cfg, config_file, camera, diagnostics, recorder)
    finally:
      if recorder is not None:
        recorder.close_all()
    if recorder is not None:
      recorder.analyze()
    return rc


def _run_observer(args, cfg, config_file, camera=None, diagnostics=None, recorder=None):
  import cv2
  from lib.detection_roi import RoiSelector, frame_roi, save_selection
  from lib.pen_pose import PenPoseEstimator, PenTagLayout
  if getattr(args, "palm_frame", "z5") == "z5":
    from lib.z5_tag_frame import (pen_in_z5_palm as _palm_lift,
                                  Z5_TAG_IN_PALM_POS as _PALM_TAG_POS,
                                  Z5_TAG_IN_PALM_QUAT as _PALM_TAG_QUAT)
    _palm_label = "z5 palm_link"
  else:
    from lib.tag_frame import (pen_in_palm as _palm_lift,
                               TAG_IN_PALM_POS as _PALM_TAG_POS,
                               TAG_IN_PALM_QUAT as _PALM_TAG_QUAT)
    _palm_label = "hand2 palm"
  cal_pos = np.asarray(_PALM_TAG_POS, dtype=np.float64).copy()
  cal_quat = np.asarray(_PALM_TAG_QUAT, dtype=np.float64).copy()

  def _lift(p, q):
    from lib.math_utils import (normalize_quat as _nq, quat_apply as _qa,
                                quat_mul as _qm)
    return (cal_pos + _qa(cal_quat, np.asarray(p, dtype=np.float64)),
            _nq(_qm(cal_quat, np.asarray(q, dtype=np.float64))))
  if args.preview:
    from lib.preview_ui import configure_qt_fonts
    configure_qt_fonts()

  K, dist = ((camera.K, camera.dist) if camera is not None
             else (get_camera_matrix(cfg), get_dist_coeffs(cfg)))
  est = PenPoseEstimator(K, dist, layout=PenTagLayout(args.tag_layout), min_tags=2)
  if args.detect_only:
    if getattr(args, 'stereo', False):
      from lib.stereo_tag_preview import run_stereo_preview
      frames = _source_frames(args, cfg, K, dist, est.layout, camera)
      return run_stereo_preview(args, frames, est, camera, diagnostics)
    from lib.tag_sheet_preview import run_detection_preview
    frames = _source_frames(args, cfg, K, dist, est.layout, camera)
    return run_detection_preview(args, frames, est, camera, diagnostics)
  world = WorldFrame(args.world_frames)
  selector = RoiSelector()
  preview_initialized = False

  ctx = zmq.Context.instance()
  pub = ctx.socket(zmq.PUB)
  pub.bind(f"tcp://*:{args.port}")
  print(f"[pen observer] source={args.source} publishing on :{args.port}")
  print(f"[pen observer] {len(est.layout)} tags, {est.layout.dictionary_name}, "
        f"inverted={est.layout.inverted}, size={est.layout.tag_size * 1000:.2f} mm")
  print(f"[pen observer] pen length={est.layout.total_length * 1000:.1f} mm, "
        f"exposed shaft={est.layout.shaft_half_length * 2000:.1f} mm")
  print("[pen observer] raw PnP; every solved frame is published as measured, "
        "nothing is smoothed, gated or held")
  print("[pen observer] visibility checks enabled; planar candidates use continuity only for similar pixel errors")
  if recorder is not None:
    print(f"[recorder] segments -> {recorder.run_dir} "
          "(keys: 1=static 2=dynamic, same key stops, 0=unmarked; "
          "summarize+plot on exit)")
  if args.no_world_tag:
    print("[pen observer] WARNING: publishing in the CAMERA frame "
          "(--no-world-tag); the policy expects the wrist-tag frame.")

  frames = _source_frames(args, cfg, K, dist, est.layout, camera)

  n = ok_n = 0
  total_n = total_ok = 0
  t_report = time.monotonic()
  # Observation throughput over the last second, including capture wait,
  # detection, and the previous frame's preview time.
  frame_times = deque()
  observer_fps = None
  phase_ms = dict(grab=0., base=0., detect=0., solve=0., output=0., preview=0.)
  processing_ms = deque(maxlen=10000)
  last_valid_t = None
  longest_no_pose_s = 0.
  run_start = time.monotonic()
  consensus_at_report = 0
  fast_at_report = full_at_report = 0
  try:
    previous_frame_end = time.perf_counter()
    for gray, _truth in frames:
      frame_start = time.perf_counter()
      # The first frame includes camera initialization; exclude it from steady-state timing.
      if total_n:
        phase_ms['grab'] += (frame_start - previous_frame_end) * 1000.
      detection_roi = frame_roi(cfg, gray.shape)
      n += 1
      total_n += 1
      # Keep the preview responsive while the operator positions the wrist tag.
      world_seen = False
      wt = None
      # Stop detecting the base after calibration, including in previews and
      # recordings; press w to sample it again.
      if not args.no_world_tag and not world.fixed:
        wt = _detect_world_tag(gray, K, dist)
        world_seen = wt is not None
        if world_seen:
          world.add(*wt)

      base_end = time.perf_counter()
      phase_ms['base'] += (base_end - frame_start) * 1000.
      frame_fixed = args.no_world_tag or world.fixed
      pose = est.estimate(gray, roi=detection_roi) if frame_fixed else None
      solve_end = time.perf_counter()
      pen_ms = (solve_end - base_end) * 1000.
      detect_ms = min(pen_ms, getattr(est, 'last_detect_ms', 0.)) if frame_fixed else 0.
      phase_ms['detect'] += detect_ms
      phase_ms['solve'] += pen_ms - detect_ms
      if pose is not None:
        faces = {est.layout.face_normal[t] for t in pose.tag_ids}
        if len(faces) < args.min_faces:
          pose = None
      if pose is not None:
        ok_n += 1
        total_ok += 1
      # Every frame is published; a failed frame carries no pen. No hold and no
      # jump gate: a gate rejects exactly the fast motion a spinning pen
      # consists of. lib/zmq_bridge.PenReceiver keeps the last valid pose and
      # ages it.
      pos = quat = None
      if pose is None:
        _publish(pub, None, frame_fixed)
      else:
        # Cache camera-frame PnP directly. Preview and publisher use the same
        # sample, with only the existing wrist-frame transform on the wire.
        if args.no_world_tag:
          pos, quat = pose.pos, pose.quat
        else:
          R, pos = world.to_world(matrix_from_quat(pose.quat), pose.pos)
          quat = quat_from_matrix(R)
        _publish(pub, (pos, quat), frame_fixed)

      preview_start = time.perf_counter()
      phase_ms['output'] += (preview_start - solve_end) * 1000.
      processing_ms.append((preview_start - frame_start) * 1000.)
      measured_t = time.monotonic()
      if frame_fixed:
        reference_t = run_start if last_valid_t is None else last_valid_t
        longest_no_pose_s = max(longest_no_pose_s, measured_t - reference_t)
        if pose is not None:
          last_valid_t = measured_t
      else:
        # Initial wrist sampling and an explicit resample are not pen dropouts.
        run_start = measured_t
        last_valid_t = None
      if diagnostics is not None or recorder is not None:
        record = dict(t_monotonic_s=measured_t, world_fixed=frame_fixed,
                      processing_ms=processing_ms[-1], source=args.source,
                      position=None if pos is None else pos.tolist(),
                      quat_wxyz=None if quat is None else quat.tolist(),
                      tag_ids=[] if pose is None else list(pose.tag_ids),
                      reproj_px=None if pose is None else float(pose.reproj_px),
                      coplanar=None if pose is None else bool(pose.coplanar))
        if recorder is not None:
          record["segment"] = recorder.label
        if camera is not None:
          record.update(capture_fps=camera.capture_fps,
                        actual_exposure_us=camera.actual_exposure_us,
                        skipped_frames=camera.skipped_frames)
        line = json.dumps(record) + "\n"
        if diagnostics is not None:
          diagnostics.write(line)
        if recorder is not None:
          recorder.write(record)
      frame_now = time.monotonic()
      frame_times.append(frame_now)
      while len(frame_times) > 2 and frame_times[1] < frame_now - 1.0:
        frame_times.popleft()
      span = frame_now - frame_times[0]
      observer_fps = (len(frame_times) - 1) / span if span > 0 else None

      if args.preview:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if not args.no_world_tag:
          if world.fixed:
            R_wc = world.R_cw.T
            _draw_world_tag(vis, (R_wc, -R_wc @ world.t_cw), K, dist, cv2,
                            fixed=True)
            _draw_palm_frame(vis, world, cal_pos, cal_quat,
                             _palm_label, K, dist, cv2)
          elif wt is not None:
            _draw_world_tag(vis, wt, K, dist, cv2)
        detections = getattr(est, 'last_detection', None) if frame_fixed else None
        if detections is None:
          # During world-frame sampling the pen pose is not solved yet; detect once for preview only.
          detections = est.detect(gray, roi=detection_roi)
        inlier_indices = set(getattr(est, 'last_inlier_indices', ())) if pose is not None else set()
        for index, (tid, quad) in enumerate(zip(*detections, strict=True)):
          accepted = index in inlier_indices
          colour = (0, 255, 0) if accepted else (0, 165, 255)
          cv2.polylines(vis, [quad.astype(int)], True, colour, 1)
          cv2.putText(vis, str(tid), tuple(quad[0].astype(int)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        if pose is not None:
          _draw_pose(vis, pose, K, dist, est.layout, cv2,
                     axis_radius_m=args.axis_points_mm * 1e-3)
          faces = sorted({est.layout.face_normal[t] for t in pose.tag_ids})
          d_mm = float(np.linalg.norm(pose.pos)) * 1000.0
          cv2.putText(vis, f"{pose.num_tags} tags / {len(faces)} faces {faces}  "
                           f"reproj {pose.reproj_px:.2f}px  {d_mm:.0f}mm"
                           f"{'  PLANAR AMBIGUOUS' if pose.ambiguous else '  COPLANAR' if pose.coplanar else ''}",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                      (0, 165, 255) if pose.coplanar else (0, 255, 0), 2)
          if pos is not None and not args.no_world_tag:
            pp, pq = _lift(pos, quat)
            r_ = args.axis_points_mm * 1e-3
            axis = matrix_from_quat(pq)[:, 2] * r_
            _draw_palm_readout(
                vis, f"{getattr(args, 'palm_frame', 'z5')} palm_link  xyz mm / |d| mm",
                [(f"+{r_*1e3:.0f}mm", pp + axis, (255, 255, 0)),
                 ("origin", pp, (255, 255, 255)),
                 (f"-{r_*1e3:.0f}mm", pp - axis, (0, 255, 255))], cv2)
        else:
          reason = getattr(est, "last_rejection", "")
          cv2.putText(vis, f"no pose: {reason}" if reason else "no pose", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        ignored = getattr(est, "last_ignored_ids", ())
        if pose is not None and ignored:
          cv2.putText(vis, f"ignored inconsistent tags: {list(ignored)}", (10, 86),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        if not args.no_world_tag:
          if world.fixed:
            txt = "world frame FIXED | base detection OFF (w = resample)"
            col = (0, 255, 0)
          else:
            got, need = world.samples, world.sample_frames
            txt = (f"sampling world frame {got}/{need} | "
                   f"{'base tag SEEN' if world_seen else 'show the wrist AprilTag id 0'}")
            col = (0, 255, 255) if world_seen else (0, 165, 255)
          cv2.putText(vis, txt, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
          if world.fixed:
            cal = (f"palm cal mm (i/k=Y j/l=X u/o=Z, shift=x10, r=reset, p=print): "
                   f"({cal_pos[0]*1e3:+.1f},{cal_pos[1]*1e3:+.1f},{cal_pos[2]*1e3:+.1f})")
            cv2.putText(vis, cal, (10, vis.shape[0] - 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        selector.draw(vis, detection_roi)
        cv2.putText(vis, 'RAW PnP', (10, vis.shape[0]-38), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 200, 0), 1)
        stats = f"Observer FPS: {observer_fps:.1f}" if observer_fps is not None else "Observer FPS: --"
        if recorder is not None and recorder.label is not None:
          elapsed = recorder.elapsed or 0.0
          limit = f"/{recorder.limit_s:.0f}s" if recorder.limit_s else ""
          badge = f"REC {recorder.label.upper()} {elapsed:.0f}s{limit}"
          (bx, by), (bw, bh) = (vis.shape[1] - 270, 10), (260, 36)
          cv2.rectangle(vis, (bx - 6, by - 6), (bx + bw, by + bh), (0, 0, 0), -1)
          cv2.rectangle(vis, (bx - 6, by - 6), (bx + bw, by + bh), (0, 0, 255), 2)
          cv2.putText(vis, badge, (bx, by + 25), cv2.FONT_HERSHEY_SIMPLEX,
                      0.7, (0, 0, 255), 2)
        if args.source == 'camera':
          stats += f" | Exposure set: {cfg['capture']['exposure_time']/1000.:g} ms"
        if camera is not None:
          cv2.putText(vis, camera.status(), (10, vis.shape[0]-86),
                      cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.putText(vis, stats, (10, vis.shape[0]-62), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0, 0, 0), 3)
        cv2.putText(vis, stats, (10, vis.shape[0]-62), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (255, 255, 255), 1)
        cv2.imshow("pen observer", vis)
        if not preview_initialized:
          cv2.setMouseCallback("pen observer", selector.mouse)
          preview_initialized = True
        k = cv2.pollKey() & 0xFF
        if k == ord("q"):
          break
        if k == ord("w"):
          world.reset()
          print("[pen observer] resampling the world frame")
        if recorder is not None and k in (ord("1"), ord("2")):
          label = SegmentRecorder.LABELS[chr(k)]
          if recorder.label == label:
            recorder.end()          # same key again stops the segment
          else:
            recorder.begin(label)
        if recorder is not None and k == ord("0"):
          recorder.end()
        if (recorder is not None and recorder.label is not None
                and recorder.limit_s and (recorder.elapsed or 0.0) >= recorder.limit_s):
          print(f"[recorder] {recorder.label} auto-stop after "
                f"{recorder.limit_s:.0f}s", flush=True)
          recorder.end()
        if k == ord("s"):
          selector.begin()
        if k in (ord("i"), ord("k"), ord("j"), ord("l"), ord("u"), ord("o"),
                 ord("I"), ord("K"), ord("J"), ord("L"), ord("U"), ord("O")):
          step = 0.01 if chr(k).isupper() else 0.001
          axis = {ord("I"): 1, ord("K"): 1, ord("J"): 0, ord("L"): 0,
                  ord("U"): 2, ord("O"): 2}[k & 0xDF]
          sign = {ord("I"): +1, ord("J"): -1, ord("U"): +1,
                  ord("K"): -1, ord("L"): +1, ord("O"): -1}[k & 0xDF]
          cal_pos[axis] += sign * step
          print(f"[palm cal] tag_in_palm_pos = {np.round(cal_pos, 5)}")
        if k == ord("r"):
          cal_pos[:] = _PALM_TAG_POS
          print(f"[palm cal] reset to default {np.round(cal_pos, 5)}")
        if k == ord("p"):
          print(f"[palm cal] Z5_TAG_IN_PALM_POS = np.array({cal_pos.tolist()})")
          print(f"[palm cal] Z5_TAG_IN_PALM_QUAT = np.array({cal_quat.tolist()})")
        if k in (ord("c"), 27):
          selector.cancel()
        selection = None
        if selector.editing and not selector.dragging and k in (10, 13, 32):
          selection = selector.rect
        elif k == ord("f"):
          selection = (0, 0, gray.shape[1], gray.shape[0])
        if selection is not None:
          try:
            saved = save_selection(cfg, selection, gray.shape, config_file=config_file)
            if saved is not None:
              cfg["fast_roi"] = saved
              selector.cancel()
              est.reset()
              print(f"[pen observer] saved and applied fast_roi={saved}; world frame unchanged")
          except (OSError, ValueError) as exc:
            print(f"[pen observer] ROI not changed: {exc}")

      phase_ms['preview'] += (time.perf_counter() - preview_start) * 1000.
      now = time.monotonic()
      if now - t_report >= 2.0:
        print(f"[pen observer] {n} frames, {ok_n} poses "
              f"({n / (now - t_report):.1f} FPS), "
              f"({100.0 * ok_n / max(1, n):.0f}%), "
              f"reproj-rejects={est.rejected_reproj}, "
              f"visibility-rejects={getattr(est, 'rejected_visibility', 0)}, "
              f"appearance-rejects={getattr(est, 'rejected_appearance', 0)}, "
              f"contour-fallbacks={getattr(est, 'contour_fallbacks', 0)}, "
              f"coplanar={est.coplanar_solves}")
        consensus_total = getattr(est, 'consensus_calls', 0)
        fast_total = getattr(est, 'consensus_fast', 0)
        full_total = getattr(est, 'consensus_full', 0)
        phases = ' '.join(f'{name}={elapsed/max(n, 1):.1f}' for name, elapsed in phase_ms.items())
        if camera is not None:
          print(f"[RealSense] {camera.status()} | valid poses={ok_n / (now - t_report):.1f} Hz")
        print(f"[pen timing] mean ms/frame: {phases} | "
              f"consensus={consensus_total-consensus_at_report}/{n} frames "
              f"fast={fast_total-fast_at_report} full={full_total-full_at_report} "
              f"ignored-last={list(getattr(est, 'last_ignored_ids', ()))}", flush=True)
        consensus_at_report = consensus_total
        fast_at_report, full_at_report = fast_total, full_total
        phase_ms = dict.fromkeys(phase_ms, 0.)
        n = ok_n = 0
        t_report = now
      previous_frame_end = time.perf_counter()
      if args.max_frames is not None and total_n >= args.max_frames:
        break
  except KeyboardInterrupt:
    print("\n[pen observer] stopped")
  finally:
    # Always say what happened. A finite source (--source video) can end
    # before the 2 s report tick, and exiting in silence tells the operator
    # nothing about whether anything was ever detected.
    print(f"[pen observer] final: {total_n} frames, {total_ok} poses, "
          f"contour-fallbacks={getattr(est, 'contour_fallbacks', 0)}, "
          f"reproj-rejects={est.rejected_reproj}, "
          f"visibility-rejects={getattr(est, 'rejected_visibility', 0)}, "
          f"coplanar={est.coplanar_solves}")
    if processing_ms:
      p50, p95 = np.percentile(processing_ms, [50, 95])
      print(f"[pen timing] processing p50={p50:.2f}ms p95={p95:.2f}ms "
            f"(last {len(processing_ms)} frames, excludes grab/preview); "
            f"longest no-pose interval after calibration={longest_no_pose_s*1000:.1f}ms")
    if camera is not None:
      print(f"[RealSense] final: {camera.status()}")
    if hasattr(frames, "close"):
      frames.close()
    pub.close(linger=0)
    if args.preview:
      cv2.destroyAllWindows()
  return 0


def _source_frames(args, cfg, K, dist, layout, camera=None):
  import cv2
  if camera is not None:
    return camera.frames()
  elif args.source == "synthetic":
    return _synthetic_frames(K, dist, args.rate_hz, layout)
  elif args.source == "video":
    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
      cap.release()
      raise RuntimeError(f"Cannot open video source {args.video!r}")
    return _video_frames(cap)
  else:
    return _camera_frames(cfg)



class SegmentRecorder:
  """Key-marked recording segments under one timestamped run directory.

  Preview keys: 1 = static segment, 2 = dynamic segment, 0 = back to unmarked.
  Each segment is its own JSONL so summarize/plot tools run per segment.
  """

  LABELS = {"1": "static", "2": "dynamic"}

  def __init__(self, root, limit_s=0.0):
    import time as _time
    self.limit_s = float(limit_s)
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    stamp = _time.strftime("run_%Y%m%d_%H%M%S")
    self.run_dir = base / stamp
    suffix = 1
    while self.run_dir.exists():
      suffix += 1
      self.run_dir = base / f"{stamp}_{suffix}"
    self.run_dir.mkdir()
    self._other = open(self.run_dir / "other.jsonl", "x", encoding="utf-8")
    self._current = None          # (label, seq, handle)
    self._t0 = None
    self.segments = []            # (label, Path) in start order

  @property
  def label(self):
    return self._current[0] if self._current else None

  @property
  def elapsed(self):
    return (time.monotonic() - self._t0) if self._t0 is not None else None

  def begin(self, label):
    self.end()
    seq = 1 + sum(1 for lab, _ in self.segments if lab == label)
    path = self.run_dir / f"{label}_{seq:02d}.jsonl"
    handle = open(path, "x", encoding="utf-8")
    self._current = (label, seq, handle)
    self._t0 = time.monotonic()
    self.segments.append((label, path))
    print(f"[recorder] segment {label}_{seq:02d} -> {path}", flush=True)

  def end(self):
    if self._current is not None:
      self._current[2].close()
      self._current = None
      self._t0 = None
      print("[recorder] segment closed", flush=True)

  def write(self, record):
    handle = self._current[2] if self._current else self._other
    handle.write(json.dumps(record) + "\n")

  def close_all(self):
    self.end()
    self._other.close()

  def analyze(self):
    tools = Path(__file__).resolve().parent.parent / "tools"
    for label, path in self.segments:
      n = sum(1 for _ in open(path, encoding="utf-8"))
      if n < 30:
        print(f"[recorder] skip {path.name}: only {n} frames", flush=True)
        continue
      print(f"[recorder] summarizing {path.name} ({n} frames)", flush=True)
      cmd = [sys.executable, str(tools / "summarize_observer.py"), str(path)]
      if label == "static":
        cmd.append("--static")
      subprocess.run(cmd)
      png = path.with_suffix(".png")
      print(f"[recorder] plotting {path.name} -> {png.name}", flush=True)
      subprocess.run([sys.executable, str(tools / "plot_observer.py"),
                      str(path), "--output", str(png)])
    print(f"[recorder] run directory: {self.run_dir}", flush=True)


def _publish(pub, pose, world_fixed: bool) -> None:
  """Wire schema — identical to what lib/zmq_bridge.PenReceiver parses."""
  if pose is None:
    msg = {"world_fixed": bool(world_fixed)}
  else:
    pos, q = pose
    msg = {"world_fixed": bool(world_fixed),
           "pen": {"position": {"x": float(pos[0]), "y": float(pos[1]),
                                "z": float(pos[2])},
                   # scipy xyzw on the wire; zmq_bridge converts to wxyz
                   "orientation": {"x": float(q[1]), "y": float(q[2]),
                                   "z": float(q[3]), "w": float(q[0])}}}
  try:
    pub.send_string(json.dumps(msg), flags=zmq.NOBLOCK)
  except zmq.Again:
    pass


def _video_frames(cap):
  import cv2
  try:
    while True:
      ok, frame = cap.read()
      if not ok:
        break
      yield (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             if frame.ndim == 3 else frame), None
  finally:
    cap.release()


def _camera_frames(cfg):
  """Hikvision MVS grab loop — copied from the cube observer that ran on this rig.

  Kept line-for-line rather than rewritten, because every step here was paid
  for on real hardware:

  * ``MvCamera.MV_CC_Initialize()`` before enumerating, ``MV_CC_Finalize()``
    on the way out. Skipping the first one is how you get a process that
    opens the device, prints its settings, and then hangs with no image.
  * ``TriggerMode = OFF`` — a camera left in trigger mode waits forever for a
    trigger nobody sends.
  * ``PixelFormat = BayerGB8`` read with ``string_at`` and demosaiced. The
    buffer is a raw Bayer plane, not gray.
  * min-channel gray rather than ``COLOR_BGR2GRAY``: white stays 255 while
    any colour collapses to ~0, which is what separates dark tags cleanly.
  """
  mvs = os.environ.get("MVS_PYTHON_PATH", "/opt/MVS/Samples/64/Python")
  if not os.path.isdir(os.path.join(mvs, "MvImport")):
    raise RuntimeError(
        f"MvImport not found at {mvs}/MvImport. Install the Hikvision MVS SDK "
        "(https://www.hikrobotics.com) or set MVS_PYTHON_PATH. For bench work "
        "without a camera use --source video <file>.")
  sys.path.insert(0, mvs)
  from ctypes import POINTER, cast, string_at

  import cv2
  from lib.camera_config import setup_camera_capture, setup_camera_roi
  from MvImport.MvCameraControl_class import (  # type: ignore
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_FRAME_OUT,
    MV_GIGE_DEVICE,
    MV_TRIGGER_MODE_OFF,
    MV_USB_DEVICE,
    MV_ACCESS_Exclusive,
    MvCamera,
    PixelType_Gvsp_BayerGB8,
  )

  print("[pen observer] initializing camera...")
  MvCamera.MV_CC_Initialize()
  devs = MV_CC_DEVICE_INFO_LIST()
  MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, devs)
  if devs.nDeviceNum == 0:
    MvCamera.MV_CC_Finalize()
    raise RuntimeError("No camera found!")

  cam = MvCamera()
  dev = cast(devs.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
  cam.MV_CC_CreateHandle(dev)
  cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
  cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
  cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerGB8)
  setup_camera_capture(cam, cfg)
  setup_camera_roi(cam, cfg)
  cam.MV_CC_StartGrabbing()
  print("[pen observer] camera ready!")

  out = MV_FRAME_OUT()
  misses = 0
  try:
    while True:
      ret = cam.MV_CC_GetImageBuffer(out, 100)
      if ret != 0:
        # The cube observer just `continue`d here. Say something instead: a
        # silent retry loop makes a dead camera look exactly like a camera
        # that is simply not pointed at the pen.
        misses += 1
        if misses in (10, 50) or misses % 200 == 0:
          print(f"[pen observer] no frame yet ({misses} timeouts, "
                f"last ret=0x{ret & 0xffffffff:x})", flush=True)
        continue
      if misses:
        print(f"[pen observer] frames flowing after {misses} timeouts", flush=True)
        misses = 0
      nH = out.stFrameInfo.nHeight
      nW = out.stFrameInfo.nWidth
      data = string_at(out.pBufAddr, out.stFrameInfo.nFrameLen)
      bayer = np.frombuffer(data, dtype=np.uint8).reshape(nH, nW)
      bgr = cv2.cvtColor(bayer, cv2.COLOR_BayerGB2BGR)
      gray = np.minimum(np.minimum(bgr[:, :, 0], bgr[:, :, 1]), bgr[:, :, 2])
      cam.MV_CC_FreeImageBuffer(out)
      yield gray.copy(), None
  finally:
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    MvCamera.MV_CC_Finalize()


if __name__ == "__main__":
  raise SystemExit(main())
