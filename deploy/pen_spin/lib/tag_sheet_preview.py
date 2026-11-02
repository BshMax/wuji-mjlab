# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Whole-sheet detection and image-space association, without any rigid-body PnP.

Repeated printed IDs are separate detections. Track numbers are local, spatial
associations, not globally unique marker identities or 3D poses.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property

import cv2
import numpy as np


@dataclass
class TagDetection:
    family: str
    marker_id: int
    corners: np.ndarray
    track_id: int = 0

    @cached_property
    def center(self):
        return self.corners.mean(axis=0)

    @cached_property
    def size_px(self):
        return float(np.linalg.norm(np.roll(self.corners, -1, axis=0) - self.corners,
                                    axis=1).mean())


@dataclass
class _Track:
    detection: TagDetection
    last_seen: float
    trail: deque = field(default_factory=lambda: deque(maxlen=12))


class TagTracker:
    """Associate nearby detections within each family/ID, expiring after 0.35s."""

    def __init__(self, max_age_s=0.35):
        self.max_age_s = max_age_s
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, now):
        self.tracks = {tid: track for tid, track in self.tracks.items()
                       if now - track.last_seen <= self.max_age_s}
        by_marker = {}
        for tid, track in self.tracks.items():
            old = track.detection
            by_marker.setdefault((old.family, old.marker_id), []).append((tid, old))
        candidates = []
        for index, detection in enumerate(detections):
            for tid, old in by_marker.get((detection.family, detection.marker_id), ()):
                distance = float(np.linalg.norm(detection.center - old.center))
                if distance <= max(30., 2 * max(old.size_px, detection.size_px)):
                    candidates.append((distance, index, tid))
        assigned, used = {}, set()
        for _, index, tid in sorted(candidates):
            if index not in assigned and tid not in used:
                assigned[index] = tid
                used.add(tid)
        for index, detection in enumerate(detections):
            tid = assigned.get(index)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = _Track(detection, now)
            track = self.tracks[tid]
            detection.track_id = tid
            track.detection, track.last_seen = detection, now
            track.trail.append(detection.center.copy())
        return detections


class TagSheetDetector:
    """Detect every 36h11 ID and every source-pen ID; never call estimate()."""

    def __init__(self, pen_estimator):
        self.pen = pen_estimator
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.april = cv2.aruco.ArucoDetector(dictionary, params)

    def detect(self, gray):
        detections = []
        corners, ids, _ = self.april.detectMarkers(gray)
        if ids is not None:
            detections.extend(TagDetection('APR36h11', int(tid), quad[0].astype(float))
                              for tid, quad in zip(ids.flatten(), corners, strict=True))
        ids, corners = self.pen.detect(gray)  # full frame, intentionally ignores saved pen ROI
        detections.extend(TagDetection('PEN', int(tid), quad.astype(float))
                          for tid, quad in zip(ids, corners, strict=True))
        return detections


def run_detection_preview(args, frames, pen_estimator, camera=None, diagnostics=None):
    """Live sheet test. No world calibration, pose solve, or pose publication."""
    detector, tracker = TagSheetDetector(pen_estimator), TagTracker()
    title = 'tag sheet test'
    count = 0
    started = report_at = time.monotonic()
    target_hz = getattr(args, 'detect_hz', None)
    period = 0. if target_hz is None else 1. / target_hz
    deadline = started
    timestamps = deque(maxlen=120)
    detected_ms = deque(maxlen=120)
    detected_instances = 0
    print('[tag sheet] FULL FRAME; APR36h11 + SOURCE_1X2_18; no wrist calibration, '
          'no rigid-body solve, no pose publication', flush=True)
    print('[tag sheet] Keep the whole sheet in view. Repeated IDs get separate T numbers. '
          'q/Esc: quit; t: toggle trails; r: reset tracks; +/-: exposure', flush=True)
    if target_hz is not None:
        print(f'[tag sheet] detection limit={target_hz:g} Hz; camera capture rate is separate; '
              'no frame duplication', flush=True)
    trails = True
    try:
        for gray, _truth in frames:
            before = time.perf_counter()
            detections = detector.detect(gray)
            now = time.monotonic()
            tracker.update(detections, now)
            detected_ms.append((time.perf_counter() - before) * 1000)
            timestamps.append(now)
            count += 1
            detected_instances += len(detections)
            april = [d for d in detections if d.family == 'APR36h11']
            pen = [d for d in detections if d.family == 'PEN']
            span = timestamps[-1] - timestamps[0]
            fps = (len(timestamps) - 1) / span if span else 0.
            if diagnostics is not None:
                diagnostics.write(json.dumps(dict(
                    mode='detection-only', t_monotonic_s=now, world_fixed=False,
                    position=None, quat_wxyz=None, tag_ids=[], reproj_px=None,
                    processing_ms=detected_ms[-1], source=args.source,
                    detections=[dict(family=d.family, id=d.marker_id, track=d.track_id,
                                     center_px=d.center.tolist(), corners_px=d.corners.tolist(),
                                     size_px=d.size_px) for d in detections])) + '\n')
            if args.preview:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                for d in detections:
                    color = (255, 220, 0) if d.family == 'APR36h11' else (0, 255, 0)
                    cv2.polylines(vis, [d.corners.astype(int)], True, color, 1)
                    label = f'{d.family}:{d.marker_id} T{d.track_id}'
                    x, y = d.corners.min(axis=0).astype(int)
                    anchor = (max(0, min(x, vis.shape[1] - len(label)*7)), max(12, y - 4))
                    cv2.putText(vis, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 3)
                    cv2.putText(vis, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
                    if trails:
                        points = np.asarray(tracker.tracks[d.track_id].trail, dtype=np.int32)
                        cv2.polylines(vis, [points], False, color, 1)
                lines = [
                    'SHEET TEST | full image | 2D tracks | NO POSE'
                    + ('' if target_hz is None else f' | limit={target_hz:g} Hz'),
                    f'APR36h11: {len(april)} tags / {len({d.marker_id for d in april})} IDs | '
                    f'PEN: {len(pen)} tags / {len({d.marker_id for d in pen})} IDs',
                    f'Processed: {fps:.1f} FPS | detect: {np.median(detected_ms):.1f} ms | '
                    'q quit, t trails, r reset, +/- exposure',
                ]
                if camera is not None:
                    lines.append(camera.status())
                for index, line in enumerate(lines):
                    y = vis.shape[0] - (len(lines) - index) * 20 + 5
                    cv2.rectangle(vis, (0, y-15), (vis.shape[1], y+4), (0, 0, 0), -1)
                    cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
                cv2.imshow(title, vis)
                key = cv2.pollKey() & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('t'):
                    trails = not trails
                if key == ord('r'):
                    tracker = TagTracker()
                if camera is not None and key in (ord('+'), ord('='), ord('-')):
                    try:
                        value = camera.adjust_exposure(.5 if key == ord('-') else 2.)
                        print(f'[tag sheet] exposure set to {value:g} us (not saved)', flush=True)
                    except (RuntimeError, ValueError) as exc:
                        print(f'[tag sheet] exposure unchanged: {exc}', flush=True)
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if now - report_at >= 2:
                print(f'[tag sheet] {fps:.1f} FPS; April={len(april)} pen={len(pen)}; '
                      f'detect median={np.median(detected_ms):.1f}ms', flush=True)
                report_at = now
            if args.max_frames is not None and count >= args.max_frames:
                break
            # Pace before acquiring the NEXT image, avoiding an intentional hold
            # of this frame. RealSense.frames() then drains any queued old frames.
            if period:
                deadline += period
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    deadline = time.monotonic()  # no burst of catch-up detections
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(frames, 'close'):
            frames.close()
        if args.preview:
            cv2.destroyAllWindows()
        print(f'[tag sheet] final: {count} frames, {detected_instances} total tag observations, '
              f'{count / max(time.monotonic()-started, 1e-6):.1f} processed FPS', flush=True)
    return 0
