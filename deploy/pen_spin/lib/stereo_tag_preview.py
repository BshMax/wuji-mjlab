# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Paired IR sheet detection with independent tracks and reference-count misses.

Repeated IDs prevent inferring exact stereo overlap from counts. We report each
view's shortfall against a known full sheet, plus bounds on joint misses, rather
than incorrectly treating every repeated ID as one physical object.
"""
from __future__ import annotations

import json
import time
from collections import Counter, deque

import cv2
import numpy as np

from lib.tag_sheet_preview import TagSheetDetector, TagTracker


def sheet_reference(sheet):
    if sheet == 'pen':
        return {('PEN', tid): 6 for tid in range(18)}
    if sheet == 'wrist':
        return {('APR36h11', 0): 12}
    if sheet == 'unknown':
        return {}
    raise ValueError(f'Unknown sheet: {sheet}')


class SheetMissStats:
    """Per-instance-count deficits, capped per ID; never substitute another ID."""

    def __init__(self, sheet='pen', measuring=False):
        self.epoch = 0
        self.reset(sheet, measuring)

    def reset(self, sheet=None, measuring=None):
        if sheet is not None:
            self.sheet = sheet
        if measuring is not None:
            self.measuring = measuring
        self.expected = sheet_reference(self.sheet)
        self.frames = 0
        self.misses = {'left': Counter(), 'right': Counter()}
        self.epoch += 1

    def update(self, left, right):
        counts = {side: Counter((d.family, d.marker_id) for d in detections)
                  for side, detections in (('left', left), ('right', right))}
        if self.measuring:
            self.frames += 1
        expected_total = sum(self.expected.values())
        result = dict(sheet=self.sheet, epoch=self.epoch, measuring=self.measuring,
                      frames=self.frames, expected_per_frame=expected_total, views={})
        for side, current in counts.items():
            missing = {key: max(0, expected - current[key])
                       for key, expected in self.expected.items()}
            if self.measuring:
                self.misses[side].update(missing)
            opportunities = self.frames * expected_total
            total = sum(self.misses[side].values())
            result['views'][side] = dict(
                count=sum(current.values()),
                april=sum(n for (family, _), n in current.items() if family == 'APR36h11'),
                pen=sum(n for (family, _), n in current.items() if family == 'PEN'),
                miss_current=sum(missing.values()) if self.expected else None,
                miss_total=total, opportunities=opportunities,
                miss_rate=total / opportunities if opportunities else None,
                missing_by_id={f'{family}:{tid}': n for (family, tid), n in missing.items()},
                miss_total_by_id={f'{family}:{tid}': self.misses[side][(family, tid)]
                                  for family, tid in self.expected})
        # Exact physical overlap is unknown when repeated IDs occur. These are
        # mathematical bounds, not a claimed stereo matching/recall measurement.
        if self.expected:
            lower = upper = 0
            for key, expected in self.expected.items():
                left_count = min(expected, counts['left'][key])
                right_count = min(expected, counts['right'][key])
                lower += max(0, expected-left_count-right_count)
                upper += expected-max(left_count, right_count)
            result['joint_miss_bounds'] = [lower, upper]
        else:
            result['joint_miss_bounds'] = None
        return result


def _text(vis, x, y, text, scale=.5):
    cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


def _panel(gray, detections, tracker, side, trails, stats, track_labels=False):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for detection in detections:
        color = (255, 220, 0) if detection.family == 'APR36h11' else (0, 255, 0)
        quad = detection.corners.astype(int)
        cv2.polylines(vis, [quad], True, color, 1)
        family = 'APR' if detection.family == 'APR36h11' else 'PEN'
        label = (f'{family}:{detection.marker_id} {side[0].upper()}{detection.track_id}'
                 if track_labels else f'{family[0]}{detection.marker_id}')
        x, y = quad.min(axis=0)
        x = max(0, min(x, vis.shape[1] - len(label)*7))
        cv2.putText(vis, label, (x, max(14, y-4)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 3)
        cv2.putText(vis, label, (x, max(14, y-4)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        if trails:
            points = np.asarray(tracker.tracks[detection.track_id].trail, np.int32)
            cv2.polylines(vis, [points], False, color, 1)
    view = stats['views'][side]
    rate = '--' if view['miss_rate'] is None else f"{view['miss_rate']:.1%}"
    expected = stats['expected_per_frame']
    missing = '--' if view['miss_current'] is None else str(view['miss_current'])
    _text(vis, 8, 22, f'{side.upper()} IR{1 if side == "left" else 2} | '
          f'APR={view["april"]} PEN={view["pen"]}')
    _text(vis, 8, 44, f'Current missing={missing}/{expected or "?"} | cumulative miss={rate}')
    return vis


def _serialize(detections):
    return [dict(family=d.family, id=d.marker_id, track=d.track_id, source=d.source, corrected_bits=d.corrected_bits,
                 decode_distance=d.decode_distance,
                 center_px=d.center.tolist(), corners_px=d.corners.tolist(), size_px=d.size_px)
            for d in detections]


def report(stats):
    parts = []
    for side in ('left', 'right'):
        view = stats['views'][side]
        rate = '--' if view['miss_rate'] is None else f'{view["miss_rate"]:.1%}'
        parts.append(f'{side}: APR={view["april"]} PEN={view["pen"]}, '
                     f'missing={view["miss_current"]}/{stats["expected_per_frame"]}, '
                     f'accumulated={view["miss_total"]}/{view["opportunities"]} ({rate})')
    return ' | '.join(parts)


def run_stereo_preview(args, frames, pen_estimator, camera, diagnostics=None):
    # Separate detector state and independent local track numbers for each view.
    from lib.pen_pose import PenPoseEstimator
    detectors = [TagSheetDetector(pen_estimator), TagSheetDetector(PenPoseEstimator(
        camera.right_K, camera.right_dist, layout=pen_estimator.layout))]
    profile = getattr(args, 'detector_profile', 'baseline')
    if profile != 'baseline':
        from lib.sheet_experiments import ExperimentalSheetDetector
        detectors = [ExperimentalSheetDetector(detector.pen, profile) for detector in detectors]
    trackers = [TagTracker(), TagTracker()]
    stats = SheetMissStats(args.sheet, getattr(args, 'measure', False))
    title = 'stereo tag sheet test'
    target_hz = args.detect_hz if args.detect_hz is not None else 50.
    period = 1 / target_hz
    count = 0
    started = deadline = report_at = time.monotonic()
    times, processing = deque(maxlen=120), deque(maxlen=120)
    trails, latest, track_labels = True, None, False
    if args.preview:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    print(f'[stereo tags] IR1+IR2 paired capture, detection limit={target_hz:g}Hz; '
          f'reference={args.sheet}, expected={sum(stats.expected.values())}/view', flush=True)
    print('[stereo tags] Entire sheet must be visible in BOTH views for a miss-rate test. '
          's: start/reset measurement, p: pause, 1: wrist12, 2: pen108, 0: unknown, '
          'r: reset tracks/stats, t: trails, +/-: shared sensor exposure, q/Esc: quit', flush=True)
    try:
        for left, right in frames:
            if right is None or left.shape != right.shape:
                raise RuntimeError('Stereo test needs a synchronized left/right image pair of equal size')
            before = time.perf_counter()
            detections = [detectors[0].detect(left), detectors[1].detect(right)]
            now = time.monotonic()
            for tracker, found in zip(trackers, detections, strict=True):
                tracker.update(found, now)
            latest = stats.update(*[[d for d in found if d.verified] for found in detections])
            processing.append((time.perf_counter()-before)*1000)
            times.append(now)
            fps = (len(times)-1)/(times[-1]-times[0]) if times[-1] > times[0] else 0.
            count += 1
            if diagnostics is not None:
                diagnostics.write(json.dumps(dict(
                    mode='stereo-detection', source=args.source, detector_profile=profile, t_monotonic_s=now,
                    processing_ms=processing[-1], frame_number=camera.frame_number,
                    sync_delta_ms=camera.sync_delta_ms, capture_fps=camera.capture_fps,
                    actual_exposure_us=camera.actual_exposure_us,
                    sync_skips=camera.sync_skips, capture_skips=camera.skipped_frames,
                    left=_serialize(detections[0]), right=_serialize(detections[1]),
                    stats=latest)) + '\n')
            if args.preview:
                panels = [_panel(image, found, tracker, side, trails, latest, track_labels)
                          for image, found, tracker, side in zip(
                              (left, right), detections, trackers, ('left', 'right'), strict=True)]
                vis = np.hstack(panels)
                status = 'MEASURING' if stats.measuring else 'PAUSED: s starts measurement'
                lines = [f'{args.sheet.upper()} / {profile} | {status} | paired {fps:.1f}/{target_hz:g} Hz | '
                         f'detect median={np.median(processing):.1f}ms | sync={camera.sync_delta_ms:g}ms',
                         's start/reset | p pause | 1 wrist12 | 2 pen108 | 0 unknown | '
                         'r reset | t trails | i labels | +/- exposure | q quit', camera.status()]
                for index, line in enumerate(lines):
                    y = vis.shape[0] - (len(lines)-index)*22 + 5
                    cv2.rectangle(vis, (0, y-17), (vis.shape[1], y+4), (0, 0, 0), -1)
                    _text(vis, 8, y, line, .48)
                cv2.imshow(title, vis)
                if count == 1:
                    width = min(1440, vis.shape[1])
                    cv2.resizeWindow(title, width, round(vis.shape[0]*width/vis.shape[1]))
                key = cv2.pollKey() & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('t'):
                    trails = not trails
                if key == ord('i'):
                    track_labels = not track_labels
                if key == ord('p'):
                    stats.measuring = False
                if key == ord('s'):
                    stats.reset(measuring=True)
                if key == ord('r'):
                    trackers = [TagTracker(), TagTracker()]
                    stats.reset()
                if key in (ord('0'), ord('1'), ord('2')):
                    args.sheet = {ord('0'): 'unknown', ord('1'): 'wrist', ord('2'): 'pen'}[key]
                    stats.reset(args.sheet, measuring=False)
                if key in (ord('+'), ord('='), ord('-')):
                    try:
                        value = camera.adjust_exposure(.5 if key == ord('-') else 2.)
                        print(f'[stereo tags] shared IR exposure={value:g}us (not saved)', flush=True)
                    except (RuntimeError, ValueError) as exc:
                        print(f'[stereo tags] exposure unchanged: {exc}', flush=True)
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if now-report_at >= 2:
                print(f'[stereo tags] {fps:.1f}/{target_hz:g} Hz | {report(latest)}', flush=True)
                report_at = now
            if args.max_frames is not None and count >= args.max_frames:
                break
            deadline += period
            remaining = deadline-time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(frames, 'close'):
            frames.close()
        if args.preview:
            cv2.destroyAllWindows()
        print(f'[stereo tags] final: {count} pairs, '
              f'{count/max(time.monotonic()-started, 1e-6):.1f} Hz', flush=True)
        if latest is not None:
            print(f'[stereo tags] {report(latest)}', flush=True)
    return 0
