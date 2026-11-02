# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""One-to-one spatial scoring; predictions never become recognition successes."""
from __future__ import annotations

import cv2
import numpy as np

VERIFIED_SOURCES = {'decoded', 'decoded_exact', 'decoded_corrected', 'tracked_redecoded'}


def quad_iou(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if (a.shape != (4, 2) or b.shape != (4, 2) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not cv2.isContourConvex(a) or not cv2.isContourConvex(b)):
        return 0.
    area_a, area_b = abs(cv2.contourArea(a)), abs(cv2.contourArea(b))
    intersection, _ = cv2.intersectConvexConvex(a, b)
    return float(intersection / max(area_a+area_b-intersection, 1e-9))


def score_frame(detections, truth, threshold=.5):
    """Spatial assignment first, independent of decoded ID; wrong IDs are FP+FN.

    Global maximum-IoU greedy matching is deterministic. Close duplicate
    detections may claim a GT slot once only; remaining duplicates count as FP.
    """
    visible = [t for t in truth if t.get('visible', True)]
    verified = [d for d in detections if d.source in VERIFIED_SOURCES]
    unknown = [d.source for d in detections if d.source not in VERIFIED_SOURCES | {'tracked_only'}]
    if unknown:
        raise ValueError(f'Unknown detection source: {unknown}')
    choices = sorted(((quad_iou(d.corners, t['corners']), i, j)
                      for i, d in enumerate(verified) for j, t in enumerate(visible)),
                     key=lambda item: (-item[0], item[1], item[2]))
    used_d, used_t, pairs = set(), set(), []
    for overlap, i, j in choices:
        if overlap >= threshold and i not in used_d and j not in used_t:
            used_d.add(i)
            used_t.add(j)
            pairs.append((i, j))
    tp = wrong_id = 0
    corners = []
    matches = []
    for i, j in pairs:
        detection, tag = verified[i], visible[j]
        if (detection.family, detection.marker_id) != (tag['family'], tag['id']):
            wrong_id += 1
            continue
        tp += 1
        # Truth corners have canonical marker orientation, so incorrect corner
        # rotation is not hidden by choosing the smallest cyclic permutation.
        error = float(np.sqrt(np.mean(np.sum(
            (detection.corners - np.asarray(tag['corners']))**2, axis=1))))
        corners.append(error)
        matches.append(dict(instance=tag['instance'], track=detection.track_id,
                            source=detection.source, corner_rmse_px=error))
    predictions = [d for d in detections if d.source == 'tracked_only']
    prediction_hits = sum(any(quad_iou(d.corners, t['corners']) >= threshold
                              and (d.family, d.marker_id) == (t['family'], t['id'])
                              for t in visible) for d in predictions)
    return dict(tp=tp, fp=len(verified)-tp, fn=len(visible)-tp, wrong_id=wrong_id,
                predictions=len(predictions), prediction_hits=prediction_hits,
                negative=len(visible) == 0, negative_fp=int(not visible and bool(verified)),
                corners=corners, matches=matches)


def summarize_scores(records):
    totals = {key: sum(r[key] for r in records) for key in
              ('tp', 'fp', 'fn', 'wrong_id', 'predictions', 'prediction_hits', 'negative_fp')}
    totals['annotated_view_frames'] = len(records)
    totals['negative_view_frames'] = sum(r['negative'] for r in records)
    totals['recall'] = totals['tp']/(totals['tp']+totals['fn']) if totals['tp']+totals['fn'] else None
    totals['false_discovery_rate'] = totals['fp']/(totals['tp']+totals['fp']) if totals['tp']+totals['fp'] else None
    totals['negative_frame_fp_rate'] = (totals['negative_fp']/totals['negative_view_frames']
                                        if totals['negative_view_frames'] else None)
    errors = [error for r in records for error in r['corners']]
    totals['corner_rmse_p95_px'] = float(np.percentile(errors, 95)) if errors else None
    return totals


def latency_summary(values):
    if not values:
        return None
    return dict(zip(('p50_ms', 'p95_ms', 'p99_ms', 'max_ms'),
                    map(float, [*np.percentile(values, [50, 95, 99]), max(values)]), strict=True),
                over_33ms_fraction=float(np.mean(np.asarray(values) > 33.)))
