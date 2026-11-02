# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Summarize --diagnostics JSONL; --static additionally measures pose jitter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(path, *, stationary=False):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if records and records[0].get('mode') == 'detection-only':
        duration = records[-1]['t_monotonic_s'] - records[0]['t_monotonic_s']
        rate = (len(records)-1) / duration if duration > 0 else 0.
        print(f'Sheet detection: {len(records)} frames, {rate:.1f} processed FPS; no 3D pose')
        p50, p95 = np.percentile([r['processing_ms'] for r in records], [50, 95])
        print(f'Detection and association: p50={p50:.2f} ms, p95={p95:.2f} ms')
        for family in ('APR36h11', 'PEN'):
            counts = [sum(d['family'] == family for d in r['detections']) for r in records]
            ids = {d['id'] for r in records for d in r['detections'] if d['family'] == family}
            print(f'{family}: mean={np.mean(counts):.1f} / max={max(counts)} instances per frame; '
                  f'unique IDs={sorted(ids)}')
        if stationary:
            print('--static pose jitter is unavailable in detection-only mode.')
        return
    ready = [r for r in records if r['world_fixed']]
    valid = [r for r in ready if r['position'] is not None]
    print(f'Frames: {len(records)}; after wrist calibration: {len(ready)}; valid poses: {len(valid)}')
    if len(ready) < 2:
        print('Not enough calibrated frames; show the wrist tag and record again.')
        return
    duration = ready[-1]['t_monotonic_s'] - ready[0]['t_monotonic_s']
    print(f'Valid pose rate: {len(valid) / duration:.1f} Hz; '
          f'valid fraction: {len(valid) / len(ready):.1%}')
    times = [ready[0]['t_monotonic_s'], *(r['t_monotonic_s'] for r in valid),
             ready[-1]['t_monotonic_s']]
    print(f'Largest observation gap including start/end: {max(np.diff(times)) * 1000:.1f} ms')
    p50, p95 = np.percentile([r['processing_ms'] for r in ready], [50, 95])
    print(f'Processing (excludes capture wait/preview): p50={p50:.2f} ms, p95={p95:.2f} ms')
    if valid:
        print(f"Reprojection error median: {np.median([r['reproj_px'] for r in valid]):.2f} px")
    if stationary and len(valid) > 1:
        positions = np.asarray([r['position'] for r in valid])
        deviations = np.linalg.norm(positions - np.median(positions, axis=0), axis=1) * 1000
        quats = np.asarray([r['quat_wxyz'] for r in valid])
        quats[(quats @ quats[0]) < 0] *= -1
        mean = quats.mean(axis=0)
        mean /= np.linalg.norm(mean)
        angles = np.degrees(2 * np.arccos(np.clip(np.abs(quats @ mean), 0, 1)))
        print(f'Stationary position deviation p95: {np.percentile(deviations, 95):.2f} mm')
        print(f'Stationary orientation deviation p95: {np.percentile(angles, 95):.2f} deg')
        print('Jitter is repeatability, not absolute accuracy; check known physical distances separately.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics', type=Path)
    parser.add_argument('--static', action='store_true', help='only for a stationary pen')
    args = parser.parse_args()
    summarize(args.diagnostics, stationary=args.static)
