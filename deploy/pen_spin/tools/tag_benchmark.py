#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Record raw IR pairs, independently annotate A4 pages, and compare detectors."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from lib.camera_config import load_camera_config
from lib.pen_pose import PenPoseEstimator
from lib.printed_tag_template import project_paper
from lib.tag_dataset import PairDataset, PairRecorder, load_truth
from lib.tag_metrics import latency_summary, score_frame, summarize_scores
from lib.tag_sheet_preview import TagSheetDetector, TagTracker


def record(args):
    from lib.realsense_camera import RealSenseCamera
    config = load_camera_config(args.camera_config)
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    with RealSenseCamera(config, stereo=True) as camera:
        metadata = dict(dataset_id=str(uuid.uuid4()), kind='real', scene=args.scene,
                        sheet=args.sheet, distance_cm=args.distance_cm, distance_source='operator_reported', config=camera.cfg,
                        serial=camera.serial, left_K=camera.K.tolist(), left_dist=camera.dist.tolist(),
                        right_K=camera.right_K.tolist(), right_dist=camera.right_dist.tolist(),
                        git_revision=revision, opencv=cv2.__version__, numpy=np.__version__,
                        expected_frames=args.frames, timing_note='Device and perf_counter clocks are NOT aligned')
        writer = PairRecorder(args.dataset, metadata)
        done = False
        started = time.monotonic()
        try:
            for index, (left, right) in enumerate(camera.frames()):
                writer.append(left, right, dict(frame_number=camera.frame_number,
                    timestamp_ms=camera.timestamp_ms, timestamp_domain=camera.timestamp_domain,
                    host_receive_ns=camera.host_receive_ns, sync_delta_ms=camera.sync_delta_ms,
                    actual_exposure_us=camera.actual_exposure_us, skipped_frames=camera.skipped_frames))
                if args.preview:
                    vis = np.hstack((left, right))
                    cv2.putText(vis, f'RAW recording {index+1}/{args.frames}; q stops (partial clip)',
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, 255, 2)
                    cv2.imshow('raw paired recording', vis)
                    if cv2.pollKey() & 0xFF == ord('q'):
                        break
                if (index+1) % 30 == 0:
                    print(f'Recorded {index+1} raw pairs', flush=True)
                if index+1 >= args.frames:
                    done = True
                    break
        except KeyboardInterrupt:
            pass
        finally:
            writer.close(complete=done)
            if args.preview:
                cv2.destroyAllWindows()
        print(f'Saved {writer.count} pairs in {time.monotonic()-started:.1f}s; complete={done}')
        if not done:
            raise RuntimeError('Partial capture marked incomplete; record the requested frame count again')


def annotate(args):
    dataset = PairDataset(args.dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    truth = (json.loads(output.read_text()) if output.exists() else dict(
        schema_version=1, dataset_id=dataset.metadata['dataset_id'], source='manual', frames={}))
    if truth['dataset_id'] != dataset.metadata['dataset_id']:
        raise ValueError('Existing annotations belong to another dataset')
    if args.negative_all:
        if dataset.metadata['scene'] != 'negative':
            raise ValueError('--negative-all requires a recording explicitly labelled --scene negative')
        for row in dataset.records:
            truth['frames'][str(row['index'])] = {side: dict(reviewed=True, tags=[], negative_confirmed=True)
                                                 for side in ('left', 'right')}
        output.write_text(json.dumps(truth, indent=2))
        print('Explicit negative confirmation saved for every paired frame')
        return
    title = 'Independent truth annotation'
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    print('Click PRINTED page TL, TR, BR, BL. c clears; right-click a tag excludes occlusion; '
          's saves/reviews this view; n skips; q exits. Detector outputs are never used.', flush=True)
    stop = False
    for row in dataset.records[args.start::args.stride]:
        images = dataset.images(row)
        for side, gray in zip(('left', 'right'), images, strict=True):
            previous = truth['frames'].get(str(row['index']), {}).get(side, {})
            points = list(previous.get('paper_corners', []))
            excluded = set(previous.get('excluded', []))

            def mouse(event, x, y, _flags, _param, points=points, excluded=excluded, gray=gray):
                if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                    points.append([x, y])
                elif event == cv2.EVENT_RBUTTONDOWN and len(points) == 4:
                    for tag in project_paper(args.sheet, points, gray.shape):
                        if cv2.pointPolygonTest(np.float32(tag['corners']), (x, y), False) >= 0:
                            if tag['instance'] in excluded:
                                excluded.remove(tag['instance'])
                            else:
                                excluded.add(tag['instance'])

            cv2.setMouseCallback(title, mouse)
            while True:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                tags = []
                if len(points) == 4:
                    try:
                        tags = project_paper(args.sheet, points, gray.shape, excluded)
                    except ValueError:
                        pass
                    for tag in tags:
                        color = (0, 255, 255) if tag['visible'] else (0, 0, 255)
                        cv2.polylines(vis, [np.int32(tag['corners'])], True, color, 1)
                for i, point in enumerate(points):
                    cv2.circle(vis, tuple(point), 4, (255, 0, 0), -1)
                    cv2.putText(vis, str(i+1), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 0), 2)
                cv2.putText(vis, f'Frame {row["index"]} {side} | {len(points)}/4 corners | s review, n skip, c clear, q quit',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
                cv2.imshow(title, vis)
                key = cv2.pollKey() & 0xFF
                if key == ord('c'):
                    points.clear()
                    excluded.clear()
                elif key == ord('s') and tags:
                    truth['frames'].setdefault(str(row['index']), {})[side] = dict(
                        reviewed=True, tags=tags, paper_corners=points.copy(), excluded=sorted(excluded),
                        sheet=args.sheet)
                    output.write_text(json.dumps(truth, indent=2))
                    break
                elif key == ord('n'):
                    break
                elif key == ord('q') or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    stop = True
                    break
                time.sleep(.01)
            if stop:
                break
        if stop:
            break
    cv2.destroyAllWindows()
    output.write_text(json.dumps(truth, indent=2))
    print(f'Saved annotations to {output}; skipped/unreviewed frames are not scored')


def detection_json(detection):
    return dict(family=detection.family, id=detection.marker_id, track=detection.track_id,
                source=detection.source, corrected_bits=detection.corrected_bits,
                decode_distance=detection.decode_distance, corners=detection.corners.tolist())


def evaluate(args):
    dataset = PairDataset(args.dataset)
    truth = load_truth(dataset, args.truth)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for profile in args.profiles:
        estimators = [PenPoseEstimator(dataset.metadata[f'{side}_K'], dataset.metadata[f'{side}_dist'])
                      for side in ('left', 'right')]
        if profile == 'baseline':
            detectors = [TagSheetDetector(est) for est in estimators]
        else:
            from lib.sheet_experiments import ExperimentalSheetDetector
            detectors = [ExperimentalSheetDetector(est, profile) for est in estimators]
        trackers = [TagTracker(), TagTracker()]
        scores = {'left': [], 'right': []}
        timings, details = [], []
        switches, previous_tracks = {'left': 0, 'right': 0}, {'left': {}, 'right': {}}
        for position, row in enumerate(dataset.records):
            left, right = dataset.images(row)  # decode/disk IO excluded and separately timed
            begin = time.perf_counter_ns()
            found = [detector.detect(image) for detector, image in zip(detectors, (left, right), strict=True)]
            association_start = time.perf_counter_ns()
            now = row['host_receive_ns']/1e9
            for tracker, detections in zip(trackers, found, strict=True):
                tracker.update(detections, now)
            end = time.perf_counter_ns()
            processing_ms = (end-begin)/1e6
            if position >= args.warmup:
                timings.append(processing_ms)
            entry = dict(index=row['index'], processing_ms=processing_ms,
                         association_ms=(end-association_start)/1e6, warmup=position < args.warmup)
            reference = truth['frames'].get(str(row['index']))
            for side, detections in zip(('left', 'right'), found, strict=True):
                entry[side] = [detection_json(d) for d in detections]
                if reference is not None:
                    score = score_frame(detections, reference[side]['tags'])
                    scores[side].append(score)
                    for match in score['matches']:
                        previous = previous_tracks[side].get(match['instance'])
                        if previous is not None and previous[1]+1 == row['index'] and previous[0] != match['track']:
                            switches[side] += 1
                        previous_tracks[side][match['instance']] = (match['track'], row['index'])
                    entry[f'{side}_score'] = score
            details.append(entry)
        report = dict(profile=profile, dataset_id=dataset.metadata['dataset_id'],
                      kind=dataset.metadata['kind'], distance_cm=dataset.metadata.get('distance_cm'),
                      processed_pairs=len(details), annotated_pairs=len(truth['frames']),
                      annotation_coverage=len(truth['frames'])/len(dataset.records),
                      opencv=cv2.__version__, numpy=np.__version__, machine=platform.machine(),
                      opencv_threads=cv2.getNumThreads(), warmup_pairs=args.warmup,
                      timing_scope='paired detection + correction + tracking + validation; '
                                   'excludes disk decode, GT scoring, preview and capture; NOT sensor-to-display latency',
                      processing_latency=latency_summary(timings),
                      views={side: dict(summarize_scores(rows), identity_switches=switches[side])
                             for side, rows in scores.items()},
                      truth_sha256=hashlib.sha256(Path(args.truth).read_bytes()).hexdigest())
        reports[profile] = report
        (output/f'{profile}.json').write_text(json.dumps(report, indent=2))
        (output/f'{profile}.frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in details))
        print(json.dumps(report, indent=2), flush=True)
    baseline = reports.get('baseline')
    comparison = {}
    for profile, report in reports.items():
        comparison[profile] = dict(latency=report['processing_latency'], views=report['views'])
        if baseline is not None:
            comparison[profile]['versus_baseline'] = {
                side: {key: (report['views'][side][key]-baseline['views'][side][key]
                             if report['views'][side][key] is not None
                             and baseline['views'][side][key] is not None else None)
                       for key in ('recall', 'false_discovery_rate', 'negative_frame_fp_rate')}
                for side in ('left', 'right')}
    (output/'comparison.json').write_text(json.dumps(comparison, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fixture = commands.add_parser('synthetic', help='generate a geometry-labelled smoke test, NOT real 50cm data')
    fixture.add_argument('dataset', type=Path)
    fixture.add_argument('--frames', type=int, default=45)
    def make_fixture(args):
        from lib.tag_fixture import generate_fixture
        print(generate_fixture(args.dataset, args.frames))
    fixture.set_defaults(run=make_fixture)
    capture = commands.add_parser('record', help='record raw paired IR PNGs and metadata')
    capture.add_argument('dataset', type=Path)
    capture.add_argument('--camera-config', default=str(Path(__file__).parents[1]/'config/realsense_sheet_max.yaml'))
    capture.add_argument('--scene', choices=('positive', 'negative'), required=True)
    capture.add_argument('--sheet', choices=('pen', 'wrist'), default='pen')
    capture.add_argument('--distance-cm', type=float, default=50.)
    capture.add_argument('--frames', type=int, default=300)
    capture.add_argument('--preview', action='store_true')
    capture.set_defaults(run=record)
    annotation = commands.add_parser('annotate', help='independently review paper corners or confirm negatives')
    annotation.add_argument('dataset', type=Path)
    annotation.add_argument('--output', type=Path, required=True)
    annotation.add_argument('--sheet', choices=('pen', 'wrist'), default='pen')
    annotation.add_argument('--start', type=int, default=0)
    annotation.add_argument('--stride', type=int, default=1)
    annotation.add_argument('--negative-all', action='store_true')
    annotation.set_defaults(run=annotate)
    evaluation = commands.add_parser('evaluate', help='replay identical frames through each profile')
    evaluation.add_argument('dataset', type=Path)
    evaluation.add_argument('--truth', type=Path, required=True)
    evaluation.add_argument('--output', type=Path, required=True)
    evaluation.add_argument('--profiles', nargs='+', choices=('baseline', 'corners', 'ecc1', 'tracked'),
                            default=['baseline'])
    evaluation.add_argument('--warmup', type=int, default=10)
    evaluation.set_defaults(run=evaluate)
    args = parser.parse_args()
    if getattr(args, 'frames', 1) <= 0 or getattr(args, 'stride', 1) <= 0:
        parser.error('frames and stride must be positive')
    args.run(args)


if __name__ == '__main__':
    main()
