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
from pathlib import Path

import cv2
import numpy as np
from lib.printed_tag_template import project_paper
from lib.tag_dataset import PairDataset, load_truth
from lib.tag_evidence import as_evidence
from lib.tag_metrics import (
    TemporalMetrics,
    latency_summary,
    score_frame,
    summarize_scores,
)
from lib.tag_sheet_preview import TagSheetDetector, TagTracker


def record(args):
    from lib.recording_session import run_capture
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    run_capture(args, dict(git_revision=revision, opencv=cv2.__version__, numpy=np.__version__))


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
                                                 for side in dataset.views}
        output.write_text(json.dumps(truth, indent=2))
        print('Explicit negative confirmation saved for every paired frame')
        return
    from lib.preview_ui import configure_qt_fonts
    configure_qt_fonts()
    title = 'Independent truth annotation'
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    print('Click PRINTED page TL, TR, BR, BL. c clears; right-click marks a fully invisible tag; '
          's saves/reviews this view; n skips; q exits. Detector outputs are never used.', flush=True)
    stop = False
    previous_by_side = {}
    for row in dataset.records[args.start::args.stride]:
        images = dataset.images(row)
        for side, gray in zip(dataset.views, images, strict=True):
            previous = truth['frames'].get(str(row['index']), {}).get(side, {})
            points = list(previous.get('paper_corners', []))
            excluded = set(previous.get('excluded', []))
            if not points and side in previous_by_side:
                old_index, old_image, old_points = previous_by_side[side]
                if 0 < row['index']-old_index <= 3:
                    seed, status, _ = cv2.calcOpticalFlowPyrLK(
                        old_image, gray, np.float32(old_points).reshape(-1, 1, 2), None,
                        winSize=(31, 31), maxLevel=3)
                    if seed is not None and status.all():
                        points = seed[:, 0].round().astype(int).tolist()
            # A propagated seed is NEVER saved as reviewed without pressing s.
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
                    previous_by_side[side] = (row['index'], gray.copy(), points.copy())
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
    detection = as_evidence(detection)
    return dict(family=detection.family, id=detection.marker_id, track=detection.track_id,
                source=detection.source, corrected_bits=detection.corrected_bits,
                decode_distance=detection.decode_distance, quality=detection.quality,
                confirmed=detection.confirmed, corners=detection.corners.tolist())


def review(args):
    """Read-only visual review of accepted FP/FN and separately coloured predictions."""
    from lib.preview_ui import configure_qt_fonts
    from lib.tag_metrics import VERIFIED_SOURCES
    dataset = PairDataset(args.dataset)
    truth = load_truth(dataset, args.truth)
    report_path = args.results.with_name(args.results.name.replace('.frames.jsonl', '.json'))
    report = json.loads(report_path.read_text())
    if report['dataset_id'] != dataset.metadata['dataset_id']:
        raise ValueError('Results belong to another dataset')
    if report['truth_sha256'] != hashlib.sha256(args.truth.read_bytes()).hexdigest():
        raise ValueError('Annotations changed after evaluation; evaluate again before reviewing')
    rows = [json.loads(line) for line in args.results.read_text().splitlines()]
    if args.only_errors:
        rows = [row for row in rows if any(row.get(f'{side}_score', {}).get('fp', 0)
                                          or row.get(f'{side}_score', {}).get('fn', 0)
                                          or row.get(f'{side}_score', {}).get('tentative_false_candidates', 0)
                                          or (row.get(f'{side}_score', {}).get('predictions', 0)
                                              > row.get(f'{side}_score', {}).get('prediction_hits', 0))
                                          for side in dataset.views)]
    if not rows:
        print('No annotated error frames in this result file')
        return
    configure_qt_fonts()
    title, position, selection = 'Benchmark error review', 0, 'both'
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    while True:
        row = rows[position]
        images = dataset.images(dataset.records[row['index']])
        panels = []
        for side, image in zip(dataset.views, images, strict=True):
            view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            reference = truth['frames'].get(str(row['index']), {}).get(side, {})
            for tag in reference.get('tags', []):
                if tag.get('visible', True):
                    cv2.polylines(view, [np.int32(tag['corners'])], True, (255, 160, 0), 1)
            score = row.get(f'{side}_score', {})
            fp_indices, verified_index = set(score.get('verified_fp_indices', [])), 0
            for detection in row[side]:
                source = detection['source']
                confirmed = detection.get('confirmed', True)
                color = (0, 165, 255)
                if confirmed and source in VERIFIED_SOURCES:
                    color = (0, 0, 255) if verified_index in fp_indices else (0, 255, 0)
                    verified_index += 1
                quad = np.int32(detection['corners'])
                cv2.polylines(view, [quad], True, color, 2)
                cv2.putText(view, f"{detection['id']}:{source}", tuple(np.maximum(quad[0], [0, 15])),
                            cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
            message = (f"{side} frame={row['index']} TP={score.get('tp', '?')} "
                       f"FP={score.get('fp', '?')} FN={score.get('fn', '?')} | "
                       f"paired processing={row['processing_ms']:.1f}ms")
            cv2.putText(view, message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 0), 1)
            panels.append(view)
        display = np.hstack(panels) if selection == 'both' else panels[dataset.views.index(selection)]
        cv2.imshow(title, display)
        cv2.resizeWindow(title, min(1440, display.shape[1]),
                         round(display.shape[0]*min(1440, display.shape[1])/display.shape[1]))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord('n'), 32):
            position = min(len(rows)-1, position+1)
        if key == ord('p'):
            position = max(0, position-1)
        if key in (ord('l'), ord('r'), ord('b')):
            chosen = {ord('l'): 'left', ord('r'): 'right', ord('b'): 'both'}[key]
            selection = chosen if chosen == 'both' or chosen in dataset.views else 'left'
    cv2.destroyAllWindows()


def evaluate(args):
    dataset = PairDataset(args.dataset)
    truth = load_truth(dataset, args.truth)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.warmup < 0 or args.warmup >= len(dataset.records):
        raise ValueError('warmup must be nonnegative and smaller than the recorded frame count')
    calibrated = all(dataset.metadata.get(f'{side}_K') is not None for side in dataset.views)
    requested = args.profiles if args.profiles is not None else (['baseline', 'corners', 'ecc1', 'tracked'] if calibrated else ['baseline'])
    if not calibrated and any(p != 'baseline' for p in requested):
        raise ValueError('Experimental profiles require calibration; use --profiles baseline for an uncalibrated camera')
    if len(requested) != len(set(requested)):
        raise ValueError('Do not repeat detector profiles')
    profiles = ['baseline'] + [p for p in requested if p != 'baseline']
    repo = Path(__file__).resolve().parents[3]
    source_hashes = {}
    for relative in ('lib/pen_pose.py', 'lib/source_patterns.py', 'lib/sheet_experiments.py',
                     'lib/tag_metrics.py', 'lib/tag_evidence.py', 'lib/tag_sheet_preview.py'):
        source = Path(__file__).parents[1]/relative
        source_hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    reports = {}
    for profile in profiles:
        from lib.recording_session import make_2d_estimator
        estimators = [make_2d_estimator(dataset.metadata, side) for side in dataset.views]
        if profile == 'baseline':
            detectors = [TagSheetDetector(est) for est in estimators]
        else:
            from lib.sheet_experiments import ExperimentalSheetDetector
            detectors = [ExperimentalSheetDetector(est, profile) for est in estimators]
        trackers = [TagTracker() for _ in dataset.views]
        scores = {side: [] for side in dataset.views}
        temporal = {side: TemporalMetrics() for side in dataset.views}
        timings, details = [], []
        switches = {side: 0 for side in dataset.views}
        comparisons = {side: 0 for side in dataset.views}
        previous_tracks = {side: {} for side in dataset.views}
        for position, row in enumerate(dataset.records):
            io_begin = time.perf_counter_ns()
            images = dataset.images(row)
            image_decode_ms = (time.perf_counter_ns()-io_begin)/1e6
            for detector in detectors:
                detector.input_time_s = row['host_receive_ns']/1e9
            begin = time.perf_counter_ns()
            found, view_ms = [], []
            for detector, image in zip(detectors, images, strict=True):
                before_view = time.perf_counter_ns()
                found.append(detector.detect(image))
                view_ms.append((time.perf_counter_ns()-before_view)/1e6)
            association_start = time.perf_counter_ns()
            now = row['host_receive_ns']/1e9
            for tracker, detections in zip(trackers, found, strict=True):
                tracker.update(detections, now)
            end = time.perf_counter_ns()
            processing_ms = (end-begin)/1e6
            if position >= args.warmup:
                timings.append(processing_ms)
            entry = dict(index=row['index'], processing_ms=processing_ms, image_decode_ms=image_decode_ms,
                         view_processing_ms=view_ms,
                         detector_audit=[getattr(d, 'last_audit', {}) for d in detectors],
                         association_ms=(end-association_start)/1e6, warmup=position < args.warmup)
            reference = truth['frames'].get(str(row['index']))
            for side, detections in zip(dataset.views, found, strict=True):
                entry[side] = [detection_json(d) for d in detections]
                if reference is not None:
                    score = score_frame(detections, reference[side]['tags'])
                    scores[side].append(score)
                    temporal[side].update(row['index'], row['host_receive_ns']/1e9,
                                          detections, reference[side]['tags'], score)
                    for match in score['matches']:
                        previous = previous_tracks[side].get(match['instance'])
                        if previous is not None and previous[1]+1 == row['index']:
                            comparisons[side] += 1
                            switches[side] += int(previous[0] != match['track'])
                        previous_tracks[side][match['instance']] = (match['track'], row['index'])
                    entry[f'{side}_score'] = score
            details.append(entry)
        report = dict(profile=profile, dataset_id=dataset.metadata['dataset_id'],
                      kind=dataset.metadata['kind'], scene=dataset.metadata.get('scene'),
                      distance_cm=dataset.metadata.get('distance_cm'), git_revision=revision,
                      detector_source_sha256=source_hashes,
                      view_names=dataset.views, pairing=dataset.metadata.get('pairing', 'hardware-paired'),
                      calibrated=calibrated, processed_pairs=len(details), annotated_pairs=len(truth['frames']),
                      input_checksum_coverage=sum(all(f'{side}_sha256' in row for side in dataset.views)
                                                  for row in dataset.records)/len(dataset.records),
                      recorded_delivery_hz=((len(dataset.records)-1)*1e9
                          /max(dataset.records[-1]['host_receive_ns']-dataset.records[0]['host_receive_ns'], 1)),
                      skipped_capture_frames=(None if dataset.metadata.get('frame_number_origin') == 'software_read_index'
                          else sum(max(0, b['frame_number']-a['frame_number']-1)
                                   for a, b in zip(dataset.records, dataset.records[1:], strict=False))),
                      annotation_coverage=len(truth['frames'])/len(dataset.records),
                      opencv=cv2.__version__, numpy=np.__version__, machine=platform.machine(),
                      opencv_threads=cv2.getNumThreads(), warmup_pairs=args.warmup,
                      timing_scope='paired detection + correction + tracking + validation; '
                                   'excludes disk decode, GT scoring, preview and capture; NOT sensor-to-display latency',
                      processing_latency=latency_summary(timings),
                      views={side: dict(summarize_scores(rows), **temporal[side].summary(),
                                       identity_switches=switches[side],
                                       identity_comparisons=comparisons[side],
                                       identity_switch_rate=switches[side]/comparisons[side] if comparisons[side] else None)
                             for side, rows in scores.items()},
                      truth_sha256=hashlib.sha256(Path(args.truth).read_bytes()).hexdigest())
        reports[profile] = report
        (output/f'{profile}.json').write_text(json.dumps(report, indent=2))
        (output/f'{profile}.frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in details))
        print(profile + ': ' + ' | '.join(
            f"{side} recall={report['views'][side]['recall']} FP={report['views'][side]['fp']}"
            for side in dataset.views) + f" | processing P95={report['processing_latency']['p95_ms']:.2f}ms", flush=True)

    baseline = reports.get('baseline')
    comparison = {}
    for profile, report in reports.items():
        comparison[profile] = dict(latency=report['processing_latency'], views=report['views'],
                                   latency_33ms_pass=report['processing_latency']['p95_ms'] <= 33.,
                                   real_50cm_validated=False)
        if baseline is not None:
            comparison[profile]['versus_baseline'] = {
                side: {key: (report['views'][side][key]-baseline['views'][side][key]
                             if report['views'][side][key] is not None
                             and baseline['views'][side][key] is not None else None)
                       for key in ('recall', 'false_discovery_rate', 'negative_frame_fp_rate')}
                for side in dataset.views}
    (output/'comparison.json').write_text(json.dumps(comparison, indent=2))
    lines = [f"Dataset kind: {dataset.metadata['kind']}; distance: {dataset.metadata.get('distance_cm')} cm",
             'This report alone does NOT establish real 50cm acceptance.',
             'Predictions and unconfirmed corrections are excluded from true recognition.',
             'Timing excludes input file decoding, GT scoring, recording and display.',
             'profile | per-view recall / FP / negative frames | P95 ms | <=33ms']
    for profile, report in reports.items():
        view_text = ' | '.join(f"{side}: recall={view['recall']}, FP={view['fp']}, "
                               f"negative FP={view['negative_fp']}/{view['negative_view_frames']}"
                               for side, view in report['views'].items())
        lines.append(f"{profile} | {view_text} | {report['processing_latency']['p95_ms']:.2f} | "
                     f"{comparison[profile]['latency_33ms_pass']}")
    lines.append(f"Annotated pairs: {len(truth['frames'])}/{len(dataset.records)}. "
                 'Sparse annotations cannot establish continuous-frame tracking stability.')
    (output/'comparison.txt').write_text('\n'.join(lines)+'\n')
    print(f'Reports and per-frame evidence saved in {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fixture = commands.add_parser('synthetic', help='generate a geometry-labelled smoke test, NOT real 50cm data')
    fixture.add_argument('dataset', type=Path)
    fixture.add_argument('--frames', type=int, default=45)
    fixture.add_argument('--full-sheet', action='store_true', help='108 analytic A4 tags for load testing')
    def make_fixture(args):
        from lib.tag_fixture import generate_fixture
        print(generate_fixture(args.dataset, args.frames, args.full_sheet))
    fixture.set_defaults(run=make_fixture)
    capture = commands.add_parser('record', help='record raw paired IR PNGs and metadata')
    capture.add_argument('dataset', type=Path)
    capture.add_argument('--camera-config', default=str(Path(__file__).parents[1]/'config/realsense_sheet_max.yaml'))
    capture.add_argument('--scene', choices=('positive', 'negative'), default='positive')
    capture.add_argument('--sheet', choices=('pen', 'wrist'), default='pen')
    capture.add_argument('--distance-cm', type=float, default=50.)
    capture.add_argument('--frames', type=int, default=None, help='optional fixed-size batch; omit for a long-lived keyboard session')
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
                            default=None)
    evaluation.add_argument('--warmup', type=int, default=10)
    evaluation.add_argument('--opencv-threads', type=int, help='apply the same OpenCV thread count to every profile')
    evaluation.set_defaults(run=evaluate)
    inspection = commands.add_parser('review', help='replay FP/FN frames with independent truth overlays')
    inspection.add_argument('dataset', type=Path)
    inspection.add_argument('--truth', type=Path, required=True)
    inspection.add_argument('--results', type=Path, required=True, help='a profile.frames.jsonl result file')
    inspection.add_argument('--only-errors', action='store_true')
    inspection.set_defaults(run=review)
    args = parser.parse_args()
    if (getattr(args, 'frames', None) is not None and args.frames <= 0) or getattr(args, 'stride', 1) <= 0:
        parser.error('frames and stride must be positive')
    if getattr(args, 'opencv_threads', None) is not None:
        if args.opencv_threads < 1:
            parser.error('opencv-threads must be positive')
        cv2.setNumThreads(args.opencv_threads)
    if getattr(args, 'start', 0) < 0:
        parser.error('start must be nonnegative')
    if hasattr(args, 'distance_cm') and (not np.isfinite(args.distance_cm) or args.distance_cm <= 0):
        parser.error('distance-cm must be finite and positive')
    args.run(args)


if __name__ == '__main__':
    main()
