# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Long-lived preview and key-controlled, individually finalized capture clips."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from lib.camera_config import load_camera_config
from lib.camera_sources import open_camera
from lib.tag_dataset import PairRecorder
from lib.tag_evidence import as_evidence
from lib.tag_sheet_preview import TagSheetDetector, TagTracker


def make_2d_estimator(metadata, side):
    from lib.pen_pose import PenPoseEstimator
    matrix, distortion = metadata.get(f'{side}_K'), metadata.get(f'{side}_dist')
    # Only the image detector is called. No identity matrix is exported as a
    # calibration and no pose may be computed for an uncalibrated source.
    return PenPoseEstimator(np.eye(3) if matrix is None else matrix,
                             np.zeros(5) if distortion is None else distortion)


class RecordingSession:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        header = self.root/'capture_session.json'
        if not header.exists():
            if any(self.root.iterdir()):
                raise ValueError('Choose an empty session folder; existing flat datasets are not overwritten')
            header.write_text(json.dumps(dict(schema_version=1, application='tag_capture_session'), indent=2))
        elif json.loads(header.read_text()).get('application') != 'tag_capture_session':
            raise ValueError('Folder is not a compatible capture session')
        self.writer = None
        self.clip = None
        self.events = (self.root/'events.jsonl').open('a', encoding='utf-8')

    def event(self, action, **details):
        self.events.write(json.dumps(dict(t_monotonic_s=time.monotonic(), action=action, **details))+'\n')
        self.events.flush()

    def start(self, metadata):
        if self.writer is not None:
            return
        number = 1
        while (self.root/f'clip_{number:04d}').exists():
            number += 1
        self.clip = self.root/f'clip_{number:04d}'
        self.writer = PairRecorder(self.clip, dict(metadata, dataset_id=str(uuid.uuid4())))
        self.event('record_start', clip=self.clip.name, backend=metadata['backend'], scene=metadata['scene'])

    def append(self, frame, **details):
        if self.writer is not None:
            self.writer.append(frame.images['left'], frame.images.get('right'), dict(frame.metadata, **details))

    def stop(self, reason='operator_pause', complete=True):
        if self.writer is None:
            return
        writer, self.writer = self.writer, None
        writer.metadata['stop_reason'] = reason
        writer.close(complete=complete)
        self.event('record_stop', clip=self.clip.name, frames=writer.count,
                   complete=writer.metadata['complete'], reason=reason)
        print(f'Saved {self.clip}: {writer.count} frames, complete={writer.metadata["complete"]}', flush=True)

    def close(self):
        self.stop('operator_exit')
        self.events.close()


def build_preview_detectors(description, profile):
    detectors, trackers = {}, {}
    for side in description['views']:
        estimator = make_2d_estimator(description, side)
        if profile == 'baseline':
            detectors[side] = TagSheetDetector(estimator)
        else:
            if description.get(f'{side}_K') is None:
                raise ValueError('Experimental profiles need supplied calibration; '
                                 'uncalibrated cameras can use baseline 2D detection')
            from lib.sheet_experiments import ExperimentalSheetDetector
            detectors[side] = ExperimentalSheetDetector(estimator, profile)
        trackers[side] = TagTracker()
    return detectors, trackers


def begin_clip(session, camera, provenance, args, scene, sheet, profile, detect):
    session.start({**provenance, **camera.describe(), 'kind': 'real', 'scene': scene,
                   'sheet': sheet, 'distance_cm': args.distance_cm,
                   'distance_source': 'operator_reported', 'preview_detection': detect,
                   'detector_profile': profile, 'expected_frames': args.frames,
                   'timing_note': 'Device and host clocks are not assumed aligned'})


def run_capture(args, provenance):
    from lib.preview_ui import configure_qt_fonts
    if args.frames is None:
        args.preview = True
    if args.preview:
        configure_qt_fonts()
        cv2.namedWindow('camera recording session', cv2.WINDOW_NORMAL)
    session = RecordingSession(args.dataset)
    scene, sheet, profile = args.scene, args.sheet, 'baseline'
    detect = False
    stop = False
    print('Window stays open. s record; p save/pause; n next clip; d detection; m algorithm; '
          '1 positive/0 negative; 2 pen/3 wrist; +/- exposure; k reload camera config; q save/quit', flush=True)
    try:
        while not stop:
            try:
                config = load_camera_config(args.camera_config)
                with open_camera(config) as camera:
                    description = camera.describe()
                    views = description['views']
                    if views not in (['left'], ['left', 'right']):
                        raise ValueError('Adapter views must be left or left+right')
                    detectors, trackers = build_preview_detectors(description, profile)
                    session.event('camera_open', description=description)
                    if args.frames is not None:
                        begin_clip(session, camera, provenance, args, scene, sheet, profile, detect)
                    for frame in camera.frames():
                        if set(frame.images) != set(views):
                            raise ValueError('Adapter frame views differ from its description')
                        if any(image.ndim != 2 or image.dtype != np.uint8 for image in frame.images.values()):
                            raise ValueError('Adapter must provide 8-bit grayscale images')
                        before = time.perf_counter_ns()
                        detections = {}
                        if detect:
                            for side in views:
                                detectors[side].input_time_s = frame.metadata['host_receive_ns']/1e9
                                found = [as_evidence(d) for d in detectors[side].detect(frame.images[side])]
                                trackers[side].update(found, frame.metadata['host_receive_ns']/1e9)
                                detections[side] = found
                        processing_ms = (time.perf_counter_ns()-before)/1e6
                        serialized = {side: [dict(id=d.marker_id, family=d.family, source=d.source,
                                                 confirmed=d.confirmed, corrected_bits=d.corrected_bits,
                                                 corners=d.corners.tolist()) for d in found]
                                      for side, found in detections.items()}
                        session.append(frame, preview_processing_ms=processing_ms, live_detections=serialized)
                        if args.frames is not None and session.writer is not None and session.writer.count >= args.frames:
                            session.stop('requested_frame_count')
                            stop = True
                            break
                        if not args.preview:
                            continue
                        panels = []
                        for side in views:
                            image = cv2.cvtColor(frame.images[side], cv2.COLOR_GRAY2BGR)
                            for detection in detections.get(side, []):
                                color = (0, 255, 0) if detection.verified else (0, 165, 255)
                                cv2.polylines(image, [detection.corners.astype(int)], True, color, 1)
                                anchor = tuple(np.maximum(detection.corners[0].astype(int), [0, 15]))
                                cv2.putText(image, f'{detection.family}:{detection.marker_id}', anchor,
                                            cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
                            panels.append(image)
                        height = min(p.shape[0] for p in panels)
                        image = np.hstack([cv2.resize(p, (round(p.shape[1]*height/p.shape[0]), height))
                                           if p.shape[0] != height else p for p in panels])
                        count = 0 if session.writer is None else session.writer.count
                        state = 'RECORDING' if session.writer is not None else 'PREVIEW / PAUSED'
                        text = [f'{state} | {scene}/{sheet} | {description["backend"]} | '
                                f'{len(views)} view(s) | {count} saved | {profile} detect={detect}',
                                's record | p save/pause | n next | d detect | m algorithm | '
                                '1 positive 0 negative | 2 pen 3 wrist | k camera reload | q exit',
                                f'{description["pairing"]} | preview detection={processing_ms:.1f}ms | +/- exposure']
                        for row, message in enumerate(text):
                            cv2.rectangle(image, (0, row*24), (image.shape[1], (row+1)*24), (0, 0, 0), -1)
                            cv2.putText(image, message, (8, row*24+17), cv2.FONT_HERSHEY_SIMPLEX,
                                        .44, (255, 255, 255), 1)
                        cv2.imshow('camera recording session', image)
                        width = min(1440, image.shape[1])
                        cv2.resizeWindow('camera recording session', width, round(image.shape[0]*width/image.shape[1]))
                        key = cv2.pollKey() & 0xFF
                        if key in (ord('q'), 27) or cv2.getWindowProperty('camera recording session', cv2.WND_PROP_VISIBLE) < 1:
                            session.stop('operator_exit')
                            stop = True
                            break
                        if key == ord('s'):
                            begin_clip(session, camera, provenance, args, scene, sheet, profile, detect)
                        if key == ord('p'):
                            session.stop()
                        if key == ord('n'):
                            session.stop('next_clip')
                            begin_clip(session, camera, provenance, args, scene, sheet, profile, detect)
                        if key in (ord('0'), ord('1'), ord('2'), ord('3')):
                            session.stop('reference_changed')
                            if key in (ord('0'), ord('1')):
                                scene = 'negative' if key == ord('0') else 'positive'
                            else:
                                sheet = 'pen' if key == ord('2') else 'wrist'
                        if key == ord('d'):
                            session.stop('preview_load_changed')
                            detect = not detect
                            detectors, trackers = build_preview_detectors(description, profile)
                        if key == ord('m'):
                            session.stop('algorithm_changed')
                            options = ['baseline', 'corners', 'ecc1', 'tracked']
                            profile = options[(options.index(profile)+1) % len(options)]
                            try:
                                detectors, trackers = build_preview_detectors(description, profile)
                            except ValueError as exc:
                                profile = 'baseline'
                                detectors, trackers = build_preview_detectors(description, profile)
                                print(exc, flush=True)
                        if key in (ord('+'), ord('='), ord('-')):
                            session.stop('exposure_changed')
                            try:
                                camera.adjust_exposure(.5 if key == ord('-') else 2.)
                            except (RuntimeError, ValueError) as exc:
                                print(f'Exposure unchanged: {exc}', flush=True)
                        if key == ord('k'):
                            session.stop('camera_reload')
                            profile = 'baseline'
                            break
            except (RuntimeError, ValueError, OSError, ImportError, TypeError) as exc:
                session.stop('camera_or_recording_error', complete=False)
                session.event('error', message=str(exc))
                if not args.preview:
                    raise
                if stop:
                    break
                print(f'{exc}\nFix device/config then press k; q exits.', flush=True)
                while True:
                    image = np.zeros((240, 1200, 3), np.uint8)
                    cv2.putText(image, str(exc)[:130], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
                    cv2.putText(image, 'Camera unavailable: k reload / q exit', (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)
                    cv2.imshow('camera recording session', image)
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('k'):
                        break
                    if key in (ord('q'), 27) or cv2.getWindowProperty('camera recording session', cv2.WND_PROP_VISIBLE) < 1:
                        stop = True
                        break
    except KeyboardInterrupt:
        session.stop('operator_interrupt')
    finally:
        session.close()
        if args.preview:
            cv2.destroyAllWindows()
