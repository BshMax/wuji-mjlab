# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Deterministic geometry-labelled smoke data; never real-distance acceptance."""
import json
import math

import cv2
import numpy as np

from lib.pen_pose import PenTagLayout
from lib.tag_dataset import PairRecorder


def generate_fixture(directory, frames=45):
    layout = PenTagLayout()
    artwork = []
    for tid in range(18):
        core = cv2.aruco.generateImageMarker(layout.dictionary, tid, 120)
        tile = np.zeros((160, 160), np.uint8)
        tile[20:140, 20:140] = 255
        tile[40:120, 40:120] = 255-core[20:100, 20:100]
        artwork.append((40+tile*(180/255)).astype(np.uint8))
    metadata = dict(dataset_id=f'synthetic-source-codes-v1:{frames}:{cv2.__version__}', kind='synthetic', scene='mixed',
                    distance_cm=None, fixture='18 small codes, blur/phase changes and empty tail; NOT full-sheet acceptance',
                    left_K=[[638., 0, 640.], [0, 638., 400.], [0, 0, 1]], left_dist=[0.]*5,
                    right_K=[[638., 0, 640.], [0, 638., 400.], [0, 0, 1]], right_dist=[0.]*5)
    recorder = PairRecorder(directory, metadata)
    truth = dict(schema_version=1, dataset_id=metadata['dataset_id'], source='synthetic_geometry', frames={})
    rng = np.random.default_rng(719)
    complete = False
    try:
        for frame in range(frames):
            images, views = [], {}
            negative = frame >= frames*4//5
            for side, shift in (('left', 0), ('right', -18)):
                canvas = np.full((800, 1280), 220, np.uint8)
                tags = []
                if not negative:
                    for tid, tile in enumerate(artwork):
                        x = 180+(tid % 6)*110+math.sin(frame*.2)*4+shift
                        y = 200+(tid // 6)*110+math.cos(frame*.15)*3
                        size = 19.+(frame % 7)*.7  # white detection edge roughly 14..18 px
                        target = np.float32([[x, y], [x+size, y], [x+size, y+size], [x, y+size]])
                        rotation = tid % 4
                        target = np.roll(target, rotation, axis=0)
                        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [159, 0], [159, 159], [0, 159]]), target)
                        # Local rendering avoids making the detector's output a reference.
                        offset = np.floor(target.min(axis=0)-4).astype(int)
                        local = H.copy()
                        local[0] -= offset[0]*local[2]
                        local[1] -= offset[1]*local[2]
                        patch = cv2.warpPerspective(tile, local, (40, 40), borderValue=220)
                        canvas[offset[1]:offset[1]+40, offset[0]:offset[0]+40] = patch
                        corners = cv2.perspectiveTransform(np.float32([[[19.5, 19.5], [139.5, 19.5],
                                                                      [139.5, 139.5], [19.5, 139.5]]]), H)[0]
                        tags.append(dict(instance=f'patch:{tid}', family='PEN', id=tid,
                                         corners=corners.tolist(), visible=True))
                    sigma = .25+(frame % 5)*.12
                    canvas = cv2.GaussianBlur(canvas, (0, 0), sigma)
                    canvas = np.clip(canvas.astype(float)+rng.normal(0, 2, canvas.shape), 0, 255).astype(np.uint8)
                images.append(canvas)
                views[side] = dict(reviewed=True, tags=tags, negative_confirmed=negative)
            truth['frames'][str(frame)] = views
            recorder.append(*images, dict(frame_number=frame, host_receive_ns=frame*33333333,
                                           timestamp_ms=frame*1000/30, timestamp_domain='synthetic',
                                           sync_delta_ms=0, actual_exposure_us=None), wait=True)
        complete = True
    finally:
        recorder.close(complete=complete)
    path = recorder.directory/'truth.json'
    path.write_text(json.dumps(truth, indent=2))
    return path
