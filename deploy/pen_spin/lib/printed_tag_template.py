# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Analytic millimetre geometry of print_tracking_tags.py's two A4 pages.

No detector outputs are used to create this reference. Paper corner annotations
must follow the PRINTED page's top-left, top-right, bottom-right, bottom-left.
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def paper_tags(sheet):
    if sheet == 'wrist':
        result = []
        for row in range(4):
            for col in range(3):
                x, y = 16+col*63.5, 30+row*63
                result.append(dict(instance=f'wrist:{row}:{col}', family='APR36h11', id=0,
                                   corners=np.array([[x, y], [x+50.4, y],
                                                     [x+50.4, y+50.4], [x, y+50.4]])))
        return result
    if sheet != 'pen':
        raise ValueError('Page template must be pen or wrist')
    path = Path(__file__).resolve().parents[1] / 'config/pen_tags.json'
    layout = json.loads(path.read_text())
    groups = defaultdict(list)
    for pattern in layout['custom_dictionary']['patterns']:
        groups[pattern['texture']].append(pattern)
    faces = []
    for patterns in groups.values():
        width = 18*(1+max(p['column'] for p in patterns))
        height = 18*(1+max(p['row'] for p in patterns))
        tag = layout['tags'][str(patterns[0]['id'])]
        corners = np.asarray(tag['corners_m'])
        right, up = corners[1]-corners[0], corners[0]-corners[3]
        turns = 0
        if width > height:
            width, height = height, width
            right, up = up, -right
            turns = 1
        flip = height == 36 and up[2] < 0
        face_tags = []
        for pattern in patterns:
            x, y = pattern['column']*18+2.25, pattern['row']*18+2.25
            quad = np.array([[x, y], [x+13.5, y], [x+13.5, y+13.5], [x, y+13.5]])
            if turns:
                quad = np.column_stack((18-quad[:, 1], quad[:, 0]))
            if flip:
                quad = np.array([width, height])-quad
            face_tags.append((pattern['id'], quad))
        faces.append((height, face_tags))
    sides = [face for face in faces if face[0] == 36]
    caps = [face for face in faces if face[0] == 18]
    result = []
    for copy in range(6):
        left, top = 11+(copy % 3)*63.5, 31+(copy // 3)*122
        for index, (_height, tags) in enumerate(sides):
            offset = np.array([left+(index % 3)*20, top+3+(index // 3)*39])
            for tid, quad in tags:
                result.append(dict(instance=f'pen:{copy}:{tid}', family='PEN', id=tid,
                                   corners=quad+offset))
        for index, (_height, tags) in enumerate(caps):
            offset = np.array([left+40, top+3+78+index*20])
            for tid, quad in tags:
                result.append(dict(instance=f'pen:{copy}:{tid}', family='PEN', id=tid,
                                   corners=quad+offset))
    return result


def project_paper(sheet, paper_corners, image_shape, excluded=()):
    quad = np.asarray(paper_corners, np.float32)
    if quad.shape != (4, 2) or not cv2.isContourConvex(quad):
        raise ValueError('Select all four paper corners in printed-page order')
    matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [210, 0], [210, 297], [0, 297]]), quad)
    result = []
    for tag in paper_tags(sheet):
        pixels = cv2.perspectiveTransform(tag['corners'][None].astype(np.float32), matrix)[0]
        visible = (tag['instance'] not in excluded and np.all(pixels >= 0)
                   and np.all(pixels[:, 0] < image_shape[1])
                   and np.all(pixels[:, 1] < image_shape[0]))
        result.append(dict(instance=tag['instance'], family=tag['family'], id=tag['id'],
                           corners=pixels.tolist(), visible=bool(visible)))
    return result
