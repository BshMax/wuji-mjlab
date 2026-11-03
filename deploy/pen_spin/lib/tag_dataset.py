# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Lossless paired-frame datasets and explicitly reviewed ground truth.

Raw frames, not drawn previews, are stored. A bounded writer queue fails rather
than silently dropping frames. Annotations are independent of detector outputs.
"""
from __future__ import annotations

import hashlib
import json
import queue
import threading
from pathlib import Path

import cv2
import numpy as np


class PairRecorder:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.views = metadata.get('views', ['left', 'right'])
        if self.views not in (['left'], ['left', 'right']):
            raise ValueError('Dataset views must be left or left+right; mono is never duplicated')
        for side in self.views:
            (self.directory / side).mkdir()
        self.metadata = dict(schema_version=1, complete=False, **metadata)
        self._write_metadata()
        self.queue = queue.Queue(maxsize=16)
        self.error = None
        self.count = 0
        self.closed = False
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write_metadata(self):
        (self.directory / 'session.json').write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')

    def append(self, left, right, metadata, *, wait=False):
        if self.closed:
            raise RuntimeError('Recorder is closed')
        if self.error is not None:
            raise RuntimeError(f'Dataset writer failed: {self.error}') from self.error
        if 'right' in self.views and right is None:
            raise ValueError('A paired recording needs an actual right image')
        if 'right' not in self.views and right is not None:
            raise ValueError('Mono recording must not contain a fabricated second view')
        record = dict(metadata, index=self.count)
        try:
            item = (left.copy(), None if right is None else right.copy(), record)
            if wait:
                self.queue.put(item, timeout=30)  # offline fixture generation only
            else:
                self.queue.put_nowait(item)
        except queue.Full as exc:
            raise RuntimeError('Lossless writer queue full: disk/encoding cannot sustain capture; '
                               'dataset remains incomplete, no silent frame loss') from exc
        self.count += 1

    def _write(self):
        try:
            with (self.directory / 'frames.jsonl').open('x', encoding='utf-8') as manifest:
                while True:
                    item = self.queue.get()
                    if item is None:
                        return
                    left, right, record = item
                    images = dict(left=left, right=right)
                    for side in self.views:
                        image = images[side]
                        if image.ndim != 2 or image.dtype != np.uint8:
                            raise ValueError('Dataset images must be raw Y8 grayscale')
                        relative = f'{side}/{record["index"]:06d}.png'
                        ok, encoded = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        if not ok:
                            raise OSError(f'Failed to write {relative}')
                        data = encoded.tobytes()
                        (self.directory / relative).write_bytes(data)
                        record[side] = relative
                        record[f'{side}_sha256'] = hashlib.sha256(data).hexdigest()
                    manifest.write(json.dumps(record) + '\n')
        except Exception as exc:
            self.error = exc

    def close(self, complete=True):
        if self.closed:
            return
        self.closed = True
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=.1)
                break
            except queue.Full:
                continue
        self.thread.join()
        self.metadata.update(complete=bool(complete and self.error is None), frame_count=self.count)
        self._write_metadata()
        if self.error is not None:
            raise RuntimeError(f'Dataset writer failed: {self.error}') from self.error


class PairDataset:
    def __init__(self, directory):
        self.directory = Path(directory)
        if (self.directory/'capture_session.json').exists() and not (self.directory/'session.json').exists():
            raise ValueError('Select a recorded clip such as clip_0001, not the capture-session root folder')
        self.metadata = json.loads((self.directory / 'session.json').read_text())
        if self.metadata.get('schema_version') != 1 or not self.metadata.get('complete'):
            raise ValueError('Unsupported or incomplete dataset; do not evaluate a partial capture')
        self.views = self.metadata.get('views', ['left', 'right'])
        if self.views not in (['left'], ['left', 'right']):
            raise ValueError('Unsupported dataset view layout')
        self.records = [json.loads(line) for line in (self.directory / 'frames.jsonl').read_text().splitlines()]
        if not self.records or len(self.records) != self.metadata['frame_count']:
            raise ValueError('Empty dataset or frame manifest length mismatch')
        if [r['index'] for r in self.records] != list(range(len(self.records))):
            raise ValueError('Frame indices must be unique and consecutive')

    def images(self, record):
        result = []
        for side in self.views:
            path = (self.directory / record[side]).resolve()
            if not path.is_relative_to(self.directory.resolve()):
                raise ValueError('Image path escapes dataset')
            data = path.read_bytes()
            expected = record.get(f'{side}_sha256')
            if expected is not None and hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f'Image checksum mismatch: {path}')
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f'Missing/corrupt image: {path}')
            if image.ndim != 2 or image.dtype != np.uint8:
                raise ValueError('Recorded source must be raw Y8, not converted RGB or 16-bit data')
            result.append(image)
        if (len(result) == 2 and result[0].shape != result[1].shape
                and self.metadata.get('pairing', 'hardware-paired') == 'hardware-paired'):
            raise ValueError('Stereo image dimensions differ')
        return result


def load_truth(dataset, path):
    truth = json.loads(Path(path).read_text())
    if truth.get('schema_version') != 1:
        raise ValueError('Unsupported annotation schema')
    if truth.get('dataset_id') != dataset.metadata['dataset_id']:
        raise ValueError('Ground truth belongs to a different dataset')
    if truth.get('source') not in ('manual', 'synthetic_geometry'):
        raise ValueError('Ground truth must be manually reviewed or independently generated geometry')
    if truth.get('source') == 'synthetic_geometry' and dataset.metadata.get('kind') != 'synthetic':
        raise ValueError('Synthetic truth cannot validate a real recording')
    entries = truth.get('frames', {})
    for key, views in entries.items():
        if not key.isdigit() or not 0 <= int(key) < len(dataset.records):
            raise ValueError('Annotation frame index outside dataset')
        for side in dataset.views:
            if side not in views or not views[side].get('reviewed'):
                raise ValueError(f'Annotation {key}/{side} is unreviewed')
            tags = views[side].get('tags', [])
            if not tags and not views[side].get('negative_confirmed'):
                raise ValueError('An empty frame must be explicitly confirmed as negative')
            instances = [tag['instance'] for tag in tags]
            if len(instances) != len(set(instances)):
                raise ValueError('Ground-truth instance IDs must be unique within each view')
            for tag in tags:
                quad = np.asarray(tag['corners'], dtype=np.float32)
                if (quad.shape != (4, 2) or not np.isfinite(quad).all()
                        or not cv2.isContourConvex(quad) or abs(cv2.contourArea(quad)) < 1):
                    raise ValueError('Ground-truth corners must form a finite convex quadrilateral')
    if not entries:
        raise ValueError('No reviewed annotations; missing annotations are NOT negative frames')
    return truth
