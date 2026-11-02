# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Opt-in A4 detector experiments. The real-pen pose path remains unchanged.

Every accepted rescue is decoded from the current image. Confidence values are
image quality measurements, not calibrated probabilities. Baseline dictionary
and source artwork are never mutated.
"""
from __future__ import annotations

import math
from collections import Counter

import cv2
import numpy as np

from lib.tag_evidence import EvidenceDetection as TagDetection
from lib.tag_evidence import as_evidence
from lib.tag_sheet_preview import TagSheetDetector


class ExperimentalSheetDetector:
    def __init__(self, estimator, profile='corners'):
        if profile not in ('corners', 'ecc1', 'tracked'):
            raise ValueError('Unknown experimental sheet detector profile')
        self.profile = profile
        self.base = TagSheetDetector(estimator)
        self.base.pen.collect_rejected = True
        self.expected_px = float(estimator.K[0, 0]*estimator.layout.tag_size/.5)
        patterns = np.asarray([p['bits'] for p in estimator.layout.custom_dictionary['patterns']], np.uint8)
        self.codes = np.asarray([np.rot90(bits, rotation).ravel()
                                 for bits in patterns for rotation in range(4)])
        self.rejections = Counter()
        self.last_retries = 0
        self.prior_decodes = []

    @staticmethod
    def refine(gray, corners):
        quad = np.asarray(corners, np.float32).copy()
        # At 50 cm each data cell spans only a few pixels. A one-pixel window
        # avoids pulling the white border toward an internal data-cell corner.
        try:
            cv2.cornerSubPix(gray, quad, (1, 1), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, .03))
        except cv2.error:
            return np.asarray(corners, float)
        if (not np.isfinite(quad).all() or not cv2.isContourConvex(quad)
                or np.max(np.linalg.norm(quad-corners, axis=1)) > 1.25):
            return np.asarray(corners, float)
        return quad.astype(float)

    def decode(self, gray, corners, max_errors=0):
        """Resample 8x8 printed cells, preserving black outer/white inner rings."""
        quad = np.asarray(corners, np.float32)
        if not cv2.isContourConvex(quad):
            return None
        H = cv2.getPerspectiveTransform(quad, np.float32([[8, 8], [56, 8], [56, 56], [8, 56]]))
        patch = cv2.warpPerspective(gray, H, (64, 64))
        cells = patch.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3)
        values = np.median(cells[:, :, 2:6, 2:6], axis=(2, 3))
        outer = np.r_[values[0], values[-1], values[1:-1, 0], values[1:-1, -1]]
        inner = np.r_[values[1, 1:7], values[6, 1:7], values[2:6, 1], values[2:6, 6]]
        white, black = np.median(inner), np.median(outer)
        contrast = float(white-black)
        threshold = (white+black)/2
        if contrast < 20 or np.mean(inner > threshold) < .85 or np.mean(outer < threshold) < .85:
            self.rejections['border_or_contrast'] += 1
            return None
        data = values[2:6, 2:6].ravel()
        bits = data < threshold  # source codebook logical 1 is printed BLACK
        distances = np.count_nonzero(self.codes != bits, axis=1)
        order = np.argsort(distances, kind='stable')
        best, runner_up = int(distances[order[0]]), int(distances[order[1]])
        if best > max_errors:
            self.rejections['code_distance'] += 1
            return None
        margin = np.abs(data-threshold)/contrast
        if best == 0 and np.min(margin) < .05:
            self.rejections['uncertain_exact_bit'] += 1
            return None
        if best == 1:
            mismatch = self.codes[order[0]] != bits
            # Only repair an uncertain cell, never a clearly printed opposite
            # bit. Reject near-competing codes and weak unmodified cells.
            if (contrast < 30 or runner_up-best < 2 or np.max(margin[mismatch]) > .30
                    or np.min(margin[~mismatch]) < .10):
                self.rejections['ecc_quality_or_ambiguity'] += 1
                return None
        tid, rotation = divmod(int(order[0]), 4)
        canonical = np.roll(np.asarray(corners, float), rotation, axis=0)
        detection = TagDetection('PEN', tid, canonical,
                                 source='decoded_corrected' if best else 'decoded_exact',
                                 corrected_bits=best, decode_distance=best)
        detection.quality = dict(contrast=contrast, runner_up_distance=runner_up,
                                 minimum_bit_margin=float(np.min(margin)))
        return detection

    @staticmethod
    def _overlaps(detection, accepted):
        return any(np.linalg.norm(detection.center-old.center) < .45*max(detection.size_px, old.size_px)
                   for old in accepted if old.family == 'PEN')

    def _rescue(self, gray, accepted):
        candidates = []
        for quad in self.base.pen.last_rejected_quads:
            edges = np.linalg.norm(np.roll(quad, -1, axis=0)-quad, axis=1)
            size = float(edges.mean())
            if (max(8., self.expected_px*.5) <= size <= self.expected_px*2.5
                    and min(edges) > 0 and max(edges)/min(edges) < 2.5):
                candidates.append((abs(math.log(size/self.expected_px)), quad))
        self.last_retries = 0
        for _, quad in sorted(candidates, key=lambda item: item[0]):
            placeholder = TagDetection('PEN', -1, quad)
            if self._overlaps(placeholder, accepted):
                continue
            if self.last_retries >= 24:
                break
            self.last_retries += 1
            refined = self.refine(gray, quad)
            detection = self.decode(gray, refined, max_errors=int(self.profile in ('ecc1', 'tracked')))
            if detection is not None and not self._overlaps(detection, accepted):
                accepted.append(detection)
        return accepted

    def _confirm(self, detections):
        for detection in detections:
            if detection.source == 'decoded_corrected':
                detection.confirmed = any(
                    old.marker_id == detection.marker_id
                    and np.linalg.norm(old.center-detection.center) < max(5., detection.size_px*.75)
                    for old in self.prior_decodes)
        # A tracked-only box never provides evidence for ECC confirmation.
        self.prior_decodes = [d for d in detections if d.family == 'PEN' and d.source != 'tracked_only']
        return detections

    def detect(self, gray):
        found = [as_evidence(d) for d in self.base.detect(gray)]
        result = []
        for detection in found:
            if detection.family == 'PEN':
                refined = self.refine(gray, detection.corners)
                checked = self.decode(gray, refined)
                if checked is not None and checked.marker_id == detection.marker_id:
                    detection = checked
                else:
                    detection.source = 'decoded_exact'  # retain baseline's exact current-frame decode
            result.append(detection)
        return self._confirm(self._rescue(gray, result))
