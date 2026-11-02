# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Opt-in A4 detector experiments. The real-pen pose path remains unchanged.

Every accepted rescue is decoded from the current image. Confidence values are
image quality measurements, not calibrated probabilities. Baseline dictionary
and source artwork are never mutated.
"""
from __future__ import annotations

import math
import time
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
        self.base.pen.batch_border_check = True
        # This experiment targets ~17-pixel codes at 50 cm. Six threshold
        # windows repeatedly generate the same dense A4 candidates; keep 3/13
        # here and expose the full baseline as a separate comparison profile.
        parameters = estimator._detector.getDetectorParameters()
        parameters.adaptiveThreshWinSizeMax = 13
        estimator._detector = cv2.aruco.ArucoDetector(estimator.layout.dictionary, parameters)
        self.expected_px = float(estimator.K[0, 0]*estimator.layout.tag_size/.5)
        patterns = np.asarray([p['bits'] for p in estimator.layout.custom_dictionary['patterns']], np.uint8)
        self.codes = np.asarray([np.rot90(bits, rotation).ravel()
                                 for bits in patterns for rotation in range(4)])
        self.rejections = Counter()
        self.last_retries = 0
        self.prior_decodes = []
        self.previous_gray = None
        self.previous = []
        self.frame_index = 0
        self.last_input_time = None
        self.last_audit = {}

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
        pens = [old for old in accepted if old.family == 'PEN']
        if not pens:
            return False
        centers = np.asarray([old.center for old in pens])
        sizes = np.asarray([old.size_px for old in pens])
        return bool(np.any(np.linalg.norm(centers-detection.center, axis=1)
                           < .45*np.maximum(sizes, detection.size_px)))

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
            if self.last_retries >= 12:
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

    def _detect_global(self, gray, roi=None):
        if roi is None:
            found = [as_evidence(d) for d in self.base.detect(gray)]
        else:
            x, y, width, height = roi
            corners, ids, _ = self.base.april.detectMarkers(np.ascontiguousarray(gray[y:y+height, x:x+width]))
            found = [] if ids is None else [
                TagDetection('APR36h11', int(tid), quad[0].astype(float)+[x, y])
                for tid, quad in zip(ids.flatten(), corners, strict=True)]
            ids, quads = self.base.pen.detect(gray, roi=roi)
            found.extend(TagDetection('PEN', tid, quad, source='tracked_redecoded')
                         for tid, quad in zip(ids, quads, strict=True))
        pens = [d for d in found if d.family == 'PEN']
        if pens:
            points = np.float32(np.concatenate([d.corners for d in pens]))
            refined = points.copy()
            try:
                cv2.cornerSubPix(gray, refined, (1, 1), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, .03))
            except cv2.error:
                refined = points
            for detection, quad, original in zip(pens, refined.reshape(-1, 4, 2),
                                                   points.reshape(-1, 4, 2), strict=True):
                if (np.isfinite(quad).all() and cv2.isContourConvex(quad)
                        and np.max(np.linalg.norm(quad-original, axis=1)) <= 1.25):
                    detection.corners = quad.astype(float)
                if roi is None:
                    detection.source = 'decoded_exact'
                detection.decode_distance = 0
        return self._rescue(gray, found)

    def _flow_candidates(self, gray):
        if self.previous_gray is None or not self.previous:
            return []
        points = np.float32(np.concatenate([d.corners for d in self.previous])).reshape(-1, 1, 2)
        options = dict(winSize=(15, 15), maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, .03))
        moved, valid, error = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, points, None, **options)
        if moved is None or valid is None:
            return []
        back, reverse_valid, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, moved, None, **options)
        if back is None or reverse_valid is None:
            return []
        consistency = np.linalg.norm(back-points, axis=2).ravel()
        result = []
        for index, old in enumerate(self.previous):
            indices = slice(index*4, index*4+4)
            quad = moved[indices, 0].astype(float)
            if (not valid[indices].all() or not reverse_valid[indices].all()
                    or not np.isfinite(quad).all() or np.max(consistency[indices]) > .7
                    or np.mean(error[indices]) > 20 or not cv2.isContourConvex(quad.astype(np.float32))
                    or np.any(quad < 2) or np.any(quad[:, 0] >= gray.shape[1]-2)
                    or np.any(quad[:, 1] >= gray.shape[0]-2)):
                self.rejections['flow_consistency'] += 1
                continue
            area_ratio = abs(cv2.contourArea(quad.astype(np.float32)))/max(
                abs(cv2.contourArea(old.corners.astype(np.float32))), 1e-6)
            if not .65 <= area_ratio <= 1.55:
                self.rejections['flow_geometry'] += 1
                continue
            age = (old.quality or {}).get('prediction_age_frames', 0)+1
            # A spatial proposal is NOT evidence of a current-frame decode.
            result.append(TagDetection('PEN', old.marker_id, quad, source='tracked_only',
                                       confirmed=False, quality=dict(prediction_age_frames=age)))
        return result

    def detect(self, gray):
        started = time.perf_counter()
        input_time = getattr(self, 'input_time_s', time.monotonic())
        if self.last_input_time is not None and (input_time-self.last_input_time > .1
                                                 or input_time <= self.last_input_time):
            self.previous = []
            self.prior_decodes = []
        self.last_input_time = input_time
        self.rejections.clear()
        self.frame_index += 1
        flow = self._flow_candidates(gray) if self.profile == 'tracked' else []
        flow_end = time.perf_counter()
        full_scan = (self.profile != 'tracked' or self.frame_index % 3 == 1
                     or not flow or len(flow) < .98*len(self.previous))
        roi = None
        if not full_scan:
            points = np.concatenate([d.corners for d in flow])
            lo = np.maximum(np.floor(points.min(axis=0)-24).astype(int), 0)
            hi = np.minimum(np.ceil(points.max(axis=0)+24).astype(int), gray.shape[::-1])
            roi = (int(lo[0]), int(lo[1]), int(hi[0]-lo[0]), int(hi[1]-lo[1]))
        result = self._detect_global(gray, roi)
        flow_retries = 0
        for proposal in flow:
            if self._overlaps(proposal, result):
                continue
            if flow_retries >= 12:
                if proposal.quality['prediction_age_frames'] <= 2:
                    result.append(proposal)
                self.rejections['flow_decode_budget'] += 1
                continue
            flow_retries += 1
            decoded = self.decode(gray, self.refine(gray, proposal.corners), max_errors=1)
            if decoded is not None and decoded.marker_id == proposal.marker_id:
                if decoded.corrected_bits == 0:
                    decoded.source = 'tracked_redecoded'
                result.append(decoded)
            elif decoded is None and proposal.quality['prediction_age_frames'] <= 2:
                result.append(proposal)
                self.rejections['flow_without_decode'] += 1
            else:
                self.rejections['flow_id_changed_or_expired'] += 1
        confirmed = self._confirm(result)
        if self.profile == 'tracked':
            self.previous_gray = gray.copy()
            self.previous = [d for d in confirmed if d.family == 'PEN'
                             and (d.verified or d.source == 'tracked_only')]
        self.last_audit = dict(full_scan=full_scan, roi=roi, batch_border_check=True,
                               threshold_windows=[3, 13], rescue_limit=12, rescue_attempts=self.last_retries,
                               flow_decode_attempts=flow_retries,
                               rejections=dict(self.rejections),
                               flow_ms=(flow_end-started)*1000,
                               total_view_ms=(time.perf_counter()-started)*1000)
        return confirmed
