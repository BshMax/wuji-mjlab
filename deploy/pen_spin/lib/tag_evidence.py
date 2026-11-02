# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Measurement evidence adapter compatible with legacy preview detections."""
from dataclasses import dataclass

from lib.tag_sheet_preview import TagDetection


@dataclass
class EvidenceDetection(TagDetection):
    source: str = 'decoded'
    corrected_bits: int = 0
    decode_distance: int | None = None
    quality: dict | None = None
    confirmed: bool = True

    @property
    def verified(self):
        return self.confirmed and self.source in {
            'decoded', 'decoded_exact', 'decoded_corrected', 'tracked_redecoded'}


def as_evidence(detection):
    if isinstance(detection, EvidenceDetection):
        return detection
    return EvidenceDetection(detection.family, detection.marker_id, detection.corners,
                             track_id=detection.track_id,
                             source=getattr(detection, 'source', 'decoded'),
                             corrected_bits=getattr(detection, 'corrected_bits', 0),
                             decode_distance=getattr(detection, 'decode_distance', None),
                             quality=getattr(detection, 'quality', None),
                             confirmed=getattr(detection, 'confirmed', True))
