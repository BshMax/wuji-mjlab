# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
import cv2
import numpy as np
from lib.pen_pose import PenPoseEstimator
from lib.sheet_experiments import ExperimentalSheetDetector


def artwork(estimator, tid=0):
    core = cv2.aruco.generateImageMarker(estimator.layout.dictionary, tid, 120)
    tile = np.zeros((160, 160), np.uint8)
    tile[20:140, 20:140] = 255
    tile[40:120, 40:120] = 255-core[20:100, 20:100]
    return np.pad(tile, 15, constant_values=255)


def test_current_image_decoder_preserves_source_id_rotation_and_dictionary():
    estimator = PenPoseEstimator(np.eye(3)*638, np.zeros(5))
    detector = ExperimentalSheetDetector(estimator, 'corners')
    before = estimator.layout.dictionary.bytesList.copy()
    quad = np.float32([[35, 35], [155, 35], [155, 155], [35, 155]])
    for tid in range(18):
        image = artwork(estimator, tid)
        for rotation in range(4):
            rotated = np.rot90(image, rotation).copy()
            result = detector.decode(rotated, quad)
            assert result is not None and result.marker_id == tid
            assert result.corrected_bits == 0 and result.source == 'decoded_exact'
            np.testing.assert_allclose(result.corners, np.roll(quad, rotation, axis=0))
    np.testing.assert_array_equal(before, estimator.layout.dictionary.bytesList)
    assert estimator.layout.dictionary.maxCorrectionBits == 0


def test_flat_and_wrong_border_images_do_not_become_source_tags():
    estimator = PenPoseEstimator(np.eye(3)*638, np.zeros(5))
    detector = ExperimentalSheetDetector(estimator)
    quad = np.float32([[35, 35], [155, 35], [155, 155], [35, 155]])
    assert detector.decode(np.full((190, 190), 127, np.uint8), quad) is None
    assert detector.decode(255-artwork(estimator), quad) is None


def test_tiny_candidate_refinement_stays_bounded():
    estimator = PenPoseEstimator(np.eye(3)*638, np.zeros(5))
    image = artwork(estimator)
    quad = np.float32([[35, 35], [155, 35], [155, 155], [35, 155]])
    refined = ExperimentalSheetDetector.refine(image, quad)
    assert np.max(np.linalg.norm(refined-quad, axis=1)) <= 1.25
