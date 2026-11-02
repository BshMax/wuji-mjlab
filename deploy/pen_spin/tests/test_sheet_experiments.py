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


def test_one_bit_repair_requires_uncertain_cell_and_temporal_confirmation():
    estimator = PenPoseEstimator(np.eye(3)*638, np.zeros(5))
    detector = ExperimentalSheetDetector(estimator, 'ecc1')
    original = artwork(estimator)
    pattern = np.asarray(estimator.layout.custom_dictionary['patterns'][0]['bits'])
    quad = np.float32([[35, 35], [155, 35], [155, 155], [35, 155]])
    accepted = None
    for index in range(16):
        row, col = divmod(index, 4)
        image = original.copy()
        image[55+row*20:75+row*20, 55+col*20:75+col*20] = 140 if pattern[row, col] else 110
        assert detector.decode(image, quad, max_errors=0) is None
        result = detector.decode(image, quad, max_errors=1)
        if result is not None:
            accepted = result
            assert result.marker_id == 0 and result.corrected_bits == 1
            assert result.source == 'decoded_corrected'
            first = detector._confirm([result])[0]
            assert not first.verified
            second = detector.decode(image, quad, max_errors=1)
            assert detector._confirm([second])[0].verified
            # A confidently opposite printed bit must not be repaired.
            image[55+row*20:75+row*20, 55+col*20:75+col*20] = 255 if pattern[row, col] else 0
            assert detector.decode(image, quad, max_errors=1) is None
            break
    assert accepted is not None


def test_flow_assisted_search_redecodes_current_image():
    estimator = PenPoseEstimator(np.eye(3)*638, np.zeros(5))
    detector = ExperimentalSheetDetector(estimator, 'tracked')
    image = artwork(estimator)
    detector.input_time_s = 1.
    first = detector.detect(image)
    assert any(d.marker_id == 0 and d.verified for d in first)
    moved = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 1]]), (190, 190), borderValue=255)
    detector.input_time_s = 1.+1/30
    later = detector.detect(moved)
    assert any(d.marker_id == 0 and d.source == 'tracked_redecoded' and d.verified for d in later)


def test_prediction_without_decode_is_explicit_and_expires(monkeypatch):
    estimator = PenPoseEstimator(np.eye(3)*638, np.zeros(5))
    detector = ExperimentalSheetDetector(estimator, 'tracked')
    image = artwork(estimator)
    detector.input_time_s = 1.
    detector.detect(image)
    monkeypatch.setattr(detector, 'decode', lambda *args, **kwargs: None)
    monkeypatch.setattr(detector, '_detect_global', lambda _image, roi=None: [])
    detector.base.pen.last_rejected_quads = []
    for frame in (1, 2):
        detector.input_time_s = 1.+frame/30
        found = detector.detect(image)
        assert found and all(d.source == 'tracked_only' and not d.verified for d in found)
    detector.input_time_s = 1.+3/30
    assert detector.detect(image) == []


def test_batch_border_check_is_identical_to_per_marker_predicate():
    estimator = PenPoseEstimator(np.eye(3)*638, np.zeros(5))
    quad = np.float32([[35, 35], [155, 35], [155, 155], [35, 155]])
    rng = np.random.default_rng(45)
    for tid in range(18):
        original = artwork(estimator, tid)
        for image in (original, 255-original,
                      np.clip(original.astype(float)+rng.normal(0, 12, original.shape), 0, 255).astype(np.uint8)):
            expected = estimator._source_border_matches(image, quad, tid)
            assert estimator._source_border_mask(image, [quad], [tid])[0] == expected
