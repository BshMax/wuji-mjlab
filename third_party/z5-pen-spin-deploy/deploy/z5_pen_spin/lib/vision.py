"""ArUco board PnP in calibrated palm and pen coordinate systems."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def quaternion_wxyz(matrix):
    """Unit quaternion from a rotation matrix, including rotations near pi."""
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    # Davenport matrix in xyzw order, followed by explicit wxyz conversion.
    k = (
        np.array(
            [
                [r[0, 0] - r[1, 1] - r[2, 2], r[0, 1] + r[1, 0], r[0, 2] + r[2, 0], r[2, 1] - r[1, 2]],
                [r[0, 1] + r[1, 0], r[1, 1] - r[0, 0] - r[2, 2], r[1, 2] + r[2, 1], r[0, 2] - r[2, 0]],
                [r[0, 2] + r[2, 0], r[1, 2] + r[2, 1], r[2, 2] - r[0, 0] - r[1, 1], r[1, 0] - r[0, 1]],
                [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
            ]
        )
        / 3
    )
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def relative_pen_pose(palm_rotation, palm_position, pen_rotation, pen_position):
    r = np.asarray(palm_rotation).T
    position = r @ (np.asarray(pen_position).reshape(3) - np.asarray(palm_position).reshape(3))
    return np.r_[position, quaternion_wxyz(r @ pen_rotation)]


def board_points(entries):
    result = {}
    for entry in entries:
        tag = int(entry["id"])
        points = np.asarray(entry["corners_m"], dtype=np.float64)
        if points.shape != (4, 3) or not np.isfinite(points).all() or tag in result:
            raise ValueError("each board tag needs a unique ID and four finite XYZ corners in meters")
        if np.linalg.matrix_rank(points[1:] - points[0]) < 2:
            raise ValueError("marker corners must span a plane")
        result[tag] = points
    if not result:
        raise ValueError("fill in measured pen_markers and palm_markers in the camera configuration")
    return result


class PenObserver:
    def __init__(self, config_path):
        import cv2

        self.cv = cv2
        self.config = json.loads(Path(config_path).read_text())
        c = self.config
        self.camera = np.asarray(c["camera_matrix"], dtype=np.float64)
        self.distortion = np.asarray(c["distortion"], dtype=np.float64)
        if (
            self.camera.shape != (3, 3)
            or not np.isfinite(self.camera).all()
            or self.camera[0, 0] <= 0
            or self.camera[1, 1] <= 0
        ):
            raise ValueError("camera_matrix must contain measured camera intrinsics")
        if self.distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(self.distortion).all():
            raise ValueError("invalid camera distortion coefficients")
        self.image_size = tuple(c["image_size"])
        self.pen = board_points(c["pen_markers"])
        self.palm = board_points(c["palm_markers"])
        self.detectors = {}
        for name in (c["pen_dictionary"], c["palm_dictionary"]):
            dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detectors[name] = cv2.aruco.ArucoDetector(dictionary, params)

    def _solve(self, board, detection):
        cv = self.cv
        corners, ids, _ = detection
        if ids is None:
            return None
        selected = [
            (board[int(tag)], pts.reshape(4, 2))
            for tag, pts in zip(ids.ravel(), corners, strict=True)
            if int(tag) in board
        ]
        if not selected:
            return None
        obj = np.concatenate([v[0] for v in selected])
        img = np.concatenate([v[1] for v in selected]).astype(np.float64)
        ok, rv, tv = cv.solvePnP(obj, img, self.camera, self.distortion, flags=cv.SOLVEPNP_SQPNP)
        if not ok:
            return None
        rv, tv = cv.solvePnPRefineLM(obj, img, self.camera, self.distortion, rv, tv)
        r = cv.Rodrigues(rv)[0]
        if np.any((obj @ r.T + tv.reshape(3))[:, 2] <= 0):
            return None
        projected = cv.projectPoints(obj, rv, tv, self.camera, self.distortion)[0].reshape(-1, 2)
        error = float(np.linalg.norm(projected - img, axis=1).mean())
        if not np.isfinite(error) or error > self.config["max_reprojection_error_px"]:
            return None
        return r, tv.reshape(3), error

    def observe(self, frame, timestamp):
        if (frame.shape[1], frame.shape[0]) != self.image_size:
            raise ValueError(f"camera resolution differs from calibrated image_size={self.image_size}")
        gray = self.cv.cvtColor(frame, self.cv.COLOR_BGR2GRAY)
        detections = {name: detector.detectMarkers(gray) for name, detector in self.detectors.items()}
        palm = self._solve(self.palm, detections[self.config["palm_dictionary"]])
        pen = self._solve(self.pen, detections[self.config["pen_dictionary"]])
        message = dict(timestamp=timestamp, frame="palm_link", quaternion_order="wxyz", valid=False)
        if palm is not None and pen is not None:
            pose = relative_pen_pose(palm[0], palm[1], pen[0], pen[1])
            message.update(
                valid=True,
                pen={"position": pose[:3].tolist(), "quaternion": pose[3:].tolist()},
                reprojection_error_px={"palm": palm[2], "pen": pen[2]},
            )
        return message
