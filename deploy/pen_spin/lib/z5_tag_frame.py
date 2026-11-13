# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Transform wrist-tag poses into the Unitree Z5 ``palm_link`` frame.

Mirror of lib/tag_frame.py for the Unitree Z5 hand (unitree_z5/z5_right_mujoco.xml).
Positions use metres and quaternions are (w, x, y, z).

palm_link frame convention (verified against the MJCF and palm_link.STL):
  origin : at the wrist end of the palm (mesh spans y -3.5 mm .. +99.8 mm)
  +Y     : toward the fingers (finger bases sit at y +40..+90 mm)
  +Z     : palm normal — the palm faces +Z (thumb side +X, right hand)
  -Y     : the wrist side; the wrist AprilTag must mount out here

The wrist AprilTag (id 0, 50.4 mm black square) rides on the printed wrist
mount from `-assembly.step1.step` (unitree_z5/wrist_tag_mount.STL): a forearm
strap plate + tall tower ending in an open square tube, with the tag lying flat
on the tower's top face. Constants below are derived from that CAD plus a photo
of the worn rig (palm down on the desk, tag facing up on the dorsal side); see
the comments for the assumptions a ruler check should confirm.
"""

import numpy as np

from lib.math_utils import normalize_quat, quat_apply, quat_inv, quat_mul

# The MJCF's fixed root_to_palm joint (rpy pi/2, 0, pi/2) — palm_link pose in
# the wrist-flange (root_link) frame. The origins coincide, so only the
# rotation matters when a mount measurement is taken at the flange instead of
# the palm frame.
Z5_PALM_IN_ROOT_QUAT = np.array([0.4999981634, 0.5, 0.5000018366, 0.5])
Z5_ROOT_IN_PALM_QUAT = quat_inv(Z5_PALM_IN_ROOT_QUAT)

# Wrist-tag pose in the Z5 palm_link frame, derived from the real printed
# wrist mount `-assembly.step1.step` (exported 2026-11-11; STL copy at
# unitree_z5/wrist_tag_mount.STL, units mm -> m).
#
# CAD facts (mm): tag pad on the tower top face at (-109.4, 0, 95.07), flat
# face normal +Z_cad. The wrist-flange interface is the mount's +X contact
# face at x=-55.7 with four M-screw holes at (y -7.37/+10.63, z 60.07/80.07) —
# the palm origin sits at the 4-hole centre on that face: (-55.7, +1.63,
# +70.07). Tag pad relative to that origin: (-53.7, -1.63, +25.0).
# CAD -> root_link axis map (photo-verified on the worn rig: palm UP, tag on
# the VOLAR side of the forearm facing the camera, mount body toward the
# elbow, tag X_w toward the elbow / Y_w toward the thumb):
#   x_cad -> +Z_root (fingers),  y_cad -> -Y_root,  z_cad -> +X_root (volar).
# Code-frame note: the observer's "wrist tag" axes are the OpenCV tag frame
# rotated 180 deg about Z (WRIST_AXES_IN_OPENCV_TAG), i.e. X_w -> -Y_palm
# (elbow), Y_w -> +X_palm (thumb), Z_w -> +Z_palm (volar/palm normal).
#
# Ruler check on the worn rig: tag centre -> wrist end-face distance along
# the arm ~54 mm, height above the flange centre ~25 mm, |d| ~59 mm.
Z5_TAG_IN_PALM_POS = np.array([0.00163, -0.0537, 0.0250])
Z5_TAG_IN_PALM_QUAT = np.array([0.7071067811865476, 0.0, 0.0, -0.7071067811865476])

# Previous estimate derived from the hand2 jig CAD (before the real mount STEP
# arrived), kept for comparison: tag normal +Z_palm, |d| = 51 mm.
#   POS  = [0.0, -0.0347, +0.0374]
#   QUAT = [0.7071, 0.0, 0.0, -0.7071]


def tag_in_root_to_palm(pos_root, quat_root):
    """Express a pose measured in the root_link (flange) frame in palm_link."""
    return (
        quat_apply(Z5_ROOT_IN_PALM_QUAT, np.asarray(pos_root, dtype=np.float64)),
        normalize_quat(quat_mul(Z5_ROOT_IN_PALM_QUAT, np.asarray(quat_root))),
    )


def pen_in_z5_palm(
    pen_pos_tag: np.ndarray, pen_quat_tag: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Lift a tag-frame pen pose into the Z5 palm_link frame.

    Drop-in Z5 counterpart of tag_frame.pen_in_palm — the Z5 policy's
    object_local_pos / object_local_quat_wxyz observation.
    """
    pos = Z5_TAG_IN_PALM_POS + quat_apply(
        Z5_TAG_IN_PALM_QUAT, np.asarray(pen_pos_tag, dtype=np.float64)
    )
    quat = normalize_quat(
        quat_mul(Z5_TAG_IN_PALM_QUAT, np.asarray(pen_quat_tag, dtype=np.float64))
    )
    return pos, quat
