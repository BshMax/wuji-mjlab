"""Live MuJoCo mirror of the real Z5 pen-spin deployment.

Shows, in one passive viewer:
  * the real hand, driven by DDS joint feedback (opaque)
  * the real pen, driven by the camera palm-link pose stream (opaque)
  * a translucent "ghost" pen at the current reference frame (orange,
    lifted 30 mm so it never hides inside the real pen)
  * the clip's full reference pen path (yellow spheres)
  * a fading trail of the observed pen (white spheres)

Frame sync comes from play_real's --frame-endpoint broadcast; without it the
ghost snaps to the reference frame nearest to the observed pen pose.

Runs in the unitree_deploy conda env (mujoco 3.15 + unitree_sdk2py + cyclonedds).
"""

from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import queue as pyqueue
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

BUNDLE = Path("third_party/z5-pen-spin-deploy/generated/deploy/z5-pen-spin")
DEPLOY = Path("third_party/z5-pen-spin-deploy/deploy/z5_pen_spin")
sys.path.insert(0, str(DEPLOY))

from lib.core import Bundle, rotation_wxyz  # noqa: E402
from lib.hand_driver import REVERSED_JOINTS, Z5HandDriver  # noqa: E402
from lib.transport import PenReceiver  # noqa: E402

GHOST_PEN_RGBA = np.array([0.15, 0.9, 1.0, 0.55])  # ghost pen in cyan, distinct from the yellow path
GHOST_HAND_RGBA = np.array([0.15, 0.9, 0.35, 0.30])  # optional translucent green reference hand
TRAIL_RGBA = np.array([1.0, 1.0, 1.0, 0.8])
PATH_RGBA = np.array([1.0, 0.85, 0.1, 0.9])
ERR_RGBA = np.array([1.0, 0.25, 0.2, 1.0])


GHOST_LIFT_M = 0.0  # default 0 so ghost pen sits at the true reference pose inside the fingers
GHOST_LIFT_DIR = np.array([0.0, 0.0, 1.0])


def _set_geom(geo, m, d, g, rgba, lift_m: float = 0.0, lift_dir=None):
    import mujoco

    gtype = int(m.geom_type[g])
    geo.type = gtype
    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        # mjvScene stores 2 entries per mesh (render mesh + convex hull).
        geo.dataid = 2 * int(m.geom_dataid[g])
    else:
        geo.dataid = int(m.geom_dataid[g])
    ldir = GHOST_LIFT_DIR if lift_dir is None else lift_dir
    geo.pos = d.geom_xpos[g] + lift_m * ldir
    geo.mat = d.geom_xmat[g].reshape(3, 3)
    s = m.geom_size[g]
    # MjModel stores [radius, half_length, 0] for cylinder/capsule, whereas
    # mjvGeom requires [radius_x, radius_y, half_length].
    if gtype in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        geo.size = np.array([s[0], s[0], s[1]], dtype=np.float64)
    else:
        geo.size = s
    geo.texid = -1  # no texture: otherwise ghost geoms inherit texture 0 (a tag pattern)
    geo.texuniform = 0
    geo.emission = 0.35  # slight glow so translucent geoms stay readable on the dark background
    geo.rgba = np.asarray(rgba, dtype=np.float64)


def _sphere(geo, pos, radius, rgba):
    import mujoco

    geo.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geo.dataid = -1
    geo.texid = -1
    geo.pos = np.asarray(pos, dtype=np.float64)
    geo.mat = np.eye(3)
    geo.size = np.array([radius, 0.0, 0.0])
    geo.emission = 0.35
    geo.rgba = np.asarray(rgba, dtype=np.float64)


def _palm_to_world(d, palm_id, pose):
    """(xyz + wxyz) palm_link pose -> world xyz + wxyz quat."""
    import mujoco

    r = d.xmat[palm_id].reshape(3, 3)
    world_r = r @ rotation_wxyz(np.asarray(pose[3:], dtype=np.float64))
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, world_r.ravel())
    return np.r_[d.xpos[palm_id] + r @ pose[:3], quat]


def _draw_top_bar(img: np.ndarray, text: str, rec_dot: bool = False) -> None:
    import cv2

    bar_h = 28
    overlay = img[:bar_h].copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], bar_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, img[:bar_h], 0.28, 0.0, img[:bar_h])
    x = 10
    if rec_dot:
        cv2.circle(img, (16, 14), 5, (0, 0, 255), -1, cv2.LINE_AA)
        x = 28
    cv2.putText(img, text, (x, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)


def _fit_panel(img: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    import cv2

    h0, w0 = img.shape[:2]
    if h0 == height and w0 == width:
        return img
    scale = min(width / w0, height / h0)
    rw, rh = max(2, int(round(w0 * scale))), max(2, int(round(h0 * scale)))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0, x0 = (height - rh) // 2, (width - rw) // 2
    canvas[y0:y0 + rh, x0:x0 + rw] = resized
    return canvas


class _RealSenseMatchedSource:
    """Background RealSense RGB grabber + AprilTag-calibrated pixel-exact MuJoCo camera."""

    def __init__(self, palm_pos_w: np.ndarray, palm_mat_w: np.ndarray,
                 panel_w: int = 640, panel_h: int = 480,
                 zoom: float = 1.5, serial: str = ""):
        import threading
        import cv2
        import mujoco
        import pyrealsense2 as rs

        # Parse Z5 wrist-tag calibration constants directly (zero lib/ module shadowing!)
        pen_spin_root = Path(__file__).resolve().parents[1]
        ns: dict = {"np": np}
        for rel, prefixes in (
            ("lib/tag_frame.py", ("WORLD_TAG_ID =", "WORLD_TAG_SIZE_M =", "WRIST_AXES_IN_OPENCV_TAG =")),
            ("lib/z5_tag_frame.py", ("Z5_TAG_IN_PALM_POS =", "Z5_TAG_IN_PALM_QUAT =")),
        ):
            for line in (pen_spin_root / rel).read_text().splitlines():
                s = line.strip()
                if any(s.startswith(pfx) for pfx in prefixes):
                    exec(s, ns)
        self.tag_in_palm_pos = np.asarray(ns["Z5_TAG_IN_PALM_POS"], dtype=np.float64)
        self.tag_in_palm_quat = np.asarray(ns["Z5_TAG_IN_PALM_QUAT"], dtype=np.float64)
        self.R_palm_tag = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(self.R_palm_tag, self.tag_in_palm_quat)
        self.R_palm_tag = self.R_palm_tag.reshape(3, 3)

        self.wrist_axes = np.asarray(ns["WRIST_AXES_IN_OPENCV_TAG"], dtype=np.float64)
        self.world_tag_id = int(ns["WORLD_TAG_ID"])
        self.world_tag_size = float(ns["WORLD_TAG_SIZE_M"])
        self.palm_pos_w = np.asarray(palm_pos_w, dtype=np.float64)
        self.palm_mat_w = np.asarray(palm_mat_w, dtype=np.float64)
        self.panel_w = int(panel_w)
        self.panel_h = int(panel_h)
        self.zoom = max(1.0, float(zoom))

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        self._pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        try:
            cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            prof = self._pipe.start(cfg)
        except Exception:
            cfg = rs.config()
            if serial:
                cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            prof = self._pipe.start(cfg)

        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.W, self.H = float(intr.width), float(intr.height)
        self.fx, self.fy = float(intr.fx), float(intr.fy)
        self.cx, self.cy = float(intr.ppx), float(intr.ppy)
        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.dist = np.zeros(5, dtype=np.float64)

        self.calibrated = False
        self.roi: tuple[int, int, int, int] = (0, 0, int(self.W), int(self.H))
        self.cam_pose: tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float] | None = None

        self._lock = threading.Lock()
        self._latest_bgr: np.ndarray | None = None
        self._stop = threading.Event()
        self._retry_counter = 0

        # Warm up auto-exposure & collect up to 12 tag samples for rock-solid median pose
        t_samples, r_samples = [], []
        for i in range(18):
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=1000)
                cf = frames.get_color_frame()
                if not cf:
                    continue
                bgr = np.asanyarray(cf.get_data()).copy()
                self._latest_bgr = bgr
                if i >= 5:
                    res = self._solve_tag(bgr)
                    if res is not None:
                        r_samples.append(res[0])
                        t_samples.append(res[1])
            except Exception:
                break

        if t_samples:
            self._commit_calibration(r_samples, t_samples)
        else:
            print(
                f"[realsense] WARN: AprilTag id={self.world_tag_id} not seen at startup "
                f"({int(self.W)}x{int(self.H)}); will auto-lock once visible.",
                flush=True,
            )

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _solve_tag(self, bgr: np.ndarray):
        import cv2

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or self.world_tag_id not in ids.flatten():
            return None
        quad = corners[list(ids.flatten()).index(self.world_tag_id)][0].astype(np.float64)
        h = self.world_tag_size / 2.0
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, quad, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        R_raw, _ = cv2.Rodrigues(rvec)
        return R_raw @ self.wrist_axes, tvec.reshape(3)

    def _commit_calibration(self, r_list: list[np.ndarray], t_list: list[np.ndarray]) -> None:
        t_rs_tag = np.median(np.stack(t_list, axis=0), axis=0)
        R_mean = np.mean(np.stack(r_list, axis=0), axis=0)
        U, _, Vt = np.linalg.svd(R_mean)
        R_rs_tag = U @ Vt
        if np.linalg.det(R_rs_tag) < 0:
            U[:, -1] *= -1.0
            R_rs_tag = U @ Vt

        R_tag_rs = R_rs_tag.T
        t_tag_rs = -R_tag_rs @ t_rs_tag
        R_palm_rs = self.R_palm_tag @ R_tag_rs
        t_palm_rs = self.tag_in_palm_pos + self.R_palm_tag @ t_tag_rs

        # Auto-center 4:3 zoom ROI on the Z5 palm + pen spinning zone
        p_hand_palm = np.array([0.0, 0.075, 0.025], dtype=np.float64)
        p_hand_rs = R_palm_rs.T @ (p_hand_palm - t_palm_rs)
        uc = self.fx * (p_hand_rs[0] / max(p_hand_rs[2], 1e-4)) + self.cx
        vc = self.fy * (p_hand_rs[1] / max(p_hand_rs[2], 1e-4)) + self.cy

        roi_h = min(self.H, self.H / self.zoom)
        roi_w = roi_h * (self.panel_w / self.panel_h)
        if roi_w > self.W:
            roi_w = self.W
            roi_h = roi_w * (self.panel_h / self.panel_w)
        x0 = int(round(np.clip(uc - roi_w / 2.0, 0.0, self.W - roi_w)))
        y0 = int(round(np.clip(vc - roi_h / 2.0, 0.0, self.H - roi_h)))
        x1 = int(min(self.W, round(x0 + roi_w)))
        y1 = int(min(self.H, round(y0 + roi_h)))
        rw, rh = max(2, x1 - x0), max(2, y1 - y0)
        self.roi = (x0, y0, x1, y1)

        sx, sy = self.panel_w / rw, self.panel_h / rh
        fx_c, fy_c = self.fx * sx, self.fy * sy
        cx_c, cy_c = (self.cx - x0) * sx, (self.cy - y0) * sy

        p_w_rs = self.palm_pos_w + self.palm_mat_w @ t_palm_rs
        R_w_rs_gl = (self.palm_mat_w @ R_palm_rs) @ np.diag([1.0, -1.0, -1.0])
        fwd = -R_w_rs_gl[:, 2]
        up = R_w_rs_gl[:, 1]
        self.cam_pose = (p_w_rs, fwd, up, fx_c, fy_c, cx_c, cy_c)
        self.calibrated = True
        print(
            f"[realsense] CALIBRATED ({int(self.W)}x{int(self.H)}, zoom={self.zoom:g}, "
            f"ROI={x0}:{x1},{y0}:{y1}, dist={np.linalg.norm(t_palm_rs):.2f}m, {len(t_list)} samples)",
            flush=True,
        )

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=500)
                cf = frames.get_color_frame()
                if cf:
                    bgr = np.asanyarray(cf.get_data()).copy()
                    with self._lock:
                        self._latest_bgr = bgr
            except Exception:
                continue

    def get_panel_and_camera(self):
        """Return (640x480 BGR image, calibrated cam_pose tuple or None)."""
        import cv2

        with self._lock:
            bgr = None if self._latest_bgr is None else self._latest_bgr.copy()
        if bgr is None:
            return None, self.cam_pose
        if not self.calibrated:
            self._retry_counter += 1
            if self._retry_counter % 15 == 1:
                res = self._solve_tag(bgr)
                if res is not None:
                    self._commit_calibration([res[0]], [res[1]])
        x0, y0, x1, y1 = self.roi
        cropped = bgr[y0:y1, x0:x1]
        panel = cv2.resize(cropped, (self.panel_w, self.panel_h), interpolation=cv2.INTER_AREA)
        return panel, self.cam_pose

    def close(self) -> None:
        self._stop.set()
        try:
            self._thread.join(timeout=1.5)
        except Exception:
            pass
        try:
            self._pipe.stop()
        except Exception:
            pass


def _apply_matched_glcamera(renderer, cam_pose, panel_w: int, panel_h: int) -> None:
    p_w_rs, fwd, up, fx_c, fy_c, cx_c, cy_c = cam_pose
    gc0 = renderer.scene.camera[0]
    znear = float(gc0.frustum_near)
    left = -znear * (cx_c / fx_c)
    right = znear * ((panel_w - cx_c) / fx_c)
    top = znear * (cy_c / fy_c)
    bottom = -znear * ((panel_h - cy_c) / fy_c)
    for i in range(2):
        gc = renderer.scene.camera[i]
        gc.pos[:] = p_w_rs
        gc.forward[:] = fwd
        gc.up[:] = up
        gc.frustum_center = 0.5 * (left + right)
        gc.frustum_width = 0.5 * (right - left)
        gc.frustum_top = top
        gc.frustum_bottom = bottom


def _recorder_worker(
    scene_path_str: str,
    ghost_lift_dir: np.ndarray,
    pen_gids: list[int],
    hand_gids: list[int],
    fixed_palm_pose: np.ndarray,
    joint_space: str,
    fps: float,
    msg_queue,
    realsense_cfg: dict | None = None,
) -> None:
    """Dedicated child process using MUJOCO_GL=egl to encode side-by-side 1280x480 H.264 MP4s.

    Running EGL in a spawned process avoids OpenGL/GLFW context conflicts with the
    parent process's interactive X11 MuJoCo passive viewer. When ``realsense_cfg``
    is enabled, simultaneously encodes ``<clip>_realsense.mp4`` pairing the color
    RealSense RGB stream with a pixel-aligned MuJoCo render from the exact
    AprilTag-calibrated RealSense viewpoint.
    """
    os.environ["MUJOCO_GL"] = "egl"
    import cv2
    import mujoco

    panel_w, panel_h = 640, 480
    model = mujoco.MjModel.from_binary_path(scene_path_str)
    data = mujoco.MjData(model)
    ghost = mujoco.MjData(model)
    if joint_space == "z5_mjcf" and model.nmocap > 0:
        for d in (data, ghost):
            d.mocap_pos[0] = fixed_palm_pose[:3]
            d.mocap_quat[0] = fixed_palm_pose[[6, 3, 4, 5]]
    mujoco.mj_forward(model, data)
    palm_id = model.body("robot/palm_link").id
    renderer = mujoco.Renderer(model, height=panel_h, width=panel_w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    rs_src = None
    if realsense_cfg and realsense_cfg.get("enabled"):
        try:
            rs_src = _RealSenseMatchedSource(
                data.xpos[palm_id].copy(),
                data.xmat[palm_id].reshape(3, 3).copy(),
                panel_w=panel_w,
                panel_h=panel_h,
                zoom=float(realsense_cfg.get("zoom", 1.5)),
                serial=str(realsense_cfg.get("serial", "")),
            )
        except Exception as exc:
            print(f"[realsense] FAILED to initialize RealSense ({exc}); continuing with Hikvision only.", flush=True)
            rs_src = None

    path_pts = np.empty((0, 3), dtype=np.float64)
    ffmpeg_proc = None
    out_path = ""
    frame_count = 0
    rs_ffmpeg_proc = None
    rs_out_path = ""
    rs_frame_count = 0
    last_left_bgr: np.ndarray | None = None

    def _spawn_ffmpeg(path_str: str):
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{panel_w * 2}x{panel_h}",
            "-r", f"{fps:g}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            path_str,
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _close_one(proc, path_str: str, n_frames: int, label: str):
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.flush()
                proc.stdin.close()
        except OSError:
            pass
        try:
            rc = proc.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
        dur = n_frames / max(fps, 1e-6)
        print(f"[{label}] SAVED {path_str} ({n_frames} frames, {dur:.1f}s, rc={rc})", flush=True)

    def _close_writer():
        nonlocal ffmpeg_proc, out_path, frame_count, rs_ffmpeg_proc, rs_out_path, rs_frame_count
        _close_one(ffmpeg_proc, out_path, frame_count, "recorder")
        ffmpeg_proc = None
        out_path = ""
        frame_count = 0
        _close_one(rs_ffmpeg_proc, rs_out_path, rs_frame_count, "recorder-rs")
        rs_ffmpeg_proc = None
        rs_out_path = ""
        rs_frame_count = 0

    def _open_writer(path_str: str):
        nonlocal ffmpeg_proc, out_path, frame_count, rs_ffmpeg_proc, rs_out_path, rs_frame_count
        _close_writer()
        ffmpeg_proc = _spawn_ffmpeg(path_str)
        out_path = path_str
        frame_count = 0
        print(f"[recorder] START -> {out_path}", flush=True)
        if rs_src is not None:
            p = Path(path_str)
            rs_path = str(p.with_name(f"{p.stem}_realsense{p.suffix}"))
            rs_ffmpeg_proc = _spawn_ffmpeg(rs_path)
            rs_out_path = rs_path
            rs_frame_count = 0
            print(f"[recorder-rs] START -> {rs_out_path}", flush=True)

    def _placeholder(msg: str) -> np.ndarray:
        img = np.full((panel_h, panel_w, 3), 24, dtype=np.uint8)
        cv2.putText(img, msg, (40, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
        return img

    try:
        while True:
            item = msg_queue.get()
            if not item:
                continue
            tag = item[0]
            if tag == "quit":
                break
            if tag == "path":
                path_pts = item[1]
            elif tag == "start":
                _open_writer(item[1])
            elif tag == "stop":
                _close_writer()
            elif tag == "frame":
                if ffmpeg_proc is None or ffmpeg_proc.stdin is None:
                    continue
                (
                    _,
                    qpos_real,
                    qpos_ghost,
                    cam_cfg,
                    cam_jpg,
                    left_hud,
                    right_hud,
                    trail_pts,
                    show_ghost_hand,
                    ghost_lift,
                ) = item

                if cam_jpg is not None:
                    arr = np.frombuffer(cam_jpg, dtype=np.uint8)
                    dec = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if dec is not None:
                        last_left_bgr = _fit_panel(dec, panel_w, panel_h)
                left = last_left_bgr.copy() if last_left_bgr is not None else _placeholder("Waiting for Hikvision camera (:5567)")

                data.qpos[:] = qpos_real
                mujoco.mj_forward(model, data)
                ghost.qpos[:] = qpos_ghost
                mujoco.mj_forward(model, ghost)

                cam.lookat[:] = cam_cfg[0]
                cam.distance = float(cam_cfg[1])
                cam.azimuth = float(cam_cfg[2])
                cam.elevation = float(cam_cfg[3])
                renderer.update_scene(data, camera=cam)

                scn = renderer.scene
                max_geom = len(scn.geoms)
                for g in pen_gids:
                    if scn.ngeom >= max_geom:
                        break
                    geo = scn.geoms[scn.ngeom]; scn.ngeom += 1
                    _set_geom(geo, model, ghost, g, GHOST_PEN_RGBA, lift_m=ghost_lift, lift_dir=ghost_lift_dir)
                if show_ghost_hand:
                    for g in hand_gids:
                        if scn.ngeom >= max_geom:
                            break
                        geo = scn.geoms[scn.ngeom]; scn.ngeom += 1
                        _set_geom(geo, model, ghost, g, GHOST_HAND_RGBA, lift_m=0.0, lift_dir=ghost_lift_dir)
                n_trail = len(trail_pts)
                for i, pt in enumerate(path_pts):
                    if scn.ngeom >= max_geom - n_trail - 8:
                        break
                    geo = scn.geoms[scn.ngeom]; scn.ngeom += 1
                    _sphere(geo, pt, 0.002, PATH_RGBA if i % 5 else np.r_[PATH_RGBA[:3], 0.45])
                for i, pt in enumerate(trail_pts):
                    if scn.ngeom >= max_geom:
                        break
                    alpha = 0.05 + 0.6 * (i + 1) / max(1, n_trail)
                    geo = scn.geoms[scn.ngeom]; scn.ngeom += 1
                    _sphere(geo, pt, 0.0025, np.r_[TRAIL_RGBA[:3], alpha])

                rgb = renderer.render()
                right = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                _draw_top_bar(left, left_hud, rec_dot=True)
                _draw_top_bar(right, right_hud, rec_dot=False)
                # Subtle 2px center divider between camera and MuJoCo views
                left[:, -1] = (90, 90, 90)
                right[:, 0] = (90, 90, 90)
                combined = np.hstack((left, right))
                try:
                    ffmpeg_proc.stdin.write(combined.tobytes())
                    frame_count += 1
                except BrokenPipeError:
                    print(f"[recorder] ffmpeg pipe broken for {out_path}", flush=True)
                    _close_writer()
                    continue

                # Simultaneous RealSense RGB + matched-viewpoint MuJoCo side-by-side video!
                if rs_src is not None and rs_ffmpeg_proc is not None and rs_ffmpeg_proc.stdin is not None:
                    rs_left, rs_cam_pose = rs_src.get_panel_and_camera()
                    if rs_left is None:
                        rs_left = _placeholder("Waiting for RealSense RGB...")
                    if rs_cam_pose is not None:
                        _apply_matched_glcamera(renderer, rs_cam_pose, panel_w, panel_h)
                        rs_rgb = renderer.render()
                        rs_right = cv2.cvtColor(rs_rgb, cv2.COLOR_RGB2BGR)
                        rs_right_hud = right_hud.replace("MuJoCo |", "MuJoCo (RS view) |")
                    else:
                        rs_right = right.copy()
                        rs_right_hud = right_hud.replace("MuJoCo |", "MuJoCo (waiting tag) |")
                    rs_left_hud = left_hud.replace("Hikvision", "RealSense RGB")
                    _draw_top_bar(rs_left, rs_left_hud, rec_dot=True)
                    _draw_top_bar(rs_right, rs_right_hud, rec_dot=False)
                    rs_left[:, -1] = (90, 90, 90)
                    rs_right[:, 0] = (90, 90, 90)
                    rs_combined = np.hstack((rs_left, rs_right))
                    try:
                        rs_ffmpeg_proc.stdin.write(rs_combined.tobytes())
                        rs_frame_count += 1
                    except BrokenPipeError:
                        print(f"[recorder-rs] ffmpeg pipe broken for {rs_out_path}", flush=True)
                        _close_one(rs_ffmpeg_proc, rs_out_path, rs_frame_count, "recorder-rs")
                        rs_ffmpeg_proc = None
    except KeyboardInterrupt:
        pass
    finally:
        _close_writer()
        if rs_src is not None:
            rs_src.close()
        try:
            renderer.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, default=BUNDLE)
    p.add_argument("--clip", help="clip id for the ghost/path (default: first clip)")
    p.add_argument("--pose-endpoint", default="tcp://127.0.0.1:5565")
    p.add_argument("--pose-max-age", type=float, default=0.5)
    p.add_argument("--frame-endpoint", default="tcp://127.0.0.1:5566", help="play_real --frame-endpoint broadcast")
    p.add_argument("--wire-convention", choices=("legacy", "urdf20261103"), default="urdf20261103")
    p.add_argument("--dds-interface")
    p.add_argument("--dds-domain", type=int, default=0)
    p.add_argument("--sdk-path", type=Path)
    p.add_argument("--no-hand", action="store_true", help="no DDS; hand stays at reference start pose")
    p.add_argument("--no-pen", action="store_true", help="no pose stream; pen tracks the reference")
    p.add_argument("--no-trail", action="store_true")
    p.add_argument("--trajectory-step", type=int, default=5, help="frames between path markers")
    p.add_argument("--trail-length", type=int, default=150)
    p.add_argument("--hz", type=float, default=60.0)
    p.add_argument("--pose-smooth", type=float, default=0.0,
                   help="EMA alpha for the displayed camera pen pose (0 = raw instant follow, default)")
    p.add_argument("--ghost-lift", type=float, default=GHOST_LIFT_M,
                   help="lift ghost pen along palm->pen normal in meters (0.0 = true reference pose)")
    p.add_argument("--ghost-hand", action="store_true",
                   help="also draw a translucent green reference hand at ref_q[ghost_frame]")
    p.add_argument("--freeze-frame", type=int, help="lock the ghost at this clip-relative frame (error-checking mode)")
    p.add_argument("--record-dir", type=Path, default=None,
                   help="directory for auto per-run side-by-side MP4s (Hikvision left + MuJoCo right)")
    p.add_argument("--cam-endpoint", default="tcp://127.0.0.1:5567",
                   help="pen_world_observer --video-pub-port ZMQ PUB endpoint")
    p.add_argument("--record-fps", type=float, default=30.0,
                   help="output side-by-side MP4 frame rate")
    p.add_argument("--record-all", action="store_true",
                   help="record continuously from startup instead of auto-splitting per play_real run")
    p.add_argument("--realsense", action="store_true",
                   help="simultaneously record color RealSense RGB + viewpoint-matched MuJoCo side-by-side MP4 (*_realsense.mp4)")
    p.add_argument("--realsense-zoom", type=float, default=1.5,
                   help="auto-centered 4:3 crop zoom around the Z5 hand for RealSense + matched MuJoCo (1.0 = full frame)")
    p.add_argument("--realsense-serial", type=str, default="",
                   help="optional Intel RealSense serial number when multiple devices are attached")
    p.add_argument("--replay", type=Path, default=None,
                   help="offline 3D MuJoCo replay of a recorded play_real.py .npz log (no hardware needed)")
    p.add_argument("--replay-speed", type=float, default=0.5,
                   help="playback speed multiplier for --replay (default 0.5x slow-motion)")
    args = p.parse_args()

    import mujoco
    import mujoco.viewer

    bundle = Bundle(args.bundle)
    replay_data = None
    if args.replay is not None:
        raw_log = np.load(args.replay, allow_pickle=True)
        replay_data = {
            "frame": np.asarray(raw_log["frame"], dtype=np.int64),
            "joint_pos": np.asarray(raw_log["joint_pos"], dtype=np.float64),
            "pen_pose": np.asarray(raw_log["pen_pose"], dtype=np.float64),
            "pos_err": np.asarray(raw_log["position_error_m"], dtype=np.float64),
            "axis_err": np.asarray(raw_log["axis_error_deg"], dtype=np.float64),
            "blind": np.asarray(raw_log["blind"], dtype=bool),
            "clip": str(raw_log["clip"]) if "clip" in raw_log else None,
        }
        if args.clip is None and replay_data["clip"] in bundle.clip_ids:
            args.clip = replay_data["clip"]
        args.no_hand = True
        args.no_pen = True
        args.frame_endpoint = ""
    clip = bundle.clip_index(args.clip)
    first, end = int(bundle.starts[clip]), int(bundle.ends[clip])
    ref_q = bundle.references["joint_pos"]
    ref_pen = bundle.references["object_pose"]

    scene_path = bundle.directory / "scene.mjb"
    if not scene_path.exists():
        for fallback in (
            bundle.directory.parent / "z5-pen-spin-v6b-antijam/scene.mjb",
            bundle.directory.parent / "z5-pen-spin/scene.mjb",
        ):
            if fallback.exists():
                scene_path = fallback
                break
    model = mujoco.MjModel.from_binary_path(str(scene_path))
    joint_space = bundle.config.get("joint_space", "z5_mjcf")
    # scene.mjb uses z5_mjcf joint signs; flip the 3 reversed joints for display when viewing a dex5s_urdf bundle.
    mjcf_signs = np.array(
        [-1.0 if (joint_space == "dex5s_urdf" and n in REVERSED_JOINTS) else 1.0 for n in bundle.joint_names],
        dtype=np.float64,
    )
    data = mujoco.MjData(model)  # real state
    ghost = mujoco.MjData(model)  # reference state
    joints = [model.joint("robot/" + n) for n in bundle.joint_names]
    qids = [int(j.qposadr[0]) for j in joints]
    palm_id = model.body("robot/palm_link").id
    pen_id = model.body("object/cube").id
    pen_qid = int(model.joint("object/cube_freejoint").qposadr[0])
    pen_gids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == pen_id]
    hand_gids = [
        g for g in range(model.ngeom)
        if model.body(int(model.geom_bodyid[g])).name.startswith("robot/")
        and int(model.geom_contype[g]) == 0
        and int(model.geom_conaffinity[g]) == 0
    ]

    fixed = np.asarray(bundle.config["fixed_palm_pose_world_xyzw"])
    global GHOST_LIFT_DIR
    for d in (data, ghost):
        if joint_space == "z5_mjcf":
            d.mocap_pos[0] = fixed[:3]
            d.mocap_quat[0] = fixed[[6, 3, 4, 5]]
        d.qpos[qids] = ref_q[first] * mjcf_signs
        mujoco.mj_forward(model, d)
    # Lift the ghost out of the hand along the mean palm->pen direction.
    lift_local = np.mean(ref_pen[:, :3], axis=0)
    lift_local /= np.linalg.norm(lift_local)
    GHOST_LIFT_DIR = data.xmat[palm_id].reshape(3, 3) @ lift_local
    # Only the pen is drawn as a ghost; hand geoms are not duplicated.

    driver = receiver = frame_sub = cam_sub = frame_ctx = None
    if not args.no_hand:
        driver = Z5HandDriver(
            bundle, convention=args.wire_convention, interface=args.dds_interface,
            domain=args.dds_domain, sdk_path=args.sdk_path,
        )
        print(f"[mirror] DDS hand state connected ({args.wire_convention})", flush=True)
    if not args.no_pen:
        receiver = PenReceiver(args.pose_endpoint, args.pose_max_age)
        print(f"[mirror] pose stream <- {args.pose_endpoint}", flush=True)
    if args.frame_endpoint or (args.record_dir is not None and args.cam_endpoint):
        import zmq

        frame_ctx = zmq.Context()
    if args.frame_endpoint:
        import zmq

        frame_sub = frame_ctx.socket(zmq.SUB)
        frame_sub.setsockopt(zmq.LINGER, 0)
        frame_sub.setsockopt(zmq.CONFLATE, 1)
        frame_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        frame_sub.connect(args.frame_endpoint)
        print(f"[mirror] frame sync <- {args.frame_endpoint}", flush=True)
    if args.record_dir is not None and args.cam_endpoint:
        import zmq

        cam_sub = frame_ctx.socket(zmq.SUB)
        cam_sub.setsockopt(zmq.LINGER, 0)
        cam_sub.setsockopt(zmq.RCVHWM, 4)
        cam_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        cam_sub.connect(args.cam_endpoint)
        print(f"[mirror] camera video <- {args.cam_endpoint}", flush=True)

    # Reference pen path in world (palm is a fixed mocap: recompute on clip change).
    def _path_pts(a, b):
        return np.array([
            data.xpos[palm_id] + data.xmat[palm_id].reshape(3, 3) @ ref_pen[f, :3]
            for f in range(a, b, max(1, args.trajectory_step))
        ])

    # Pen shaft axis (+X) per reference frame, for orientation-aware snapping.
    def _seg_axes(a, b):
        return np.stack([rotation_wxyz(ref_pen[f, 3:7])[:, 0] for f in range(a, b)])

    path_pts = _path_pts(first, end)
    seg_axes = _seg_axes(first, end)
    trail = deque(maxlen=args.trail_length)

    rec_queue = None
    rec_proc = None
    if args.record_dir is not None:
        args.record_dir.mkdir(parents=True, exist_ok=True)
        mp_ctx = mp.get_context("spawn")
        rec_queue = mp_ctx.Queue(maxsize=12)
        rs_cfg = {
            "enabled": bool(args.realsense),
            "zoom": float(args.realsense_zoom),
            "serial": str(args.realsense_serial),
        }
        rec_proc = mp_ctx.Process(
            target=_recorder_worker,
            args=(
                str(scene_path.resolve()),
                GHOST_LIFT_DIR.copy(),
                pen_gids,
                hand_gids,
                fixed.copy(),
                joint_space,
                float(args.record_fps),
                rec_queue,
                rs_cfg,
            ),
            daemon=False,
        )
        rec_proc.start()
        rec_queue.put(("path", path_pts.copy()))
        mode_desc = "continuous (--record-all)" if args.record_all else "auto per play_real run"
        rs_desc = f" + RealSense RGB (zoom={args.realsense_zoom:g}x)" if args.realsense else ""
        print(f"[mirror] side-by-side recorder -> {args.record_dir} ({mode_desc}, {args.record_fps:g} fps{rs_desc})", flush=True)

    def _rec_send_control(msg):
        if rec_queue is None:
            return
        for _ in range(16):
            try:
                rec_queue.put(msg, timeout=0.2)
                return
            except pyqueue.Full:
                try:
                    rec_queue.get_nowait()
                except pyqueue.Empty:
                    pass

    rec_active = False
    rec_clip_id: str | None = None
    rec_start_t = 0.0
    rec_last_frame_idx = -1
    rec_next_t = 0.0
    rec_period = 1.0 / max(float(args.record_fps), 1.0)
    last_cam_jpg: bytes | None = None

    def _start_rec(clip_name_str: str):
        nonlocal rec_active, rec_clip_id, rec_start_t, rec_last_frame_idx, rec_next_t
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short = clip_name_str.split("/")[-1]
        out_file = args.record_dir / f"{ts}_{short}.mp4"
        _rec_send_control(("start", str(out_file)))
        rec_active = True
        rec_clip_id = clip_name_str
        rec_start_t = time.monotonic()
        rec_last_frame_idx = -1
        rec_next_t = rec_start_t

    def _stop_rec():
        nonlocal rec_active, rec_clip_id, rec_last_frame_idx
        if rec_active:
            _rec_send_control(("stop",))
            rec_active = False
            rec_clip_id = None
            rec_last_frame_idx = -1

    if args.record_dir is not None and args.record_all:
        _start_rec(bundle.clip_ids[clip])

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.lookat[:] = data.xpos[palm_id]
    viewer.cam.distance = 0.4
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25
    us = viewer.user_scn

    stale_pen = True
    last_pose = ref_pen[first].copy()
    smooth_pose = last_pose.copy()
    ghost_frame = first
    last_info = {}
    last_info_t = -np.inf  # monotonic time of the newest broadcast message
    hud_next = 0.0
    period = (bundle.dt / max(float(args.replay_speed), 0.05)) if replay_data is not None else (1.0 / args.hz)
    replay_idx = 0
    clip_len = end - first
    if replay_data is not None:
        n_rep = len(replay_data["frame"])
        print(f"[mirror] REPLAY {args.replay} ({n_rep} ticks = {n_rep * bundle.dt:.2f}s @ {args.replay_speed:g}x, looping)", flush=True)
    print(f"[mirror] clip={bundle.clip_ids[clip]} frames={clip_len} ({clip_len * bundle.dt:.1f}s)", flush=True)
    print("[mirror] running - close the window or Ctrl-C to quit", flush=True)
    try:
        while viewer.is_running():
            t0 = time.monotonic()
            if replay_data is not None:
                n_rep = len(replay_data["frame"])
                if replay_idx >= n_rep:
                    replay_idx = 0
                    trail.clear()
                data.qpos[qids] = replay_data["joint_pos"][replay_idx] * mjcf_signs
                smooth_pose = replay_data["pen_pose"][replay_idx].copy()
                stale_pen = False
                ghost_frame = int(replay_data["frame"][replay_idx])
                last_info = {
                    "state": "BLIND" if replay_data["blind"][replay_idx] else f"replay {replay_idx * bundle.dt:.2f}s",
                    "frame": ghost_frame,
                    "pos_err": float(replay_data["pos_err"][replay_idx]),
                    "axis_err": float(replay_data["axis_err"][replay_idx]),
                }
                last_info_t = time.monotonic()
                replay_idx += 1
            else:
                if driver is not None:
                    q, dq = driver.read()
                    data.qpos[qids] = q * mjcf_signs
                was_stale_pen = stale_pen
                if receiver is not None:
                    try:
                        last_pose = receiver.latest()
                        stale_pen = False
                    except TimeoutError:
                        stale_pen = True
                else:
                    last_pose = ref_pen[ghost_frame]
                # Display-only EMA: default 0.0 (instant follow); also snap immediately after dropped frames.
                a = min(max(args.pose_smooth, 0.0), 1.0)
                if stale_pen or was_stale_pen or a <= 0.0:
                    smooth_pose = last_pose.copy()
                else:
                    q = last_pose[3:7].copy()
                    if np.dot(q, smooth_pose[3:7]) < 0.0:
                        q = -q
                    smooth_pose[:3] += a * (last_pose[:3] - smooth_pose[:3])
                    nq = smooth_pose[3:7] + a * (q - smooth_pose[3:7])
                    smooth_pose[3:7] = nq / np.linalg.norm(nq)
            data.qpos[pen_qid:pen_qid + 7] = _palm_to_world(data, palm_id, smooth_pose)
            mujoco.mj_forward(model, data)

            # Frame sync: broadcast wins; otherwise nearest reference frame.
            if frame_sub is not None:
                import zmq

                while True:
                    try:
                        last_info = json.loads(frame_sub.recv_string(flags=zmq.NOBLOCK))
                        last_info_t = time.monotonic()
                    except zmq.Again:
                        break
            if cam_sub is not None:
                import zmq

                while True:
                    try:
                        parts = cam_sub.recv_multipart(flags=zmq.NOBLOCK)
                        if len(parts) >= 2:
                            last_cam_jpg = parts[-1]
                    except zmq.Again:
                        break
            frame_fresh = time.monotonic() - last_info_t < 0.5
            bclip = last_info.get("clip") if frame_fresh else None
            if bclip and bclip in bundle.clip_ids and bclip != bundle.clip_ids[clip]:
                clip = bundle.clip_index(bclip)
                first, end = int(bundle.starts[clip]), int(bundle.ends[clip])
                clip_len = end - first
                path_pts = _path_pts(first, end)
                seg_axes = _seg_axes(first, end)
                if rec_queue is not None:
                    _rec_send_control(("path", path_pts.copy()))
                print(f"[mirror] following broadcast clip -> {bclip} ({clip_len} frames)", flush=True)
            if args.freeze_frame is not None:
                ghost_frame = first + args.freeze_frame
            elif frame_fresh and "frame" in last_info:
                ghost_frame = int(last_info["frame"])
            elif receiver is not None and not stale_pen:
                # Snap by position AND shaft orientation (axis is unsigned);
                # position-only matching picks frames whose pose differs wildly.
                dpos = np.linalg.norm(ref_pen[first:end, :3] - last_pose[:3], axis=1)
                obs_ax = rotation_wxyz(np.asarray(last_pose[3:7], dtype=np.float64))[:, 0]
                daxis = np.arccos(np.clip(np.abs(seg_axes @ obs_ax), -1.0, 1.0))
                cost = dpos + 0.05 * daxis
                cand = int(np.argmin(cost))
                # Hysteresis keeps the ghost from hopping between distant
                # frames when no frame broadcast is available.
                cur = min(max(ghost_frame - first, 0), len(cost) - 1)
                if cost[cand] < cost[cur] - 0.005:
                    ghost_frame = first + cand
            ghost_frame = min(max(ghost_frame, first), end - 1)

            if args.ghost_hand:
                ghost.qpos[qids] = ref_q[ghost_frame] * mjcf_signs
            ghost.qpos[pen_qid:pen_qid + 7] = _palm_to_world(ghost, palm_id, ref_pen[ghost_frame])
            mujoco.mj_forward(model, ghost)

            if not args.no_trail and not stale_pen:
                trail.append(data.xpos[pen_id].copy())

            # Rebuild decorations.
            max_geom = len(us.geoms)
            us.ngeom = 0
            for g in pen_gids:
                if us.ngeom >= max_geom:
                    break
                geo = us.geoms[us.ngeom]; us.ngeom += 1
                _set_geom(geo, model, ghost, g, GHOST_PEN_RGBA, lift_m=args.ghost_lift)
            if args.ghost_hand:
                for g in hand_gids:
                    if us.ngeom >= max_geom:
                        break
                    geo = us.geoms[us.ngeom]; us.ngeom += 1
                    _set_geom(geo, model, ghost, g, GHOST_HAND_RGBA, lift_m=0.0)
            for i, pt in enumerate(path_pts):
                if us.ngeom >= max_geom - args.trail_length - 8:
                    break
                geo = us.geoms[us.ngeom]; us.ngeom += 1
                _sphere(geo, pt, 0.002, PATH_RGBA if i % 5 else np.r_[PATH_RGBA[:3], 0.45])
            n_trail = len(trail)
            for i, pt in enumerate(trail):
                if us.ngeom >= max_geom:
                    break
                a = 0.05 + 0.6 * (i + 1) / max(1, n_trail)
                geo = us.geoms[us.ngeom]; us.ngeom += 1
                _sphere(geo, pt, 0.0025, np.r_[TRAIL_RGBA[:3], a])

            viewer.sync()

            if rec_queue is not None:
                now_rec = time.monotonic()
                if not args.record_all:
                    bstate = last_info.get("state") if frame_fresh else None
                    cur_clip_id = bundle.clip_ids[clip]
                    silent_s = now_rec - last_info_t
                    rel_frame = ghost_frame - first
                    if rec_active:
                        if (
                            bstate in ("stopped", "move", "wait", "settle")
                            or silent_s > 1.2
                            or (bstate in ("hold", "play") and cur_clip_id != rec_clip_id)
                            or (bstate == "play" and rec_last_frame_idx >= 0 and rel_frame < rec_last_frame_idx - 20)
                        ):
                            _stop_rec()
                    if not rec_active and frame_fresh and bstate in ("hold", "play"):
                        _start_rec(cur_clip_id)

                if rec_active and now_rec >= rec_next_t:
                    rec_next_t = max(rec_next_t + rec_period, now_rec)
                    elapsed = now_rec - rec_start_t
                    short_clip = bundle.clip_ids[clip].split("/")[-1]
                    st_txt = last_info.get("state", "live") if frame_fresh else "live"
                    left_hud = f"REC {elapsed:5.1f}s | Hikvision | {short_clip} [{st_txt}]"
                    err_txt = (
                        f" err={last_info['pos_err'] * 1000:.0f}mm/{last_info['axis_err']:.1f}deg"
                        if (frame_fresh and "pos_err" in last_info)
                        else ""
                    )
                    right_hud = f"MuJoCo | f={ghost_frame - first:4d}/{clip_len}{err_txt}"
                    cam_cfg = (
                        viewer.cam.lookat.copy(),
                        float(viewer.cam.distance),
                        float(viewer.cam.azimuth),
                        float(viewer.cam.elevation),
                    )
                    trail_snap = np.array(trail, dtype=np.float64) if len(trail) else np.empty((0, 3), dtype=np.float64)
                    try:
                        rec_queue.put_nowait((
                            "frame",
                            data.qpos.copy(),
                            ghost.qpos.copy(),
                            cam_cfg,
                            last_cam_jpg,
                            left_hud,
                            right_hud,
                            trail_snap,
                            args.ghost_hand,
                            args.ghost_lift,
                        ))
                    except pyqueue.Full:
                        pass
                    if frame_fresh and last_info.get("state") == "play":
                        rec_last_frame_idx = ghost_frame - first

            if time.monotonic() >= hud_next:
                hud_next = time.monotonic() + 0.25
                err = ""
                if "pos_err" in last_info:
                    err = f" err={last_info['pos_err'] * 1000:.0f}mm/{last_info['axis_err']:.1f}deg"
                flags = (" STALE_PEN" if stale_pen and receiver else "") + \
                        (f" [{last_info.get('state', '?')}]" if frame_fresh else " [no frame stream]") + \
                        (" REC" if rec_active else "") + \
                        (" FROZEN" if args.freeze_frame is not None else "")
                print(f"[mirror] {bundle.clip_ids[clip].split('/')[-1]} "
                      f"f={ghost_frame - first:4d}/{clip_len}{err}{flags}", flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if rec_queue is not None and rec_proc is not None:
            _stop_rec()
            _rec_send_control(("quit",))
            rec_proc.join(timeout=8.0)
            if rec_proc.is_alive():
                rec_proc.terminate()
                rec_proc.join(timeout=2.0)
        try:
            viewer.close()
        except Exception:
            pass
        for x in (driver, receiver, frame_sub, cam_sub):
            if x is not None:
                x.close()
        if frame_ctx is not None:
            frame_ctx.term()

    os._exit(0)  # skip interpreter teardown; mujoco viewer teardown segfaults otherwise


if __name__ == "__main__":
    main()
