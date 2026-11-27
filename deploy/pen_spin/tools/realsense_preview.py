"""Live RealSense RGB framing & AprilTag alignment preview for Dex5S pen-spin recording.

Uses Tkinter + PIL (works even when opencv-python-headless is installed).
Shows the live 1280x720 RGB feed, AprilTag id=0 detection status (required for
automatic pixel-exact MuJoCo viewpoint alignment in dex5s_mirror.py --realsense),
and the exact 4:3 crop box corresponding to --realsense-zoom.

Keys:
  + / = : zoom in (+0.1)
  -     : zoom out (-0.1, down to 1.0 = full frame)
  q/ESC : quit and release the camera before launching dex5s_mirror.py --realsense
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


def _load_tag_constants():
    pen_spin_root = Path(__file__).resolve().parents[1]
    ns: dict = {"np": np}
    for rel, prefixes in (
        ("lib/tag_frame.py", ("WORLD_TAG_ID =", "WORLD_TAG_SIZE_M =", "WRIST_AXES_IN_OPENCV_TAG =")),
        ("lib/dex5s_tag_frame.py", ("DEX5S_TAG_IN_PALM_POS =", "DEX5S_TAG_IN_PALM_QUAT =")),
    ):
        for line in (pen_spin_root / rel).read_text().splitlines():
            s = line.strip()
            if any(s.startswith(pfx) for pfx in prefixes):
                exec(s, ns)
    w, x, y, z = [float(v) for v in ns["DEX5S_TAG_IN_PALM_QUAT"]]
    R_palm_tag = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return (
        int(ns["WORLD_TAG_ID"]),
        float(ns["WORLD_TAG_SIZE_M"]),
        np.asarray(ns["WRIST_AXES_IN_OPENCV_TAG"], dtype=np.float64),
        np.asarray(ns["DEX5S_TAG_IN_PALM_POS"], dtype=np.float64),
        R_palm_tag,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zoom", type=float, default=1.5, help="initial crop zoom (matches dex5s_mirror --realsense-zoom)")
    ap.add_argument("--serial", type=str, default="", help="optional RealSense serial number")
    args = ap.parse_args()

    import pyrealsense2 as rs

    tag_id, tag_size, wrist_axes, tag_in_palm_pos, R_palm_tag = _load_tag_constants()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    pipe = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    try:
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        prof = pipe.start(cfg)
    except Exception:
        cfg = rs.config()
        if args.serial:
            cfg.enable_device(args.serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        prof = pipe.start(cfg)

    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    W, H = float(intr.width), float(intr.height)
    fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    h = tag_size / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    p_hand_palm = np.array([0.0, 0.075, 0.025], dtype=np.float64)

    state = {"zoom": max(1.0, float(args.zoom)), "running": True, "photo": None}
    root = tk.Tk()
    root.title("RealSense Framing Preview (+/- zoom, q/ESC quit)")
    label = tk.Label(root)
    label.pack()

    def _quit(_event=None):
        state["running"] = False
        root.quit()

    def _on_key(event):
        ch = event.char
        if ch in ("q", "Q", "b", "B") or event.keysym == "Escape":
            _quit()
        elif ch in ("+", "="):
            state["zoom"] = min(3.0, round(state["zoom"] + 0.1, 2))
        elif ch in ("-", "_"):
            state["zoom"] = max(1.0, round(state["zoom"] - 0.1, 2))

    root.protocol("WM_DELETE_WINDOW", _quit)
    root.bind("<Key>", _on_key)
    print(f"[realsense_preview] {int(W)}x{int(H)} live (Tkinter). Press +/- to adjust zoom, q/ESC to quit.", flush=True)

    def _tick():
        if not state["running"]:
            return
        try:
            frames = pipe.poll_for_frames()
            if not frames:
                frames = pipe.wait_for_frames(timeout_ms=200)
            cf = frames.get_color_frame()
        except Exception:
            cf = None

        if cf:
            vis = np.asanyarray(cf.get_data()).copy()
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            tag_ok = False
            dist_m = 0.0
            uc, vc = W / 2.0, H / 2.0
            if ids is not None and tag_id in ids.flatten():
                quad = corners[list(ids.flatten()).index(tag_id)][0].astype(np.float64)
                ok, rvec, tvec = cv2.solvePnP(obj, quad, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    tag_ok = True
                    R_raw, _ = cv2.Rodrigues(rvec)
                    R_rs_tag = R_raw @ wrist_axes
                    t_rs_tag = tvec.reshape(3)
                    R_tag_rs = R_rs_tag.T
                    t_tag_rs = -R_tag_rs @ t_rs_tag
                    R_palm_rs = R_palm_tag @ R_tag_rs
                    t_palm_rs = tag_in_palm_pos + R_palm_tag @ t_tag_rs
                    dist_m = float(np.linalg.norm(t_palm_rs))
                    p_hand_rs = R_palm_rs.T @ (p_hand_palm - t_palm_rs)
                    uc = fx * (p_hand_rs[0] / max(p_hand_rs[2], 1e-4)) + cx
                    vc = fy * (p_hand_rs[1] / max(p_hand_rs[2], 1e-4)) + cy
                    pts = quad.astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(vis, (int(round(uc)), int(round(vc))), 5, (0, 255, 255), -1, cv2.LINE_AA)

            zoom = state["zoom"]
            roi_h = min(H, H / zoom)
            roi_w = roi_h * (640.0 / 480.0)
            if roi_w > W:
                roi_w = W
                roi_h = roi_w * (480.0 / 640.0)
            x0 = int(round(np.clip(uc - roi_w / 2.0, 0.0, W - roi_w)))
            y0 = int(round(np.clip(vc - roi_h / 2.0, 0.0, H - roi_h)))
            x1 = int(min(W, round(x0 + roi_w)))
            y1 = int(min(H, round(y0 + roi_h)))
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 255), 2, cv2.LINE_AA)

            status = (
                f"TAG id={tag_id} OK | dist={dist_m:.2f}m | zoom={zoom:.1f}x (crop {x1-x0}x{y1-y0}) | +/- zoom, q quit"
                if tag_ok
                else f"TAG id={tag_id} NOT VISIBLE (adjust angle/unblock wrist tag!) | zoom={zoom:.1f}x | q quit"
            )
            color = (80, 255, 80) if tag_ok else (60, 60, 255)
            cv2.rectangle(vis, (0, 0), (int(W), 34), (20, 20, 20), -1)
            cv2.putText(vis, status, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            state["photo"] = photo
            label.configure(image=photo)

        if state["running"]:
            root.after(15, _tick)

    try:
        root.after(10, _tick)
        root.mainloop()
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass
        print(f"[realsense_preview] closed (use --realsense-zoom {state['zoom']:.1f} in dex5s_mirror.py).", flush=True)


if __name__ == "__main__":
    main()
