# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Report the pen's +/-radius shaft-axis endpoint positions for accuracy checks.

The policy observes the pen as two axis points: the pen origin extended
+-axis_point_radius_m (0.10 m) toward each end. This tool prints those two
points — and the pen origin — in a chosen reference frame, so the numbers can
be compared against a ruler measurement on the rig.

Frames:
  tag   (default)  wrist AprilTag frame — verifies intrinsics + tag size + PnP.
                   Measure from the printed tag's centre along its drawn axes.
  hand2            Wuji Hand 2 right_palm_link frame (TAG_IN_PALM_* constants).
  z5               Unitree Z5 palm_link frame (Z5_TAG_IN_PALM_* constants —
                   placeholder until the tag mount is measured on the rig).

Sources: a recorded diagnostics JSONL, or the live observer stream.

  python measure_axis_points.py outputs/tag_eval/run_*/static_01.jsonl
  python measure_axis_points.py --port 5556 --frame hand2 --window-s 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from lib.math_utils import matrix_from_quat  # noqa: E402

LIFT = {}
def _lifts():
    from lib.tag_frame import pen_in_palm
    LIFT["hand2"] = pen_in_palm
    try:
        from lib.z5_tag_frame import pen_in_z5_palm
        LIFT["z5"] = pen_in_z5_palm
    except ImportError:
        pass
    return LIFT


def endpoints(pos, quat_wxyz, radius_m):
    """Pen origin and the +/-Z(source) shaft points at radius, in the pose's frame."""
    R = matrix_from_quat(np.asarray(quat_wxyz, dtype=np.float64))
    pos = np.asarray(pos, dtype=np.float64)
    axis = R @ np.array([0.0, 0.0, radius_m])
    return pos, pos + axis, pos - axis


def summarise(name, rows, frame, radius_m):
    pos = np.array([r["position"] for r in rows])
    quat = np.array([r["quat_wxyz"] for r in rows])
    if frame != "tag":
        lift = _lifts()[frame]
        lifted = [lift(p, q) for p, q in zip(pos, quat)]
        pos = np.array([p for p, _ in lifted])
        quat = np.array([q for _, q in lifted])
    origin, plus, minus = [], [], []
    for p, q in zip(pos, quat):
        o, a, b = endpoints(p, q, radius_m)
        origin.append(o); plus.append(a); minus.append(b)
    origin = np.array(origin) * 1e3
    plus = np.array(plus) * 1e3
    minus = np.array(minus) * 1e3
    span = np.linalg.norm(plus - minus, axis=1)
    valid_rate = np.nan
    if rows[0].get("t_monotonic_s") is not None and len(rows) > 1:
        t = np.array([r["t_monotonic_s"] for r in rows])
        valid_rate = len(rows) / max(t[-1] - t[0], 1e-9)
    print(f"\n[{name}] n={len(rows)}" +
          (f"  valid ~{valid_rate:.1f} Hz" if np.isfinite(valid_rate) else ""),
          flush=True)
    print(f"  frame = {frame}   radius = {radius_m*1e3:.0f} mm   (units: mm)",
          flush=True)
    for lbl, pts in (("pen origin", origin),
                     (f"+{radius_m*1e3:.0f}mm end", plus),
                     (f"-{radius_m*1e3:.0f}mm end", minus)):
        med = np.median(pts, axis=0)
        sig = pts.std(axis=0)
        print(f"    {lbl:14s} xyz=({med[0]:7.1f},{med[1]:7.1f},{med[2]:7.1f})"
              f"  |d|={np.linalg.norm(med):7.1f}  sigma=({sig[0]:.2f},{sig[1]:.2f},{sig[2]:.2f})",
              flush=True)
    print(f"    end-to-end dist = {np.median(span):.1f} mm  (nominally {2*radius_m*1e3:.0f})",
          flush=True)


def from_jsonl(path, frame, radius_m):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows = [r for r in rows
            if isinstance(r.get("position"), list) and isinstance(r.get("quat_wxyz"), list)]
    if not rows:
        raise SystemExit(f"{path}: no valid poses")
    summarise(path.name, rows, frame, radius_m)


def from_port(port, host, frame, radius_m, window_s):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib.zmq_bridge import PenReceiver
    rx = PenReceiver(port=port, host=host)
    print(f"listening on tcp://{host}:{port} — Ctrl-C to stop", flush=True)
    buf = []
    t_flush = t_start = time.monotonic()
    try:
        while True:
            pos, quat, age = rx.latest()
            if rx.seen:
                buf.append({"position": list(pos), "quat_wxyz": list(quat),
                            "t_monotonic_s": time.monotonic()})
            now = time.monotonic()
            if now - t_flush >= window_s and buf:
                summarise(f"live {len(buf)} samples", buf, frame, radius_m)
                buf.clear()
                t_flush = now
            if not rx.seen and now - t_start > 5.0:
                print("still no pen poses on the wire — the observer only "
                      "publishes them once its preview reads "
                      "'world frame FIXED': show the wrist AprilTag (id 0) to "
                      "the camera, or press 'w' to resample.", flush=True)
                t_start = now
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="?", type=Path, help="recorded diagnostics JSONL")
    ap.add_argument("--port", type=int, help="live ZMQ port instead of a file")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--frame", choices=["tag", "hand2", "z5"], default="tag")
    ap.add_argument("--radius-mm", type=float, default=100.0,
                    help="axis point radius; training uses 100 mm")
    ap.add_argument("--window-s", type=float, default=2.0,
                    help="live stats window")
    args = ap.parse_args()
    if args.radius_mm <= 0:
        ap.error("--radius-mm must be positive")
    radius_m = args.radius_mm * 1e-3
    if args.frame != "tag":
        _lifts()
        if args.frame not in LIFT:
            ap.error(f"{args.frame} lift unavailable — check lib imports")
        if args.frame == "z5":
            print("NOTE: z5 constants were derived from the shared jig CAD — "
                  "verify once with a ruler (see Z5_TAG_IN_PALM_* comments in "
                  "lib/z5_tag_frame.py).", flush=True)
    if args.jsonl is not None:
        from_jsonl(args.jsonl, args.frame, radius_m)
    elif args.port:
        from_port(args.port, args.host, args.frame, radius_m, args.window_s)
    else:
        ap.error("give a JSONL path or --port")


if __name__ == "__main__":
    main()
