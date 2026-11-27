# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Republish the pen observer's wrist-tag pose in the Dex5S palm_link frame.

The dex5s-pen-spin deploy bundle (third_party/dex5s-pen-spin-deploy) consumes
single-frame ZMQ JSON messages in a fixed schema — the same contract as its
own scripts/pen_world_observer.py:

  {"timestamp": <unix s>, "frame": "palm_link", "quaternion_order": "wxyz",
   "valid": true, "pen": {"position": [x,y,z], "quaternion": [w,x,y,z]}}

This bridge subscribes to our observer stream (pen pose in the fixed
wrist-tag frame, source pen quat), lifts it into ``palm_link`` with
lib.dex5s_tag_frame, converts the quat to the tracking pen frame (+X shaft)
with lib.pen_frame, and republishes on the bundle's expected port.

Run the pen observer on a different port than the bundle endpoint:

  python deploy/pen_spin/scripts/pen_world_observer.py --port 5556 ...
  python deploy/pen_spin/tools/dex5s_pose_bridge.py --in-port 5556 --out-port 5555
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for lib.*


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-port", type=int, default=5556,
                    help="our pen observer stream (wrist-tag frame)")
    ap.add_argument("--out-port", type=int, default=5555,
                    help="port the dex5s bundle PenReceiver connects to")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--frame", choices=["dex5s"], default="dex5s",
                    help="palm frame constants (currently only the Dex5S mount)")
    ap.add_argument("--pos-offset-mm", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    metavar=("DX", "DY", "DZ"),
                    help="static calibration offset added in palm_link (mm), e.g. 6.0 -6.0 0.0 to remove wrist-tag mount bias")
    args = ap.parse_args()

    import zmq
    from lib.pen_frame import source_to_tracking
    from lib.dex5s_tag_frame import DEX5S_TAG_IN_PALM_POS, pen_in_dex5s_palm
    pos_offset_m = np.asarray(args.pos_offset_mm, dtype=np.float64) * 1e-3

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.host}:{args.in_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.CONFLATE, 1)
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{args.out_port}")

    print(f"[dex5s bridge] tag-frame :{args.in_port} -> palm_link :{args.out_port}",
          flush=True)
    published = skipped = 0
    while True:
        try:
            msg = json.loads(sub.recv_string())
        except KeyboardInterrupt:
            break
        except Exception:
            continue
        body = msg.get("pen") if msg.get("world_fixed") else None
        now = time.time()
        raw_ts = msg.get("timestamp")
        if isinstance(raw_ts, (int, float)) and np.isfinite(raw_ts) and abs(now - float(raw_ts)) < 10.0:
            stamp = float(raw_ts)
        else:
            stamp = now
        out = {"timestamp": stamp, "frame": "palm_link",
               "quaternion_order": "wxyz", "valid": False}
        if body is not None:
            try:
                p, o = body["position"], body["orientation"]
                pos_tag = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
                quat_tag = np.array([o["w"], o["x"], o["y"], o["z"]],
                                    dtype=np.float64)
                pos_palm, quat_palm = pen_in_dex5s_palm(pos_tag, quat_tag)
                live_cal = msg.get("tag_in_palm_pos")
                if live_cal is not None and len(live_cal) == 3:
                    pos_palm = pos_palm + (np.asarray(live_cal, dtype=np.float64) - DEX5S_TAG_IN_PALM_POS)
                pos_palm = pos_palm + pos_offset_m
                quat_palm = source_to_tracking(quat_palm)
                if np.isfinite(pos_palm).all() and np.isfinite(quat_palm).all():
                    out["valid"] = True
                    out["pen"] = {"position": pos_palm.tolist(),
                                  "quaternion": quat_palm.tolist()}
            except (KeyError, TypeError, ValueError):
                pass
        if out["valid"]:
            published += 1
        else:
            skipped += 1
        try:
            pub.send_string(json.dumps(out), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        if (published + skipped) % 500 == 0:
            print(f"[dex5s bridge] published={published} invalid={skipped}",
                  flush=True)


if __name__ == "__main__":
    main()
