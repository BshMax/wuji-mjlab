# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Republish the pen observer's wrist-tag pose in the Z5 palm_link frame.

The z5-pen-spin deploy bundle (third_party/z5-pen-spin-deploy) consumes
single-frame ZMQ JSON messages in a fixed schema — the same contract as its
own scripts/pen_world_observer.py:

  {"timestamp": <unix s>, "frame": "palm_link", "quaternion_order": "wxyz",
   "valid": true, "pen": {"position": [x,y,z], "quaternion": [w,x,y,z]}}

This bridge subscribes to our observer stream (pen pose in the fixed
wrist-tag frame, source pen quat), lifts it into ``palm_link`` with
lib.z5_tag_frame, converts the quat to the tracking pen frame (+X shaft)
with lib.pen_frame, and republishes on the bundle's expected port.

Run the pen observer on a different port than the bundle endpoint:

  python deploy/pen_spin/scripts/pen_world_observer.py --port 5556 ...
  python deploy/pen_spin/tools/z5_pose_bridge.py --in-port 5556 --out-port 5555
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
                    help="port the z5 bundle PenReceiver connects to")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--frame", choices=["z5"], default="z5",
                    help="palm frame constants (currently only the Z5 mount)")
    args = ap.parse_args()

    import zmq
    from lib.pen_frame import source_to_tracking
    from lib.z5_tag_frame import pen_in_z5_palm

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.host}:{args.in_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.CONFLATE, 1)
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{args.out_port}")

    print(f"[z5 bridge] tag-frame :{args.in_port} -> palm_link :{args.out_port}",
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
        out = {"timestamp": time.time(), "frame": "palm_link",
               "quaternion_order": "wxyz", "valid": False}
        if body is not None:
            try:
                p, o = body["position"], body["orientation"]
                pos_tag = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
                quat_tag = np.array([o["w"], o["x"], o["y"], o["z"]],
                                    dtype=np.float64)
                pos_palm, quat_palm = pen_in_z5_palm(pos_tag, quat_tag)
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
            print(f"[z5 bridge] published={published} invalid={skipped}",
                  flush=True)


if __name__ == "__main__":
    main()
