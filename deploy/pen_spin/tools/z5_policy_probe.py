# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Camera-only action probe for the exported Z5 pen-spin policy.

Runs the real vision chain — pen_world_observer -> z5_pose_bridge ->
this tool — and feeds the live palm_link pen pose into the exported ONNX
policy. A perfect mock hand tracks the commanded targets so the joint
observation terms stay meaningful; no DDS/hardware is ever touched.

Pipeline:

  pen-observer --preview --port 5556 --record-dir outputs/tag_eval
  z5-bridge   (observer :5556 -> palm_link :5555)
  z5-probe    (consumes :5555, prints live action diagnostics)

The reference clip advances at the 50 Hz control rate (wrapping inside
the clip); the printed tracking error compares the measured pen pose with
the demonstrated pose at the same frame.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLE_ROOT = REPO_ROOT / "third_party/z5-pen-spin-deploy"
sys.path.insert(0, str(BUNDLE_ROOT))

from deploy.z5_pen_spin.lib.control import tracking_error  # noqa: E402
from deploy.z5_pen_spin.lib.core import Bundle, Policy, ResidualController  # noqa: E402
from deploy.z5_pen_spin.lib.hand_driver import TargetGuard  # noqa: E402
from deploy.z5_pen_spin.lib.simulation import MockHandDriver  # noqa: E402
from deploy.z5_pen_spin.lib.transport import PenReceiver  # noqa: E402


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path,
                    default=BUNDLE_ROOT / "generated/deploy/z5-pen-spin")
    ap.add_argument("--pose-endpoint", default="tcp://127.0.0.1:5555")
    ap.add_argument("--pose-max-age", type=float, default=0.3)
    ap.add_argument("--startup-timeout", type=float, default=10.0)
    ap.add_argument("--clip", help="reference clip id (default: first)")
    ap.add_argument("--engine", choices=["onnx", "torchscript"], default="onnx")
    ap.add_argument("--steps", type=int, default=0,
                    help="stop after N control ticks (0: until Ctrl-C)")
    ap.add_argument("--hold-seconds", type=float, default=0.0,
                    help="freeze the reference frame this long before playback")
    ap.add_argument("--freeze-hand", action="store_true",
                    help="hold joints at the clip start pose instead of "
                         "letting the mock hand track policy targets")
    ap.add_argument("--no-realtime", action="store_true")
    ap.add_argument("--max-target-speed", type=float, default=8.0)
    ap.add_argument("--log", type=Path, help="write NPZ telemetry")
    ap.add_argument("--print-every", type=int, default=25,
                    help="summary line period in control ticks")
    ap.add_argument("--verbose", action="store_true",
                    help="print the full 22-dim action/target every tick")
    return ap


def main():
    args = parser().parse_args()
    bundle = Bundle(args.bundle)
    clip = bundle.clip_index(args.clip)
    first, end = int(bundle.starts[clip]), int(bundle.ends[clip])
    policy = Policy(bundle, args.engine)
    controller = ResidualController(bundle.config)
    guard = TargetGuard(bundle, "legacy", args.max_target_speed)
    mock = MockHandDriver(bundle, first)
    receiver = PenReceiver(args.pose_endpoint, args.pose_max_age)
    frozen_q = bundle.references["joint_pos"][first].copy()
    frozen_dq = np.zeros(22, dtype=np.float32)

    print(f"[probe] clip={bundle.clip_ids[clip]} frames={end - first} "
          f"engine={args.engine} endpoint={args.pose_endpoint}", flush=True)

    # Wait for the first fresh pose from the bridge.
    deadline = time.monotonic() + args.startup_timeout
    while True:
        try:
            pose = receiver.latest()
            break
        except TimeoutError:
            if time.monotonic() >= deadline:
                receiver.close()
                raise SystemExit("[probe] no fresh palm_link pose within "
                                 f"{args.startup_timeout}s — is z5-bridge running?")
            time.sleep(0.01)
    print(f"[probe] first pose xyz(mm)={np.round(pose[:3] * 1000, 1)}", flush=True)

    hold_ticks = int(round(args.hold_seconds / bundle.dt))
    stale = 0
    trace = []
    deadline = time.monotonic()
    try:
        tick = 0
        while not args.steps or tick < args.steps:
            frame = first + max(0, tick - hold_ticks) % (end - first)
            started = time.monotonic()
            if args.freeze_hand:
                q, dq = frozen_q.copy(), frozen_dq.copy()
            else:
                q, dq = mock.read()
            try:
                pose = receiver.latest()
                pose_stale = False
            except TimeoutError:
                pose_stale = True
                stale += 1
            pos_err, axis_err = tracking_error(
                pose, bundle.references["object_pose"][frame])
            obs = bundle.observation(frame, q, dq, pose, controller.last_action)
            action = policy(obs)
            desired = controller.target(bundle.references["joint_pos"][frame],
                                        action)
            target = guard.apply(desired)
            if not args.freeze_hand:
                mock.send(target)
            compute_ms = (time.monotonic() - started) * 1000.0
            trace.append((frame, obs, q, dq, pose.copy(), action, desired,
                          target, compute_ms / 1000.0, pos_err, axis_err,
                          pose_stale))

            if args.verbose:
                print(f"t={tick:5d} f={frame} a={np.array2string(action, precision=2, max_line_width=200)}",
                      flush=True)
                print(f"      tgt={np.array2string(target, precision=2, max_line_width=200)}",
                      flush=True)
            elif tick % args.print_every == 0:
                flag = " STALE" if pose_stale else ""
                print(f"t={tick * bundle.dt:6.1f}s f={frame:4d} "
                      f"pen=({pose[0] * 1e3:5.0f},{pose[1] * 1e3:4.0f},"
                      f"{pose[2] * 1e3:4.0f})mm "
                      f"err={pos_err * 1e3:5.0f}mm/{axis_err:4.1f}deg "
                      f"|a|max={np.abs(action).max():.2f} "
                      f"|res|max={np.abs(controller.residual).max():.3f} "
                      f"cmp={compute_ms:4.1f}ms{flag}", flush=True)
            tick += 1
            deadline = max(deadline + bundle.dt, time.monotonic())
            if not args.no_realtime:
                time.sleep(max(0.0, deadline - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
        if args.log:
            fields = ("frame", "observation", "joint_pos", "joint_vel",
                      "pen_pose", "action", "desired_target", "sent_target",
                      "compute_seconds", "position_error_m", "axis_error_deg",
                      "pose_stale")
            args.log.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.log,
                **{n: np.array([r[i] for r in trace])
                   for i, n in enumerate(fields)},
                clip=np.array(bundle.clip_ids[clip]))

    actions = np.array([r[5] for r in trace])
    print(json.dumps({
        "clip": bundle.clip_ids[clip],
        "steps": len(trace),
        "seconds": len(trace) * bundle.dt,
        "stale_poses": stale,
        "guard_interventions": guard.interventions,
        "action_abs_max": float(np.abs(actions).max()) if len(trace) else 0.0,
        "max_compute_ms": max((r[8] * 1000 for r in trace), default=0.0),
        "max_position_error_m": max((r[9] for r in trace), default=0.0),
        "max_axis_error_deg": max((r[10] for r in trace), default=0.0),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
