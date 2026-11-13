# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Sim-to-sim full-chain test: virtual observer -> real bridge -> policy -> MuJoCo.

Mirrors the physical deployment one-for-one, but the camera is replaced by
ground truth read out of the MuJoCo scene:

  MuJoCo pen pose (palm frame, +X tracking quat)
    -> invert through Z5_TAG_IN_PALM and the source(+Z)/tracking(+X) basis
    -> publish on --obs-port in the observer's legacy schema
    -> z5_pose_bridge subprocess (the real binary used with the camera)
    -> PenReceiver on --pose-endpoint (same freshness/schema checks as real)
    -> Bundle.observation -> ONNX policy -> ResidualController -> TargetGuard
    -> MujocoHandDriver actuators (scene.mjb)

If this loop tracks the reference, the complete software chain is good and
only the camera + DDS drivers remain untested for the real run.

Requires mujoco==3.15 for scene.mjb; run via the `z5-sim-chain` pixi task.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLE_ROOT = REPO_ROOT / "third_party/z5-pen-spin-deploy"
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE_ROOT))
sys.path.insert(0, str(TOOLS_DIR.parent))  # makes lib.* importable

from deploy.z5_pen_spin.lib.control import tracking_error  # noqa: E402
from deploy.z5_pen_spin.lib.core import Bundle, Policy, ResidualController  # noqa: E402
from deploy.z5_pen_spin.lib.hand_driver import TargetGuard  # noqa: E402
from deploy.z5_pen_spin.lib.simulation import MujocoHandDriver  # noqa: E402
from deploy.z5_pen_spin.lib.transport import PenReceiver  # noqa: E402
from lib.math_utils import normalize_quat, quat_apply, quat_inv, quat_mul  # noqa: E402
from lib.pen_frame import TRACKING_IN_SOURCE_QUAT  # noqa: E402
from lib.z5_tag_frame import Z5_TAG_IN_PALM_POS, Z5_TAG_IN_PALM_QUAT  # noqa: E402

PALM_IN_TAG_QUAT = quat_inv(Z5_TAG_IN_PALM_QUAT)
SOURCE_IN_TRACKING_QUAT = quat_inv(TRACKING_IN_SOURCE_QUAT)


def pen_palm_to_observer_wire(pen_pos_palm, pen_quat_palm):
    """Palm-frame tracking pose -> legacy observer message (tag frame, +Z quat).

    Inverse of z5_pose_bridge: pose_in_tag = T_tag_in_palm^-1 . pose_in_palm,
    then tracking(+X) -> source(+Z) shaft convention, wire order xyzw.
    """
    pos_tag = quat_apply(PALM_IN_TAG_QUAT, pen_pos_palm - Z5_TAG_IN_PALM_POS)
    quat_tag = quat_mul(PALM_IN_TAG_QUAT, pen_quat_palm)
    quat_tag = normalize_quat(quat_mul(quat_tag, SOURCE_IN_TRACKING_QUAT))
    return {
        "position": dict(zip("xyz", pos_tag.tolist())),
        "orientation": dict(zip("xyzw", quat_tag[[1, 2, 3, 0]].tolist())),
    }


def add_pose_noise(pos, quat, pos_std, axis_std, rng):
    """Gaussian position jitter + random small rotation, camera-like."""
    if pos_std > 0:
        pos = pos + rng.normal(0.0, pos_std, 3)
    if axis_std > 0:
        axis = rng.normal(0.0, 1.0, 3)
        axis /= np.linalg.norm(axis)
        half = 0.5 * rng.normal(0.0, axis_std)
        quat = normalize_quat(quat_mul(quat, np.r_[np.cos(half), np.sin(half) * axis]))
    return pos, quat


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path,
                    default=BUNDLE_ROOT / "generated/deploy/z5-pen-spin")
    ap.add_argument("--clip")
    ap.add_argument("--engine", choices=["onnx", "torchscript"], default="onnx")
    ap.add_argument("--steps", type=int, default=0,
                    help="0: complete clip including --hold-seconds")
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--hold-seconds", type=float, default=3.0)
    ap.add_argument("--obs-port", type=int, default=5576,
                    help="virtual observer publish port (observer legacy schema)")
    ap.add_argument("--pose-endpoint", default="tcp://127.0.0.1:5575",
                    help="bridge republish endpoint consumed by PenReceiver")
    ap.add_argument("--bridge-out-port", type=int, default=5575)
    ap.add_argument("--pose-max-age", type=float, default=0.3)
    ap.add_argument("--noise-pos-m", type=float, default=0.0,
                    help="extra Gaussian position noise, metres (e.g. 0.0005)")
    ap.add_argument("--noise-axis-deg", type=float, default=0.0,
                    help="extra random rotation noise, degrees")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--no-realtime", action="store_true")
    ap.add_argument("--max-target-speed", type=float, default=8.0)
    ap.add_argument("--max-position-error", type=float, default=0.05)
    ap.add_argument("--max-axis-error-deg", type=float, default=60.0)
    ap.add_argument("--no-tracking-guard", action="store_true")
    ap.add_argument("--log", type=Path)
    ap.add_argument("--print-every", type=int, default=50)
    return ap


def main():
    args = parser().parse_args()
    import zmq

    bundle = Bundle(args.bundle)
    clip = bundle.clip_index(args.clip)
    first, end = int(bundle.starts[clip]), int(bundle.ends[clip])
    if not 0.0 <= args.hold_seconds:
        raise SystemExit("--hold-seconds must be >= 0")

    policy = Policy(bundle, args.engine)
    controller = ResidualController(bundle.config)
    guard = TargetGuard(bundle, "legacy", args.max_target_speed)
    rng = np.random.default_rng(0)

    # Virtual observer socket (observer legacy schema, like --port on the real one).
    ctx = zmq.Context.instance()
    obs_pub = ctx.socket(zmq.PUB)
    obs_pub.bind(f"tcp://*:{args.obs_port}")

    # The real bridge binary, exactly as used with the physical camera.
    bridge = subprocess.Popen(
        [sys.executable, str(TOOLS_DIR / "z5_pose_bridge.py"),
         "--in-port", str(args.obs_port),
         "--out-port", str(args.bridge_out_port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    receiver = PenReceiver(args.pose_endpoint, args.pose_max_age)

    def publish_pose(driver):
        pose = driver.pen_pose()
        pos, quat = add_pose_noise(pose[:3].astype(np.float64),
                                   pose[3:].astype(np.float64),
                                   args.noise_pos_m,
                                   np.radians(args.noise_axis_deg), rng)
        body = pen_palm_to_observer_wire(pos, quat)
        obs_pub.send_json({"timestamp": time.time(), "world_fixed": True,
                           "pen": body})

    driver = failure = None
    trace = []
    transform_err = []
    try:
        driver = MujocoHandDriver(bundle, first, args.viewer)
        q, dq = driver.read()
        guard.reset(q)
        time.sleep(0.4)  # let the bridge connect both sockets
        publish_pose(driver)
        # Wait for the first converted pose, same as the real bring-up.
        deadline = time.monotonic() + 5.0
        while True:
            try:
                receiver.latest()
                break
            except TimeoutError:
                publish_pose(driver)
                if time.monotonic() >= deadline:
                    raise TimeoutError("no palm_link pose from the bridge")
                time.sleep(0.01)

        hold_steps = int(round(args.hold_seconds / bundle.dt))
        total = hold_steps + end - first
        if args.steps:
            total = min(total, args.steps)
        if args.duration:
            total = min(total, max(1, int(args.duration / bundle.dt)))
        realtime = not args.no_realtime
        print(f"[sim-chain] clip={bundle.clip_ids[clip]} steps={total} "
              f"noise=({args.noise_pos_m * 1e3:.1f}mm,{args.noise_axis_deg:.2f}deg) "
              f"guard={'off' if args.no_tracking_guard else 'on'}", flush=True)

        deadline = time.monotonic()
        for tick in range(total):
            started = time.monotonic()
            frame = first + max(0, tick - hold_steps)
            q, dq = driver.read()
            pose_seen = receiver.latest()
            truth = driver.pen_pose()
            transform_err.append(float(np.linalg.norm(pose_seen[:3] - truth[:3])))
            pos_err, axis_err = tracking_error(
                pose_seen, bundle.references["object_pose"][frame])
            if not args.no_tracking_guard and (
                    pos_err > args.max_position_error
                    or axis_err > args.max_axis_error_deg):
                raise RuntimeError(
                    f"pen tracking error: {pos_err:.4f} m, {axis_err:.1f} deg")
            obs = bundle.observation(frame, q, dq, pose_seen,
                                     controller.last_action)
            action = policy(obs)
            desired = controller.target(bundle.references["joint_pos"][frame],
                                        action)
            target = guard.apply(desired)
            compute_ms = (time.monotonic() - started) * 1000.0
            if realtime:
                time.sleep(max(0.0, deadline + bundle.dt - time.monotonic()))
            driver.send(target)
            publish_pose(driver)  # fresh observation for the next tick
            deadline = max(deadline + bundle.dt, time.monotonic())
            trace.append((frame, obs, q, dq, pose_seen, action, desired,
                          target, compute_ms / 1000.0, pos_err, axis_err))
            if tick % args.print_every == 0:
                print(f"t={tick * bundle.dt:6.1f}s f={frame:4d} "
                      f"err={pos_err * 1e3:5.0f}mm/{axis_err:4.1f}deg "
                      f"|a|max={np.abs(action).max():.2f} "
                      f"xform={transform_err[-1] * 1e6:5.1f}um "
                      f"cmp={compute_ms:4.1f}ms", flush=True)
    except BaseException as exc:
        failure = exc
    finally:
        try:
            if driver is not None:
                driver.close()
        finally:
            receiver.close()
            bridge.terminate()
            bridge.wait(timeout=3)
        if args.log:
            fields = ("frame", "observation", "joint_pos", "joint_vel",
                      "pen_pose", "action", "desired_target", "sent_target",
                      "compute_seconds", "position_error_m", "axis_error_deg")
            args.log.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.log,
                **{n: np.array([r[i] for r in trace])
                   for i, n in enumerate(fields)},
                clip=np.array(bundle.clip_ids[clip]),
                transform_error_m=np.asarray(transform_err),
                error=np.array("" if failure is None else str(failure)))
    if failure is not None:
        print(f"[sim-chain] FAILED after {len(trace)} steps: {failure}",
              flush=True)
        raise SystemExit(1) from failure
    print(json.dumps({
        "backend": "mujoco+full-chain",
        "clip": bundle.clip_ids[clip],
        "steps": len(trace),
        "simulated_seconds": len(trace) * bundle.dt,
        "target_guard_interventions": guard.interventions,
        "max_transform_error_um": max(transform_err, default=0.0) * 1e6,
        "max_compute_ms": max((r[8] * 1000 for r in trace), default=0.0),
        "max_position_error_m": max((r[9] for r in trace), default=0.0),
        "max_axis_error_deg": max((r[10] for r in trace), default=0.0),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
