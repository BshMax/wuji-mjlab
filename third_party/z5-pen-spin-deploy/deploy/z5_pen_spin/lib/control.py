"""One-shot reference playback using mock, MuJoCo or real DDS feedback."""

from __future__ import annotations

import select
import sys
import time
from pathlib import Path

import numpy as np

from .core import Bundle, Policy, ResidualController, rotation_wxyz
from .hand_driver import TargetGuard, Z5HandDriver
from .simulation import MockHandDriver, MujocoHandDriver
from .transport import PenReceiver


def tracking_error(current, reference):
    position = float(np.linalg.norm(current[:3] - reference[:3]))
    dot = rotation_wxyz(current[3:])[:, 0] @ rotation_wxyz(reference[3:])[:, 0]
    return position, float(np.degrees(np.arccos(np.clip(dot, -1, 1))))


def _pace(deadline, realtime):
    if realtime:
        time.sleep(max(0.0, deadline - time.monotonic()))


_KEYS = {"enabled": False, "fd": -1, "saved": None}


def _keys_enable():
    if not sys.stdin.isatty():
        return
    try:
        import termios
        import tty
    except ImportError:
        return
    fd = sys.stdin.fileno()
    _KEYS["saved"] = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    _KEYS.update(enabled=True, fd=fd)


def _keys_disable():
    if _KEYS["enabled"]:
        import termios

        termios.tcsetattr(_KEYS["fd"], termios.TCSADRAIN, _KEYS["saved"])
        _KEYS.update(enabled=False, fd=-1, saved=None)


def _poll_key():
    if not _KEYS["enabled"]:
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return sys.stdin.read(1) if ready else None


_FINGER_PREFIXES = {
    "thumb": "right_thumb_",
    "index": "right_index_finger_",
    "middle": "right_middle_finger_",
    "ring": "right_ring_finger_",
    "little": "right_little_finger_",
}


def _clip_list(bundle, clip_arg):
    names = [c.strip() for c in (clip_arg or "").split(",") if c.strip()]
    if names == ["all"]:
        names = list(bundle.clip_ids)
    return [bundle.clip_index(name) for name in names] if names else [bundle.clip_index(None)]


def _move_mask(bundle, group):
    prefix = _FINGER_PREFIXES.get(group or "all")
    if prefix is None:
        return None
    mask = np.array([name.startswith(prefix) for name in bundle.joint_names])
    if not mask.any():
        raise ValueError(f"--joint-group {group!r} matched no joints")
    return mask


def _rest_pose(args, limits, measured):
    src = getattr(args, "flat_pose", "zero") or "zero"
    if src == "measured":
        return np.clip(measured, *limits.T)
    if src == "zero":
        return np.zeros(22)
    v = np.load(src, allow_pickle=True)
    if hasattr(v, "files"):
        v = v["rest_pose"] if "rest_pose" in v.files else v[v.files[0]]
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.shape != (22,) or not np.isfinite(v).all():
        raise ValueError(f"--flat-pose file must contain a finite 22-vector: {src}")
    return np.clip(v, *limits.T)


def run(args):
    bundle = Bundle(args.bundle)
    if args.list_clips:
        return {
            "clips": [
                {"id": name, "frames": int(end - start), "seconds": float((end - start) * bundle.dt)}
                for name, start, end in zip(bundle.clip_ids, bundle.starts, bundle.ends, strict=True)
            ]
        }
    clips = _clip_list(bundle, args.clip)
    move_mask = _move_mask(bundle, getattr(args, "joint_group", "all"))
    first = int(bundle.starts[clips[0]])
    if args.backend == "real" and args.wire_convention is None:
        raise ValueError("real DDS requires --wire-convention legacy or urdf20261103")
    if args.backend == "real" and args.no_realtime:
        raise ValueError("real DDS control requires real-time pacing")
    if args.backend != "mujoco" and args.viewer:
        raise ValueError("--viewer is available with --backend mujoco")
    if args.backend == "real" and args.no_tracking_guard:
        raise ValueError("real DDS requires the pen tracking guard")
    policy = Policy(bundle, args.engine)
    controller = ResidualController(bundle.config)
    driver = receiver = None
    trace = []
    trace_clip = []
    rest_pose = None
    failure = None
    guard_bad = 0
    blind_ticks = 0
    blind_since = None
    guard = TargetGuard(bundle, args.wire_convention or "legacy", args.max_target_speed)
    realtime = not args.no_realtime
    frame_ctx = frame_pub = None
    if args.frame_endpoint:
        import json as _json
        import zmq

        frame_ctx = zmq.Context()
        frame_pub = frame_ctx.socket(zmq.PUB)
        frame_pub.setsockopt(zmq.LINGER, 0)
        frame_pub.bind(args.frame_endpoint)

        clip_id = bundle.clip_ids[clips[0]]

        def publish_frame(**fields):
            frame_pub.send_string(_json.dumps({"t": time.time(), "clip": clip_id, **fields}))

    else:

        def publish_frame(**fields):
            pass

    def _latest_pose(frame_idx):
        """Fresh camera pose, or the reference pen pose during a bounded blackout.

        While the stream is fresh the live pose is returned. Once it goes
        stale, the reference pose at frame_idx stands in for at most
        args.pose_fallback seconds, then the TimeoutError is re-raised and
        the run aborts as before. pose_fallback <= 0 disables the fallback.
        The tracking guard sees ~zero error during the blackout because the
        stand-in is the reference itself — real divergence resumes counting
        as soon as the camera recovers.
        """
        nonlocal blind_since
        try:
            pose = receiver.latest()
            if blind_since is not None:
                print(f"[pose] camera pose recovered after "
                      f"{time.monotonic() - blind_since:.2f}s blackout", flush=True)
                blind_since = None
            return pose, False
        except TimeoutError:
            if args.pose_fallback <= 0:
                raise
            now = time.monotonic()
            if blind_since is None:
                blind_since = now
                print(f"[pose] camera pose lost at frame {frame_idx}; "
                      f"feeding reference for up to {args.pose_fallback:.2f}s",
                      flush=True)
            elif now - blind_since > args.pose_fallback:
                raise
            return bundle.references["object_pose"][frame_idx].copy(), True

    try:
        _keys_enable()
        if args.backend == "real":
            if not args.no_pen:
                receiver = PenReceiver(args.pose_endpoint, args.pose_max_age)
                deadline = time.monotonic() + args.startup_timeout
                while True:
                    try:
                        receiver.latest()
                        break
                    except TimeoutError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.01)
            driver = Z5HandDriver(
                bundle,
                convention=args.wire_convention,
                interface=args.dds_interface,
                domain=args.dds_domain,
                sdk_path=args.sdk_path,
                kp=args.kp,
                kd=args.kd,
                state_timeout=args.state_max_age,
                startup_timeout=args.startup_timeout,
                temperature_limit=args.temperature_limit,
            )
        elif args.backend == "mujoco":
            driver = MujocoHandDriver(bundle, first, args.viewer)
        else:
            driver = MockHandDriver(bundle, first)
        measured, _ = driver.read()
        guard.reset(measured, getattr(args, "takeover_tolerance", 0.02))
        rest_pose = _rest_pose(args, guard.limits, measured) if move_mask is not None else None
        if move_mask is not None:
            # Settle every joint into the flat rest pose first: a previous
            # finger leaves the hand curled, which would otherwise poison the
            # held-joint pose and can block the moving finger's start pose.
            ticks = max(1, int(np.ceil(args.move_seconds / bundle.dt)))
            deadline = time.monotonic()
            for tick in range(ticks):
                if receiver is not None:
                    _latest_pose(first)
                driver.read()
                fraction = (tick + 1) / ticks
                blend = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
                driver.send(guard.apply(measured + blend * (rest_pose - measured)))
                publish_frame(state="settle", frame=first)
                deadline = max(deadline + bundle.dt, time.monotonic())
                _pace(deadline, realtime)
            for _ in range(max(1, int(np.ceil(args.settle_seconds / bundle.dt)))):
                driver.send(guard.apply(rest_pose))
                publish_frame(state="settle", frame=first)
                deadline = max(deadline + bundle.dt, time.monotonic())
                _pace(deadline, realtime)
            if args.backend == "real":
                actual, _ = driver.read()
                if np.max(np.abs(actual - rest_pose)) > args.start_joint_tolerance:
                    raise RuntimeError("hand did not reach the flat rest pose")
            held = [bundle.joint_names[i] for i in range(22) if not move_mask[i]]
            print(f"joint-group run: holding {len(held)} joints flat; "
                  f"rest_pose={np.array2string(rest_pose, precision=3, suppress_small=True)}",
                  flush=True)
        # Take over from measured encoders, then move smoothly into each clip's
        # first demonstrated pose before starting its reference clock.
        for clip_index_pos, clip in enumerate(clips):
            clip_id = bundle.clip_ids[clip]
            first, end = int(bundle.starts[clip]), int(bundle.ends[clip])
            ref_first = bundle.references["joint_pos"][first]
            move_target = ref_first if move_mask is None else np.where(move_mask, ref_first, rest_pose)
            measured, _ = driver.read()
            ticks = max(1, int(np.ceil(args.move_seconds / bundle.dt)))
            deadline = time.monotonic()
            for tick in range(ticks):
                if receiver is not None:
                    _latest_pose(first)
                driver.read()
                fraction = (tick + 1) / ticks
                blend = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
                target = measured + blend * (move_target - measured)
                driver.send(guard.apply(target))
                publish_frame(state="move", frame=first)
                # Avoid a burst of catch-up commands after an overrun.
                deadline = max(deadline + bundle.dt, time.monotonic())
                _pace(deadline, realtime)
            actual, _ = driver.read()
            if args.backend == "real" and np.max(np.abs(actual - move_target)) > args.start_joint_tolerance:
                raise RuntimeError("hand did not reach the reference start pose")
            if args.wait_start and clip_index_pos == 0:
                print("start pose reached - ENTER plays the clip, q aborts", flush=True)
                if _KEYS["enabled"]:
                    deadline = time.monotonic()
                    while True:
                        key = _poll_key()
                        if key == "\n":
                            break
                        if key == "q":
                            raise RuntimeError("aborted by 'q' before playback")
                        driver.send(guard.apply(move_target))
                        publish_frame(state="wait", frame=first)
                        deadline = max(deadline + bundle.dt, time.monotonic())
                        _pace(deadline, realtime)
                else:
                    sys.stdin.readline()
            hold_steps = int(round(args.hold_seconds / bundle.dt))
            total_steps = hold_steps + end - first
            if args.steps:
                total_steps = min(total_steps, args.steps)
            if args.duration:
                total_steps = min(total_steps, max(1, int(args.duration / bundle.dt)))
            deadline = time.monotonic()
            last_send = deadline - bundle.dt
            for tick in range(total_steps):
                if _poll_key() == "q":
                    raise RuntimeError("aborted by 'q' key")
                if realtime and time.monotonic() - deadline > args.max_overrun:
                    raise TimeoutError("control loop missed its deadline")
                started = time.monotonic()
                frame = first + max(0, tick - hold_steps)
                phase = "hold" if tick < hold_steps else "play"
                publish_frame(state=phase, frame=frame)
                q, dq = driver.read()
                blind = False
                if receiver is not None:
                    pose, blind = _latest_pose(frame)
                    blind_ticks += int(blind)
                elif args.backend == "mujoco" and not args.no_pen:
                    pose = driver.pen_pose()
                else:
                    pose = bundle.references["object_pose"][frame].copy()
                position_error, angle_error = tracking_error(pose, bundle.references["object_pose"][frame])
                if not args.no_tracking_guard and (
                    position_error > args.max_position_error or angle_error > args.max_axis_error_deg
                ):
                    guard_bad += 1
                    if guard_bad >= args.guard_patience:
                        raise RuntimeError(
                            f"pen tracking error sustained {guard_bad} ticks: "
                            f"{position_error:.4f} m, {angle_error:.1f} deg"
                        )
                else:
                    guard_bad = 0
                publish_frame(state=phase, frame=frame, pos_err=position_error, axis_err=angle_error)
                obs = bundle.observation(frame, q, dq, pose, controller.last_action)
                action = policy(obs)
                desired = controller.target(bundle.references["joint_pos"][frame], action)
                if move_mask is not None:
                    desired = np.where(move_mask, desired, rest_pose)
                target = guard.apply(desired)
                if realtime and time.monotonic() - started > args.max_overrun:
                    raise TimeoutError("policy inference exceeded the control deadline")
                _pace(last_send + bundle.dt, realtime)
                # Check freshness again after inference and before publishing.
                if receiver is not None:
                    _latest_pose(frame)
                driver.send(target)
                last_send = time.monotonic()
                trace.append(
                    (
                        frame,
                        obs,
                        q,
                        dq,
                        pose,
                        action,
                        desired,
                        target,
                        time.monotonic() - started,
                        position_error,
                        angle_error,
                        blind,
                    )
                )
                trace_clip.append(clip_id)
                deadline = max(deadline + bundle.dt, time.monotonic())
                _pace(deadline, realtime)
    except BaseException as exc:
        failure = exc
    finally:
        _keys_disable()
        if frame_pub is not None:
            publish_frame(state="stopped", frame=frame if "frame" in dir() else first)
            frame_pub.close()
            frame_ctx.term()
        try:
            if driver is not None:
                driver.close()
        finally:
            if receiver is not None:
                receiver.close()
        if args.log:
            path = Path(args.log)
            path.parent.mkdir(parents=True, exist_ok=True)
            fields = (
                "frame",
                "observation",
                "joint_pos",
                "joint_vel",
                "pen_pose",
                "policy_action",
                "desired_target",
                "sent_target",
                "compute_seconds",
                "position_error_m",
                "axis_error_deg",
                "blind",
            )
            rows = {name: np.array([row[i] for row in trace]) for i, name in enumerate(fields)}
            # Plain names for sysid: state/action are motor angles in radians.
            rows["state"] = rows["joint_pos"]
            rows["action"] = rows["sent_target"]
            np.savez_compressed(
                path,
                **rows,
                clip_id=np.array(trace_clip),
                backend=np.array(args.backend),
                clip=np.array(bundle.clip_ids[clips[0]]),
                clips=np.array([bundle.clip_ids[c] for c in clips]),
                joint_group=np.array(getattr(args, "joint_group", "all") or "all"),
                rest_pose=np.array([]) if rest_pose is None else np.asarray(rest_pose),
                kp=np.array(getattr(driver, "kp", np.nan)),
                kd=np.array(getattr(driver, "kd", np.nan)),
                error=np.array("" if failure is None else str(failure)),
            )
    if failure is not None:
        raise failure
    return {
        "backend": args.backend,
        "clip": bundle.clip_ids[clips[0]],
        "clips": [bundle.clip_ids[c] for c in clips],
        "joint_group": getattr(args, "joint_group", "all") or "all",
        "steps": len(trace),
        "simulated_seconds": len(trace) * bundle.dt,
        "target_guard_interventions": guard.interventions,
        "max_compute_ms": max((row[8] * 1000 for row in trace), default=0),
        "max_position_error_m": max((row[9] for row in trace), default=0),
        "max_axis_error_deg": max((row[10] for row in trace), default=0),
        "blind_ticks": blind_ticks,
    }
