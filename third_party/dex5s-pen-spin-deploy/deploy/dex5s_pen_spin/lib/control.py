"""One-shot reference playback using mock, MuJoCo or real DDS feedback."""

from __future__ import annotations

import datetime
import select
import sys
import time
from pathlib import Path

import numpy as np

from .core import Bundle, Policy, ResidualController, rotation_wxyz
from .hand_driver import TargetGuard, Dex5SHandDriver
from .simulation import MockHandDriver, MujocoHandDriver
from .transport import PenReceiver


def tracking_error(current, reference):
    position = float(np.linalg.norm(current[:3] - reference[:3]))
    dot = rotation_wxyz(current[3:])[:, 0] @ rotation_wxyz(reference[3:])[:, 0]
    return position, float(np.degrees(np.arccos(np.clip(dot, -1, 1))))


def _blend_pose(pose_a: np.ndarray, pose_b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear position + shortest-arc quaternion NLERP from pose_a to pose_b."""
    a = float(np.clip(alpha, 0.0, 1.0))
    pos = (1.0 - a) * pose_a[:3] + a * pose_b[:3]
    qa = np.asarray(pose_a[3:7], dtype=np.float64)
    qb = np.asarray(pose_b[3:7], dtype=np.float64)
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb
    q = (1.0 - a) * qa + a * qb
    q /= max(float(np.linalg.norm(q)), 1e-8)
    return np.r_[pos, q].astype(np.float32)


def _quat_mul(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = qa
    w2, x2, y2, z2 = qb
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _propagate_occluded_pose(
    last_fresh_pose: np.ndarray,
    ref_at_fresh: np.ndarray,
    ref_now: np.ndarray,
    stale_ticks: int,
    decay_per_tick: float = 0.985,
) -> np.ndarray:
    """Propagate last measured pen-vs-reference offset along the moving reference during tag occlusion.

    Strips out the physically irrelevant axial roll around the cylindrical pen shaft (+X)
    before propagating orientation error in the moving reference body frame. Otherwise, a
    large shaft-roll difference (~100-180 deg) left-multiplied onto a rotated ref_now would
    fabricate a huge fake axis flip during occlusion.
    """
    w = float(decay_per_tick ** min(max(stale_ticks, 1), 30))
    dp = np.asarray(last_fresh_pose[:3], dtype=np.float64) - np.asarray(ref_at_fresh[:3], dtype=np.float64)
    pos = np.asarray(ref_now[:3], dtype=np.float64) + w * dp
    r_ref0 = rotation_wxyz(ref_at_fresh[3:7])
    u_meas = rotation_wxyz(last_fresh_pose[3:7])[:, 0]
    u_body = r_ref0.T @ u_meas
    u_body /= max(float(np.linalg.norm(u_body)), 1e-8)
    dot = float(np.clip(u_body[0], -1.0, 1.0))
    if dot > 0.999999:
        q_ax = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif dot < -0.999999:
        ang = np.pi * w
        q_ax = np.array([np.cos(ang / 2.0), 0.0, np.sin(ang / 2.0), 0.0], dtype=np.float64)
    else:
        axis = np.array([0.0, -u_body[2], u_body[1]], dtype=np.float64)  # [1,0,0] x u_body
        axis /= max(float(np.linalg.norm(axis)), 1e-8)
        ang = float(np.arccos(dot)) * w
        q_ax = np.array([np.cos(ang / 2.0), *(axis * np.sin(ang / 2.0))], dtype=np.float64)
    q_pred = _quat_mul(np.asarray(ref_now[3:7], dtype=np.float64), q_ax)
    q_pred /= max(float(np.linalg.norm(q_pred)), 1e-8)
    return np.r_[pos, q_pred].astype(np.float32)


_ROLL1_JOINTS = {
    "right_index_finger_roll1_joint",
    "right_middle_finger_roll1_joint",
    "right_ring_finger_roll1_joint",
    "right_little_finger_roll1_joint",
}

# Exact least-squares joint-space offset (rad) on urdf20261103 thumb & little finger
# so 3D link/tip positions match the training MJCF across pen-spinning trajectories
# (including the 4.6-deg axis tilt on right_little_finger_pitch_joint: sin(4.6deg)=0.0799).
_URDF20261103_THUMB_KIN_OFFSET = {
    "right_thumb_pitch_joint": +0.0138,
    "right_thumb_roll_joint": -0.0321,
    "right_thumb_yaw1_joint": +0.0072,
    "right_little_finger_pitch_joint": -0.0034,
    "right_little_finger_roll1_joint": -0.0057,
    "right_little_finger_yaw_joint": +0.0006,
}
_LITTLE_PITCH_TO_YAW_TILT = -0.0799


class PoseOutlierFilter:
    """Reject short single-frame PnP glitches (e.g., 10+ cm occlusion flips).

    Holds the last accepted pose for up to ``max_streak`` consecutive outlier
    ticks; if a large jump persists beyond ``max_streak`` ticks, accepts the
    new pose so real motion or recovery can never lock up the filter.
    Set ``max_jump_m <= 0`` and ``max_jump_deg <= 0`` to disable.
    """

    def __init__(self, max_jump_m: float = 0.03, max_jump_deg: float = 30.0, max_streak: int = 5):
        self.max_jump_m = float(max_jump_m)
        self.max_jump_deg = float(max_jump_deg)
        self.max_streak = max(1, int(max_streak))
        self.last_pose: np.ndarray | None = None
        self.streak = 0
        self.rejected = 0

    def reset(self):
        self.last_pose = None
        self.streak = 0

    def apply(self, pose: np.ndarray) -> tuple[np.ndarray, bool]:
        pose = np.asarray(pose, dtype=np.float32)
        if self.last_pose is None or (self.max_jump_m <= 0.0 and self.max_jump_deg <= 0.0):
            self.last_pose = pose.copy()
            self.streak = 0
            return pose, False
        pos_jump, ang_jump = tracking_error(pose, self.last_pose)
        is_bad = (
            (self.max_jump_m > 0.0 and pos_jump > self.max_jump_m)
            or (self.max_jump_deg > 0.0 and ang_jump > self.max_jump_deg)
        )
        if is_bad and self.streak < self.max_streak:
            self.streak += 1
            self.rejected += 1
            return self.last_pose.copy(), True
        self.last_pose = pose.copy()
        self.streak = 0
        return pose, False


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
    ctrl_cfg = dict(bundle.config)
    ema_override = getattr(args, "ema_alpha", None)
    if ema_override is not None:
        if not (0.0 < float(ema_override) <= 1.0):
            raise ValueError("--ema-alpha must be in (0, 1]")
        ctrl_cfg["ema_alpha"] = float(ema_override)
    controller = ResidualController(ctrl_cfg)
    pose_filter = PoseOutlierFilter(
        max_jump_m=float(getattr(args, "max_pose_jump_m", 0.03)),
        max_jump_deg=float(getattr(args, "max_pose_jump_deg", 30.0)),
    )
    recover_ticks_cfg = max(0, int(getattr(args, "pose_recover_ticks", 0)))
    occluded_mode = str(getattr(args, "pose_occluded_mode", "hold") or "hold")
    joint_space = bundle.config.get("joint_space", "dex5s_mjcf")
    roll1_bias_val = float(getattr(args, "roll1_bias_comp", 0.0))
    roll1_comp = np.array(
        [
            (-roll1_bias_val if (joint_space == "dex5s_urdf" and n == "right_little_finger_roll1_joint") else roll1_bias_val)
            if n in _ROLL1_JOINTS
            else 0.0
            for n in bundle.joint_names
        ],
        dtype=np.float32,
    )
    use_thumb_kin_comp = bool(
        getattr(args, "thumb_kin_comp", True)
        and joint_space == "dex5s_mjcf"
        and args.wire_convention == "urdf20261103"
        and args.backend == "real"
    )
    thumb_kin_comp = np.array(
        [_URDF20261103_THUMB_KIN_OFFSET.get(n, 0.0) if use_thumb_kin_comp else 0.0 for n in bundle.joint_names],
        dtype=np.float32,
    )
    little_pitch_idx = bundle.joint_names.index("right_little_finger_pitch_joint")
    little_yaw_idx = bundle.joint_names.index("right_little_finger_yaw_joint")
    max_tracking_lead = float(getattr(args, "max_tracking_lead", 0.0) or 0.0)
    other_tracking_lead = float(getattr(args, "other_max_tracking_lead", 0.0) or 0.0)
    roll1_mask = np.array([n in _ROLL1_JOINTS for n in bundle.joint_names], dtype=bool)
    lead_vec = np.where(
        roll1_mask,
        max_tracking_lead if max_tracking_lead > 0.0 else np.inf,
        other_tracking_lead if other_tracking_lead > 0.0 else np.inf,
    ).astype(np.float32)
    has_lead_clamp = bool(np.any(np.isfinite(lead_vec)))
    lead_clamp_ticks = 0
    rs485_stale_ticks = 0
    pose_stale_ticks_total = 0
    max_pose_stale_ticks = 0
    stale_pred_ticks = 0
    joint_freeze_sec = float(getattr(args, "joint_freeze_seconds", 0.8) or 0.0)
    joint_freeze_ticks = int(round(joint_freeze_sec / bundle.dt)) if (args.backend == "real" and joint_freeze_sec > 0.0) else 0
    was_blind = False
    recover_left = 0
    last_output_pose: np.ndarray | None = None
    last_fresh_pose: np.ndarray | None = None
    driver = receiver = None
    trace = []
    trace_clip = []
    trace_motor_bits = []
    trace_tau_est = []
    rest_pose = None
    failure = None
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_log_path: str | None = None
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
                      f"{'holding last pose' if occluded_mode == 'hold' else 'feeding reference'} "
                      f"for up to {args.pose_fallback:.2f}s",
                      flush=True)
            elif now - blind_since > args.pose_fallback:
                raise
            if occluded_mode == "hold" and last_fresh_pose is not None:
                return last_fresh_pose.copy(), True
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
            driver = Dex5SHandDriver(
                bundle,
                convention=args.wire_convention,
                interface=args.dds_interface,
                domain=args.dds_domain,
                sdk_path=args.sdk_path,
                kp=args.kp,
                kd=args.kd,
                large_kp=getattr(args, "large_kp", None),
                large_kd=getattr(args, "large_kd", None),
                thumb_kp=getattr(args, "thumb_kp", None),
                thumb_kd=getattr(args, "thumb_kd", None),
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
                diff = actual - move_target
                worst = np.argsort(-np.abs(diff))[:3]
                detail = ", ".join(
                    f"{bundle.joint_names[i]}({actual[i]:+.2f} vs {move_target[i]:+.2f}, err={diff[i]:+.2f})"
                    for i in worst
                )
                raise RuntimeError(f"hand did not reach the reference start pose: {detail}")
            if args.wait_start and clip_index_pos == 0:
                print("start pose reached - 'c' plays the clip, 'b' aborts", flush=True)
                if _KEYS["enabled"]:
                    deadline = time.monotonic()
                    wait_tick = 0
                    while True:
                        key = _poll_key()
                        if key in ("c", "C"):
                            break
                        if key in ("b", "B"):
                            raise RuntimeError("aborted by 'b' before playback")
                        driver.send(guard.apply(move_target))
                        if receiver is not None and wait_tick % 50 == 0:
                            try:
                                p_now, _ = _latest_pose(first)
                                pe0, ae0 = tracking_error(p_now, bundle.references["object_pose"][first])
                                print(f"[wait_start] pen align err: pos={pe0*100:.2f} cm, axis={ae0:.1f} deg (target < 1.2 cm)", flush=True)
                            except TimeoutError:
                                pass
                        wait_tick += 1
                        publish_frame(state="wait", frame=first)
                        deadline = max(deadline + bundle.dt, time.monotonic())
                        _pace(deadline, realtime)
                else:
                    sys.stdin.readline()
            if clip_index_pos == 0:
                run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            hold_steps = int(round(args.hold_seconds / bundle.dt))
            pose_filter.reset()
            was_blind = False
            recover_left = 0
            last_output_pose = None
            last_fresh_stamp = -np.inf
            last_fresh_pose = None
            last_fresh_frame = first
            stale_ticks = 0
            prev_raw_q: np.ndarray | None = None
            prev_raw_dq: np.ndarray | None = None
            joint_stale_streak = np.zeros(22, dtype=np.int32)
            joint_frozen_ticks = np.zeros(22, dtype=np.int32)
            joint_frozen_tgt_min = np.zeros(22, dtype=np.float32)
            joint_frozen_tgt_max = np.zeros(22, dtype=np.float32)
            joint_frozen_max_err = np.zeros(22, dtype=np.float32)
            last_sent_target: np.ndarray | None = None
            total_steps = hold_steps + end - first
            if args.steps:
                total_steps = min(total_steps, args.steps)
            if args.duration:
                total_steps = min(total_steps, max(1, int(args.duration / bundle.dt)))
            deadline = time.monotonic()
            last_send = deadline - bundle.dt
            for tick in range(total_steps):
                if _poll_key() in ("b", "B"):
                    raise RuntimeError("aborted by 'b' key")
                if realtime and time.monotonic() - deadline > args.max_overrun:
                    raise TimeoutError("control loop missed its deadline")
                started = time.monotonic()
                frame = first + max(0, tick - hold_steps)
                phase = "hold" if tick < hold_steps else "play"
                publish_frame(state=phase, frame=frame)
                q, dq = driver.read()
                dq_obs = dq.copy()
                q_lead_ref = q.copy()
                if prev_raw_q is not None and prev_raw_dq is not None:
                    is_joint_stale = (q == prev_raw_q) & (dq == prev_raw_dq) & (np.abs(dq) > 1e-3)
                    joint_stale_streak = np.where(is_joint_stale, joint_stale_streak + 1, 0)
                    if np.any(joint_stale_streak >= 1):
                        rs485_stale_ticks += int(np.any(joint_stale_streak >= 2))
                        decay = np.power(0.5, joint_stale_streak).astype(np.float32)
                        dq_obs = dq * decay
                        if last_sent_target is not None:
                            step_allowance = np.minimum(0.04 * joint_stale_streak, 0.08)
                            q_lead_ref = q + np.clip(last_sent_target - q, -step_allowance, step_allowance)
                if joint_freeze_ticks > 0 and prev_raw_q is not None and last_sent_target is not None:
                    same_q = q == prev_raw_q
                    for j_idx in range(22):
                        if same_q[j_idx]:
                            err_j = float(abs(last_sent_target[j_idx] - q[j_idx]))
                            if joint_frozen_ticks[j_idx] == 0:
                                joint_frozen_tgt_min[j_idx] = last_sent_target[j_idx]
                                joint_frozen_tgt_max[j_idx] = last_sent_target[j_idx]
                                joint_frozen_max_err[j_idx] = err_j
                            else:
                                if last_sent_target[j_idx] < joint_frozen_tgt_min[j_idx]:
                                    joint_frozen_tgt_min[j_idx] = last_sent_target[j_idx]
                                if last_sent_target[j_idx] > joint_frozen_tgt_max[j_idx]:
                                    joint_frozen_tgt_max[j_idx] = last_sent_target[j_idx]
                                if err_j > joint_frozen_max_err[j_idx]:
                                    joint_frozen_max_err[j_idx] = err_j
                            joint_frozen_ticks[j_idx] += 1
                            span_j = float(joint_frozen_tgt_max[j_idx] - joint_frozen_tgt_min[j_idx])
                            merr_j = float(joint_frozen_max_err[j_idx])
                            if joint_frozen_ticks[j_idx] >= joint_freeze_ticks and (
                                (merr_j >= 0.25 and span_j >= 0.15) or merr_j >= 0.35
                            ):
                                raise RuntimeError(
                                    f"joint feedback frozen: {bundle.joint_names[j_idx]} stuck at "
                                    f"{q[j_idx]:+.3f} rad for {joint_frozen_ticks[j_idx] * bundle.dt:.2f}s "
                                    f"(max|err|={merr_j:.3f} rad, tgt_span={span_j:.3f} rad)"
                                )
                        else:
                            joint_frozen_ticks[j_idx] = 0
                prev_raw_q = q.copy()
                prev_raw_dq = dq.copy()
                blind = False
                pose_age_ms = 0.0
                if receiver is not None:
                    pose, blind = _latest_pose(frame)
                    blind_ticks += int(blind)
                    cur_stamp = float(receiver.cache.timestamp)
                    if np.isfinite(cur_stamp):
                        pose_age_ms = max(0.0, (time.time() - cur_stamp) * 1000.0)
                    is_fresh = (not blind) and (cur_stamp > last_fresh_stamp + 1e-9)
                    if blind or (not is_fresh and last_fresh_pose is not None):
                        stale_ticks += 1
                        if stale_ticks >= 2:
                            pose_stale_ticks_total += 1
                        if stale_ticks > max_pose_stale_ticks:
                            max_pose_stale_ticks = stale_ticks
                        if occluded_mode == "propagate" and stale_ticks >= 2 and last_fresh_pose is not None:
                            pose = _propagate_occluded_pose(
                                last_fresh_pose,
                                bundle.references["object_pose"][last_fresh_frame],
                                bundle.references["object_pose"][frame],
                                stale_ticks=stale_ticks,
                            )
                            stale_pred_ticks += 1
                        elif last_fresh_pose is not None:
                            # Hold strictly at the last valid camera pose in place.
                            pose = last_fresh_pose.copy()
                        elif last_output_pose is not None:
                            pose = last_output_pose.copy()
                        if blind or stale_ticks >= 1:
                            was_blind = True
                    else:
                        if was_blind:
                            was_blind = False
                            recover_left = recover_ticks_cfg
                            pose_filter.reset()
                            if recover_left == 0:
                                # Snap immediately to the newly arrived valid frame without outlier-jump delay.
                                pose_filter.last_pose = pose.copy()
                                pose_filter.streak = 0
                        elif recover_left > 0 and last_output_pose is not None:
                            pass
                        if recover_left > 0 and last_output_pose is not None:
                            alpha = 1.0 / (recover_left + 1)
                            pose = _blend_pose(last_output_pose, pose, alpha)
                            recover_left -= 1
                            pose_filter.last_pose = pose.copy()
                            pose_filter.streak = 0
                        else:
                            pose, is_outlier = pose_filter.apply(pose)
                            if is_outlier and pose_filter.rejected <= 5:
                                print(
                                    f"[pose] rejected single-frame PnP outlier at frame {frame} "
                                    f"(total rejected={pose_filter.rejected})",
                                    flush=True,
                                )
                        last_fresh_stamp = cur_stamp
                        last_fresh_pose = pose.copy()
                        last_fresh_frame = frame
                        stale_ticks = 0
                    last_output_pose = pose.copy()
                elif args.backend == "mujoco" and not args.no_pen:
                    pose = driver.pen_pose()
                else:
                    pose = bundle.references["object_pose"][frame].copy()
                position_error, angle_error = tracking_error(pose, bundle.references["object_pose"][frame])
                pose_is_stale = blind or (receiver is not None and stale_ticks >= 2)
                if not args.no_tracking_guard and not pose_is_stale:
                    if position_error > args.max_position_error or angle_error > args.max_axis_error_deg:
                        guard_bad += 1
                        if guard_bad >= args.guard_patience:
                            raise RuntimeError(
                                f"pen tracking error sustained {guard_bad} ticks: "
                                f"{position_error:.4f} m, {angle_error:.1f} deg"
                            )
                    else:
                        guard_bad = 0
                publish_frame(state=phase, frame=frame, pos_err=position_error, axis_err=angle_error)
                obs = bundle.observation(frame, q, dq_obs, pose, controller.last_action)
                action = policy(obs)
                desired = controller.target(bundle.references["joint_pos"][frame], action)
                if roll1_bias_val != 0.0:
                    desired = desired + roll1_comp
                if use_thumb_kin_comp:
                    desired = desired + thumb_kin_comp
                    desired[little_yaw_idx] += _LITTLE_PITCH_TO_YAW_TILT * desired[little_pitch_idx]
                if has_lead_clamp:
                    clamped_lead = np.clip(desired, q_lead_ref - lead_vec, q_lead_ref + lead_vec)
                    if np.any(np.abs(clamped_lead - desired) > 1e-6):
                        lead_clamp_ticks += 1
                    desired = clamped_lead
                if move_mask is not None:
                    desired = np.where(move_mask, desired, rest_pose)
                target = guard.apply(desired)
                last_sent_target = target.copy()
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
                        pose_age_ms,
                        stale_ticks,
                    )
                )
                trace_clip.append(clip_id)
                if hasattr(driver, "diagnostics"):
                    m_bits, m_tau = driver.diagnostics()
                    if m_bits is not None:
                        trace_motor_bits.append(m_bits)
                    if m_tau is not None:
                        trace_tau_est.append(m_tau)
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
            raw_path = Path(args.log)
            clip_short = bundle.clip_ids[clips[0]].split("/")[-1]
            if raw_path.is_dir() or raw_path.suffix.lower() != ".npz":
                path = raw_path / f"{run_ts}_{clip_short}.npz"
            elif raw_path.exists():
                path = raw_path.with_name(f"{raw_path.stem}_{run_ts}{raw_path.suffix}")
            else:
                path = raw_path
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
                "pose_age_ms",
                "pose_stale_ticks",
            )
            rows = {name: np.array([row[i] for row in trace]) for i, name in enumerate(fields)}
            # Plain names for sysid: state/action are motor angles in radians.
            rows["state"] = rows["joint_pos"]
            rows["action"] = rows["sent_target"]
            if len(trace_motor_bits) == len(trace) and len(trace) > 0:
                rows["motor_state_bits"] = np.asarray(trace_motor_bits, dtype=np.uint32)
            if len(trace_tau_est) == len(trace) and len(trace) > 0:
                rows["tau_est"] = np.asarray(trace_tau_est, dtype=np.float32)
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
            saved_log_path = str(path)
            print(f"[play_real] saved log -> {path}", flush=True)
    if failure is not None:
        raise failure
    return {
        "backend": args.backend,
        "clip": bundle.clip_ids[clips[0]],
        "clips": [bundle.clip_ids[c] for c in clips],
        "log_path": saved_log_path,
        "joint_group": getattr(args, "joint_group", "all") or "all",
        "steps": len(trace),
        "simulated_seconds": len(trace) * bundle.dt,
        "target_guard_interventions": guard.interventions,
        "max_compute_ms": max((row[8] * 1000 for row in trace), default=0),
        "max_position_error_m": max((row[9] for row in trace), default=0),
        "max_axis_error_deg": max((row[10] for row in trace), default=0),
        "blind_ticks": blind_ticks,
        "outlier_ticks": pose_filter.rejected,
        "lead_clamp_ticks": lead_clamp_ticks,
        "rs485_stale_ticks": rs485_stale_ticks,
        "pose_stale_ticks": pose_stale_ticks_total,
        "max_pose_stale_ticks": max_pose_stale_ticks,
        "stale_pred_ticks": stale_pred_ticks,
    }
