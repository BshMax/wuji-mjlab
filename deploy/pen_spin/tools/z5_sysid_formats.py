# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Shared npz schemas for the Z5 sysid toolchain (deploy side).

Mirror of Dex-Supermimic ``src/dex_supermimic/sysid/z5/formats.py`` — the two
files implement the identical contract as copies rather than a shared
dependency. Everything is plain numpy dtypes so files load with
``allow_pickle=False``; strings are numpy unicode scalars.

Plan ``z5_excitation_plan_v1``:

- ``schema``         0-d str, this schema id
- ``joint_names``    (22,) str, model order (thumb first == bundle joint_names)
- ``rate_hz``        0-d float64, command rate, 50..1000
- ``q`` ``dq`` ``tau`` ``kp`` ``kd``   (N,22) float32, model sign convention.
  kp==0 & tau!=0 marks a torque-mode frame.
- ``segment``        (N,) int32, segment index per frame
- ``segments_json``  0-d str: JSON list of {"name", "kind" in
  steps|chirp|multisine|ramps|hold|replay|transition,
  "moving_joints": [names], "params": {...}, "prompt": str|null}
- ``meta_json``      0-d str: {"generator", "created_utc", "park_pose": [22],
  "palm_quat_wxyz": [4], "recipe": {...}}

Validity (checked by :func:`validate_plan`): all channels finite; frames
continuous (|dq_cmd| per frame <= 8 rad/s / rate); within joint limits.

Hardware log ``z5_hw_log_v1``:

- ``schema``, ``joint_names`` (22,), ``plan_sha256`` 0-d str (sha256 of the
  plan file bytes, "" when unknown)
- ``meta_json`` 0-d str: {"backend": real|mock|synthetic, "wire_convention",
  "rate_hz", "abort_reason": null|str, "guard_interventions",
  "send_jitter_us": {"p50","p99","max"}, "state_rate_hz_measured",
  "started_utc", "notes", and for synthetic a "truth" dict}
- Command stream, one row per Write():
  ``cmd_t_ns`` (M,) int64 (time.monotonic_ns at send),
  ``cmd_frame`` (M,) int32 (plan frame index, -1 for homing/hold/release),
  ``cmd_q`` ``cmd_dq`` ``cmd_tau`` ``cmd_kp`` ``cmd_kd`` (M,22) float32,
  model convention, after the runtime guard.
- State stream, one row per received HandState_:
  ``st_t_ns`` (K,) int64 (monotonic_ns in the callback),
  ``st_q`` ``st_dq`` ``st_ddq`` ``st_tau_est`` (K,22) float32 model
  convention (wire signs applied to q,dq,ddq,tau_est and to commanded
  q,dq,tau),
  ``st_temperature`` (K,22,2) float32,
  ``st_motorstate`` (K,22) uint32, ``st_sensor`` (K,22,2) uint32 raw,
  ``st_vol`` (K,22) float32 bus voltage (NaN where not recorded; files
  written before the field existed load as NaN).
- ``events_json`` 0-d str: list of {"t_ns", "type", "detail"} with type in
  segment_start|segment_end|prompt|guard|abort|release.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PLAN_SCHEMA = "z5_excitation_plan_v1"
LOG_SCHEMA = "z5_hw_log_v1"

NUM_JOINTS = 22
MAX_TARGET_SPEED = 8.0  # rad/s, plan continuity bound
SEGMENT_KINDS = (
    "steps",
    "chirp",
    "multisine",
    "ramps",
    "hold",
    "replay",
    "transition",
)


@dataclass
class ExcitationPlan:
    """In-memory form of ``z5_excitation_plan_v1``."""

    joint_names: tuple
    rate_hz: float
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    segment: np.ndarray
    segments: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def num_frames(self):
        return int(self.q.shape[0])

    @property
    def duration_s(self):
        return self.num_frames / self.rate_hz


@dataclass
class HwLog:
    """In-memory form of ``z5_hw_log_v1``."""

    joint_names: tuple
    plan_sha256: str
    meta: dict
    cmd_t_ns: np.ndarray
    cmd_frame: np.ndarray
    cmd_q: np.ndarray
    cmd_dq: np.ndarray
    cmd_tau: np.ndarray
    cmd_kp: np.ndarray
    cmd_kd: np.ndarray
    st_t_ns: np.ndarray
    st_q: np.ndarray
    st_dq: np.ndarray
    st_ddq: np.ndarray
    st_tau_est: np.ndarray
    st_temperature: np.ndarray
    st_motorstate: np.ndarray
    st_sensor: np.ndarray
    st_vol: np.ndarray
    events: list = field(default_factory=list)


def _str_array(values):
    return np.asarray([str(v) for v in values])


def _json_scalar(obj):
    return np.asarray(json.dumps(obj))


def save_plan(plan, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(PLAN_SCHEMA),
        joint_names=_str_array(plan.joint_names),
        rate_hz=np.asarray(float(plan.rate_hz)),
        q=np.asarray(plan.q, dtype=np.float32),
        dq=np.asarray(plan.dq, dtype=np.float32),
        tau=np.asarray(plan.tau, dtype=np.float32),
        kp=np.asarray(plan.kp, dtype=np.float32),
        kd=np.asarray(plan.kd, dtype=np.float32),
        segment=np.asarray(plan.segment, dtype=np.int32),
        segments_json=_json_scalar(plan.segments),
        meta_json=_json_scalar(plan.meta),
    )
    return path


def load_plan(path):
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema"])
        if schema != PLAN_SCHEMA:
            raise ValueError(f"unsupported plan schema {schema!r} in {path}")
        plan = ExcitationPlan(
            joint_names=tuple(str(v) for v in data["joint_names"]),
            rate_hz=float(data["rate_hz"]),
            q=data["q"].astype(np.float32),
            dq=data["dq"].astype(np.float32),
            tau=data["tau"].astype(np.float32),
            kp=data["kp"].astype(np.float32),
            kd=data["kd"].astype(np.float32),
            segment=data["segment"].astype(np.int32),
            segments=json.loads(str(data["segments_json"])),
            meta=json.loads(str(data["meta_json"])),
        )
    _require_plan_shapes(plan)
    return plan


def _require_plan_shapes(plan):
    n = plan.num_frames
    if len(plan.joint_names) != NUM_JOINTS:
        raise ValueError(f"plan must have {NUM_JOINTS} joint names")
    for name in ("q", "dq", "tau", "kp", "kd"):
        arr = getattr(plan, name)
        if arr.shape != (n, NUM_JOINTS):
            raise ValueError(f"plan {name} has shape {arr.shape}, expected ({n}, {NUM_JOINTS})")
    if plan.segment.shape != (n,):
        raise ValueError(f"plan segment has shape {plan.segment.shape}, expected ({n},)")
    if not 50.0 <= plan.rate_hz <= 1000.0:
        raise ValueError(f"plan rate_hz {plan.rate_hz} outside 50..1000")
    if plan.segment.size and (plan.segment.min() < 0 or plan.segment.max() >= len(plan.segments)):
        raise ValueError("plan segment indices out of range")


def validate_plan(plan, limits):
    """Return a list of validation errors ([] = valid).

    ``limits`` is a (22,2) array of [lo, hi] joint limits in the plan's model
    convention; callers pick the table (bundle limits, the urdf20261103 thumb
    yaw1 cap, ...).
    """
    _require_plan_shapes(plan)
    limits = np.asarray(limits, dtype=np.float64)
    if limits.shape != (NUM_JOINTS, 2):
        raise ValueError(f"limits shape {limits.shape}, expected ({NUM_JOINTS}, 2)")
    errors = []
    for name in ("q", "dq", "tau", "kp", "kd"):
        if not np.isfinite(getattr(plan, name)).all():
            errors.append(f"plan {name} contains non-finite values")
    if errors:
        return errors
    below = plan.q < limits[:, 0] - 1e-6
    above = plan.q > limits[:, 1] + 1e-6
    for idx in np.argwhere(below | above):
        k, j = int(idx[0]), int(idx[1])
        errors.append(
            f"frame {k} joint {plan.joint_names[j]}: q={plan.q[k, j]:.4f} "
            f"outside [{limits[j, 0]:.4f}, {limits[j, 1]:.4f}]"
        )
        if len(errors) > 20:
            errors.append("... (further limit violations suppressed)")
            return errors
    if plan.num_frames > 1:
        step = np.abs(np.diff(plan.q.astype(np.float64), axis=0)).max()
        max_step = MAX_TARGET_SPEED / plan.rate_hz
        if step > max_step + 1e-9:
            errors.append(
                f"plan discontinuity: max per-frame |dq| {step:.5f} rad exceeds "
                f"{MAX_TARGET_SPEED} rad/s at {plan.rate_hz:g} Hz ({max_step:.5f} rad)"
            )
    for i, seg in enumerate(plan.segments):
        if seg.get("kind") not in SEGMENT_KINDS:
            errors.append(f"segment {i} has unknown kind {seg.get('kind')!r}")
        for joint in seg.get("moving_joints", []):
            if joint not in plan.joint_names:
                errors.append(f"segment {i} moves unknown joint {joint!r}")
    return errors


def save_log(log, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(LOG_SCHEMA),
        joint_names=_str_array(log.joint_names),
        plan_sha256=np.asarray(str(log.plan_sha256)),
        meta_json=_json_scalar(log.meta),
        cmd_t_ns=np.asarray(log.cmd_t_ns, dtype=np.int64),
        cmd_frame=np.asarray(log.cmd_frame, dtype=np.int32),
        cmd_q=np.asarray(log.cmd_q, dtype=np.float32),
        cmd_dq=np.asarray(log.cmd_dq, dtype=np.float32),
        cmd_tau=np.asarray(log.cmd_tau, dtype=np.float32),
        cmd_kp=np.asarray(log.cmd_kp, dtype=np.float32),
        cmd_kd=np.asarray(log.cmd_kd, dtype=np.float32),
        st_t_ns=np.asarray(log.st_t_ns, dtype=np.int64),
        st_q=np.asarray(log.st_q, dtype=np.float32),
        st_dq=np.asarray(log.st_dq, dtype=np.float32),
        st_ddq=np.asarray(log.st_ddq, dtype=np.float32),
        st_tau_est=np.asarray(log.st_tau_est, dtype=np.float32),
        st_temperature=np.asarray(log.st_temperature, dtype=np.float32),
        st_motorstate=np.asarray(log.st_motorstate, dtype=np.uint32),
        st_sensor=np.asarray(log.st_sensor, dtype=np.uint32),
        st_vol=np.asarray(log.st_vol, dtype=np.float32),
        events_json=_json_scalar(log.events),
    )
    return path


def load_log(path):
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema"])
        if schema != LOG_SCHEMA:
            raise ValueError(f"unsupported log schema {schema!r} in {path}")
        log = HwLog(
            joint_names=tuple(str(v) for v in data["joint_names"]),
            plan_sha256=str(data["plan_sha256"]),
            meta=json.loads(str(data["meta_json"])),
            cmd_t_ns=data["cmd_t_ns"].astype(np.int64),
            cmd_frame=data["cmd_frame"].astype(np.int32),
            cmd_q=data["cmd_q"].astype(np.float32),
            cmd_dq=data["cmd_dq"].astype(np.float32),
            cmd_tau=data["cmd_tau"].astype(np.float32),
            cmd_kp=data["cmd_kp"].astype(np.float32),
            cmd_kd=data["cmd_kd"].astype(np.float32),
            st_t_ns=data["st_t_ns"].astype(np.int64),
            st_q=data["st_q"].astype(np.float32),
            st_dq=data["st_dq"].astype(np.float32),
            st_ddq=data["st_ddq"].astype(np.float32),
            st_tau_est=data["st_tau_est"].astype(np.float32),
            st_temperature=data["st_temperature"].astype(np.float32),
            st_motorstate=data["st_motorstate"].astype(np.uint32),
            st_sensor=data["st_sensor"].astype(np.uint32),
            st_vol=(
                data["st_vol"].astype(np.float32)
                if "st_vol" in data.files
                else np.full((len(data["st_t_ns"]), NUM_JOINTS), np.nan, dtype=np.float32)
            ),
            events=json.loads(str(data["events_json"])),
        )
    _require_log_shapes(log)
    return log


def _require_log_shapes(log):
    if len(log.joint_names) != NUM_JOINTS:
        raise ValueError(f"log must have {NUM_JOINTS} joint names")
    m, k = log.cmd_t_ns.shape[0], log.st_t_ns.shape[0]
    for name in ("cmd_q", "cmd_dq", "cmd_tau", "cmd_kp", "cmd_kd"):
        arr = getattr(log, name)
        if arr.shape != (m, NUM_JOINTS):
            raise ValueError(f"log {name} has shape {arr.shape}, expected ({m}, {NUM_JOINTS})")
    if log.cmd_frame.shape != (m,):
        raise ValueError("log cmd_frame must be (M,)")
    for name in ("st_q", "st_dq", "st_ddq", "st_tau_est"):
        arr = getattr(log, name)
        if arr.shape != (k, NUM_JOINTS):
            raise ValueError(f"log {name} has shape {arr.shape}, expected ({k}, {NUM_JOINTS})")
    if log.st_temperature.shape != (k, NUM_JOINTS, 2):
        raise ValueError("log st_temperature must be (K,22,2)")
    if log.st_motorstate.shape != (k, NUM_JOINTS):
        raise ValueError("log st_motorstate must be (K,22)")
    if log.st_sensor.shape != (k, NUM_JOINTS, 2):
        raise ValueError("log st_sensor must be (K,22,2)")
    if log.st_vol.shape != (k, NUM_JOINTS):
        raise ValueError("log st_vol must be (K,22)")


def plan_sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
