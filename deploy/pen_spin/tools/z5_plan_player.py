#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Play a ``z5_excitation_plan_v1`` on the Dex5-S (or a mock) -> ``z5_hw_log_v1``.

Flow: connect -> wait for state -> cosine home from the measured pose to
plan frame 0 with the home gains -> "press Enter" (skipped with --no-prompt
or --backend mock) -> absolute-time schedule over the plan frames (monotonic
clock, sleep then spin, never skip a frame, actual send times + jitter
logged; a frame sent > --max-late-ms late slips the whole remaining schedule
forward instead of bursting catch-up sends) -> per-segment prompts hold the
current target and re-base the schedule -> hold 0.5 s -> release
(kp=kd=tau=dq=0) -> close. On any exception or KeyboardInterrupt: hold the
measured pose 0.3 s with home gains, release, and always save the partial
log with ``meta.abort_reason``.

One log row per Write() (``cmd_frame`` = plan frame index, -1 for
homing/hold/release rows) and one row per received HandState_ with all
MotorState_ fields and a monotonic_ns stamp. Aborts on temperature over
--temperature-limit, state stale beyond --stale-timeout, or tracking error
over --track-error-limit sustained for 0.2 s on position-mode joints.

The mock backend runs a per-joint second-order response at 1 kHz in WIRE
convention (so the sign mapping is exercised), with a 20 ms command delay
and a temperature-spike test hook.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = REPO_ROOT / "third_party/z5-pen-spin-deploy"
sys.path.insert(0, str(VENDOR_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deploy.z5_pen_spin.lib.core import Bundle  # noqa: E402
from deploy.z5_pen_spin.lib.hand_driver import wire_mapping  # noqa: E402
import z5_sysid_formats as fmt  # noqa: E402

NUM_JOINTS = fmt.NUM_JOINTS
POSITION_MODE = 1
HOME_LEAD_S = 0.05
END_HOLD_S = 0.5
ABORT_HOLD_S = 0.3
PROMPT_HOLD_PERIOD_S = 0.05
TORQUE_WALL_MARGIN = 0.05  # rad; torque-mode joints hold at the wall past it
TRACK_ERROR_WINDOW_S = 0.2
LOG_CHUNK_ROWS = 65536  # rows per preallocated LogSink chunk


class AbortError(RuntimeError):
    """Guard abort: temperature, stale state, or sustained tracking error."""


class _ChunkedStore:
    """Rows appended into preallocated chunk arrays (no per-row objects).

    ``columns`` maps a column name to ``(dtype, tail_shape)``; each chunk is a
    dict of ``(rows_per_chunk, *tail)`` ndarrays. ``column()`` concatenates the
    chunks once at build time.
    """

    def __init__(self, columns: dict, rows_per_chunk: int = LOG_CHUNK_ROWS):
        self.columns = columns
        self.rows_per_chunk = int(rows_per_chunk)
        self._chunks: list[dict] = []
        self._cur = self._new_chunk()
        self._n = 0

    def _new_chunk(self) -> dict:
        return {
            name: np.empty((self.rows_per_chunk,) + tuple(tail), dtype=dtype)
            for name, (dtype, tail) in self.columns.items()
        }

    def append(self, **cols) -> None:
        if self._n >= self.rows_per_chunk:
            self._chunks.append(self._cur)
            self._cur = self._new_chunk()
            self._n = 0
        for name, value in cols.items():
            self._cur[name][self._n] = value
        self._n += 1

    def column(self, name: str) -> np.ndarray:
        dtype, tail = self.columns[name]
        parts = [c[name] for c in self._chunks]
        if self._n:
            parts.append(self._cur[name][: self._n])
        if not parts:
            return np.zeros((0,) + tuple(tail), dtype=dtype)
        return np.concatenate(parts)

    def __len__(self) -> int:
        return len(self._chunks) * self.rows_per_chunk + self._n


class LogSink:
    """Thread-safe chunked cmd/state/event buffers -> ``fmt.HwLog``."""

    _CMD_COLUMNS = {
        "t_ns": (np.int64, ()),
        "frame": (np.int32, ()),
        "q": (np.float32, (NUM_JOINTS,)),
        "dq": (np.float32, (NUM_JOINTS,)),
        "tau": (np.float32, (NUM_JOINTS,)),
        "kp": (np.float32, (NUM_JOINTS,)),
        "kd": (np.float32, (NUM_JOINTS,)),
    }
    _STATE_COLUMNS = {
        "t_ns": (np.int64, ()),
        "q": (np.float32, (NUM_JOINTS,)),
        "dq": (np.float32, (NUM_JOINTS,)),
        "ddq": (np.float32, (NUM_JOINTS,)),
        "tau_est": (np.float32, (NUM_JOINTS,)),
        "temperature": (np.float32, (NUM_JOINTS, 2)),
        "motorstate": (np.uint32, (NUM_JOINTS,)),
        "sensor": (np.uint32, (NUM_JOINTS, 2)),
        "vol": (np.float32, (NUM_JOINTS,)),
    }

    def __init__(self, rows_per_chunk: int = LOG_CHUNK_ROWS):
        self.lock = threading.Lock()
        self.cmd = _ChunkedStore(self._CMD_COLUMNS, rows_per_chunk)
        self.st = _ChunkedStore(self._STATE_COLUMNS, rows_per_chunk)
        self.jit = _ChunkedStore({"ns": (np.int64, ())}, rows_per_chunk)
        self.events: list[dict] = []

    def event(self, t_ns: int, kind: str, detail: str) -> None:
        with self.lock:
            self.events.append({"t_ns": int(t_ns), "type": kind, "detail": detail})

    def log_cmd(self, t_ns, frame, q, dq, tau, kp, kd):
        with self.lock:
            self.cmd.append(
                t_ns=int(t_ns), frame=int(frame), q=q, dq=dq, tau=tau, kp=kp, kd=kd
            )

    def log_state(self, t_ns, q, dq, ddq, tau_est, temperature, motorstate, sensor, vol):
        with self.lock:
            self.st.append(
                t_ns=int(t_ns), q=q, dq=dq, ddq=ddq, tau_est=tau_est,
                temperature=temperature, motorstate=motorstate, sensor=sensor, vol=vol,
            )

    def log_jitter(self, ns: int) -> None:
        with self.lock:
            self.jit.append(ns=int(ns))

    @property
    def cmd_t_ns(self) -> np.ndarray:
        return self.cmd.column("t_ns")

    @property
    def cmd_frame(self) -> np.ndarray:
        return self.cmd.column("frame")

    @property
    def st_t_ns(self) -> np.ndarray:
        return self.st.column("t_ns")

    @property
    def jitter_ns(self) -> np.ndarray:
        return self.jit.column("ns")

    def build_log(self, joint_names, plan_sha256, meta):
        with self.lock:
            return fmt.HwLog(
                joint_names=joint_names,
                plan_sha256=plan_sha256,
                meta=meta,
                cmd_t_ns=self.cmd.column("t_ns"),
                cmd_frame=self.cmd.column("frame"),
                cmd_q=self.cmd.column("q"),
                cmd_dq=self.cmd.column("dq"),
                cmd_tau=self.cmd.column("tau"),
                cmd_kp=self.cmd.column("kp"),
                cmd_kd=self.cmd.column("kd"),
                st_t_ns=self.st.column("t_ns"),
                st_q=self.st.column("q"),
                st_dq=self.st.column("dq"),
                st_ddq=self.st.column("ddq"),
                st_tau_est=self.st.column("tau_est"),
                st_temperature=self.st.column("temperature"),
                st_motorstate=self.st.column("motorstate"),
                st_sensor=self.st.column("sensor"),
                st_vol=self.st.column("vol"),
                events=list(self.events),
            )


class MockBackend:
    """Per-joint second-order mock in WIRE convention, 1 kHz state thread.

    Internally holds the 22 motors in wire order; ``send`` maps the
    model-convention command through ``wire_mapping`` signs and the state
    thread maps measured wire state back, so both sign paths are exercised.
    """

    def __init__(
        self,
        bundle: Bundle,
        convention: str,
        sink: LogSink,
        *,
        delay_s: float = 0.020,
        rate_hz: float = 1000.0,
        start_q_model=None,
        armature: float = 0.02,
        damping: float = 0.003,
        frictionloss: float = 0.04,
    ):
        self.indices, self.signs = wire_mapping(bundle.joint_names, convention)
        self.sink = sink
        self.delay_s = float(delay_s)
        self.period = 1.0 / float(rate_hz)
        self.armature = float(armature)
        self.damping = float(damping)
        self.frictionloss = float(frictionloss)
        start = np.zeros(NUM_JOINTS) if start_q_model is None else np.asarray(start_q_model, float)
        self.q_w = np.zeros(NUM_JOINTS)
        self.q_w[self.indices] = start * self.signs
        self.dq_w = np.zeros(NUM_JOINTS)
        self._temperature = np.full((NUM_JOINTS, 2), 35.0)
        self._cmd_w = (np.zeros(NUM_JOINTS),) * 5  # kp, kd, q, dq, tau (wire)
        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _to_wire(self, values):
        wire = np.zeros(NUM_JOINTS)
        wire[self.indices] = np.asarray(values, dtype=np.float64) * self.signs
        return wire

    def _reorder(self, values):  # gains: reorder only, no sign
        wire = np.zeros(NUM_JOINTS)
        wire[self.indices] = np.asarray(values, dtype=np.float64)
        return wire

    def connect(self, timeout=5.0):
        self._thread = threading.Thread(target=self._run, daemon=True, name="mock-z5")
        self._thread.start()
        deadline = time.monotonic() + timeout
        while self.latest() is None:
            if time.monotonic() > deadline:
                raise TimeoutError("mock backend produced no state")
            time.sleep(0.001)
        return self.latest()["q"].copy()

    def _run(self):
        tau_w = np.zeros(NUM_JOINTS)
        ddq_w = np.zeros(NUM_JOINTS)
        t_next = time.monotonic()
        t_prev = t_next
        while not self._stop.is_set():
            now = time.monotonic()
            # Integrate over the real elapsed time: if the publisher thread
            # cannot hold rate_hz under load the response stays wall-clock
            # correct and the published rate just honestly degrades.
            sim_dt = min(now - t_prev, 0.004)
            t_prev = now
            with self._lock:
                while self._queue and self._queue[0][0] <= now - self.delay_s:
                    self._cmd_w = self._queue.popleft()[1]
                cmd = self._cmd_w
            kp_w, kd_w, q_t, dq_t, tau_ff = cmd
            force = kp_w * (q_t - self.q_w) + kd_w * (dq_t - self.dq_w) + tau_ff
            tau_w = force.copy()
            net = force - self.damping * self.dq_w - self.frictionloss * np.tanh(self.dq_w / 0.01)
            ddq_w = net / self.armature
            self.dq_w += ddq_w * sim_dt
            self.q_w += self.dq_w * sim_dt
            idx, sgn = self.indices, self.signs
            q_m = self.q_w[idx] * sgn
            dq_m = self.dq_w[idx] * sgn
            with self._lock:
                temp = self._temperature.copy()
            self.sink.log_state(
                time.monotonic_ns(),
                q_m,
                dq_m,
                ddq_w[idx] * sgn,
                tau_w[idx] * sgn,
                temp,
                np.full(NUM_JOINTS, POSITION_MODE, dtype=np.uint32),
                np.zeros((NUM_JOINTS, 2), dtype=np.uint32),
                np.full(NUM_JOINTS, 24.0, dtype=np.float32),
            )
            with self._lock:
                self._latest = {"q": q_m.copy(), "dq": dq_m.copy(), "temperature": temp, "stamp": now}
            t_next += self.period
            sleep_t = t_next - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                # Fell behind: drop the catch-up rather than burst-publishing.
                t_next = time.monotonic() + self.period

    def send(self, q, dq, tau, kp, kd):
        cmd = (
            self._reorder(kp),
            self._reorder(kd),
            self._to_wire(q),
            self._to_wire(dq),
            self._to_wire(tau),
        )
        with self._lock:
            self._queue.append((time.monotonic(), cmd))

    def release(self):
        self.send(
            np.zeros(NUM_JOINTS),
            np.zeros(NUM_JOINTS),
            np.zeros(NUM_JOINTS),
            np.zeros(NUM_JOINTS),
            np.zeros(NUM_JOINTS),
        )

    def read(self):
        latest = self.latest()
        if latest is None:
            raise TimeoutError("mock backend has no state yet")
        return latest["q"].copy(), latest["dq"].copy()

    def latest(self):
        with self._lock:
            return self._latest

    def inject_temperature(self, value: float = 90.0, joint: int | None = None):
        """Test hook: force motor temperatures to ``value`` (C)."""
        with self._lock:
            if joint is None:
                self._temperature[:] = value
            else:
                self._temperature[int(joint)] = value

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class RealBackend:
    """Dex5-S DDS backend mirroring ``Z5HandDriver`` conventions.

    Sends full per-joint MIT commands (q/dq/tau/kp/kd on ``rt/dex5/right/cmd``)
    and logs every ``rt/dex5/right/state`` HandState_ message with a
    monotonic_ns stamp. unitree_sdk2py is imported lazily so the mock path
    and tests never need it.
    """

    CMD_TOPIC = "rt/dex5/right/cmd"
    STATE_TOPIC = "rt/dex5/right/state"

    def __init__(
        self,
        bundle: Bundle,
        convention: str,
        sink: LogSink,
        *,
        interface=None,
        domain: int = 0,
        sdk_path=None,
        startup_timeout: float = 5.0,
    ):
        self.indices, self.signs = wire_mapping(bundle.joint_names, convention)
        self.sink = sink
        if sdk_path:
            sys.path.insert(0, str(sdk_path))
        from unitree_sdk2py.core.channel import (  # noqa: PLC0415
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (  # noqa: PLC0415
            HandCmd_,
            HandState_,
            MotorCmd_,
        )

        self._MotorCmd = MotorCmd_
        self._message = HandCmd_(
            [MotorCmd_(POSITION_MODE, 0.0, 0.0, 0.0, 0.0, 0.0, 0) for _ in range(NUM_JOINTS)],
            [0] * 4,
        )
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._error: str | None = None
        ChannelFactoryInitialize(domain, interface)
        self.publisher = ChannelPublisher(self.CMD_TOPIC, HandCmd_)
        self.subscriber = ChannelSubscriber(self.STATE_TOPIC, HandState_)
        try:
            self.publisher.Init()
            self.subscriber.Init(self._on_state)
            deadline = time.monotonic() + startup_timeout
            while self.latest() is None:
                if self._error:
                    raise RuntimeError(self._error)
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"no HandState_ on {self.STATE_TOPIC} within {startup_timeout:.1f} s"
                    )
                time.sleep(0.001)
        except BaseException:
            self.close()
            raise

    def _on_state(self, message):
        try:
            motors = message.motor_state
            q_w = np.asarray([v.q for v in motors], dtype=np.float64)
            dq_w = np.asarray([v.dq for v in motors], dtype=np.float64)
            ddq_w = np.asarray([v.ddq for v in motors], dtype=np.float64)
            tau_w = np.asarray([v.tau_est for v in motors], dtype=np.float64)
            temp = np.asarray([v.temperature for v in motors], dtype=np.float64)
            motorstate = np.asarray([v.motorstate for v in motors], dtype=np.uint32)
            sensor = np.asarray([v.sensor for v in motors], dtype=np.uint32)
            vol = np.asarray([v.vol for v in motors], dtype=np.float64)
            idx, sgn = self.indices, self.signs
            q_m = q_w[idx] * sgn
            dq_m = dq_w[idx] * sgn
            self.sink.log_state(
                time.monotonic_ns(),
                q_m,
                dq_m,
                ddq_w[idx] * sgn,
                tau_w[idx] * sgn,
                temp,
                motorstate,
                sensor,
                vol[idx],
            )
            with self._lock:
                self._latest = {
                    "q": q_m,
                    "dq": dq_m,
                    "temperature": temp,
                    "stamp": time.monotonic(),
                }
        except (AttributeError, TypeError, ValueError) as exc:
            with self._lock:
                self._error = str(exc)

    def _write(self, kp, kd, q, dq, tau):
        idx, sgn = self.indices, self.signs
        wq = np.zeros(NUM_JOINTS)
        wq[idx] = q * sgn
        wdq = np.zeros(NUM_JOINTS)
        wdq[idx] = dq * sgn
        wtau = np.zeros(NUM_JOINTS)
        wtau[idx] = tau * sgn
        wkp = np.zeros(NUM_JOINTS)
        wkp[idx] = kp
        wkd = np.zeros(NUM_JOINTS)
        wkd[idx] = kd
        for i, command in enumerate(self._message.motor_cmd):
            command.mode = POSITION_MODE
            command.q = float(wq[i])
            command.dq = float(wdq[i])
            command.tau = float(wtau[i])
            command.kp = float(wkp[i])
            command.kd = float(wkd[i])
        self.publisher.Write(self._message)

    def send(self, q, dq, tau, kp, kd):
        self._write(
            np.asarray(kp, dtype=np.float64),
            np.asarray(kd, dtype=np.float64),
            np.asarray(q, dtype=np.float64),
            np.asarray(dq, dtype=np.float64),
            np.asarray(tau, dtype=np.float64),
        )

    def release(self):
        z = np.zeros(NUM_JOINTS)
        self._write(z, z, z, z, z)

    def read(self):
        latest = self.latest()
        if latest is None:
            raise TimeoutError("no Z5 state yet")
        return latest["q"].copy(), latest["dq"].copy()

    def connect(self, timeout=5.0):
        # The DDS channels are opened in __init__ (first state awaited there).
        return self.read()[0]

    def latest(self):
        with self._lock:
            return self._latest

    def close(self):
        for channel in (getattr(self, "subscriber", None), getattr(self, "publisher", None)):
            if channel is not None:
                try:
                    channel.Close()
                except Exception:
                    pass
        self.publisher = self.subscriber = None


def load_and_validate_plan(
    plan_path,
    bundle: Bundle,
    convention: str,
    *,
    allow_torque_mode: bool = False,
    max_tau: float = 0.15,
    max_speed: float = 8.0,
    max_kp: float = 4.0,
    max_kd: float = 0.2,
):
    """Load a plan and enforce the playback contract; returns (plan, limits)."""
    plan = fmt.load_plan(plan_path)
    if tuple(plan.joint_names) != tuple(bundle.joint_names):
        raise ValueError("plan joint names do not match the bundle joint names")
    limits = bundle.limits.astype(np.float64).copy()
    if convention == "urdf20261103":
        i = bundle.joint_names.index("right_thumb_yaw1_joint")
        limits[i, 1] = min(limits[i, 1], 1.57)
    errors = fmt.validate_plan(plan, limits)
    if np.any(plan.kp < 0) or np.any(plan.kd < 0):
        errors.append("plan contains negative gains")
    kp_max = float(plan.kp.max()) if plan.num_frames else 0.0
    kd_max = float(plan.kd.max()) if plan.num_frames else 0.0
    if kp_max > max_kp + 1e-9:
        errors.append(f"plan kp max {kp_max:.3f} exceeds --max-kp {max_kp}")
    if kd_max > max_kd + 1e-9:
        errors.append(f"plan kd max {kd_max:.3f} exceeds --max-kd {max_kd}")
    n_tau = int((np.abs(plan.tau) > 0.0).any(axis=1).sum()) if plan.num_frames else 0
    if n_tau:
        if not allow_torque_mode:
            errors.append(
                f"{n_tau} frames with nonzero tau; rerun with --allow-torque-mode"
            )
        elif float(np.abs(plan.tau).max()) > max_tau + 1e-9:
            errors.append(
                f"tau max {float(np.abs(plan.tau).max()):.4f} "
                f"exceeds --max-tau {max_tau}"
            )
    dq_max = float(np.abs(plan.dq).max()) if plan.num_frames else 0.0
    if dq_max > max_speed + 1e-9:
        errors.append(f"plan dq max {dq_max:.4f} rad/s exceeds --max-speed {max_speed}")
    step = np.abs(np.diff(plan.q.astype(np.float64), axis=0)).max() if plan.num_frames > 1 else 0.0
    if step > max_speed / plan.rate_hz + 1e-9:
        errors.append(
            f"plan exceeds --max-speed {max_speed}: per-frame dq {step:.5f} rad"
        )
    if errors:
        raise ValueError("plan rejected: " + "; ".join(errors))
    return plan, limits


class PlanPlayer:
    """Absolute-time plan playback with guard, prompts and abort handling."""

    def __init__(
        self,
        plan,
        backend,
        sink: LogSink,
        limits: np.ndarray,
        *,
        home_seconds: float = 2.0,
        home_kp: float = 2.0,
        home_kd: float = 0.04,
        max_speed: float = 8.0,
        max_kp: float = 4.0,
        max_kd: float = 0.2,
        max_late_s: float | None = None,
        temperature_limit: float = 65.0,
        track_error_limit: float = 0.6,
        stale_timeout: float = 0.2,
        allow_torque_mode: bool = False,
        max_tau: float = 0.15,
        max_home_delta: float = 0.8,
        allow_large_home: bool = False,
        is_real: bool = False,
        no_prompt: bool = False,
        input_fn=input,
    ):
        self.plan = plan
        self.backend = backend
        self.sink = sink
        self.limits = np.asarray(limits, dtype=np.float64)
        self.rate = float(plan.rate_hz)
        self.dt = 1.0 / self.rate
        self.home_seconds = float(home_seconds)
        self.home_kp = float(home_kp)
        self.home_kd = float(home_kd)
        self.max_kp = float(max_kp)
        self.max_kd = float(max_kd)
        if self.home_kp > self.max_kp + 1e-9 or self.home_kd > self.max_kd + 1e-9:
            raise ValueError(
                f"home gains (kp={self.home_kp}, kd={self.home_kd}) exceed "
                f"--max-kp/--max-kd ({self.max_kp}/{self.max_kd})"
            )
        self.max_speed = float(max_speed)
        self.max_step = self.max_speed * self.dt
        self.max_late = self.dt if max_late_s is None else float(max_late_s)
        self.temperature_limit = float(temperature_limit)
        self.track_error_limit = float(track_error_limit)
        self.stale_timeout = float(stale_timeout)
        self.max_tau = float(max_tau)
        self.allow_torque_mode = bool(allow_torque_mode)
        self.max_home_delta = float(max_home_delta)
        self.allow_large_home = bool(allow_large_home)
        self.is_real = bool(is_real)
        self.prompts = not no_prompt
        self.input_fn = input_fn
        self.interventions = 0
        self.guard_events = 0
        self.schedule_slips = 0
        self.slip_total_s = 0.0
        self._slip_s = 0.0  # schedule offset shared by every timed loop
        self._last_send_q: np.ndarray | None = None
        self._last_send_t: float | None = None
        self._mode_prev = np.ones(NUM_JOINTS, dtype=bool)  # True = position
        self._torque_hold: dict[int, float] = {}
        self._over_since: dict[int, float] = {}
        self._last_cmd: dict | None = None

    # --- timing ----------------------------------------------------------
    def _sleep_until(self, due: float):
        # Blocking sleeps overshoot by several ms on a loaded scheduler —
        # even the nominal 1 ms coarse nap was observed ~5 ms late. Waits up
        # to ~2 plan periods therefore spin entirely; only longer waits
        # (slow plans, holds) coarse-sleep first.
        while True:
            left = due - time.monotonic()
            if left <= 0:
                return
            if left > 0.005:
                time.sleep(left - 0.004)
            else:
                # Spin the last ~5 ms; sleep(0) yields the GIL so the state
                # publisher thread is not starved by the busy wait.
                while time.monotonic() < due:
                    time.sleep(0)
                return

    # --- guard -----------------------------------------------------------
    def _guard(self, q, dq, tau, kp, kd):
        q = np.asarray(q, dtype=np.float64)
        clipped = np.clip(q, self.limits[:, 0], self.limits[:, 1])
        moved = np.abs(clipped - q) > 1e-9
        if moved.any():
            self.interventions += int(moved.sum())
            q = clipped
        dq = np.asarray(dq, dtype=np.float64).copy()
        over_dq = np.abs(dq) > self.max_speed + 1e-9
        if over_dq.any():
            self.interventions += int(over_dq.sum())
            dq = np.clip(dq, -self.max_speed, self.max_speed)
        tau = np.asarray(tau, dtype=np.float64).copy()
        over_tau = np.abs(tau) > self.max_tau + 1e-9
        if over_tau.any():
            self.interventions += int(over_tau.sum())
            tau = np.clip(tau, -self.max_tau, self.max_tau)
        kp = np.asarray(kp, dtype=np.float64).copy()
        kd = np.asarray(kd, dtype=np.float64).copy()
        over_gains = (kp > self.max_kp + 1e-9) | (kd > self.max_kd + 1e-9)
        if over_gains.any():
            self.interventions += int(over_gains.sum())
            kp = np.clip(kp, 0.0, self.max_kp)
            kd = np.clip(kd, 0.0, self.max_kd)
        latest = self.backend.latest()
        # Torque-mode joints: latch to a position hold at the wall once the
        # measured pose leaves [lo+margin, hi-margin] (until segment change).
        torque = (kp <= 0.0) & (np.abs(tau) > 0.0)
        if np.any(torque) and latest is not None:
            measured = latest["q"]
            lo_w = self.limits[:, 0] + TORQUE_WALL_MARGIN
            hi_w = self.limits[:, 1] - TORQUE_WALL_MARGIN
            for j in np.flatnonzero(torque):
                if j in self._torque_hold:
                    wall = self._torque_hold[j]
                elif measured[j] < lo_w[j]:
                    wall = lo_w[j]
                elif measured[j] > hi_w[j]:
                    wall = hi_w[j]
                else:
                    continue
                if j not in self._torque_hold:
                    self._torque_hold[j] = wall
                    self.guard_events += 1
                    self.sink.event(
                        time.monotonic_ns(),
                        "guard",
                        f"torque-mode joint {self.plan.joint_names[j]} left "
                        f"[{lo_w[j]:.3f},{hi_w[j]:.3f}]; holding at wall {wall:.3f}",
                    )
                q[j], dq[j], tau[j] = wall, 0.0, 0.0
                kp[j], kd[j] = self.home_kp, self.home_kd
        position = kp > 0.0
        # Torque -> position switch: the plan's q for the torque stretch says
        # nothing about where the joint actually is. Seed the rate limiter
        # from the measured position so the first position-mode target ramps
        # away from reality at <= max_speed per frame instead of jumping.
        if self._last_send_q is not None and latest is not None:
            switched = position & ~self._mode_prev
            for j in np.flatnonzero(switched):
                self._last_send_q[j] = latest["q"][j]
                self.guard_events += 1
                self.sink.event(
                    time.monotonic_ns(),
                    "guard",
                    f"joint {self.plan.joint_names[j]} torque->position; "
                    f"rate limit seeded at measured q {latest['q'][j]:.3f}",
                )
        if self._last_send_q is not None:
            clamped = np.clip(q, self._last_send_q - self.max_step, self._last_send_q + self.max_step)
            moved = np.abs(clamped - q) > 1e-9
            if np.any(moved & position):
                self.interventions += int((moved & position).sum())
                q = np.where(position, clamped, q)
        self._mode_prev = position.copy()
        self._last_send_q = q.copy()
        return q, dq, tau, kp, kd

    def _check_safety(self):
        latest = self.backend.latest()
        if latest is None:
            raise AbortError("no state received")
        if time.monotonic() - latest["stamp"] > self.stale_timeout:
            raise AbortError(f"state stale > {self.stale_timeout} s")
        if np.any(latest["temperature"] > self.temperature_limit):
            raise AbortError(f"motor temperature over {self.temperature_limit} C")
        if self._last_cmd is not None:
            kp = self._last_cmd["kp"]
            pos = kp > 0.0
            if pos.any():
                err = np.abs(latest["q"] - self._last_cmd["q"])
                now = time.monotonic()
                for j in np.flatnonzero(pos & (err > self.track_error_limit)):
                    self._over_since.setdefault(j, now)
                    if now - self._over_since[j] > TRACK_ERROR_WINDOW_S:
                        raise AbortError(
                            f"tracking error {err[j]:.3f} rad on joint "
                            f"{self.plan.joint_names[j]} sustained > {TRACK_ERROR_WINDOW_S} s"
                        )
                for j in list(self._over_since):
                    if not (pos[j] and err[j] > self.track_error_limit):
                        del self._over_since[j]

    # --- sending ---------------------------------------------------------
    def _send_at(self, due: float, frame: int, q, dq, tau, kp, kd):
        """Send at the scheduled time (``due`` + the shared slip offset).

        If the frame is already more than ``max_late`` behind schedule, the
        schedule is shifted so this frame is due one period after the last
        actual send (a ``slip`` event; ``schedule_slips``/``slip_total_ms``
        in the log meta). Frames are never skipped and never sent
        back-to-back chasing a stale deadline.
        """
        due += self._slip_s
        self._sleep_until(due)
        late = time.monotonic() - due
        # A stall can also overshoot a slip-rescheduled send, so re-check
        # lateness after every sleep (bounded: each slip shortens the wait).
        for _ in range(3):
            if late <= self.max_late:
                break
            last = self._last_send_t
            new_due = (
                max(time.monotonic(), last + self.dt)
                if last is not None
                else time.monotonic()
            )
            shift = new_due - due
            self._slip_s += shift
            self.schedule_slips += 1
            self.slip_total_s += shift
            self.sink.event(
                time.monotonic_ns(),
                "slip",
                f"send schedule slipped {shift * 1e3:.1f} ms at frame {frame}",
            )
            due = new_due
            self._sleep_until(due)
            late = time.monotonic() - due
        anchored = 0.0
        if late > 0.5 * self.dt:
            # Lateness under the slip threshold still compresses the next
            # send below half a period. Re-anchor by the lateness (no slip
            # event, no frame lost) so spacing never drops under 0.5*dt.
            self._slip_s += late
            anchored = late
        q, dq, tau, kp, kd = self._guard(q, dq, tau, kp, kd)
        self.backend.send(q, dq, tau, kp, kd)
        t_ns = time.monotonic_ns()
        sent_late = t_ns / 1e9 - due
        if sent_late > 0.5 * self.dt:
            # A stall inside _guard/send lands the command on the wire late
            # even though the schedule check already passed; anchor the
            # remaining schedule to the actual send time so the next frame
            # keeps >= 0.5*dt spacing.
            self._slip_s += sent_late - anchored
            if sent_late > self.max_late:
                self.schedule_slips += 1
                self.slip_total_s += sent_late - anchored
                self.sink.event(
                    t_ns,
                    "slip",
                    f"send stalled {sent_late * 1e3:.1f} ms past due at frame {frame}",
                )
        self.sink.log_cmd(t_ns, frame, q, dq, tau, kp, kd)
        self.sink.log_jitter(t_ns - int(due * 1e9))
        self._last_send_t = t_ns / 1e9
        self._last_cmd = {"q": q, "dq": dq, "tau": tau, "kp": kp, "kd": kd}
        self._check_safety()

    def _rebase_after_prompt(self, due_raw: float) -> None:
        """Re-anchor the schedule so the next frame is due one period out."""
        self._slip_s = (time.monotonic() + self.dt) - due_raw
        self.sink.event(
            time.monotonic_ns(), "slip", "schedule re-based after prompt"
        )

    def _send_now(self, frame: int, q, dq, tau, kp, kd):
        # Undo the slip offset: "now" is the deadline, so hold/prompt sends
        # are not delayed by an accumulated schedule shift.
        self._send_at(time.monotonic() - self._slip_s, frame, q, dq, tau, kp, kd)

    def _hold(self, seconds: float, q, dq, tau, kp, kd, frame=-1):
        """Keep sending ``(q,...)`` at the plan rate for ``seconds``."""
        # Fresh now-anchored schedule: an accumulated slip must not delay a
        # (possibly emergency) hold's first send.
        self._slip_s = 0.0
        n = max(1, round(seconds * self.rate))
        t0 = time.monotonic()
        for i in range(n):
            self._send_at(t0 + i * self.dt, frame, q, dq, tau, kp, kd)

    def _hold_current(self, seconds: float):
        cmd = self._last_cmd
        if cmd is None:
            return
        self._hold(seconds, cmd["q"], cmd["dq"], cmd["tau"], cmd["kp"], cmd["kd"])

    # --- flow ------------------------------------------------------------
    def _prompt(self, text: str):
        if not self.prompts:
            return
        self.sink.event(time.monotonic_ns(), "prompt", text)
        done = threading.Event()

        def reader():
            try:
                self.input_fn(f"{text} [Enter] ")
            except EOFError:
                pass
            done.set()

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        while not done.wait(PROMPT_HOLD_PERIOD_S):
            try:
                self._send_now(-1, *self._held_cmd())
            except AbortError:
                done.set()
                raise

    def _held_cmd(self):
        cmd = self._last_cmd
        if cmd is None:
            z = np.zeros(NUM_JOINTS)
            return z, z, z, np.full(NUM_JOINTS, self.home_kp), np.full(NUM_JOINTS, self.home_kd)
        return cmd["q"], cmd["dq"], cmd["tau"], cmd["kp"], cmd["kd"]

    def _home(self):
        q_meas = np.asarray(self.backend.read()[0], dtype=np.float64)
        target0 = self.plan.q[0].astype(np.float64)
        delta = float(np.abs(q_meas - target0).max())
        print(f"homing delta: max |q_meas - frame 0| = {delta:.3f} rad", flush=True)
        if delta > self.max_home_delta and self.is_real:
            # Mock is unaffected; real hardware gates the large move.
            if self.prompts:
                reply = str(
                    self.input_fn(
                        f"homing move of {delta:.3f} rad exceeds "
                        f"--max-home-delta {self.max_home_delta} - type 'yes' to continue: "
                    )
                )
                if reply.strip().lower() != "yes":
                    raise AbortError("operator declined the large homing move")
                self.sink.event(
                    time.monotonic_ns(), "prompt", "large homing move confirmed"
                )
            elif not self.allow_large_home:
                raise AbortError(
                    f"homing delta {delta:.3f} rad exceeds --max-home-delta "
                    f"{self.max_home_delta}; rerun with prompts or --allow-large-home"
                )
        n = max(2, round(self.home_seconds * self.rate))
        kp = np.full(NUM_JOINTS, self.home_kp)
        kd = np.full(NUM_JOINTS, self.home_kd)
        z = np.zeros(NUM_JOINTS)
        t0 = time.monotonic() + HOME_LEAD_S
        for i in range(n):
            blend = 0.5 * (1.0 - math.cos(math.pi * (i + 1) / n))
            self._send_at(t0 + i * self.dt, -1, q_meas + (target0 - q_meas) * blend, z, z, kp, kd)

    def _play_frames(self):
        n = self.plan.num_frames
        t0 = time.monotonic() + HOME_LEAD_S
        prev_seg = -1
        for k in range(n):
            seg = int(self.plan.segment[k])
            if seg != prev_seg:
                info = self.plan.segments[seg]
                self.sink.event(
                    time.monotonic_ns(), "segment_start", info.get("name", str(seg))
                )
                if info.get("prompt"):
                    self._prompt(info["prompt"])
                    # The wait stalled the schedule; re-base so the next frame
                    # is due one period after the prompt returns (no burst).
                    self._rebase_after_prompt(t0 + k * self.dt)
                prev_seg = seg
                self._torque_hold.clear()
            self._send_at(
                t0 + k * self.dt,
                k,
                self.plan.q[k],
                self.plan.dq[k],
                self.plan.tau[k],
                self.plan.kp[k],
                self.plan.kd[k],
            )
        if prev_seg >= 0:
            self.sink.event(
                time.monotonic_ns(),
                "segment_end",
                self.plan.segments[prev_seg].get("name", str(prev_seg)),
            )

    def _release(self):
        z = np.zeros(NUM_JOINTS)
        try:
            self.backend.release()
        except AttributeError:
            self.backend.send(z, z, z, z, z)
        self.sink.log_cmd(time.monotonic_ns(), -1, z, z, z, z, z)
        self.sink.event(time.monotonic_ns(), "release", "kp=kd=tau=dq=0")

    def _fail_safe(self):
        try:
            q_meas = np.asarray(self.backend.read()[0], dtype=np.float64)
        except Exception:
            cmd = self._last_cmd
            q_meas = cmd["q"] if cmd is not None else np.zeros(NUM_JOINTS)
        z = np.zeros(NUM_JOINTS)
        kp = np.full(NUM_JOINTS, self.home_kp)
        kd = np.full(NUM_JOINTS, self.home_kd)
        try:
            self._hold(ABORT_HOLD_S, q_meas, z, z, kp, kd)
        except AbortError:
            # The state stream is the thing that failed; hold blind instead.
            n = max(1, round(ABORT_HOLD_S * self.rate))
            t0 = time.monotonic()
            for i in range(n):
                self._sleep_until(t0 + i * self.dt)
                try:
                    self.backend.send(q_meas, z, z, kp, kd)
                    self.sink.log_cmd(time.monotonic_ns(), -1, q_meas, z, z, kp, kd)
                except Exception:
                    break

    def run(self) -> str | None:
        """Run the full flow; returns ``abort_reason`` (None on success).

        The abort path is robust by construction: whatever raised, we try to
        fail-safe, then always try to release, and always close the backend —
        a failure in any of those steps (including a second KeyboardInterrupt)
        does not skip the remaining ones. The cyclic GC is frozen/disabled
        for the timed run so collector passes over the multi-million-row log
        buffers cannot stall a send; it is re-enabled in ``finally``.
        """
        abort_reason = None
        gc_was_enabled = gc.isenabled()
        try:
            gc.freeze()
            gc.disable()
            try:
                self._home()
                self._prompt("homing complete - press Enter to start playback")
                self._play_frames()
                self._hold_current(END_HOLD_S)
            except KeyboardInterrupt:
                abort_reason = "keyboard interrupt"
            except AbortError as exc:
                abort_reason = str(exc)
            except BaseException as exc:  # noqa: BLE001 - abort path must always run
                abort_reason = f"{type(exc).__name__}: {exc}"
            if abort_reason is not None:
                self.sink.event(time.monotonic_ns(), "abort", abort_reason)
                try:
                    self._fail_safe()
                except BaseException as exc2:  # noqa: BLE001
                    self.sink.event(
                        time.monotonic_ns(),
                        "abort",
                        f"fail-safe raised {type(exc2).__name__}: {exc2}",
                    )
            try:
                self._release()
            except BaseException as exc2:  # noqa: BLE001 - still close below
                self.sink.event(
                    time.monotonic_ns(),
                    "release_error",
                    f"{type(exc2).__name__}: {exc2}",
                )
        finally:
            try:
                self.backend.close()
            except BaseException:  # noqa: BLE001 - nothing left to do
                pass
            if gc_was_enabled:
                gc.enable()
            gc.unfreeze()
        return abort_reason


def _finish_log(
    sink: LogSink,
    plan_path,
    plan,
    backend_name,
    convention,
    guard_interventions,
    started_utc,
    notes,
    abort_reason,
    schedule_slips=0,
    slip_total_s=0.0,
):
    import hashlib
    import resource

    plan_sha = hashlib.sha256(Path(plan_path).read_bytes()).hexdigest()
    jit = np.asarray(sink.jitter_ns, dtype=np.float64) / 1e3
    if len(jit):
        jitter = {
            "p50": float(np.percentile(np.abs(jit), 50)),
            "p99": float(np.percentile(np.abs(jit), 99)),
            "max": float(np.abs(jit).max()),
        }
        max_late_ms = float(jit.max() / 1e3)
    else:
        jitter = {"p50": 0.0, "p99": 0.0, "max": 0.0}
        max_late_ms = 0.0
    # Spacing between consecutive plan-frame sends (catch-up detector).
    # ``over_*ms`` counts absolute inter-send gaps; ``excess_*ms`` counts gaps
    # beyond the nominal plan period.
    cmd_t = np.asarray(sink.cmd_t_ns, dtype=np.int64)
    cmd_f = np.asarray(sink.cmd_frame, dtype=np.int64)
    gaps = np.diff(cmd_t[cmd_f >= 0]) / 1e6 if (cmd_f >= 0).sum() > 1 else np.zeros(0)
    period_ms = 1e3 / float(plan.rate_hz)
    excess = gaps - period_ms
    spacing = {
        "period_ms": float(period_ms),
        "p50_ms": float(np.percentile(gaps, 50)) if len(gaps) else 0.0,
        "p99_ms": float(np.percentile(gaps, 99)) if len(gaps) else 0.0,
        "max_ms": float(gaps.max()) if len(gaps) else 0.0,
        "min_ms": float(gaps.min()) if len(gaps) else 0.0,
        "over_2ms": int((gaps > 2.0).sum()),
        "over_5ms": int((gaps > 5.0).sum()),
        "excess_2ms": int((excess > 2.0).sum()),
        "excess_5ms": int((excess > 5.0).sum()),
    }
    st = np.asarray(sink.st_t_ns, dtype=np.int64)
    measured_hz = 1e9 / float(np.median(np.diff(st))) if len(st) > 1 else 0.0
    meta = {
        "backend": backend_name,
        "wire_convention": convention,
        "rate_hz": float(plan.rate_hz),
        "abort_reason": abort_reason,
        "guard_interventions": int(guard_interventions),
        "send_jitter_us": jitter,
        "send_spacing": spacing,
        "max_lateness_ms": max_late_ms,
        "schedule_slips": int(schedule_slips),
        "slip_total_ms": float(slip_total_s * 1e3),
        "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "state_rate_hz_measured": measured_hz,
        "started_utc": started_utc,
        "notes": notes or "",
    }
    return sink.build_log(plan.joint_names, plan_sha, meta)


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=("real", "mock"), default="real")
    p.add_argument(
        "--wire-convention", choices=("legacy", "urdf20261103"), default="urdf20261103"
    )
    p.add_argument("--interface", default=None)
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--sdk-path", type=Path, default=None)
    p.add_argument(
        "--bundle",
        type=Path,
        default=VENDOR_ROOT / "generated/deploy/z5-pen-spin",
    )
    p.add_argument("--max-speed", type=float, default=8.0)
    p.add_argument("--max-kp", type=float, default=4.0)
    p.add_argument("--max-kd", type=float, default=0.2)
    p.add_argument(
        "--max-late-ms",
        type=float,
        default=None,
        help="slip the schedule when a send is later than this (default: one plan period)",
    )
    p.add_argument("--temperature-limit", type=float, default=65.0)
    p.add_argument("--track-error-limit", type=float, default=0.6)
    p.add_argument("--stale-timeout", type=float, default=0.2)
    p.add_argument("--home-seconds", type=float, default=2.0)
    p.add_argument("--home-kp", type=float, default=2.0)
    p.add_argument("--home-kd", type=float, default=0.04)
    p.add_argument("--max-home-delta", type=float, default=0.8)
    p.add_argument("--allow-large-home", action="store_true")
    p.add_argument("--no-prompt", action="store_true")
    p.add_argument("--allow-torque-mode", action="store_true")
    p.add_argument("--max-tau", type=float, default=0.15)
    p.add_argument("--rt-priority", type=int, default=None,
                   help="try SCHED_FIFO with this priority (warns on failure)")
    p.add_argument("--mock-delay-ms", type=float, default=20.0)
    p.add_argument("--mock-state-hz", type=float, default=1000.0)
    p.add_argument("--startup-timeout", type=float, default=5.0)
    p.add_argument("--notes", default="")
    return p


def _try_rt_priority(priority) -> bool:
    """Best-effort SCHED_FIFO; returns False (and warns) on any failure."""
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(priority)))
        print(f"SCHED_FIFO priority {priority} set", flush=True)
        return True
    except (AttributeError, OSError) as exc:
        print(
            f"warning: could not set SCHED_FIFO priority {priority}: {exc}",
            file=sys.stderr,
        )
        return False


def run_playback(args) -> tuple:
    """Validate, connect, play; returns (log, abort_reason, player)."""
    from datetime import datetime, timezone

    started_utc = datetime.now(timezone.utc).isoformat()
    if getattr(args, "rt_priority", None):
        _try_rt_priority(args.rt_priority)
    bundle = Bundle(args.bundle)
    plan, limits = load_and_validate_plan(
        args.plan,
        bundle,
        args.wire_convention,
        allow_torque_mode=args.allow_torque_mode,
        max_tau=args.max_tau,
        max_speed=args.max_speed,
        max_kp=getattr(args, "max_kp", 4.0),
        max_kd=getattr(args, "max_kd", 0.2),
    )
    sink = LogSink()
    if args.backend == "mock":
        start = plan.meta.get("park_pose")
        backend = MockBackend(
            bundle,
            args.wire_convention,
            sink,
            delay_s=args.mock_delay_ms / 1e3,
            rate_hz=args.mock_state_hz,
            start_q_model=np.asarray(start, dtype=np.float64) if start is not None else None,
        )
    else:
        backend = RealBackend(
            bundle,
            args.wire_convention,
            sink,
            interface=args.interface,
            domain=args.domain,
            sdk_path=args.sdk_path,
            startup_timeout=args.startup_timeout,
        )
    player = PlanPlayer(
        plan,
        backend,
        sink,
        limits,
        home_seconds=args.home_seconds,
        home_kp=args.home_kp,
        home_kd=args.home_kd,
        max_speed=args.max_speed,
        max_kp=getattr(args, "max_kp", 4.0),
        max_kd=getattr(args, "max_kd", 0.2),
        max_late_s=(args.max_late_ms / 1e3) if getattr(args, "max_late_ms", None) else None,
        temperature_limit=args.temperature_limit,
        track_error_limit=args.track_error_limit,
        stale_timeout=args.stale_timeout,
        allow_torque_mode=args.allow_torque_mode,
        max_tau=args.max_tau,
        max_home_delta=getattr(args, "max_home_delta", 0.8),
        allow_large_home=getattr(args, "allow_large_home", False),
        is_real=args.backend == "real",
        no_prompt=args.no_prompt or args.backend == "mock",
    )
    try:
        backend.connect(timeout=args.startup_timeout)
    except Exception as exc:  # noqa: BLE001 - still save a (nearly empty) log
        abort_reason = f"connect: {type(exc).__name__}: {exc}"
        try:
            backend.close()
        except Exception:
            pass
        log = _finish_log(
            sink, args.plan, plan, args.backend, args.wire_convention,
            0, started_utc, args.notes, abort_reason,
        )
        return log, abort_reason, player
    try:
        abort_reason = player.run()
    except BaseException as exc:  # noqa: BLE001 - always persist the partial log
        abort_reason = f"run: {type(exc).__name__}: {exc}"
    log = _finish_log(
        sink,
        args.plan,
        plan,
        args.backend,
        args.wire_convention,
        player.interventions,
        started_utc,
        args.notes,
        abort_reason,
        schedule_slips=player.schedule_slips,
        slip_total_s=player.slip_total_s,
    )
    return log, abort_reason, player


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        log, abort_reason, _player = run_playback(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    fmt.save_log(log, args.out)
    jit = log.meta["send_jitter_us"]
    sp = log.meta.get("send_spacing", {})
    print(
        f"wrote {args.out} ({len(log.cmd_t_ns)} cmd rows, {len(log.st_t_ns)} state rows, "
        f"state {log.meta['state_rate_hz_measured']:.0f} Hz, "
        f"jitter us p50={jit['p50']:.0f} p99={jit['p99']:.0f} max={jit['max']:.0f}, "
        f"spacing ms p50={sp.get('p50_ms', 0):.2f} p99={sp.get('p99_ms', 0):.2f} "
        f"max={sp.get('max_ms', 0):.2f} min={sp.get('min_ms', 0):.2f} "
        f"over5ms={sp.get('over_5ms', 0)} excess5ms={sp.get('excess_5ms', 0)}, "
        f"slips={log.meta.get('schedule_slips', 0)} "
        f"({log.meta.get('slip_total_ms', 0):.1f} ms), "
        f"peak RSS {log.meta.get('peak_rss_kb', 0) // 1024} MB)"
    )
    if abort_reason:
        print(f"ABORTED: {abort_reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
