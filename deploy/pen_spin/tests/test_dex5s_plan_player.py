# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Mock-backend checks for the Dex5S sysid plan player (no hardware)."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import dex5s_plan_player as zp  # noqa: E402
import dex5s_sysid_formats as fmt  # noqa: E402
from deploy.dex5s_pen_spin.lib.core import Bundle  # noqa: E402

BUNDLE_PATH = zp.VENDOR_ROOT / "generated/deploy/dex5s-ctx1126-supermimic-t1-emf-u0800"
RATE = 100.0
HOME_KP, HOME_KD = 2.0, 0.04


@pytest.fixture(scope="module")
def bundle():
    return Bundle(BUNDLE_PATH)


def _neutral_q(bundle, n=None):
    """Per-joint pose close to zero, strictly inside the bundle's calibrated limits."""
    lo = bundle.limits[:, 0].astype(np.float64) + 0.02
    hi = bundle.limits[:, 1].astype(np.float64) - 0.02
    q0 = np.clip(np.zeros(22, dtype=np.float64), lo, hi)
    return q0 if n is None else np.tile(q0, (n, 1))


def _plan(path, q, segments, *, rate_hz=RATE, kp=2.0, kd=0.04, tau=None, bundle=None):
    """Write a minimal dex5s_excitation_plan_v1; ``segments`` carry an extra ``n``.

    ``kp``/``kd``/``tau`` may be scalars or full (n, 22) arrays.
    """
    bundle = bundle or Bundle(BUNDLE_PATH)
    n = len(q)

    def rows(value):
        return np.broadcast_to(np.asarray(value, dtype=np.float32), (n, 22)).copy()

    plan = fmt.ExcitationPlan(
        joint_names=bundle.joint_names,
        rate_hz=rate_hz,
        q=np.asarray(q, dtype=np.float32),
        dq=np.zeros((n, 22), dtype=np.float32),
        tau=np.zeros((n, 22), dtype=np.float32) if tau is None else rows(tau),
        kp=rows(kp),
        kd=rows(kd),
        segment=np.repeat(np.arange(len(segments)), [s["n"] for s in segments]).astype(np.int32),
        segments=[{k: v for k, v in s.items() if k != "n"} for s in segments],
        meta={"generator": "test", "park_pose": _neutral_q(bundle).tolist()},
    )
    fmt.save_plan(plan, path)
    return path


def _smooth_two_segment(bundle):
    """Two short segments: joint 5 moves up, joint 6 moves down (continuous)."""
    n_seg = 20
    q0 = _neutral_q(bundle)
    a1 = np.tile(q0, (n_seg, 1))
    a1[:, 5] = q0[5] + 0.1 * (0.5 - 0.5 * np.cos(np.pi * np.arange(n_seg) / n_seg))
    q_mid = q0.copy()
    q_mid[5] = q0[5] + 0.1
    a2 = np.tile(q_mid, (n_seg, 1))
    a2[:, 6] = q0[6] - 0.08 * (0.5 - 0.5 * np.cos(np.pi * np.arange(n_seg) / n_seg))
    q = np.vstack([np.tile(q0, (2, 1)), a1, a2])
    segments = [
        {"name": "seg0", "kind": "ramps", "moving_joints": [bundle.joint_names[5]],
         "params": {}, "prompt": None, "n": n_seg + 2},
        {"name": "seg1", "kind": "ramps", "moving_joints": [bundle.joint_names[6]],
         "params": {}, "prompt": None, "n": n_seg},
    ]
    return q, segments


def _player(plan_path, bundle, sink=None, start_q=None, **kw):
    """PlanPlayer + MockBackend wired together (no CLI layer)."""
    plan, limits = zp.load_and_validate_plan(plan_path, bundle, "urdf20261103")
    sink = sink or zp.LogSink()
    backend = zp.MockBackend(
        bundle, "urdf20261103", sink, delay_s=0.02, start_q_model=start_q
    )
    kw.setdefault("home_seconds", 0.2)
    kw.setdefault("home_kp", HOME_KP)
    kw.setdefault("home_kd", HOME_KD)
    kw.setdefault("no_prompt", True)
    player = zp.PlanPlayer(plan, backend, sink, limits, **kw)
    return player, backend, sink


def _plan_frame_gaps_ms(log):
    """ms gaps between consecutive plan-frame sends (cmd_frame >= 0)."""
    t = log.cmd_t_ns[log.cmd_frame >= 0]
    return np.diff(t.astype(np.float64)) / 1e6


def _args(tmp_path, plan_path, **over):
    argv = [
        "--plan", str(plan_path),
        "--out", str(tmp_path / "log.npz"),
        "--backend", "mock",
        "--wire-convention", "urdf20261103",
        "--no-prompt",
        "--home-seconds", "0.2",
        "--bundle", str(BUNDLE_PATH),
    ]
    for key, value in over.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is not False:
            argv += [flag, str(value)]
    return zp.build_arg_parser().parse_args(argv)


def test_mock_end_to_end(tmp_path, bundle):
    q, segments = _smooth_two_segment(bundle)
    plan_path = _plan(tmp_path / "plan.npz", q, segments)
    args = _args(tmp_path, plan_path)
    log, abort_reason, _player = zp.run_playback(args)
    fmt.save_log(log, args.out)
    assert abort_reason is None
    loaded = fmt.load_log(args.out)

    assert loaded.meta["backend"] == "mock"
    assert loaded.meta["abort_reason"] is None
    assert loaded.meta["wire_convention"] == "urdf20261103"
    assert loaded.plan_sha256

    # Monotonic timestamps on both streams.
    assert np.all(np.diff(loaded.cmd_t_ns) > 0)
    assert np.all(np.diff(loaded.st_t_ns) > 0)

    # Every plan frame sent exactly once.
    sent = loaded.cmd_frame[loaded.cmd_frame >= 0]
    counts = np.bincount(sent, minlength=len(q))
    assert (counts == 1).all(), f"frames sent != once: {np.flatnonzero(counts != 1)}"

    # Homing rows precede the first plan-frame row; release row is last.
    first0 = int(np.flatnonzero(loaded.cmd_frame == 0)[0])
    assert first0 > 0
    assert (loaded.cmd_frame[:first0] == -1).all()
    assert loaded.cmd_frame[-1] == -1
    assert np.all(loaded.cmd_kp[-1] == 0)
    assert np.all(loaded.cmd_kd[-1] == 0)
    assert np.all(loaded.cmd_tau[-1] == 0)
    assert any(e["type"] == "release" for e in loaded.events)

    # 1 kHz state stream over the ~1 s run.
    assert len(loaded.st_t_ns) > 500
    assert 500.0 < loaded.meta["state_rate_hz_measured"] < 2000.0

    # The mock tracked the final commanded pose (segment events recorded).
    assert any(e["type"] == "segment_start" for e in loaded.events)
    final = loaded.st_q[-1]
    last_target = loaded.cmd_q[np.flatnonzero(loaded.cmd_frame >= 0)[-1]]
    assert np.max(np.abs(final - last_target)) < 0.05


def test_urdf20261103_sign_roundtrip(tmp_path, bundle):
    """A reversed joint (thumb pitch, sign -1) must track in model convention."""
    n = 40
    q = _neutral_q(bundle, n)
    q[:, 0] = -0.05 - 0.08 * (0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n))
    segments = [
        {"name": "move", "kind": "ramps", "moving_joints": [bundle.joint_names[0]],
         "params": {}, "prompt": None, "n": n},
    ]
    plan_path = _plan(tmp_path / "plan.npz", q, segments)
    args = _args(tmp_path, plan_path)
    log, abort_reason, _player = zp.run_playback(args)
    assert abort_reason is None
    # Model-convention state follows the model-convention command: the wire
    # sign flip was applied on both send and receive paths.
    tail = slice(int(0.8 * len(log.st_t_ns)), None)
    err = np.abs(log.st_q[tail, 0] - np.median(log.cmd_q[log.cmd_frame >= 0, 0][-10:]))
    assert np.median(err) < 0.02
    assert log.st_q[-1, 0] < -0.10  # moved negative in model convention


def test_limit_violation_rejected(tmp_path, bundle):
    q = np.tile(bundle.limits[:, 0].astype(np.float64), (5, 1))
    q[:, 5] = bundle.limits[5, 1] + 0.2  # above the upper limit, held constant
    segments = [
        {"name": "bad", "kind": "hold", "moving_joints": [bundle.joint_names[5]],
         "params": {}, "prompt": None, "n": 5},
    ]
    plan_path = _plan(tmp_path / "plan.npz", q, segments)
    args = _args(tmp_path, plan_path)
    with pytest.raises(ValueError, match="plan rejected"):
        zp.run_playback(args)


def test_torque_mode_gating(tmp_path, bundle):
    n = 150  # 1.5 s at 100 Hz; enough for tau=-0.1 to push joint 5 to the wall
    q = _neutral_q(bundle, n)
    kp = np.full((n, 22), HOME_KP, dtype=np.float32)
    kp[:, 5] = 0.0  # joint 5 in torque mode (kp==0 & tau!=0)
    tau = np.zeros((n, 22))
    tau[:, 5] = -0.1
    segments = [
        {"name": "torque", "kind": "hold", "moving_joints": [bundle.joint_names[5]],
         "params": {}, "prompt": None, "n": n},
    ]
    plan_path = _plan(tmp_path / "plan.npz", q, segments, kp=kp, tau=tau)
    args = _args(tmp_path, plan_path)
    with pytest.raises(ValueError, match="torque-mode"):
        zp.run_playback(args)

    args = _args(tmp_path, plan_path, allow_torque_mode=True, max_tau=0.15)
    log, abort_reason, player = zp.run_playback(args)
    assert abort_reason is None
    # After the wall latch, joint 5 rows carry position gains again and the
    # target settles at the lower wall (lo + TORQUE_WALL_MARGIN).
    first_plan_row = int(np.flatnonzero(log.cmd_frame >= 0)[0])
    latched = np.flatnonzero((log.cmd_kp[:, 5] > 0.0) & (np.arange(len(log.cmd_kp)) > first_plan_row))
    assert len(latched) > 0
    wall = bundle.limits[5, 0] + zp.TORQUE_WALL_MARGIN
    assert np.allclose(log.cmd_q[latched[-3:], 5], wall, atol=1e-3)
    assert np.allclose(log.cmd_tau[latched[-3:], 5], 0.0)
    assert any(e["type"] == "guard" for e in log.events)


def test_temperature_spike_aborts(tmp_path, bundle):
    n = 300  # 3 s plan; the spike hits mid-playback
    q, segments = _smooth_two_segment(bundle)
    q = np.vstack([q, np.tile(q[-1], (n - len(q), 1))])
    segments.append(
        {"name": "tail", "kind": "hold", "moving_joints": [],
         "params": {}, "prompt": None, "n": n - 42}
    )
    plan_path = _plan(tmp_path / "plan.npz", q, segments)

    plan, limits = zp.load_and_validate_plan(plan_path, bundle, "urdf20261103")
    sink = zp.LogSink()
    backend = zp.MockBackend(bundle, "urdf20261103", sink, delay_s=0.02)
    player = zp.PlanPlayer(
        plan, backend, sink, limits,
        home_seconds=0.2, home_kp=HOME_KP, home_kd=HOME_KD,
        no_prompt=True,
    )
    backend.connect()
    threading.Timer(0.8, lambda: backend.inject_temperature(95.0)).start()
    abort_reason = player.run()
    log = sink.build_log(
        plan.joint_names, "",
        {"backend": "mock", "abort_reason": abort_reason},
    )
    assert abort_reason is not None
    assert "temperature" in abort_reason
    # Partial log: not all 300 plan frames were sent.
    sent = log.cmd_frame[log.cmd_frame >= 0]
    assert len(np.unique(sent)) < 300
    # Fail-safe ran: release row + release + abort events recorded.
    assert log.cmd_frame[-1] == -1
    assert np.all(log.cmd_kp[-1] == 0) and np.all(log.cmd_tau[-1] == 0)
    assert any(e["type"] == "release" for e in log.events)
    assert any(e["type"] == "abort" for e in log.events)


def test_state_log_has_vol(tmp_path, bundle):
    """R6: the hardware log carries a per-joint bus-voltage channel."""
    q, segments = _smooth_two_segment(bundle)
    plan_path = _plan(tmp_path / "plan.npz", q, segments)
    log, abort_reason, _ = zp.run_playback(_args(tmp_path, plan_path))
    assert abort_reason is None
    assert log.st_vol.shape == (len(log.st_t_ns), 22)
    assert np.isfinite(log.st_vol).all()


def test_prompt_wait_never_bursts(tmp_path, bundle):
    """R1(a): 0.3 s prompts (first segment + mid-plan) cause no catch-up burst."""
    n_seg = 40
    q0 = _neutral_q(bundle)
    a1 = np.tile(q0, (n_seg, 1))
    a1[:, 5] = q0[5] + 0.06 * (0.5 - 0.5 * np.cos(np.pi * np.arange(n_seg) / n_seg)) + 0.03
    a2 = np.tile(a1[-1], (n_seg, 1))
    a2[:, 6] = q0[6] - 0.05 * (0.5 - 0.5 * np.cos(np.pi * np.arange(n_seg) / n_seg))
    q = np.vstack([a1, a2])
    segments = [
        {"name": "seg0", "kind": "ramps", "moving_joints": [bundle.joint_names[5]],
         "params": {}, "prompt": "first finger - press Enter", "n": n_seg},
        {"name": "seg1", "kind": "ramps", "moving_joints": [bundle.joint_names[6]],
         "params": {}, "prompt": "mid-plan prompt - press Enter", "n": n_seg},
    ]
    plan_path = _plan(tmp_path / "plan.npz", q, segments)
    plan, limits = zp.load_and_validate_plan(plan_path, bundle, "urdf20261103")
    sink = zp.LogSink()
    backend = zp.MockBackend(bundle, "urdf20261103", sink, delay_s=0.02)

    def slow_enter(*_a):
        time.sleep(0.3)
        return ""

    player = zp.PlanPlayer(
        plan, backend, sink, limits, home_seconds=0.2,
        no_prompt=False, input_fn=slow_enter,
    )
    backend.connect()
    abort = player.run()
    log = sink.build_log(player.plan.joint_names, "", {"backend": "mock"})
    assert abort is None
    gaps = _plan_frame_gaps_ms(log)
    dt_ms = 1e3 / RATE
    assert len(gaps) == len(q) - 1
    assert gaps.min() >= 0.5 * dt_ms, f"burst send: min gap {gaps.min():.2f} ms"


def test_send_stall_slips_not_bursts(tmp_path, bundle):
    """R1(b): a single 30 ms send stall -> slip event, still no burst spacing."""
    q, segments = _smooth_two_segment(bundle)
    plan_path = _plan(tmp_path / "plan.npz", q, segments)
    player, backend, sink = _player(plan_path, bundle)
    orig_send = backend.send
    calls = {"n": 0}

    def stalling_send(*a):
        calls["n"] += 1
        if calls["n"] == 30:  # ~10 frames into playback
            time.sleep(0.030)
        return orig_send(*a)

    backend.send = stalling_send
    backend.connect()
    abort = player.run()
    log = sink.build_log(player.plan.joint_names, "", {"backend": "mock"})
    assert abort is None
    assert player.schedule_slips >= 1
    assert any(e["type"] == "slip" for e in log.events)
    gaps = _plan_frame_gaps_ms(log)
    dt_ms = 1e3 / RATE
    assert gaps.min() >= 0.5 * dt_ms, f"burst send: min gap {gaps.min():.2f} ms"


def test_gain_bounds_rejected(tmp_path, bundle):
    """R3: plans exceeding --max-kp / --max-kd are refused at load."""
    q = np.zeros((5, 22))
    segments = [
        {"name": "s", "kind": "hold", "moving_joints": [], "params": {}, "prompt": None, "n": 5}
    ]
    p_kp = _plan(tmp_path / "kp.npz", q, segments, kp=4.5)
    with pytest.raises(ValueError, match="max-kp"):
        zp.load_and_validate_plan(p_kp, bundle, "urdf20261103")
    p_kd = _plan(tmp_path / "kd.npz", q, segments, kd=0.25)
    with pytest.raises(ValueError, match="max-kd"):
        zp.load_and_validate_plan(p_kd, bundle, "urdf20261103")
    # Home gains must respect the caps too.
    with pytest.raises(ValueError, match="home gains"):
        zp.PlanPlayer(
            fmt.ExcitationPlan(
                joint_names=bundle.joint_names, rate_hz=RATE,
                q=np.zeros((5, 22), np.float32), dq=np.zeros((5, 22), np.float32),
                tau=np.zeros((5, 22), np.float32),
                kp=np.full((5, 22), 2.0, np.float32), kd=np.full((5, 22), 0.04, np.float32),
                segment=np.zeros(5, np.int32),
                segments=[{"name": "s", "kind": "hold", "moving_joints": [], "params": {}}],
            ),
            None, zp.LogSink(), bundle.limits, home_kp=5.0, max_kp=4.0,
        )


def test_nonzero_tau_requires_flag(tmp_path, bundle):
    """R3: ANY nonzero tau (not only kp==0 frames) needs --allow-torque-mode."""
    n = 10
    q = np.zeros((n, 22))
    tau = np.zeros((n, 22))
    tau[:, 5] = 0.05  # tau with position gains still on
    segments = [
        {"name": "s", "kind": "hold", "moving_joints": [bundle.joint_names[5]],
         "params": {}, "prompt": None, "n": n}
    ]
    plan_path = _plan(tmp_path / "plan.npz", q, segments, tau=tau)
    with pytest.raises(ValueError, match="nonzero tau"):
        zp.load_and_validate_plan(plan_path, bundle, "urdf20261103")
    # tau above --max-tau is refused even with the flag.
    plan, _limits = None, None
    with pytest.raises(ValueError, match="max-tau"):
        zp.load_and_validate_plan(
            plan_path, bundle, "urdf20261103",
            allow_torque_mode=True, max_tau=0.03,
        )


def test_dq_over_speed_rejected(tmp_path, bundle):
    """R3: |dq| above --max-speed is refused at load."""
    n = 10
    dq = np.zeros((n, 22), dtype=np.float32)
    dq[:, 5] = 9.0
    bundle_ = bundle
    plan = fmt.ExcitationPlan(
        joint_names=bundle_.joint_names, rate_hz=RATE,
        q=np.zeros((n, 22), np.float32), dq=dq,
        tau=np.zeros((n, 22), np.float32),
        kp=np.full((n, 22), 2.0, np.float32), kd=np.full((n, 22), 0.04, np.float32),
        segment=np.zeros(n, np.int32),
        segments=[{"name": "s", "kind": "hold", "moving_joints": [bundle_.joint_names[5]],
                   "params": {}, "prompt": None}],
        meta={},
    )
    plan_path = fmt.save_plan(plan, tmp_path / "plan.npz")
    with pytest.raises(ValueError, match="dq max"):
        zp.load_and_validate_plan(plan_path, bundle, "urdf20261103", max_speed=8.0)


def test_torque_to_position_handoff(tmp_path, bundle):
    """R4: after a torque segment, position targets ramp from measured q."""
    n_torque, n_pos = 80, 60
    q0 = _neutral_q(bundle, n_torque)
    q0[:, 5] = 0.30
    kp = np.full((n_torque + n_pos, 22), HOME_KP, dtype=np.float32)
    kd = np.full((n_torque + n_pos, 22), HOME_KD, dtype=np.float32)
    tau = np.zeros((n_torque + n_pos, 22))
    kp[:n_torque, 5] = 0.0       # joint 5 in torque mode
    tau[:n_torque, 5] = -0.05    # drifts the joint ~0.15 rad negative, mid-range
    # Position segment: continuous ramp 0.3 -> 0.6 rad; the measured joint is
    # ~0.45 rad away from the far end, so the rate limiter must engage.
    ramp = _neutral_q(bundle, n_pos)
    ramp[:, 5] = 0.30 + 0.3 * (0.5 - 0.5 * np.cos(np.pi * np.arange(n_pos) / n_pos))
    q = np.vstack([q0, ramp])
    segments = [
        {"name": "t", "kind": "hold", "moving_joints": [bundle.joint_names[5]],
         "params": {}, "prompt": None, "n": n_torque},
        {"name": "p", "kind": "hold", "moving_joints": [bundle.joint_names[5]],
         "params": {}, "prompt": None, "n": n_pos},
    ]
    plan_path = _plan(tmp_path / "plan.npz", q, segments, kp=kp, kd=kd, tau=tau)
    args = _args(tmp_path, plan_path, allow_torque_mode=True, max_tau=0.15)
    log, abort_reason, _ = zp.run_playback(args)
    assert abort_reason is None
    # First position-mode send for joint 5: target within max_step of measured.
    pos_rows = np.flatnonzero(
        (log.cmd_frame >= n_torque) & (log.cmd_kp[:, 5] > 0)
    )
    assert len(pos_rows) > 0
    first = pos_rows[0]
    meas = log.st_q[np.searchsorted(log.st_t_ns, log.cmd_t_ns[first]), 5]
    max_step = 8.0 / RATE
    assert abs(log.cmd_q[first, 5] - meas) <= max_step + 0.02
    assert any("torque->position" in e["detail"] for e in log.events)


def test_fail_safe_failure_still_releases(tmp_path, bundle):
    """R5: a _fail_safe crash still logs the release row and closes the backend."""
    q, segments = _smooth_two_segment(bundle)
    plan_path = _plan(tmp_path / "plan.npz", q, segments)
    player, backend, sink = _player(plan_path, bundle)

    def boom(*_a, **_kw):
        raise RuntimeError("simulated fail-safe crash")

    player._fail_safe = boom

    def crash_play():
        raise RuntimeError("simulated mid-play crash")

    player._play_frames = crash_play
    backend.connect()
    abort = player.run()
    log = sink.build_log(player.plan.joint_names, "", {"backend": "mock"})
    assert abort and "simulated mid-play crash" in abort
    assert backend._thread is None  # closed
    # Release row logged anyway (zeros, frame -1, last row).
    assert log.cmd_frame[-1] == -1
    assert np.all(log.cmd_kp[-1] == 0) and np.all(log.cmd_tau[-1] == 0)
    assert any(e["type"] == "release" for e in log.events)


def test_large_home_mock_unaffected_and_real_gates(tmp_path, bundle):
    """R14: a >0.8 rad homing move gates only on the real backend."""
    n = 20
    q = _neutral_q(bundle, n)
    segments = [
        {"name": "s", "kind": "hold", "moving_joints": [], "params": {}, "prompt": None, "n": n}
    ]
    plan_path = _plan(tmp_path / "plan.npz", q, segments)
    start = _neutral_q(bundle)
    start[5] = q[0, 5] + 0.9  # 0.9 rad away from frame 0 -> over the 0.8 default

    # Mock: proceeds untouched (no prompt, no flag).
    player, backend, sink = _player(plan_path, bundle, start_q=start)
    backend.connect()
    assert player.run() is None

    # is_real + no_prompt + no --allow-large-home -> abort before homing.
    player, backend, sink = _player(plan_path, bundle, start_q=start, is_real=True)
    backend.connect()
    abort = player.run()
    assert abort is not None and "max-home-delta" in abort
    log = sink.build_log(player.plan.joint_names, "", {"backend": "mock"})
    assert (log.cmd_frame >= 0).sum() == 0  # never reached playback

    # is_real + prompts + operator types 'yes' -> homes.
    player, backend, sink = _player(
        plan_path, bundle, start_q=start, is_real=True,
        no_prompt=False, input_fn=lambda *a: "yes",
    )
    backend.connect()
    assert player.run() is None

    # is_real + prompts + wrong answer -> declined.
    player, backend, sink = _player(
        plan_path, bundle, start_q=start, is_real=True,
        no_prompt=False, input_fn=lambda *a: "no",
    )
    backend.connect()
    abort = player.run()
    assert abort is not None and "declined" in abort
