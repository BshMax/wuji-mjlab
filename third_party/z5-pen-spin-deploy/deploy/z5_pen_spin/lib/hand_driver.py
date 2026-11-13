"""Unitree DDS Z5 adapter. Model order and wire convention are explicit."""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

from .core import finite_vector

WIRE_NAMES = tuple(
    f"right_{finger}_finger_{joint}_joint"
    for finger in ("index", "middle", "ring", "little")
    for joint in (("pitch",) if finger == "little" else ()) + ("roll1", "yaw", "roll2", "roll3")
) + tuple(f"right_thumb_{joint}_joint" for joint in ("pitch", "roll", "yaw1", "yaw2", "yaw3"))
REVERSED_JOINTS = {"right_thumb_pitch_joint", "right_little_finger_pitch_joint", "right_little_finger_roll1_joint"}


def wire_mapping(joint_names, convention):
    if len(joint_names) != 22 or set(joint_names) != set(WIRE_NAMES):
        raise ValueError("policy/DDS joint names do not match")
    if convention not in ("legacy", "urdf20261103"):
        raise ValueError("wire convention must be legacy or urdf20261103")
    indices = np.array([WIRE_NAMES.index(name) for name in joint_names])
    signs = np.array(
        [-1.0 if convention == "urdf20261103" and name in REVERSED_JOINTS else 1.0 for name in joint_names],
        dtype=np.float32,
    )
    return indices, signs


class TargetGuard:
    def __init__(self, bundle, convention="legacy", max_speed=8.0):
        self.limits = bundle.limits.copy()
        if convention == "urdf20261103":
            # The newer thumb yaw1 has a narrower range than the training MJCF.
            i = bundle.joint_names.index("right_thumb_yaw1_joint")
            self.limits[i, 1] = min(self.limits[i, 1], 1.57)
        if not np.isfinite(max_speed) or max_speed <= 0:
            raise ValueError("max target speed must be positive")
        self.delta = max_speed * bundle.dt
        self.last = None
        self.interventions = 0

    def reset(self, measured, tolerance=0.02):
        measured = finite_vector(measured, 22, "takeover pose")
        if np.any(measured < self.limits[:, 0] - tolerance) or np.any(measured > self.limits[:, 1] + tolerance):
            bad = [i for i in range(22) if measured[i] < self.limits[i, 0] - tolerance
                   or measured[i] > self.limits[i, 1] + tolerance]
            detail = ", ".join(f"{i}:{measured[i]:+.3f} not in [{self.limits[i,0]:+.3f},{self.limits[i,1]:+.3f}]" for i in bad)
            raise ValueError(f"measured takeover pose exceeds configured model/hardware limits ({detail})")
        self.last = np.clip(measured, *self.limits.T)

    def apply(self, target):
        target = finite_vector(target, 22, "joint target")
        limited = np.clip(target, *self.limits.T)
        if self.last is not None:
            limited = np.clip(limited, self.last - self.delta, self.last + self.delta)
        self.interventions += int(np.any(np.abs(limited - target) > 1e-6))
        self.last = limited.copy()
        return limited


class Z5HandDriver:
    def __init__(
        self,
        bundle,
        *,
        convention,
        interface=None,
        domain=0,
        sdk_path=None,
        kp=None,
        kd=None,
        state_timeout=0.2,
        startup_timeout=5.0,
        temperature_limit=65.0,
    ):
        self.indices, self.signs = wire_mapping(bundle.joint_names, convention)
        self.kp = float(bundle.config["nominal_kp"] if kp is None else kp)
        self.kd = float(bundle.config["nominal_kd"] if kd is None else kd)
        if not all(np.isfinite(v) and v >= 0 for v in (self.kp, self.kd)):
            raise ValueError("DDS gains must be finite and nonnegative")
        self.timeout, self.temperature_limit = state_timeout, temperature_limit
        self.condition = threading.Condition()
        self.state = None
        self.stamp = -np.inf
        self.error = None
        self.publisher = self.subscriber = self.message = None
        if sdk_path:
            sys.path.insert(0, str(sdk_path))
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, MotorCmd_

        try:
            ChannelFactoryInitialize(domain, interface)
            self.publisher = ChannelPublisher("rt/dex5/right/cmd", HandCmd_)
            self.subscriber = ChannelSubscriber("rt/dex5/right/state", HandState_)
            self.message = HandCmd_([MotorCmd_(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0) for _ in range(22)], [0] * 4)
            self.publisher.Init()
            self.subscriber.Init(self._on_state)
            deadline = time.monotonic() + startup_timeout
            warned = False
            with self.condition:
                while self.state is None and self.error is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"no Z5 HandState_ within {startup_timeout:.1f} seconds - "
                            "check that dex5_lonicera is still running (terminal A)"
                        )
                    if not warned and time.monotonic() > deadline - startup_timeout + 2.0:
                        print("waiting for rt/dex5/right/state ...", file=sys.stderr, flush=True)
                        warned = True
                    self.condition.wait(min(remaining, 1.0))
            self.read()
            print(f"Z5HandDriver ready: 22 motors, kp={self.kp}, kd={self.kd} per MotorCmd_ each tick",
                  file=sys.stderr, flush=True)
        except BaseException:
            self.close()
            raise

    def _on_state(self, message):
        try:
            motors = message.motor_state
            q = finite_vector([v.q for v in motors], 22, "DDS q")
            dq = finite_vector([v.dq for v in motors], 22, "DDS dq")
            temperature = np.asarray([v.temperature for v in motors], dtype=float)
            if temperature.shape != (22, 2) or not np.isfinite(temperature).all():
                raise ValueError("invalid DDS motor temperatures")
            if np.any(temperature > self.temperature_limit):
                raise RuntimeError("Z5 motor temperature limit exceeded")
            with self.condition:
                self.state = q[self.indices] * self.signs, dq[self.indices] * self.signs
                self.stamp = time.monotonic()
                self.condition.notify_all()
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def read(self):
        with self.condition:
            if self.error:
                raise RuntimeError(self.error)
            if self.state is None or time.monotonic() - self.stamp > self.timeout:
                raise TimeoutError("Z5 joint feedback is stale")
            return tuple(v.copy() for v in self.state)

    def send(self, target):
        self.read()
        target = finite_vector(target, 22, "DDS target")
        wire = np.empty(22, dtype=np.float32)
        wire[self.indices] = target * self.signs
        for command, q in zip(self.message.motor_cmd, wire, strict=True):
            command.mode, command.q = 1, float(q)
            command.dq = command.tau = 0.0
            command.kp, command.kd = self.kp, self.kd
        self.publisher.Write(self.message)

    def close(self):
        if self.publisher is not None and self.message is not None:
            try:
                for command in self.message.motor_cmd:
                    command.kp = command.kd = command.tau = command.dq = 0.0
                self.publisher.Write(self.message)
            except Exception:
                pass
        for channel in (self.subscriber, self.publisher):
            if channel is not None:
                try:
                    channel.Close()
                except Exception:
                    pass
        self.publisher = self.subscriber = self.message = None
