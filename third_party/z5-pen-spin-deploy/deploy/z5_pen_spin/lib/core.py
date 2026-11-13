"""The exported pen-spin observation and residual controller contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def finite_vector(value, size, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return value


def rotation_wxyz(quat):
    q = finite_vector(quat, 4, "quaternion").astype(np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError("zero quaternion")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def bounded_points(pose, radius, bound):
    """Pose is palm-local [x,y,z,w,qx,qy,qz]; point order is +X, -X."""
    pose = finite_vector(pose, 7, "pen pose")
    direction = rotation_wxyz(pose[3:])[:, 0] * radius
    points = pose[:3] + np.stack((direction, -direction))
    return (bound * np.tanh(points / bound)).astype(np.float32).reshape(6)


class Bundle:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.config = json.loads((self.directory / "policy_config.json").read_text())
        c = self.config
        if c["schema"] != "dex_supermimic_z5_pen_spin:1" or c["task"] != "z5-pen-spin":
            raise ValueError("unsupported deployment bundle")
        self.joint_names = tuple(c["joint_names"])
        if len(self.joint_names) != 22 or len(set(self.joint_names)) != 22 or c["observation_dim"] != 163:
            raise ValueError("bundle must describe the 163-input, 22-action Z5 policy")
        self.dt = float(c["control_dt"])
        if not np.isclose(self.dt, 0.02):
            raise ValueError("Z5 pen-spin requires 50 Hz")
        self.limits = np.asarray(c["joint_limits"], dtype=np.float32)
        if self.limits.shape != (22, 2) or not np.isfinite(self.limits).all():
            raise ValueError("invalid bundle joint limits")
        if np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError("invalid bundle joint limit order")
        with np.load(self.directory / "reference.npz", allow_pickle=False) as data:
            self.references = {key: data[key].copy() for key in data.files}
        self.clip_ids = tuple(str(v) for v in self.references["clip_ids"])
        lengths = self.references["lengths"]
        if len(lengths) != len(self.clip_ids) or np.any(lengths < 1):
            raise ValueError("invalid reference clip lengths")
        self.starts = np.r_[0, np.cumsum(lengths)[:-1]]
        self.ends = np.cumsum(lengths)
        for key, width in (("joint_pos", (22,)), ("joint_vel", (22,)), ("object_pose", (7,)), ("command", (91,))):
            value = self.references[key]
            if value.shape != (int(self.ends[-1]), *width) or not np.isfinite(value).all():
                raise ValueError(f"invalid reference {key}")

    def clip_index(self, clip=None):
        if clip is None:
            return 0
        if clip not in self.clip_ids:
            raise ValueError(f"unknown clip {clip!r}; available: {', '.join(self.clip_ids)}")
        return self.clip_ids.index(clip)

    def observation(self, frame, q, dq, pen_pose, last_action):
        c = self.config
        q = finite_vector(q, 22, "joint positions")
        dq = finite_vector(dq, 22, "joint velocities")
        action = finite_vector(last_action, 22, "last raw action")
        points = bounded_points(pen_pose, c["axis_point_radius_m"], c["object_position_tanh_scale_m"])
        return np.concatenate(
            (
                self.references["command"][frame],
                q - c["default_joint_pos"],
                (dq - c["default_joint_vel"]) * c["joint_velocity_scale"],
                points,
                action,
            )
        ).astype(np.float32)


class ResidualController:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.last_action = np.zeros(22, dtype=np.float32)
        self.residual = np.zeros(22, dtype=np.float32)

    def target(self, reference, action):
        action = finite_vector(action, 22, "policy action")
        if np.any(np.abs(action) > 1.00001):
            raise ValueError("policy action outside tanh range")
        c = self.config
        residual = np.clip(action * c["action_scale"], *c["residual_clip"])
        self.residual = c["ema_alpha"] * residual + (1 - c["ema_alpha"]) * self.residual
        self.last_action = action.copy()
        # Encoder feedback is already in the hardware encoder convention. No
        # simulated encoder-bias subtraction, noise, dropout or action mask.
        return finite_vector(reference, 22, "reference joints") + self.residual


class Policy:
    def __init__(self, bundle, engine="torchscript"):
        self.engine = engine
        if engine == "torchscript":
            import torch

            torch.set_num_threads(1)
            self.torch = torch
            self.model = torch.jit.load(str(bundle.directory / "policy.ts"), map_location="cpu").eval()
            self(np.zeros(163, dtype=np.float32))
        elif engine == "onnx":
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            self.model = ort.InferenceSession(
                str(bundle.directory / "policy.onnx"), options, providers=["CPUExecutionProvider"]
            )
        else:
            raise ValueError(f"unknown inference engine: {engine}")

    def __call__(self, observation):
        observation = finite_vector(observation, 163, "observation")
        if self.engine == "torchscript":
            with self.torch.inference_mode():
                action = self.model(self.torch.from_numpy(observation[None])).numpy()[0]
        else:
            action = self.model.run(["action"], {"policy": observation[None]})[0][0]
        return finite_vector(action, 22, "inferred action")
