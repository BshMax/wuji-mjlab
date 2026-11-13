"""Export the trained actor and the exact training references without a simulator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .core import bounded_points

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = ROOT / (
    "logs/dex_supermimic/z5-pen-spin/z5-com-xi-native-resume20000-e8192-utd1-20261105-101255/model_final.pt"
)
DEFAULT_BUNDLE = ROOT / "generated/deploy/z5-pen-spin"


def _local_path(value):
    path = Path(value)
    if path.exists():
        return path
    # Recorded configs may have been produced in a different checkout.
    for marker in ("src/dex_supermimic/", "logs/"):
        if marker in str(path):
            candidate = ROOT / (marker + str(path).split(marker, 1)[1])
            if candidate.exists():
                return candidate
    raise FileNotFoundError(path)


def export_bundle(checkpoint=DEFAULT_CHECKPOINT, output=DEFAULT_BUNDLE, *, onnx=True):
    import mujoco

    from dex_supermimic.rl.dex_supermimic.networks import Actor
    from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingReader
    from dex_supermimic.tasks.pen_spin.mdp.observations import (
        bounded_object_pose_to_points_wrist_local,
        make_axis_points,
    )
    from dex_supermimic.tasks.pen_spin.tooling.recipe import resolve_recipe

    torch.set_num_threads(1)
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["config"]["task"] != "z5-pen-spin" or state["format_version"] not in (3, 4):
        raise ValueError("requires a Z5 pen-spin checkpoint with joint metadata")
    snapshot_path = checkpoint.parent / "environment_config.json"
    env = json.loads(snapshot_path.read_text())
    motion, action = env["commands"]["motion"], env["actions"]["joint_pos"]
    names = state["joint_names"]
    if names != motion["joint_names"] or len(names) != 22:
        raise ValueError("checkpoint/reference joint order mismatch")
    if motion.get("lookahead_steps") or motion["object_points_local"] is not None:
        raise ValueError("this exporter requires the original Z5 pen-spin observation layout")
    actor = Actor(163, 22).eval()
    actor.load_state_dict(state["modules"]["actor"], strict=True)
    if action["offset"] != 0 or action["clip"] is not None or not isinstance(action["scale"], (float, int)):
        raise ValueError("unsupported residual action configuration")
    if not np.isclose(action["alpha"], state["config"]["action_alpha"]):
        raise ValueError("checkpoint and environment EMA alpha differ")
    resolved = resolve_recipe(
        _local_path(motion["recipe_path"]), clips_root=_local_path(motion["clips_root"]), verify_clip_hashes=True
    ).clip_sets[0]
    reference = SingleHandTrackingReader(
        resolved,
        joint_names=tuple(names),
        num_links=len(motion["link_names"]),
        wrist_to_palm_quat=motion["clip_wrist_to_palm_quat"],
    ).read()
    digest = hashlib.sha256()
    for field in ("joint_pos", "object_pose", "link_pos"):
        digest.update(reference[field].numpy().tobytes())
    digest.update(torch.cumsum(reference["trajectory_lengths"], 0).numpy().tobytes())
    if digest.hexdigest() != state.get("reference_signature"):
        raise ValueError("reference data differs from the trajectories used by this checkpoint")
    model = mujoco.MjModel.from_binary_path(str(_local_path(env["canonical_model_path"])))
    joints = [model.joint("robot/" + name) for name in names]
    qids = [int(j.qposadr[0]) for j in joints]
    dt = model.opt.timestep * env["decimation"]
    if not np.isclose(dt, resolved.frame_dt) or not np.isclose(dt, 0.02):
        raise ValueError("model and reference must both run at 50 Hz")
    pose_xyzw = reference["object_pose"].numpy()
    poses = pose_xyzw[:, [0, 1, 2, 6, 3, 4, 5]]
    runtime_points = np.stack(
        [bounded_points(p, motion["axis_point_radius_m"], motion["object_position_tanh_scale_m"]) for p in poses]
    )
    points = (
        bounded_object_pose_to_points_wrist_local(
            reference["object_pose"],
            make_axis_points(motion["axis_point_radius_m"]),
            position_tanh_scale_m=motion["object_position_tanh_scale_m"],
        )
        .reshape(-1, 6)
        .numpy()
    )
    np.testing.assert_allclose(runtime_points, points, atol=1e-7, rtol=1e-6)
    command = np.concatenate(
        (reference["joint_pos"].numpy(), points, reference["link_pos"].numpy().reshape(len(poses), -1)), axis=1
    )
    if command.shape[1] != 91:
        raise ValueError("expected a 91-dimensional reference command")
    config = dict(
        schema="dex_supermimic_z5_pen_spin:1",
        task="z5-pen-spin",
        observation_dim=163,
        observation_terms={"motion": 91, "joint_pos": 22, "joint_vel": 22, "object_pose": 6, "last_action": 22},
        joint_names=names,
        link_names=motion["link_names"],
        history=1,
        control_dt=dt,
        default_joint_pos=model.qpos0[qids].tolist(),
        default_joint_vel=[0.0] * 22,
        joint_velocity_scale=0.25,
        action_type="reference_plus_ema_residual",
        action_scale=action["scale"],
        residual_clip=action["residual_clip"],
        ema_alpha=action["alpha"],
        axis_point_radius_m=motion["axis_point_radius_m"],
        object_position_tanh_scale_m=motion["object_position_tanh_scale_m"],
        pose_frame="palm_link",
        quaternion_order="wxyz",
        pen_shaft_axis="+X",
        joint_limits=[j.range.tolist() for j in joints],
        nominal_kp=2.0,
        nominal_kd=state["config"]["actuator_kv"],
        fixed_palm_pose_world_xyzw=motion["fixed_wrist_pose_w"],
        checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        iteration=state["iteration"],
        reference_signature=state.get("reference_signature"),
        contact_sensor=state["config"]["contact_sensor"],
        mujoco_version=mujoco.__version__,
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "reference.npz",
        clip_ids=np.array([c.clip_id for c in resolved.clips]),
        lengths=reference["trajectory_lengths"].numpy(),
        command=command.astype(np.float32),
        joint_pos=reference["joint_pos"].numpy(),
        joint_vel=reference["joint_vel"].numpy(),
        object_pose=poses,
    )
    example = torch.zeros(1, 163)
    traced = torch.jit.trace(actor, example)
    traced.save(str(output / "policy.ts"))
    sample = torch.randn(32, 163, generator=torch.Generator().manual_seed(42))
    with torch.inference_mode():
        expected = actor(sample).numpy()
        ts_error = float(np.max(np.abs(traced(sample).numpy() - expected)))
    validation = {
        "torchscript_max_abs_error": ts_error,
        "object_points_max_abs_error": float(np.max(np.abs(runtime_points - points))),
    }
    if onnx:
        import onnx as onnx_lib
        from onnx.reference import ReferenceEvaluator

        torch.onnx.export(
            actor,
            example,
            str(output / "policy.onnx"),
            input_names=["policy"],
            output_names=["action"],
            dynamic_axes={"policy": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        onnx_model = onnx_lib.load(str(output / "policy.onnx"))
        onnx_lib.checker.check_model(onnx_model)
        actual = ReferenceEvaluator(onnx_model).run(None, {"policy": sample.numpy()})[0]
        validation["onnx_max_abs_error"] = float(np.max(np.abs(actual - expected)))
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    # Keep a canonical model for an optional CPU MuJoCo smoke run / mirror.
    mujoco.mj_saveModel(model, str(output / "scene.mjb"))
    validation.update(clips=len(resolved.clips), frames=len(poses), observation_dim=163, action_dim=22)
    (output / "policy_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return validation
