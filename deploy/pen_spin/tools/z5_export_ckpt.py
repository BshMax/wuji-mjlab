"""Export a raw dex_supermimic .pt checkpoint into a deployment bundle.

The training-side exporter (deploy/z5_pen_spin/scripts/export_policy.py) needs the
full Dex-SuperMimic checkout. This tool rebuilds the exact Actor graph instead:

    obs -> RunningNorm((x - mean) / sqrt(var + 1e-4))
        -> Linear 163-512 -> ELU -> Linear 512-256 -> ELU
        -> Linear 256-128 -> ELU -> Linear 128-22 -> tanh

It writes policy.ts / policy.onnx and copies reference.npz + scene.mjb from an
existing bundle (the reference signature is checked so shared references are
safe to reuse). Run inside the unitree_deploy conda env:

    python deploy/pen_spin/tools/z5_export_ckpt.py \
        --checkpoint third_party/z5-pen-spin-deploy/checkpoints/model_20000.pt \
        --output third_party/z5-pen-spin-deploy/generated/deploy/z5-pen-spin-20000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE_BUNDLE = ROOT / "third_party/z5-pen-spin-deploy/generated/deploy/z5-pen-spin"


class RunningNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(163))
        self.register_buffer("var", torch.ones(163))
        self.register_buffer("count", torch.zeros(()))

    def forward(self, obs):
        return (obs - self.mean) / torch.sqrt(self.var + 1e-4)


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RunningNorm()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(163, 512), torch.nn.ELU(),
            torch.nn.Linear(512, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 22),
        )

    def forward(self, obs):
        return torch.tanh(self.net(self.norm(obs)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-bundle", type=Path, default=BASE_BUNDLE,
                   help="bundle to copy reference.npz/scene.mjb/policy_config.json from")
    args = p.parse_args()

    torch.set_num_threads(1)
    checkpoint = args.checkpoint.resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["config"]["task"] != "z5-pen-spin" or state["format_version"] not in (3, 4):
        raise SystemExit("not a Z5 pen-spin checkpoint")

    base = json.loads((args.base_bundle / "policy_config.json").read_text())
    if state["joint_names"] != base["joint_names"]:
        raise SystemExit("checkpoint joint order differs from the base bundle")
    if state.get("reference_signature") != base["reference_signature"]:
        raise SystemExit(
            "reference_signature mismatch - this checkpoint was trained on different "
            "reference clips; reference.npz cannot be reused"
        )

    actor = Actor().eval()
    actor.load_state_dict(state["modules"]["actor"], strict=True)

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("reference.npz", "scene.mjb"):
        shutil.copy2(args.base_bundle / name, out / name)

    example = torch.zeros(1, 163)
    traced = torch.jit.trace(actor, example)
    traced.save(str(out / "policy.ts"))
    sample = torch.randn(32, 163, generator=torch.Generator().manual_seed(42))
    with torch.inference_mode():
        expected = actor(sample).numpy()
        ts_error = float(np.max(np.abs(traced(sample).numpy() - expected)))
    validation = {"torchscript_max_abs_error": ts_error}

    torch.onnx.export(
        actor, example, str(out / "policy.onnx"),
        input_names=["policy"], output_names=["action"],
        dynamic_axes={"policy": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    import onnx as onnx_lib
    from onnx.reference import ReferenceEvaluator

    onnx_model = onnx_lib.load(str(out / "policy.onnx"))
    onnx_lib.checker.check_model(onnx_model)
    actual = ReferenceEvaluator(onnx_model).run(None, {"policy": sample.numpy()})[0]
    onnx_error = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    validation["onnx_max_abs_error"] = onnx_error

    config = dict(base)
    config.update(
        checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        iteration=int(state["iteration"]),
        nominal_kd=float(state["config"].get("actuator_kv", base["nominal_kd"])),
    )
    validation.update(
        clips=len(base.get("joint_names", [])) and int(np.load(out / "reference.npz")["lengths"].shape[0]),
        frames=int(np.load(out / "reference.npz")["joint_pos"].shape[0]),
        observation_dim=163, action_dim=22,
    )
    (out / "policy_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"bundle": str(out), "iteration": config["iteration"], **validation}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
