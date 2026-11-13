"""Apply z5_right_fitted_params.json per-joint dynamics to a bundle scene.mjb.

The fitted file was produced by the dex_supermimic sysid run from the five
per-finger recordings (kp=2.0/kd=0.1 position commands). MuJoCo has no Stribeck
friction, so the mapping keeps the Coulomb + viscous parts:

    actuator  gainprm[0]   = stiffness          (position gain)
    actuator  biasprm[1]   = -stiffness         (-kp * q)
    actuator  biasprm[2]   = -damping           (-kd * dq)
    actuator  forcerange   = +-effort_limit
    dof       armature     = armature
    dof       frictionloss = dynamic_friction   (stribeck Coulomb level)
    dof       damping      = viscous_friction   (stribeck viscous term)

Real-hardware gains are untouched - keep commanding kp=2.0/kd=0.1; the fitted
model reproduces the hand's response to those commands.

Usage (unitree_deploy env, needs mujoco):
    python deploy/pen_spin/tools/z5_apply_fitted.py \
        --bundle third_party/z5-pen-spin-deploy/generated/deploy/z5-pen-spin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--params", type=Path, help="defaults to <bundle>/z5_right_fitted_params.json")
    p.add_argument("--dry-run", action="store_true", help="print the table without saving")
    args = p.parse_args()

    import mujoco

    bundle = args.bundle.resolve()
    params_path = args.params or bundle / "z5_right_fitted_params.json"
    params = json.loads(params_path.read_text())
    if params.get("schema") != "dex_supermimic_z5_sysid_v1":
        raise SystemExit(f"unsupported params schema: {params_path}")

    mjb = bundle / "scene.mjb"
    model = mujoco.MjModel.from_binary_path(str(mjb))

    rows = []
    for entry in params["joints"]:
        name = entry["name"]
        fit = entry["fitted"]
        joint = model.joint("robot/" + name)
        actuator = model.actuator("robot/" + name + "_actuator")
        dof = int(joint.dofadr[0])
        old = (
            float(actuator.gainprm[0]), float(-actuator.biasprm[2]),
            float(model.dof_armature[dof]), float(model.dof_frictionloss[dof]),
            float(model.dof_damping[dof]), float(actuator.forcerange[1]),
        )
        actuator.gainprm[0] = fit["stiffness"]
        actuator.biasprm[1] = -fit["stiffness"]
        actuator.biasprm[2] = -fit["damping"]
        actuator.forcerange[:] = (-fit["effort_limit"], fit["effort_limit"])
        model.dof_armature[dof] = fit["armature"]
        model.dof_frictionloss[dof] = fit["dynamic_friction"]
        model.dof_damping[dof] = fit["viscous_friction"]
        rows.append((name, entry["identified"], old,
                     (fit["stiffness"], fit["damping"], fit["armature"],
                      fit["dynamic_friction"], fit["viscous_friction"], fit["effort_limit"])))

    print(f"{'joint':38s} {'id':>3s} | {'kp':>5s} {'kd':>6s} {'arm':>8s} {'fric':>6s} {'visc':>6s} {'Fmax':>5s}"
          "   <- fitted (was: same order)")
    for name, ident, old, new in rows:
        mark = "Y" if ident else "-"
        print(f"{name:38s} {mark:>3s} | {new[0]:5.3f} {new[1]:6.4f} {new[2]:8.5f} {new[3]:6.4f} "
              f"{new[4]:6.4f} {new[5]:5.2f}   (was {old[0]:.2f} {old[1]:.4f} {old[2]:.5f} "
              f"{old[3]:.4f} {old[4]:.4f} {old[5]:.2f})")

    if args.dry_run:
        print("dry-run: scene.mjb unchanged")
        return 0
    mujoco.mj_saveModel(model, str(mjb))
    print(f"saved fitted dynamics -> {mjb}")
    print("note: stribeck static 'friction' term is not representable in MuJoCo; "
          "frictionloss=dynamic_friction is used (static overshoot dropped).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
