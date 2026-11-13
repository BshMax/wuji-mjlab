"""Capture the measured flat rest pose into a file usable by --flat-pose.

Run this while the physical hand is in its flat/open pose (e.g. right after
``dex5_lonicera -k`` calibration). It only subscribes to joint state, averages
a short window, prints the pose per joint, saves an .npy vector, and releases
the hand — no motion commands are sent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/z5-pen-spin-deploy"))

from deploy.z5_pen_spin.lib.core import Bundle
from deploy.z5_pen_spin.lib.hand_driver import Z5HandDriver


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path,
                   default=ROOT / "third_party/z5-pen-spin-deploy/generated/deploy/z5-pen-spin")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/flat_pose.npy")
    p.add_argument("--wire-convention", default="urdf20261103",
                   choices=("legacy", "urdf20261103"))
    p.add_argument("--sdk-path", type=Path, default="/home/unitree/Softwares/unitree_sdk2_python")
    p.add_argument("--dds-interface")
    p.add_argument("--dds-domain", type=int, default=0)
    p.add_argument("--startup-timeout", type=float, default=10.0)
    p.add_argument("--samples", type=int, default=25, help="state samples to average")
    args = p.parse_args()

    bundle = Bundle(args.bundle)
    driver = Z5HandDriver(
        bundle,
        convention=args.wire_convention,
        interface=args.dds_interface,
        domain=args.dds_domain,
        sdk_path=args.sdk_path,
        startup_timeout=args.startup_timeout,
    )
    try:
        samples = []
        for _ in range(max(1, args.samples)):
            q, _ = driver.read()
            samples.append(q)
        flat = np.mean(samples, axis=0)
        dev = np.max(np.abs(np.asarray(samples) - flat))
        print(f"flat pose captured ({len(samples)} samples, max deviation {dev:.4f} rad):")
        for name, v in zip(bundle.joint_names, flat):
            flag = "  <-- nonzero" if abs(v) > 0.05 else ""
            print(f"  {name:38s} {v:+.4f}{flag}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, flat.astype(np.float64))
        print(f"saved -> {args.out}")
        print("use with: --flat-pose", args.out)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
