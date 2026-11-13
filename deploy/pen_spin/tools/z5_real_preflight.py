# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Pre-flight check for real-hardware Z5 pen-spin deployment.

Verifies everything that does NOT need the hand plugged in:

  - unitree_deploy conda env: unitree_sdk2py + cyclonedds + onnxruntime
  - Z5 bundle + camera calibration files
  - observer / bridge ZMQ ports free
  - dex5_lonicera daemon binary + bundled DDS libs
  - DDS network interfaces present
  - joint-name <-> wire convention mapping sanity

Run:  python deploy/pen_spin/tools/z5_real_preflight.py
Exit 0 = ready for the hand; 1 = something must be fixed first.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BUNDLE = REPO / "third_party/z5-pen-spin-deploy/generated/deploy/z5-pen-spin"
CAMERA = REPO / "deploy/pen_spin/config/camera.yaml"
PEN_TAGS = REPO / "deploy/pen_spin/config/pen_tags.json"
SDK_DIR = Path("/home/unitree/Softwares/unitree_sdk2_python")
CONDA_PY = Path("/home/unitree/miniconda3/envs/unitree_deploy/bin/python")
LONICERA = Path("/home/unitree/Documents/dex5_lonicera_package_x86_64_net/bin/dex5_lonicera")

CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


@check("unitree_deploy python")
def _python():
    if not CONDA_PY.exists():
        return False, f"missing {CONDA_PY}"
    out = subprocess.check_output([str(CONDA_PY), "--version"], text=True).strip()
    return True, out


@check("unitree_sdk2py + cyclonedds + onnxruntime")
def _sdk():
    code = (
        "import sys; sys.path.insert(0, '%s');"
        "import cyclonedds, onnxruntime;"
        "from unitree_sdk2py.core.channel import ChannelPublisher;"
        "from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_;"
        "print('ok')" % SDK_DIR)
    r = subprocess.run([str(CONDA_PY), "-c", code],
                       capture_output=True, text=True)
    ok = r.stdout.strip() == "ok"
    return ok, r.stderr.strip().splitlines()[-1] if not ok else "imports pass"


@check("Z5 bundle files")
def _bundle():
    need = ["policy.onnx", "policy_config.json", "reference.npz", "scene.mjb"]
    missing = [n for n in need if not (BUNDLE / n).exists()]
    return (not missing), f"missing: {missing}" if missing else "all present"


@check("camera calibration + pen tags")
def _calib():
    ok = CAMERA.exists() and PEN_TAGS.exists()
    if not ok:
        return False, f"missing {CAMERA} or {PEN_TAGS}"
    cfg = json.loads(PEN_TAGS.read_text())
    return True, f"{len(cfg.get('tags', cfg))} pen tags configured"


@check("ZMQ ports 5556/5565 free")
def _ports():
    busy = []
    for port in (5556, 5565):
        s = socket.socket()
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            busy.append(port)
        finally:
            s.close()
    return (not busy), f"in use: {busy}" if busy else "free"


@check("dex5_lonicera daemon")
def _lonicera():
    if not LONICERA.exists():
        return False, f"missing {LONICERA}"
    r = subprocess.run(["ldd", str(LONICERA)], capture_output=True, text=True)
    missing = [l.split()[0] for l in r.stdout.splitlines() if "not found" in l]
    return (not missing), f"missing libs: {missing}" if missing else "deps ok"


@check("DDS network interfaces")
def _ifaces():
    ifaces = [p.name.split("@")[0]
              for p in Path("/sys/class/net").iterdir()
              if p.name != "lo"]
    return bool(ifaces), ",".join(ifaces) if ifaces else "none found"


@check("wire convention mapping")
def _wire():
    sys.path.insert(0, str(REPO / "third_party/z5-pen-spin-deploy"))
    from deploy.z5_pen_spin.lib.core import Bundle
    from deploy.z5_pen_spin.lib.hand_driver import wire_mapping
    b = Bundle(BUNDLE)
    for conv in ("legacy", "urdf20261103"):
        idx, signs = wire_mapping(b.joint_names, conv)
        assert len(idx) == 22
    flipped = int((signs < 0).sum())
    return True, f"22 joints mapped, {flipped} sign-flipped in urdf20261103"


def main():
    failed = 0
    print("Z5 real-hardware pre-flight\n" + "-" * 50)
    for name, fn in CHECKS:
        try:
            ok, note = fn()
        except Exception as exc:  # noqa: BLE001 - report, don't die
            ok, note = False, f"{type(exc).__name__}: {exc}"
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: {note}")
        failed += 0 if ok else 1
    print("-" * 50)
    if failed:
        print(f"{failed} check(s) failed — fix before connecting the hand")
        return 1
    print("All checks pass — ready to connect the hand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
