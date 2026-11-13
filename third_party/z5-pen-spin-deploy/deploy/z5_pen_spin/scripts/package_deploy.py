"""Package deployment code and an exported bundle for a standalone handoff."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RELEASE_NAME = "z5-pen-spin-deploy"
BUNDLE_FILES = ("policy_config.json", "reference.npz", "policy.ts", "policy.onnx", "scene.mjb")


def add_text(archive, path, text):
    data = text.encode("utf-8")
    info = tarfile.TarInfo(str(Path(RELEASE_NAME) / path))
    info.size = len(data)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def package(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    for name in BUNDLE_FILES:
        if not (bundle / name).is_file():
            raise FileNotFoundError(f"missing exported bundle file {name}; run export_policy.py first")
    config = json.loads((bundle / "policy_config.json").read_text())
    if config["schema"] != "dex_supermimic_z5_pen_spin:1":
        raise ValueError("unsupported deployment bundle")
    version = config["mujoco_version"]
    if not version or any(char not in "0123456789." for char in version):
        raise ValueError("invalid exported MuJoCo version")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for file in sorted((ROOT / "deploy").rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                archive.add(file, arcname=str(Path(RELEASE_NAME) / file.relative_to(ROOT)), recursive=False)
        for name in (*BUNDLE_FILES, "validation.json", "deployment_checks.json"):
            file = bundle / name
            if file.is_file():
                archive.add(file, arcname=f"{RELEASE_NAME}/generated/deploy/z5-pen-spin/{name}", recursive=False)
        for file in (ROOT / "NOTICE", *sorted((ROOT / "LICENSES").glob("*.txt"))):
            archive.add(file, arcname=str(Path(RELEASE_NAME) / file.relative_to(ROOT)), recursive=False)
        add_text(archive, "README.md", (ROOT / "deploy/z5_pen_spin/PORTABLE.md").read_text())
        add_text(archive, "requirements-sim.txt", f"-r deploy/z5_pen_spin/requirements.txt\nmujoco=={version}\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "generated/deploy/z5-pen-spin")
    parser.add_argument("--output", type=Path, default=ROOT / "generated/releases/z5-pen-spin-deploy.tar.gz")
    args = parser.parse_args()
    output = package(args.bundle, args.output)
    print(f"Archive: {output}\nSize: {output.stat().st_size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
