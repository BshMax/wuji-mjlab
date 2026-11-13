"""Export a deployment bundle with the project's training Python environment."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deploy.z5_pen_spin.lib.export import DEFAULT_BUNDLE, DEFAULT_CHECKPOINT, export_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--no-onnx", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.checkpoint, args.output, onnx=not args.no_onnx), indent=2))
    print(f"Bundle: {args.output.resolve()}")


if __name__ == "__main__":
    main()
