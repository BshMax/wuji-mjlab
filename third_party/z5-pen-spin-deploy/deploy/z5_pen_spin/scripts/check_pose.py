"""Read-only comparison of live vision and a selected reference frame."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deploy.z5_pen_spin.lib.control import tracking_error
from deploy.z5_pen_spin.lib.core import Bundle
from deploy.z5_pen_spin.lib.transport import PenReceiver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, default=Path(__file__).resolve().parents[3] / "generated/deploy/z5-pen-spin"
    )
    parser.add_argument("--clip")
    parser.add_argument("--frame", type=int, default=0, help="index relative to the selected clip")
    parser.add_argument("--pose-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    bundle = Bundle(args.bundle)
    clip = bundle.clip_index(args.clip)
    if not 0 <= args.frame < bundle.ends[clip] - bundle.starts[clip]:
        parser.error("--frame is outside the selected clip")
    reference = bundle.references["object_pose"][int(bundle.starts[clip]) + args.frame]
    receiver = PenReceiver(args.pose_endpoint)
    deadline = time.monotonic() + args.duration
    print(json.dumps({"reference_pose_palm_wxyz": reference.tolist()}, ensure_ascii=False))
    try:
        while time.monotonic() < deadline:
            try:
                pose = receiver.latest()
                position, angle = tracking_error(pose, reference)
                result = dict(valid=True, pose_palm_wxyz=pose.tolist(), position_error_m=position, axis_error_deg=angle)
            except TimeoutError as exc:
                result = dict(valid=False, error=str(exc))
            print(json.dumps(result, ensure_ascii=False), flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
