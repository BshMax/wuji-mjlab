"""Run the exported Z5 policy; the default mock backend does not access hardware."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from deploy.z5_pen_spin.lib.control import run

ROOT = Path(__file__).resolve().parents[3]


def positive(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def nonnegative(value):
    number = float(value)
    if not 0 <= number < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=ROOT / "generated/deploy/z5-pen-spin")
    p.add_argument("--backend", choices=("mock", "mujoco", "real"), default="mock")
    p.add_argument("--engine", choices=("torchscript", "onnx"), default="torchscript")
    p.add_argument("--clip", help="group/clip ID from --list-clips; comma-separated list or 'all' plays clips back-to-back")
    p.add_argument("--list-clips", action="store_true")
    p.add_argument("--steps", type=int, default=0, help="0: run the complete clip including initial hold")
    p.add_argument("--duration", type=nonnegative, default=0)
    p.add_argument("--hold-seconds", type=nonnegative, default=3.0)
    p.add_argument("--move-seconds", type=positive, default=3.0)
    p.add_argument("--no-realtime", action="store_true", help="mock/MuJoCo only")
    p.add_argument("--viewer", action="store_true", help="MuJoCo passive viewer")
    p.add_argument("--log", type=Path, help="NPZ telemetry path or directory (directory auto-names YYYYMMDD_HHMMSS_<clip>.npz; existing .npz auto-suffixes timestamp)")
    p.add_argument("--wire-convention", choices=("legacy", "urdf20261103"))
    p.add_argument("--dds-interface")
    p.add_argument("--dds-domain", type=int, default=0)
    p.add_argument("--sdk-path", type=Path, help="directory containing unitree_sdk2py")
    p.add_argument("--kp", type=nonnegative)
    p.add_argument("--kd", type=nonnegative)
    p.add_argument("--large-kp", type=nonnegative,
                   help="optional higher kp for the 4 finger roll1 motors (and thumb if --thumb-kp is omitted)")
    p.add_argument("--large-kd", type=nonnegative,
                   help="optional kd override for the large proximal motors")
    p.add_argument("--thumb-kp", type=nonnegative,
                   help="optional dedicated kp for the 3 proximal thumb motors (pitch/roll/yaw1, e.g. 3.0)")
    p.add_argument("--thumb-kd", type=nonnegative,
                   help="optional dedicated kd for the 3 proximal thumb motors (pitch/roll/yaw1, e.g. 0.08)")
    p.add_argument("--no-thumb-kin-comp", dest="thumb_kin_comp", action="store_false", default=True,
                   help="disable urdf20261103 thumb kinematic offset alignment (+0.0138, -0.0321, +0.0072 rad)")
    p.add_argument("--roll1-bias-comp", type=float, default=0.0,
                   help="static feedforward bias (rad) added to index/middle/ring/little roll1 targets (e.g. 0.025)")
    p.add_argument("--ema-alpha", type=positive,
                   help="override policy_config.json ema_alpha in (0, 1] (e.g. 0.20 to damp 6Hz chatter)")
    p.add_argument("--pose-endpoint", default="tcp://127.0.0.1:5555")
    p.add_argument("--pose-max-age", type=positive, default=0.12)
    p.add_argument("--max-pose-jump-m", type=nonnegative, default=0.03,
                   help="reject single-frame pen PnP position jumps larger than this (m; 0 disables)")
    p.add_argument("--max-pose-jump-deg", type=nonnegative, default=30.0,
                   help="reject single-frame pen PnP shaft-angle jumps larger than this (deg; 0 disables)")
    p.add_argument("--pose-recover-ticks", type=int, default=0,
                   help="ticks to smoothly blend pose after a blind camera blackout recovers (0 = instant snap, default)")
    p.add_argument("--pose-occluded-mode", choices=("hold", "propagate"), default="hold",
                   help="behavior during dropped/occluded camera frames: 'hold' keeps last valid pose in place (default), 'propagate' extrapolates along reference")
    p.add_argument("--pose-fallback", type=nonnegative, default=1.0,
                   help="seconds the reference pen pose may stand in for a lost "
                        "camera stream before the run aborts (0 = abort immediately)")
    p.add_argument("--state-max-age", type=positive, default=0.35)
    p.add_argument(
        "--joint-freeze-seconds",
        type=nonnegative,
        default=0.8,
        help="abort real run if any single joint encoder stays bit-identical under >0.25 rad tracking error for this long (0 disables)",
    )
    p.add_argument("--startup-timeout", type=positive, default=5.0)
    p.add_argument("--temperature-limit", type=positive, default=65.0)
    p.add_argument("--max-target-speed", type=positive, default=8.0, help="rad/s per target joint")
    p.add_argument("--max-tracking-lead", type=nonnegative, default=0.0,
                   help="anti-catapult compliance guard on the 4 finger roll1 joints: clamp desired within [q - lead, q + lead] (rad; 0 disables)")
    p.add_argument("--other-max-tracking-lead", type=nonnegative, default=0.0,
                   help="optional lead clamp for thumb and distal/yaw joints (0 = unclamped so thumb_yaw1/index_yaw never starve)")
    p.add_argument("--start-joint-tolerance", type=positive, default=0.2)
    p.add_argument("--takeover-tolerance", type=nonnegative, default=0.02,
                   help="per-joint slack beyond limits allowed at takeover (rad)")
    p.add_argument("--max-position-error", type=positive, default=0.5)
    p.add_argument("--max-axis-error-deg", type=positive, default=150.0)
    p.add_argument("--guard-patience", type=int, default=15,
                   help="consecutive over-threshold ticks before the tracking guard aborts "
                        "(rides out single detection glitches; 1 = stop immediately)")
    p.add_argument("--max-overrun", type=positive, default=0.2)
    p.add_argument("--no-tracking-guard", action="store_true", help="mock/MuJoCo diagnostics only")
    p.add_argument("--no-pen", action="store_true", help="skip the pose stream and feed the reference pen pose (parameter-ID runs without a pen)")
    p.add_argument("--joint-group", choices=("all", "thumb", "index", "middle", "ring", "little"), default="all",
                   help="only drive one finger's joints; the rest hold the clip start pose (self-collision-safe sysid)")
    p.add_argument("--flat-pose", default="zero",
                   help="held-joint pose for --joint-group runs: 'zero' (all joints 0), 'measured' (takeover pose), "
                        "or path to a .npy/.npz file containing a 22-vector or a 'rest_pose' field")
    p.add_argument("--settle-seconds", type=nonnegative, default=1.5,
                   help="seconds to hold the flat rest pose before clip playback (joint-group runs)")
    p.add_argument("--wait-start", action="store_true", help="hold the start pose until 'c' is pressed ('b' aborts)")
    p.add_argument("--frame-endpoint", help="ZMQ PUB endpoint broadcasting current frame/errors for z5_mirror")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.steps < 0 or args.dds_domain < 0:
        p.error("--steps and --dds-domain must be nonnegative")
    try:
        print(json.dumps(run(args), indent=2, ensure_ascii=False))
    except KeyboardInterrupt:
        print("Stopped; driver closed.")
    except (ValueError, RuntimeError, TimeoutError, ImportError, OSError) as exc:
        p.exit(1, f"Deployment stopped: {exc}\n")


if __name__ == "__main__":
    main()
