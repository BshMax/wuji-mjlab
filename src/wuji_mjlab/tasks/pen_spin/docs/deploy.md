# PenSpin · Deploy

[中文版](deploy_zh.md) · [Task home](../README.md) · [Setup](setup.md) · [Train](train.md) · [Deploy](deploy.md)

Run all commands from the **repository root**. Hardware and printable resources are in [Setup](setup.md); training and export are in [Train](train.md).

**Startup order: start camera observations, confirm a valid pen pose, then start the policy. `pen-policy` opens the live mirror by default.** See [Start the camera and policy](#start-the-camera-and-policy) for commands.

## Policy models

| Group | Motion |
| --- | --- |
| `01_single_axis` | Single-axis rotation |
| `02_continuous_single_axis` | Continuous single-axis rotation |
| `03_continuous_multi_axis` | Continuous multi-axis rotation |
| `04_stable_grasp_rotation` | Stable grasp rotation |
| `05_multi_axis_two_rotations` | Multi-axis two rotations |

The [v2026.10.22 asset bundle](https://github.com/wuji-technology/wuji-mjlab/releases/download/v2026.10.22/wuji-mjlab-v2026.10.22-assets.zip) contains PenSpin hardware, motion data and five matched checkpoint sets under `pen_spin/checkpoints/<group>/`. Copy the bundle's `pen_spin/checkpoints/` into `src/wuji_mjlab/tasks/pen_spin/data/checkpoints/`, preserving group directories and all companion files as shown below. Dataset downloads and training steps are in [Train](train.md).

```text
src/wuji_mjlab/tasks/pen_spin/data/checkpoints/<group>/
├── model.pt
├── policy.onnx
└── config.json
```

The release includes `model.pt`, `policy.onnx` and `config.json` for each group. Keep these files together and deploy `policy.onnx` with a reference trajectory from the same group. For your own training runs, retain the original `params/` directory separately to reproduce the training configuration.

Deployment uses the supplied [`pen_spin_policy.json`](../../../../../deploy/pen_spin/config/pen_spin_policy.json) at 50 Hz. Keep the exported `config.json` with the model; the runtime does not load it automatically.

## Hardware and environment

Prepare Wuji Hand 2, a 250 mm tagged pen, a wrist AprilTag, a Hikrobot camera and the vendor MVS SDK. The pen has a 178 mm exposed shaft and a 14.9 mm shaft diameter. Install the separate deployment environment:

```bash
pixi install -e pen-spin-deploy
```

Check [camera calibration](../../../../../deploy/pen_spin/config/camera.yaml), [pen tag layout](../../../../../deploy/pen_spin/config/pen_tags.json) and [control settings](../../../../../deploy/pen_spin/config/control.yaml). Camera intrinsics must match the actual device. MVS SDK installation and setup checks are covered in the [hardware setup guide](setup.md).

## Camera placement and observation quality

**Pen spinning depends heavily on camera observations.** Establish a camera position that provides stable, continuous pen poses with minimal occlusion and motion blur before running the policy. See the assembly photo and camera preview in the [PenSpin hardware guide](setup.md#5-camera-placement-and-lighting).

1. Keep both tagged pen ends, the full motion envelope and the wrist reference tag in view. Avoid sustained finger/mount occlusion; adjust focus, exposure and lighting to keep moving tags sharp.
2. Prefer an oblique view showing multiple tag faces. A `1 faces / COPLANAR` label indicates single-face coplanar observations; adjusting the viewpoint can improve pose stability.
3. Confirm the wireframe matches the physical pen's position, axis and ends. Check the entire motion for jumps, flips and sustained dropouts.
4. Keep the camera and hand mount fixed after establishing the world frame. If either moves, press `w` in the preview to resample the wrist reference and check alignment again.

## Start the camera and policy

The default deployment uses **two terminals**, both at the repository root. `pen-policy` starts **real-hand control and the live MuJoCo mirror together**, showing measured joints, the camera-observed pen and the reference motion. The mirror visualizes the physical run; enabling it does not make deployment simulation-only.

### 1. Start camera observations first and keep them running

```bash
# Terminal 1: start first and keep running throughout deployment
pixi run -e pen-spin-deploy pen-observer --preview
```

Wait for wrist-tag sampling to finish and the world frame to become fixed. Confirm that the pen outline is aligned and poses update continuously before starting the policy. An open camera preview alone does not establish readiness. If observations are unavailable, the policy exits before connecting the hand; restore camera observations, then restart the policy.

### 2. Start policy control with its live mirror

Prepare an ONNX and NPZ from the same group. This example uses Group 3; both paths must point to existing files.

```bash
# Terminal 2: real-hand control + live mirror; the window opens by default
pixi run -e pen-spin-deploy pen-policy \
  --policy src/wuji_mjlab/tasks/pen_spin/data/checkpoints/03_continuous_multi_axis/policy.onnx \
  --motion src/wuji_mjlab/tasks/pen_spin/data/rawdata/clips/03_continuous_multi_axis/clip_001.npz
```

**Do not start a separate `pen-sim` for the default workflow or switch to `python -m runtime --sim`.** `pen-policy` already enables the mirror. If the observer is running, proceed directly to step 2. Run only one policy process per hand; stop the previous policy before switching groups. The camera observer can remain running.

For another group, change both ONNX and NPZ paths. `--motion` selects one supplied `.npz` trajectory. Use `--hand-sn "<serial>"` to select a device when multiple hands are connected.

`pen-sim` is an optional standalone read-only observer for checking observations before policy startup, not a required third process.

**The control process drives the physical hand: clear its motion area before startup.** It moves slowly toward the initial hand pose, then enters `FROZEN`:

| Input | Effect |
|---|---|
| First Enter | Check pen placement, enter `HOLD` and hold the first frame |
| Second Enter | Enter `POLICY` and advance along the reference trajectory |
| `r` | Reset and open the hand |
| Ctrl+C | Exit the process; not a replacement for a hardware emergency stop |

Use `--hold-last-frame` to hold the final reference after the motion. Additional runtime options are described below.

## Policy contract and runtime options

Use the ready-to-use `.npz` reference trajectories from the release and the policy for the same motion group. Runtime observation and action settings are defined in [`pen_spin_policy.json`](../../../../../deploy/pen_spin/config/pen_spin_policy.json).

The first Enter is refused if the pen was not seen, the sample is too old, its
position differs by over 50 mm, its shaft differs by over 60 degrees, or the
pen is end-for-end.

`--hold-last-frame` keeps the final reference frame, with zero reference
velocity, until `r` or Ctrl+C. `--open-loop` feeds joint state from the clip
for policy inference. `--pen-axis-offset-m d`
shifts the live pen observation by `d` along the pen axis toward A_top.

## Pen tag layouts

`deploy/pen_spin/config/pen_tags.json` already contains the layout for the 250 mm pen (178 mm exposed shaft). Use it directly. Check that it loads and print the tag count and total length:

```bash
pixi run -e pen-spin-deploy python - <<'PY'
from lib.pen_pose import PenTagLayout
layout = PenTagLayout("deploy/pen_spin/config/pen_tags.json")
print(f"tags={len(layout)}, length_mm={layout.total_length * 1000:.1f}")
PY
```

Expected output: `tags=18, length_mm=250.0`. Physical dimensions and textures must match the layout.

## D435 / D435i global-shutter IR source

Use `pixi run -e pen-spin-deploy pen-observer-realsense --preview` instead of the Hikvision observer. The source reads rectified left-IR Y8, with intrinsics from the active SDK stream. RGB-to-gray conversion does not turn the rolling-shutter RGB sensor into a global-shutter sensor. No depth stream is needed for the existing multi-marker PnP estimator.

Settings live in `deploy/pen_spin/config/realsense.yaml`: 848×480 at 90 FPS, manual 1000 us exposure, gain 16, emitter off. Use `--camera-serial` for multiple devices, `--camera-config` for a separate profile, and `--exposure-us` for a runtime override. D435i 1280×720 supports at most 30 FPS on the checked device. Unsupported profiles fail explicitly. Re-select the detection ROI after changing resolution; ROI saves affect only the selected config.

Resample the wrist tag after moving/replacing the camera. Keep camera and wrist mount fixed; moving the wrist tag relative to the hand requires re-establishing the transform in `tag_frame.py`. Wrist-tag black-square size is 50.4 mm. Keep the pen's custom source dictionary and physical geometry unchanged.

For a camera-only check add `--max-frames 300`. Record per-frame measurements with `--diagnostics /tmp/pen-static.jsonl` (must be a new file), then run `pixi run -e pen-spin-deploy python deploy/pen_spin/tools/summarize_observer.py /tmp/pen-static.jsonl --static`. Omit `--static` for moving-pen data. Processing times exclude capture wait and preview; stationary repeatability does not establish absolute accuracy. Verify known distances, overlay alignment, tag pixel size, and valid pose rate before enabling the hand policy. The receiver retains the last valid sample: monitor pose age as well as the visualization.

### Whole-sheet detection test

Run `pixi run -e pen-spin-deploy pen-tag-test --exposure-us 5000` to test an entire printed A4 sheet. This is equivalent to `--source realsense --detect-only --preview`. It scans the full image, ignores but preserves the saved pen ROI, skips wrist calibration and rigid-body PnP, and never creates a pose publisher. All standard 36h11 IDs and custom pen IDs 0–17 are supported. Duplicate printed IDs remain separate detections with session-local spatial track numbers (`T1`, `T2`, etc.). These are image-space associations, not unique physical identities or 3D poses; crossing, occlusion and large displacement can change associations.

The wrist sheet has 12 instances of ID 0; the pen sheet has 108 instances of 18 distinct IDs. Real-camera detection depends on pixel size, lighting and blur. Press `q`/Esc to quit, `t` to toggle trails, `r` to reset tracks, and `+`/`-` to adjust runtime exposure without changing YAML. Use longer exposure for initial stationary-paper tests; use shorter exposure and more light for fast motion. The existing `--diagnostics` and summary tool also support this mode.
