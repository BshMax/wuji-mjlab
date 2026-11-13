# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Vision-in-the-loop sim-to-sim test: render -> real detection -> policy -> MuJoCo.

The most complete pre-hardware check. Two MuJoCo models run side by side:

  dynamics : the exported scene.mjb via MujocoHandDriver (identical to the
             deployment's --backend mujoco, canonical training dynamics)
  render   : the project's z5 hand + real wrist mount + an AprilTag face +
             the textured aruco_pen250 asset + a fixed camera matching
             config/camera.yaml intrinsics

Every control tick the render model is synced to the dynamics state, the
camera frame is rendered, and the REAL observer code runs on it —
scripts/pen_world_observer._detect_world_tag for the wrist tag and
lib.pen_pose.PenPoseEstimator for the pen — then the pose goes out on the
observer's legacy schema, through the real z5_pose_bridge subprocess, into
PenReceiver, the 163-obs, ONNX policy, ResidualController, TargetGuard, and
finally back into the dynamics actuators.

Nothing in this loop is faked except photon transport: the detection,
calibration, transforms, transport, policy and actuation path are the same
code that will run on the physical rig.

Requires mujoco==3.15 + opencv + onnxruntime + pyzmq: run via the
`z5-vision-sim` pixi task.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLE_ROOT = REPO_ROOT / "third_party/z5-pen-spin-deploy"
PEN_SPIN = REPO_ROOT / "deploy/pen_spin"
TOOLS_DIR = Path(__file__).resolve().parent
for _p in (str(BUNDLE_ROOT), str(PEN_SPIN), str(PEN_SPIN / "scripts")):
    sys.path.insert(0, _p)

from deploy.z5_pen_spin.lib.control import tracking_error  # noqa: E402
from deploy.z5_pen_spin.lib.core import Bundle, Policy, ResidualController  # noqa: E402
from deploy.z5_pen_spin.lib.hand_driver import TargetGuard  # noqa: E402
from deploy.z5_pen_spin.lib.simulation import MujocoHandDriver  # noqa: E402
from deploy.z5_pen_spin.lib.transport import PenReceiver  # noqa: E402
from lib.camera_config import get_camera_matrix, load_camera_config  # noqa: E402
from lib.math_utils import (matrix_from_quat, normalize_quat, quat_apply,  # noqa: E402
                            quat_from_matrix, quat_inv, quat_mul)
from lib.pen_pose import PenPoseEstimator  # noqa: E402
from lib.tag_frame import compose  # noqa: E402
from lib.z5_tag_frame import (Z5_ROOT_IN_PALM_QUAT, Z5_TAG_IN_PALM_POS,  # noqa: E402
                              Z5_TAG_IN_PALM_QUAT)
from pen_world_observer import WorldFrame, _detect_world_tag  # noqa: E402

PREVIEW_SCENE = PEN_SPIN / "scenes/z5_jig_preview.xml"
PEN_MJCF = (REPO_ROOT / "src/wuji_mjlab/assets/objects/aruco_pen250"
            / "xmls/pen_tracking.xml")
CAMERA_CFG = PEN_SPIN / "config/camera.yaml"
TAG_PNG = REPO_ROOT / "outputs/vision_sim/world_tag_id0.png"


def _lookat_quat(pos, target, up=(0.0, 0.0, 1.0)):
    """Camera frame quat (wxyz): MuJoCo cameras look along local -Z, +Y up."""
    fwd = np.asarray(target) - np.asarray(pos)
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(up, dtype=np.float64)
    right = np.cross(up, fwd)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, -down, -fwd], axis=1)  # cols: cam +X,+Y(up),+Z(back)
    return quat_from_matrix(R)


def build_render_scene(bundle, cam_cfg):
    """Hand + mount + printed tag + textured pen + camera, at the bundle's
    fixed palm pose. Returns (model, data, joint_qids, pen_qid, cam_name)."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(PREVIEW_SCENE))
    # MSAA so synthetic edges are not pixel-step staircases — the real
    # AprilTag corner refinement expects the smooth gradients a lens makes.
    spec.visual.quality.offsamples = 4
    fixed = np.asarray(bundle.config["fixed_palm_pose_world_xyzw"],
                       dtype=np.float64)
    palm_pos, palm_quat = fixed[:3], fixed[[6, 3, 4, 5]]

    mount = spec.body("z5_mount")
    mount.pos = palm_pos.copy()
    mount.quat = quat_mul(palm_quat, Z5_ROOT_IN_PALM_QUAT)

    # Printed wrist tag: OpenCV +X -> +Y_palm (fingers), +Y -> -X_palm.
    tag_pos_world, tag_quat_world = compose(
        palm_pos, palm_quat, Z5_TAG_IN_PALM_POS, Z5_TAG_IN_PALM_QUAT)
    tag_quat_world = quat_mul(tag_quat_world, np.array([0.0, 0.0, 0.0, 1.0]))
    import cv2
    TAG_PNG.parent.mkdir(parents=True, exist_ok=True)
    tag_img = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        0, 640, 0)  # face edge == black border edge: a crisp silhouette
    cv2.imwrite(str(TAG_PNG), cv2.cvtColor(tag_img, cv2.COLOR_GRAY2BGR))
    spec.add_texture(name="wrist_tag_tex",
                     type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(TAG_PNG))
    mat = spec.add_material(name="wrist_tag_mat")
    mat.texuniform = False  # per-face texcoords so the marker is not stretched
    mat.specular = 0.0      # printed paper: no specular blowout hiding the code
    mat.shininess = 0.0
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "wrist_tag_tex"
    # Sit the printed tag a hair above the mount's pad surface so the STL
    # face does not swallow it (the ~2 mm lift is negligible for pose).
    # A white paper plate ~20 mm larger than the black square is the quiet
    # zone the AprilTag contour fit needs — same as the real sticker.
    tag_lift = quat_apply(tag_quat_world, np.array([0.0, 0.0, 0.002]))
    tb = spec.worldbody.add_body(pos=tag_pos_world + tag_lift,
                               quat=tag_quat_world)
    tb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.036, 0.036, 0.0002],
                pos=[0, 0, -0.0001], rgba=[1, 1, 1, 1],
                contype=0, conaffinity=0)
    tb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0252, 0.0252, 0.0003],
                material="wrist_tag_mat", contype=0, conaffinity=0)

    # Textured pen asset (freejoint body synced to the dynamics each tick).
    pen_spec = mujoco.MjSpec.from_file(str(PEN_MJCF))
    spec.attach(pen_spec, frame=spec.worldbody.add_frame(), prefix="pen_")

    # Camera matching config/camera.yaml (pinhole; renderer has no distortion).
    K = get_camera_matrix(cam_cfg)
    width, height = 1280, 1024
    fovy = float(np.degrees(2 * np.arctan(height / 2.0 / K[1, 1])))
    cam_pos = palm_pos + np.array([-0.10, -0.30, 0.38])
    cam_quat = _lookat_quat(cam_pos, palm_pos + np.array([0.0, 0.04, 0.0]))
    spec.worldbody.add_camera(name="sim_cam", pos=cam_pos, quat=cam_quat,
                              fovy=fovy)

    model = spec.compile()
    data = mujoco.MjData(model)
    qids = []
    for name in bundle.joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    "z5_" + name)
        if jid < 0:
            raise RuntimeError(f"render scene missing joint {name}")
        qids.append(int(model.jnt_qposadr[jid]))
    pen_qid = int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                          "pen_cube_freejoint")])
    return model, data, qids, pen_qid, "sim_cam", K, (width, height), fovy


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path,
                    default=BUNDLE_ROOT / "generated/deploy/z5-pen-spin")
    ap.add_argument("--clip")
    ap.add_argument("--engine", choices=["onnx", "torchscript"], default="onnx")
    ap.add_argument("--steps", type=int, default=0,
                    help="0: complete clip including --hold-seconds")
    ap.add_argument("--hold-seconds", type=float, default=3.0)
    ap.add_argument("--calib-frames", type=int, default=30,
                    help="wrist-tag fix samples (sim equivalent of pressing w)")
    ap.add_argument("--obs-port", type=int, default=5578)
    ap.add_argument("--bridge-out-port", type=int, default=5577)
    ap.add_argument("--pose-endpoint", default="tcp://127.0.0.1:5577")
    ap.add_argument("--pose-max-age", type=float, default=0.3)
    ap.add_argument("--max-target-speed", type=float, default=8.0)
    ap.add_argument("--max-position-error", type=float, default=0.05)
    ap.add_argument("--max-axis-error-deg", type=float, default=60.0)
    ap.add_argument("--no-tracking-guard", action="store_true")
    ap.add_argument("--log", type=Path)
    ap.add_argument("--save-frames", type=Path,
                    help="dump every --save-every-th rendered frame here")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--print-every", type=int, default=50)
    ap.add_argument("--min-faces", type=int, default=1,
                    help="same default as pen_world_observer")
    ap.add_argument("--viewer", action="store_true",
                    help="open a MuJoCo 3D view of the render scene")
    ap.add_argument("--preview", action="store_true",
                    help="show the rendered camera frame as the detector "
                         "sees it (OpenCV window)")
    return ap


def main():
    args = parser().parse_args()
    import cv2
    import mujoco
    import zmq

    bundle = Bundle(args.bundle)
    clip = bundle.clip_index(args.clip)
    first, end = int(bundle.starts[clip]), int(bundle.ends[clip])
    cam_cfg = load_camera_config(str(CAMERA_CFG))
    model, data, qids, pen_qid, cam_name, K, (w, h), _ = build_render_scene(
        bundle, cam_cfg)
    dist = np.zeros(5)  # renderer is an ideal pinhole
    est = PenPoseEstimator(K, dist)
    world = WorldFrame(sample_frames=args.calib_frames)

    policy = Policy(bundle, args.engine)
    controller = ResidualController(bundle.config)
    guard = TargetGuard(bundle, "legacy", args.max_target_speed)

    ctx = zmq.Context.instance()
    obs_pub = ctx.socket(zmq.PUB)
    obs_pub.bind(f"tcp://*:{args.obs_port}")
    bridge = subprocess.Popen(
        [sys.executable, str(TOOLS_DIR / "z5_pose_bridge.py"),
         "--in-port", str(args.obs_port),
         "--out-port", str(args.bridge_out_port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    receiver = PenReceiver(args.pose_endpoint, args.pose_max_age)
    renderer = mujoco.Renderer(model, h, w)

    palm_pos = np.asarray(bundle.config["fixed_palm_pose_world_xyzw"])[:3]
    palm_quat = np.asarray(bundle.config["fixed_palm_pose_world_xyzw"])[
        [6, 3, 4, 5]]

    stats = {"detect_ok": 0, "detect_miss": 0, "stale": 0}

    def sync_and_render(driver):
        q, _ = driver.read()
        data.qpos[qids] = q
        pen_local = driver.pen_pose()
        wp, wq = compose(palm_pos, palm_quat, pen_local[:3], pen_local[3:])
        data.qpos[pen_qid:pen_qid + 7] = np.r_[wp, wq]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam_name)
        rgb = renderer.render()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def publish(gray, stamp):
        body = {"timestamp": stamp, "world_fixed": world.fixed}
        if world.fixed:
            pose = est.estimate(gray)
            if pose is not None:
                faces = {est.layout.face_normal[t] for t in pose.tag_ids}
                if len(faces) >= args.min_faces:
                    R, pos = world.to_world(matrix_from_quat(pose.quat),
                                            pose.pos)
                    quat = quat_from_matrix(R)
                    body["pen"] = {
                        "position": dict(zip("xyz", pos.tolist())),
                        "orientation": dict(zip("xyzw", quat[[1, 2, 3, 0]]))}
                    stats["detect_ok"] += 1
                else:
                    stats["detect_miss"] += 1
            else:
                stats["detect_miss"] += 1
        obs_pub.send_json(body)
        return "pen" in body

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)

    def show(gray, seen):
        if viewer is not None:
            if not viewer.is_running():
                raise SystemExit("viewer closed")
            viewer.sync()
        if args.preview:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.putText(vis, "pen " + ("TRACKED" if seen else "missed"),
                        (20, 40), 0, 0.9, (0, 255, 0) if seen else (0, 0, 255), 2)
            cv2.imshow("vision-sim", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                raise SystemExit("preview closed")

    driver = failure = None
    trace = []
    pose_err = []       # mm position, deg orientation — detection vs truth
    try:
        driver = MujocoHandDriver(bundle, first)
        q, dq = driver.read()
        guard.reset(q)
        time.sleep(0.5)  # let bridge sockets connect

        # Calibration phase: fix the wrist tag exactly like pressing 'w',
        # automatically, on a static scene.
        print(f"[vision-sim] calibrating wrist tag "
              f"({args.calib_frames} frames)...", flush=True)
        while not world.fixed:
            gray = sync_and_render(driver)
            wt = _detect_world_tag(gray, K, dist)
            show(gray, wt is not None)
            if wt is not None:
                world.add(*wt)
        print(f"[vision-sim] tag fixed after {world.samples} samples",
              flush=True)
        if args.save_frames:
            args.save_frames.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_frames / "calib.png"), gray)

        publish(sync_and_render(driver), time.time())
        deadline = time.monotonic() + 5.0
        while True:
            try:
                receiver.latest()
                break
            except TimeoutError:
                publish(sync_and_render(driver), time.time())
                if time.monotonic() >= deadline:
                    raise TimeoutError("no palm_link pose from the bridge")
                time.sleep(0.05)

        hold_steps = int(round(args.hold_seconds / bundle.dt))
        total = hold_steps + end - first
        if args.steps:
            total = min(total, args.steps)
        print(f"[vision-sim] clip={bundle.clip_ids[clip]} steps={total} "
              f"guard={'off' if args.no_tracking_guard else 'on'}", flush=True)

        for tick in range(total):
            started = time.monotonic()
            frame = first + max(0, tick - hold_steps)
            q, dq = driver.read()
            pose_seen = receiver.latest()
            truth = driver.pen_pose()
            qdiff = quat_mul(normalize_quat(pose_seen[3:]),
                             quat_inv(normalize_quat(truth[3:])))
            pose_err.append((float(np.linalg.norm(
                pose_seen[:3] - truth[:3])),
                float(np.degrees(2 * np.arccos(
                    np.clip(abs(qdiff[0]), 0, 1))))))
            pos_err, axis_err = tracking_error(
                pose_seen, bundle.references["object_pose"][frame])
            if not args.no_tracking_guard and (
                    pos_err > args.max_position_error
                    or axis_err > args.max_axis_error_deg):
                raise RuntimeError(
                    f"pen tracking error: {pos_err:.4f} m, {axis_err:.1f} deg")
            obs = bundle.observation(frame, q, dq, pose_seen,
                                     controller.last_action)
            action = policy(obs)
            desired = controller.target(bundle.references["joint_pos"][frame],
                                        action)
            target = guard.apply(desired)
            compute_ms = (time.monotonic() - started) * 1000.0
            driver.send(target)
            gray = sync_and_render(driver)
            seen = publish(gray, time.time())
            show(gray, seen)
            if not seen:
                stats["stale"] += 1
            trace.append((frame, obs, q, dq, pose_seen, action, desired,
                          target, compute_ms / 1000.0, pos_err, axis_err))
            if args.save_frames and tick % args.save_every == 0:
                cv2.imwrite(str(args.save_frames / f"f{tick:05d}.png"),
                            sync_and_render(driver))
            if tick % args.print_every == 0:
                print(f"t={tick * bundle.dt:6.1f}s f={frame:4d} "
                      f"err={pos_err * 1e3:5.0f}mm/{axis_err:4.1f}deg "
                      f"|a|max={np.abs(action).max():.2f} "
                      f"cam_err={pose_err[-1][0] * 1e3:5.1f}mm/"
                      f"{pose_err[-1][1]:4.1f}deg "
                      f"det={stats['detect_ok']}/{stats['detect_ok'] + stats['detect_miss']} "
                      f"cmp={compute_ms:4.1f}ms", flush=True)
    except BaseException as exc:
        failure = exc
    finally:
        try:
            if viewer is not None:
                viewer.close()
            if args.preview:
                cv2.destroyAllWindows()
            if driver is not None:
                driver.close()
        finally:
            receiver.close()
            bridge.terminate()
            bridge.wait(timeout=3)
        if args.log:
            fields = ("frame", "observation", "joint_pos", "joint_vel",
                      "pen_pose", "action", "desired_target", "sent_target",
                      "compute_seconds", "position_error_m", "axis_error_deg")
            args.log.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.log,
                **{n: np.array([r[i] for r in trace])
                   for i, n in enumerate(fields)},
                clip=np.array(bundle.clip_ids[clip]),
                camera_error_m=np.asarray([e[0] for e in pose_err]),
                camera_error_deg=np.asarray([e[1] for e in pose_err]),
                error=np.array("" if failure is None else str(failure)))
    if failure is not None:
        print(f"[vision-sim] FAILED after {len(trace)} steps: {failure}",
              flush=True)
        raise SystemExit(1) from failure
    print(json.dumps({
        "backend": "mujoco+vision-chain",
        "clip": bundle.clip_ids[clip],
        "steps": len(trace),
        "simulated_seconds": len(trace) * bundle.dt,
        "pen_detections": stats["detect_ok"],
        "pen_misses": stats["detect_miss"],
        "target_guard_interventions": guard.interventions,
        "max_camera_error_mm": max(
            (e[0] for e in pose_err), default=0.0) * 1e3,
        "max_camera_error_deg": max(
            (e[1] for e in pose_err), default=0.0),
        "max_compute_ms": max((r[8] * 1000 for r in trace), default=0.0),
        "max_position_error_m": max((r[9] for r in trace), default=0.0),
        "max_axis_error_deg": max((r[10] for r in trace), default=0.0),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
