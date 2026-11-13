"""Live MuJoCo mirror of the real Z5 pen-spin deployment.

Shows, in one passive viewer:
  * the real hand, driven by DDS joint feedback (opaque)
  * the real pen, driven by the camera palm-link pose stream (opaque)
  * a translucent "ghost" pen at the current reference frame (orange,
    lifted 30 mm so it never hides inside the real pen)
  * the clip's full reference pen path (yellow spheres)
  * a fading trail of the observed pen (white spheres)

Frame sync comes from play_real's --frame-endpoint broadcast; without it the
ghost snaps to the reference frame nearest to the observed pen pose.

Runs in the unitree_deploy conda env (mujoco 3.15 + unitree_sdk2py + cyclonedds).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

BUNDLE = Path("third_party/z5-pen-spin-deploy/generated/deploy/z5-pen-spin")
DEPLOY = Path("third_party/z5-pen-spin-deploy/deploy/z5_pen_spin")
sys.path.insert(0, str(DEPLOY))

from lib.core import Bundle, rotation_wxyz  # noqa: E402
from lib.hand_driver import Z5HandDriver  # noqa: E402
from lib.transport import PenReceiver  # noqa: E402

GHOST_PEN_RGBA = np.array([0.15, 0.9, 1.0, 0.55])  # ghost pen in cyan, distinct from the yellow path
TRAIL_RGBA = np.array([1.0, 1.0, 1.0, 0.8])
PATH_RGBA = np.array([1.0, 0.85, 0.1, 0.9])
ERR_RGBA = np.array([1.0, 0.25, 0.2, 1.0])


GHOST_LIFT_M = 0.05  # ghost hovers this far along the palm normal
GHOST_LIFT_DIR = np.array([0.0, 0.0, 1.0])


def _set_geom(geo, m, d, g, rgba):
    geo.type = int(m.geom_type[g])
    geo.dataid = int(m.geom_dataid[g])
    geo.pos = d.geom_xpos[g] + GHOST_LIFT_M * GHOST_LIFT_DIR
    geo.mat = d.geom_xmat[g].reshape(3, 3)
    geo.size = m.geom_size[g]
    geo.texid = -1  # no texture: otherwise ghost geoms inherit texture 0 (a tag pattern)
    geo.texuniform = 0
    geo.emission = 0.35  # slight glow so translucent geoms stay readable on the dark background
    geo.rgba = np.asarray(rgba, dtype=np.float64)


def _sphere(geo, pos, radius, rgba):
    import mujoco

    geo.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geo.dataid = -1
    geo.texid = -1
    geo.pos = np.asarray(pos, dtype=np.float64)
    geo.mat = np.eye(3)
    geo.size = np.array([radius, 0.0, 0.0])
    geo.emission = 0.35
    geo.rgba = np.asarray(rgba, dtype=np.float64)


def _palm_to_world(d, palm_id, pose):
    """(xyz + wxyz) palm_link pose -> world xyz + wxyz quat."""
    import mujoco

    r = d.xmat[palm_id].reshape(3, 3)
    world_r = r @ rotation_wxyz(np.asarray(pose[3:], dtype=np.float64))
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, world_r.ravel())
    return np.r_[d.xpos[palm_id] + r @ pose[:3], quat]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, default=BUNDLE)
    p.add_argument("--clip", help="clip id for the ghost/path (default: first clip)")
    p.add_argument("--pose-endpoint", default="tcp://127.0.0.1:5565")
    p.add_argument("--pose-max-age", type=float, default=0.5)
    p.add_argument("--frame-endpoint", default="tcp://127.0.0.1:5566", help="play_real --frame-endpoint broadcast")
    p.add_argument("--wire-convention", choices=("legacy", "urdf20261103"), default="urdf20261103")
    p.add_argument("--dds-interface")
    p.add_argument("--dds-domain", type=int, default=0)
    p.add_argument("--sdk-path", type=Path)
    p.add_argument("--no-hand", action="store_true", help="no DDS; hand stays at reference start pose")
    p.add_argument("--no-pen", action="store_true", help="no pose stream; pen tracks the reference")
    p.add_argument("--no-trail", action="store_true")
    p.add_argument("--trajectory-step", type=int, default=5, help="frames between path markers")
    p.add_argument("--trail-length", type=int, default=150)
    p.add_argument("--hz", type=float, default=60.0)
    p.add_argument("--pose-smooth", type=float, default=0.3,
                   help="EMA alpha for the displayed camera pen pose (0 = raw stream)")
    p.add_argument("--freeze-frame", type=int, help="lock the ghost at this clip-relative frame (error-checking mode)")
    args = p.parse_args()

    import mujoco
    import mujoco.viewer

    bundle = Bundle(args.bundle)
    clip = bundle.clip_index(args.clip)
    first, end = int(bundle.starts[clip]), int(bundle.ends[clip])
    ref_q = bundle.references["joint_pos"]
    ref_pen = bundle.references["object_pose"]

    model = mujoco.MjModel.from_binary_path(str(bundle.directory / "scene.mjb"))
    data = mujoco.MjData(model)  # real state
    ghost = mujoco.MjData(model)  # reference state
    joints = [model.joint("robot/" + n) for n in bundle.joint_names]
    qids = [int(j.qposadr[0]) for j in joints]
    palm_id = model.body("robot/palm_link").id
    pen_id = model.body("object/cube").id
    pen_qid = int(model.joint("object/cube_freejoint").qposadr[0])
    pen_gids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == pen_id]

    fixed = np.asarray(bundle.config["fixed_palm_pose_world_xyzw"])
    global GHOST_LIFT_DIR
    for d in (data, ghost):
        d.mocap_pos[0] = fixed[:3]
        d.mocap_quat[0] = fixed[[6, 3, 4, 5]]
        d.qpos[qids] = ref_q[first]
        mujoco.mj_forward(model, d)
    # Lift the ghost out of the hand along the mean palm->pen direction.
    lift_local = np.mean(ref_pen[:, :3], axis=0)
    lift_local /= np.linalg.norm(lift_local)
    GHOST_LIFT_DIR = data.xmat[palm_id].reshape(3, 3) @ lift_local
    # Only the pen is drawn as a ghost; hand geoms are not duplicated.

    driver = receiver = frame_sub = frame_ctx = None
    if not args.no_hand:
        driver = Z5HandDriver(
            bundle, convention=args.wire_convention, interface=args.dds_interface,
            domain=args.dds_domain, sdk_path=args.sdk_path,
        )
        print(f"[mirror] DDS hand state connected ({args.wire_convention})", flush=True)
    if not args.no_pen:
        receiver = PenReceiver(args.pose_endpoint, args.pose_max_age)
        print(f"[mirror] pose stream <- {args.pose_endpoint}", flush=True)
    if args.frame_endpoint:
        import zmq

        frame_ctx = zmq.Context()
        frame_sub = frame_ctx.socket(zmq.SUB)
        frame_sub.setsockopt(zmq.LINGER, 0)
        frame_sub.setsockopt(zmq.CONFLATE, 1)
        frame_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        frame_sub.connect(args.frame_endpoint)
        print(f"[mirror] frame sync <- {args.frame_endpoint}", flush=True)

    # Reference pen path in world (palm is a fixed mocap: recompute on clip change).
    def _path_pts(a, b):
        return np.array([
            data.xpos[palm_id] + data.xmat[palm_id].reshape(3, 3) @ ref_pen[f, :3]
            for f in range(a, b, max(1, args.trajectory_step))
        ])

    # Pen shaft axis (+X) per reference frame, for orientation-aware snapping.
    def _seg_axes(a, b):
        return np.stack([rotation_wxyz(ref_pen[f, 3:7])[:, 0] for f in range(a, b)])

    path_pts = _path_pts(first, end)
    seg_axes = _seg_axes(first, end)
    trail = deque(maxlen=args.trail_length)

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.lookat[:] = fixed[:3]
    viewer.cam.distance = 0.4
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25
    us = viewer.user_scn

    stale_pen = True
    last_pose = ref_pen[first].copy()
    smooth_pose = last_pose.copy()
    ghost_frame = first
    last_info = {}
    last_info_t = -np.inf  # monotonic time of the newest broadcast message
    hud_next = 0.0
    period = 1.0 / args.hz
    clip_len = end - first
    print(f"[mirror] clip={bundle.clip_ids[clip]} frames={clip_len} ({clip_len * bundle.dt:.1f}s)", flush=True)
    print("[mirror] running - close the window or Ctrl-C to quit", flush=True)
    try:
        while viewer.is_running():
            t0 = time.monotonic()
            if driver is not None:
                q, dq = driver.read()
                data.qpos[qids] = q
            if receiver is not None:
                try:
                    last_pose = receiver.latest()
                    stale_pen = False
                except TimeoutError:
                    stale_pen = True
            else:
                last_pose = ref_pen[ghost_frame]
            # Display-only EMA: raw PnP poses jitter a few mm at stream rate.
            a = min(max(args.pose_smooth, 0.0), 1.0)
            if stale_pen or a <= 0.0:
                smooth_pose = last_pose.copy()
            else:
                q = last_pose[3:7].copy()
                if np.dot(q, smooth_pose[3:7]) < 0.0:
                    q = -q
                smooth_pose[:3] += a * (last_pose[:3] - smooth_pose[:3])
                nq = smooth_pose[3:7] + a * (q - smooth_pose[3:7])
                smooth_pose[3:7] = nq / np.linalg.norm(nq)
            data.qpos[pen_qid:pen_qid + 7] = _palm_to_world(data, palm_id, smooth_pose)
            mujoco.mj_forward(model, data)

            # Frame sync: broadcast wins; otherwise nearest reference frame.
            if frame_sub is not None:
                import zmq

                while True:
                    try:
                        last_info = json.loads(frame_sub.recv_string(flags=zmq.NOBLOCK))
                        last_info_t = time.monotonic()
                    except zmq.Again:
                        break
            frame_fresh = time.monotonic() - last_info_t < 0.5
            bclip = last_info.get("clip") if frame_fresh else None
            if bclip and bclip in bundle.clip_ids and bclip != bundle.clip_ids[clip]:
                clip = bundle.clip_index(bclip)
                first, end = int(bundle.starts[clip]), int(bundle.ends[clip])
                clip_len = end - first
                path_pts = _path_pts(first, end)
                seg_axes = _seg_axes(first, end)
                print(f"[mirror] following broadcast clip -> {bclip} ({clip_len} frames)", flush=True)
            if args.freeze_frame is not None:
                ghost_frame = first + args.freeze_frame
            elif frame_fresh and "frame" in last_info:
                ghost_frame = int(last_info["frame"])
            elif receiver is not None and not stale_pen:
                # Snap by position AND shaft orientation (axis is unsigned);
                # position-only matching picks frames whose pose differs wildly.
                dpos = np.linalg.norm(ref_pen[first:end, :3] - last_pose[:3], axis=1)
                obs_ax = rotation_wxyz(np.asarray(last_pose[3:7], dtype=np.float64))[:, 0]
                daxis = np.arccos(np.clip(np.abs(seg_axes @ obs_ax), -1.0, 1.0))
                cost = dpos + 0.05 * daxis
                cand = int(np.argmin(cost))
                # Hysteresis keeps the ghost from hopping between distant
                # frames when no frame broadcast is available.
                cur = min(max(ghost_frame - first, 0), len(cost) - 1)
                if cost[cand] < cost[cur] - 0.005:
                    ghost_frame = first + cand
            ghost_frame = min(max(ghost_frame, first), end - 1)

            ghost.qpos[pen_qid:pen_qid + 7] = _palm_to_world(ghost, palm_id, ref_pen[ghost_frame])
            mujoco.mj_forward(model, ghost)

            if not args.no_trail and not stale_pen:
                trail.append(data.xpos[pen_id].copy())

            # Rebuild decorations.
            max_geom = len(us.geoms)
            us.ngeom = 0
            for g in pen_gids:
                if us.ngeom >= max_geom:
                    break
                geo = us.geoms[us.ngeom]; us.ngeom += 1
                _set_geom(geo, model, ghost, g, GHOST_PEN_RGBA)
            for i, pt in enumerate(path_pts):
                if us.ngeom >= max_geom - args.trail_length - 8:
                    break
                geo = us.geoms[us.ngeom]; us.ngeom += 1
                _sphere(geo, pt, 0.002, PATH_RGBA if i % 5 else np.r_[PATH_RGBA[:3], 0.45])
            n_trail = len(trail)
            for i, pt in enumerate(trail):
                if us.ngeom >= max_geom:
                    break
                a = 0.05 + 0.6 * (i + 1) / max(1, n_trail)
                geo = us.geoms[us.ngeom]; us.ngeom += 1
                _sphere(geo, pt, 0.0025, np.r_[TRAIL_RGBA[:3], a])

            viewer.sync()
            if time.monotonic() >= hud_next:
                hud_next = time.monotonic() + 0.25
                err = ""
                if "pos_err" in last_info:
                    err = f" err={last_info['pos_err'] * 1000:.0f}mm/{last_info['axis_err']:.1f}deg"
                flags = (" STALE_PEN" if stale_pen and receiver else "") + \
                        (f" [{last_info.get('state', '?')}]" if frame_fresh else " [no frame stream]") + \
                        (" FROZEN" if args.freeze_frame is not None else "")
                print(f"[mirror] {bundle.clip_ids[clip].split('/')[-1]} "
                      f"f={ghost_frame - first:4d}/{clip_len}{err}{flags}", flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            viewer.close()
        except Exception:
            pass
        for x in (driver, receiver, frame_sub):
            if x is not None:
                x.close()
        if frame_ctx is not None:
            frame_ctx.term()
    import os

    os._exit(0)  # skip interpreter teardown; mujoco viewer teardown segfaults otherwise


if __name__ == "__main__":
    main()
