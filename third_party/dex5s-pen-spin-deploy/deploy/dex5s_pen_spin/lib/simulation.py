"""No-hardware adapters for the same policy/controller loop."""

from __future__ import annotations

import numpy as np

from .core import rotation_wxyz


class MockHandDriver:
    """Encoder/pose fixture for pipeline checks; does not model pen dynamics."""

    def __init__(self, bundle, first_frame):
        self.bundle = bundle
        self.q = bundle.references["joint_pos"][first_frame].copy()
        self.dq = np.zeros(22, dtype=np.float32)

    def read(self):
        return self.q.copy(), self.dq.copy()

    def send(self, target):
        self.dq = (target - self.q) / self.bundle.dt
        self.q = target.copy()

    def close(self):
        pass


class MujocoHandDriver:
    """CPU MuJoCo rollout of the exported canonical training model."""

    def __init__(self, bundle, first_frame, viewer=False):
        import mujoco

        self.mj, self.bundle = mujoco, bundle
        self.model = mujoco.MjModel.from_binary_path(str(bundle.directory / "scene.mjb"))
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint("robot/" + name) for name in bundle.joint_names]
        self.qids = [int(j.qposadr[0]) for j in joints]
        self.vids = [int(j.dofadr[0]) for j in joints]
        self.aids = [self.model.actuator("robot/" + name + "_actuator").id for name in bundle.joint_names]
        self.palm_id, self.pen_id = self.model.body("robot/palm_link").id, self.model.body("object/cube").id
        fixed = np.asarray(bundle.config["fixed_palm_pose_world_xyzw"])
        if self.model.nmocap != 1:
            raise ValueError("expected exactly one hand mocap root")
        self.data.mocap_pos[0] = fixed[:3]
        self.data.mocap_quat[0] = fixed[[6, 3, 4, 5]]
        self.data.qpos[self.qids] = bundle.references["joint_pos"][first_frame]
        self.data.ctrl[self.aids] = self.data.qpos[self.qids]
        mujoco.mj_forward(self.model, self.data)
        pose = bundle.references["object_pose"][first_frame]
        r = self.data.xmat[self.palm_id].reshape(3, 3)
        world_r = r @ rotation_wxyz(pose[3:])
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, world_r.ravel())
        obj_qid = int(self.model.joint("object/cube_freejoint").qposadr[0])
        self.data.qpos[obj_qid : obj_qid + 7] = np.r_[self.data.xpos[self.palm_id] + r @ pose[:3], quat]
        mujoco.mj_forward(self.model, self.data)
        self.substeps = round(bundle.dt / self.model.opt.timestep)
        self.viewer = None
        if viewer:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = fixed[:3]
            self.viewer.cam.distance = 0.45

    def read(self):
        return self.data.qpos[self.qids].astype(np.float32), self.data.qvel[self.vids].astype(np.float32)

    def pen_pose(self):
        r = self.data.xmat[self.palm_id].reshape(3, 3).T
        pos = r @ (self.data.xpos[self.pen_id] - self.data.xpos[self.palm_id])
        quat = np.empty(4)
        self.mj.mju_mat2Quat(quat, (r @ self.data.xmat[self.pen_id].reshape(3, 3)).ravel())
        return np.r_[pos, quat].astype(np.float32)

    def send(self, target):
        self.data.ctrl[self.aids] = target
        self.mj.mj_step(self.model, self.data, nstep=self.substeps)
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
