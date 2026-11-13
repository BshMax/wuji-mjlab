"""Latest-sample ZMQ transport, with acquisition-time freshness checks."""

from __future__ import annotations

import time

import numpy as np

from .core import finite_vector, rotation_wxyz


class PoseCache:
    def __init__(self, max_age=0.3):
        if not np.isfinite(max_age) or max_age <= 0:
            raise ValueError("pose max_age must be positive")
        self.max_age = max_age
        self.pose = None
        self.timestamp = -np.inf
        self.received_at = -np.inf

    def update(self, message, *, wall_time=None, monotonic_time=None):
        now = time.time() if wall_time is None else wall_time
        stamp = float(message["timestamp"])
        if message.get("frame") != "palm_link" or message.get("quaternion_order") != "wxyz":
            raise ValueError("pose message must declare frame=palm_link and quaternion_order=wxyz")
        if not np.isfinite(stamp) or stamp > now + 0.05 or now - stamp > self.max_age:
            raise ValueError("pen acquisition timestamp is stale or in the future; synchronize host clocks")
        if not message.get("valid", False):
            return  # A missed detection must not refresh the last valid pose.
        if stamp <= self.timestamp:
            return
        pen = message["pen"]
        pose = np.r_[
            finite_vector(pen["position"], 3, "pen position"), finite_vector(pen["quaternion"], 4, "pen quaternion")
        ].astype(np.float32)
        rotation_wxyz(pose[3:])
        pose[3:] /= np.linalg.norm(pose[3:])
        self.pose, self.timestamp = pose, stamp
        self.received_at = time.monotonic() if monotonic_time is None else monotonic_time

    def latest(self, *, wall_time=None, monotonic_time=None):
        wall = time.time() if wall_time is None else wall_time
        mono = time.monotonic() if monotonic_time is None else monotonic_time
        if (
            self.pose is None
            or wall - self.timestamp > self.max_age
            or mono - self.received_at > self.max_age
            or self.timestamp > wall + 0.05
        ):
            raise TimeoutError("no fresh, valid palm-local pen pose")
        return self.pose.copy()


class PenReceiver:
    def __init__(self, endpoint, max_age=0.3):
        import zmq

        self.zmq = zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.connect(endpoint)
        self.cache = PoseCache(max_age)
        self.last_error = None

    def latest(self):
        try:
            message = self.socket.recv_json(flags=self.zmq.NOBLOCK)
        except self.zmq.Again:
            pass
        else:
            try:
                self.cache.update(message)
                self.last_error = None
            except (KeyError, TypeError, ValueError) as exc:
                self.last_error = str(exc)
        try:
            return self.cache.latest()
        except TimeoutError as exc:
            if self.last_error:
                raise TimeoutError(f"{exc}: {self.last_error}") from exc
            raise

    def close(self):
        self.socket.close()
        self.context.term()
