# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""D435/D435i global-shutter IR1 capture, with calibration from the active stream.

The SDK is imported only when this source is selected. Images and intrinsics
always describe the same rectified Y8 stream; no RGB or depth alignment is used.
"""
from __future__ import annotations

import copy
import math
from collections import deque

import numpy as np


def opencv_intrinsics(intr, rs):
    """Accept only distortion models that OpenCV can consume unchanged."""
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]], dtype=float)
    if intr.model == rs.distortion.none:
        dist = np.zeros(5)
    elif intr.model == rs.distortion.brown_conrady:
        dist = np.asarray(intr.coeffs, dtype=float)
    else:
        raise ValueError(f"Unsupported RealSense distortion model: {intr.model}; "
                         "use rectified IR1 Y8, not raw or color images")
    if not np.isfinite(K).all() or not np.isfinite(dist).all() or min(intr.fx, intr.fy) <= 0:
        raise ValueError("Invalid RealSense calibration")
    return K, dist


class RealSenseCamera:
    """Context-managed live capture. Starting/stopping never touches the hand."""

    def __init__(self, cfg, *, serial=None, rs_module=None):
        self.cfg = copy.deepcopy(cfg)
        self.serial = serial or self.cfg['realsense'].get('serial')
        self.rs = rs_module
        self.pipeline = None
        self.capture_fps = None
        self.actual_exposure_us = None
        self.skipped_frames = 0
        self._samples = deque(maxlen=90)
        self._last_number = None

    def __enter__(self):
        if self.rs is None:
            try:
                import pyrealsense2 as rs
            except ImportError as exc:
                raise RuntimeError("RealSense source needs pyrealsense2; "
                                   "install the pen-spin-deploy environment") from exc
            self.rs = rs
        rs = self.rs
        stream = self.cfg['realsense']
        width, height, fps = (int(stream[k]) for k in ('width', 'height', 'fps'))
        if min(width, height, fps) <= 0:
            raise ValueError("RealSense width, height and fps must be positive")
        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError("No RealSense camera found; check USB connection and permissions")
        if not self.serial and len(devices) != 1:
            raise RuntimeError("Multiple RealSense devices found; select --camera-serial")
        if not self.serial:
            self.serial = devices[0].get_info(rs.camera_info.serial_number)
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(config)
            self.pipeline = pipeline
            device = profile.get_device()
            name = device.get_info(rs.camera_info.name)
            if 'D435' not in name.upper():
                raise ValueError(f"This source is validated for D435/D435i, found {name}")
            active = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            if (active.width(), active.height(), active.fps(), active.format()) != (
                    width, height, fps, rs.format.y8):
                raise ValueError("RealSense did not activate the requested IR1 Y8 profile")
            intr = active.get_intrinsics()
            self.K, self.dist = opencv_intrinsics(intr, rs)
            # All controls below belong to the stereo sensor, not the RGB sensor.
            sensor = device.first_depth_sensor()
            self.sensor = sensor
            if sensor.supports(rs.option.hdr_enabled):
                self._set(sensor, rs.option.hdr_enabled, 0)
            self._set(sensor, rs.option.frames_queue_size, 1)
            self._set(sensor, rs.option.enable_auto_exposure, 0)
            exposure = float(self.cfg['capture']['exposure_time'])
            if exposure > 1e6 / fps:
                raise ValueError("Exposure exceeds the requested frame period; lower it to retain FPS")
            self._set(sensor, rs.option.exposure, exposure)
            self._set(sensor, rs.option.gain, float(self.cfg['capture']['gain']))
            self._set(sensor, rs.option.emitter_enabled, 0)
            self.exposure_us = sensor.get_option(rs.option.exposure)
            self.gain = sensor.get_option(rs.option.gain)
            # Sensor controls take several frames to take effect.
            for _ in range(10):
                pipeline.wait_for_frames(1000)
            self.cfg['sensor'] = dict(full_width=width, full_height=height)
            self.cfg['roi'] = dict(offset_x=0, offset_y=0, width=width, height=height)
            print(f"[RealSense] {name} serial={self.serial} IR1 Y8 {width}x{height}@{fps}; "
                  f"exposure readback={self.exposure_us:g}us gain={self.gain:g} emitter=OFF")
            print(f"[RealSense] active-stream K={self.K.tolist()} dist={self.dist.tolist()}")
            return self
        except Exception as exc:
            self.close()
            raise RuntimeError(f"Cannot initialize RealSense IR1 {width}x{height}@{fps}: {exc}") from exc

    @staticmethod
    def _set(sensor, option, value):
        if not sensor.supports(option):
            raise ValueError(f"Required RealSense control not supported: {option}")
        bounds = sensor.get_option_range(option)
        if not math.isfinite(value) or not bounds.min <= value <= bounds.max:
            raise ValueError(f"{option}={value} outside [{bounds.min}, {bounds.max}]")
        sensor.set_option(option, value)

    def frames(self):
        rs = self.rs
        misses = 0
        while True:
            try:
                frames = self.pipeline.wait_for_frames(500)
            except RuntimeError as exc:
                misses += 1
                if misses >= 10:
                    raise RuntimeError("RealSense: no frames for 5 seconds; check camera connection") from exc
                continue
            misses = 0
            # Drop queued frames instead of processing an increasingly old stream.
            for _ in range(8):
                newer = self.pipeline.poll_for_frames()
                if not newer:
                    break
                frames = newer
            ir = frames.get_infrared_frame(1)
            if not ir:
                raise RuntimeError("RealSense frameset is missing the requested IR1 image")
            number, timestamp = ir.get_frame_number(), ir.get_timestamp()
            if self._last_number is not None:
                self.skipped_frames += max(0, number - self._last_number - 1)
            self._last_number = number
            self._samples.append((number, timestamp))
            first_number, first_time = self._samples[0]
            elapsed = timestamp - first_time
            if elapsed > 0:
                self.capture_fps = 1000 * (number - first_number) / elapsed
            if ir.supports_frame_metadata(rs.frame_metadata_value.actual_exposure):
                self.actual_exposure_us = ir.get_frame_metadata(rs.frame_metadata_value.actual_exposure)
            yield np.asanyarray(ir.get_data()).copy(), None

    def status(self):
        fps = '--' if self.capture_fps is None else f'{self.capture_fps:.1f}'
        exposure = self.actual_exposure_us
        label = 'metadata' if exposure is not None else 'readback'
        if exposure is None:
            exposure = self.exposure_us
        return (f"IR capture={fps} FPS | Exposure {label}={exposure:g}us "
                f"| skipped={self.skipped_frames}")

    def adjust_exposure(self, factor):
        """Preview-only manual adjustment; never write the user's YAML."""
        bounds = self.sensor.get_option_range(self.rs.option.exposure)
        upper = min(bounds.max, 1e6 / self.cfg['realsense']['fps'])
        value = max(bounds.min, min(upper, self.exposure_us * factor))
        value = bounds.min + round((value - bounds.min) / bounds.step) * bounds.step
        value = max(bounds.min, min(upper, value))
        self._set(self.sensor, self.rs.option.exposure, value)
        self.exposure_us = self.sensor.get_option(self.rs.option.exposure)
        return self.exposure_us

    def close(self):
        if self.pipeline is not None:
            pipeline, self.pipeline = self.pipeline, None
            pipeline.stop()

    def __exit__(self, *_exc):
        self.close()
