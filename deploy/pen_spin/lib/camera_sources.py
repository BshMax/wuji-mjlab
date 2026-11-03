# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Camera interchange for 2D recording; no shutter or synchronization is invented.

An SDK adapter supplies a context manager with describe(), frames(), and close().
frames() yields CameraFrame: named Y8 views and acquisition metadata. A trusted
local plugin can be selected as camera.backend=python, camera.adapter=module:factory.
"""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraFrame:
    images: dict[str, np.ndarray]
    metadata: dict


class RealSenseSource:
    def __init__(self, cfg):
        from lib.realsense_camera import RealSenseCamera
        mode = cfg.get('camera', {}).get('views', 'stereo')
        if mode not in ('mono', 'stereo'):
            raise ValueError('RealSense camera.views must be mono or stereo')
        self.stereo = mode == 'stereo'
        self.camera = RealSenseCamera(cfg, stereo=self.stereo)

    def __enter__(self):
        self.camera.__enter__()
        return self

    def describe(self):
        camera = self.camera
        result = dict(backend='realsense', serial=camera.serial, config=camera.cfg,
                      views=['left', 'right'] if self.stereo else ['left'],
                      left_K=camera.K.tolist(), left_dist=camera.dist.tolist(),
                      shutter='global', pairing='hardware-paired' if self.stereo else 'single',
                      exposure_readback_us=camera.exposure_us, gain_readback=camera.gain,
                      exposure_units='microseconds', frame_number_origin='hardware')
        if self.stereo:
            result.update(right_K=camera.right_K.tolist(), right_dist=camera.right_dist.tolist())
        return result

    def frames(self):
        camera = self.camera
        for left, right in camera.frames():
            images = dict(left=left)
            if self.stereo:
                images['right'] = right
            yield CameraFrame(images, dict(frame_number=camera.frame_number,
                timestamp_ms=camera.timestamp_ms, timestamp_domain=camera.timestamp_domain,
                host_receive_ns=camera.host_receive_ns, sync_delta_ms=camera.sync_delta_ms,
                actual_exposure_us=camera.actual_exposure_us, skipped_frames=camera.skipped_frames))

    def adjust_exposure(self, factor):
        return self.camera.adjust_exposure(factor)

    def close(self):
        self.camera.close()

    def __exit__(self, *_exc):
        self.close()


def _calibration(config, side, shape):
    calibration = config.get('calibration', {}).get(side)
    if calibration is None:
        return None, None
    if (calibration['width'], calibration['height']) != (shape[1], shape[0]):
        raise ValueError(f'{side} calibration size does not match the actual camera image')
    matrix = np.asarray(calibration['K'], float)
    if 'dist' not in calibration:
        raise ValueError('Calibration must explicitly provide dist; [] is only for known rectified images')
    distortion = np.asarray(calibration['dist'], float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0
            or not np.allclose(matrix[2], [0, 0, 1]) or distortion.size not in (0, 4, 5, 8, 12, 14)
            or not np.isfinite(distortion).all()):
        raise ValueError(f'Invalid OpenCV pinhole calibration for {side}')
    if calibration.get('model', 'opencv_pinhole') != 'opencv_pinhole':
        raise ValueError('Use an OpenCV pinhole distortion model; fisheye needs its own adapter/rectification')
    return matrix.tolist(), distortion.tolist()


class OpenCVSource:
    """UVC/video devices. Two independent devices are NOT a synchronized stereo rig."""

    def __init__(self, cfg, capture_factory=None):
        self.cfg = cfg
        self.settings = cfg.get('camera', {})
        self.capture_factory = capture_factory or cv2.VideoCapture
        self.captures = {}
        self.info = {}

    def __enter__(self):
        devices = dict(left=self.settings.get('device', 0))
        if self.settings.get('right_device') is not None:
            devices['right'] = self.settings['right_device']
        devices = {side: int(device) if isinstance(device, str) and device.isdigit() else device
                   for side, device in devices.items()}
        keys = [str(Path(f'/dev/video{device}' if isinstance(device, int) else device).resolve())
                for device in devices.values()]
        if len(keys) != len(set(keys)):
            raise ValueError('Left and right devices must be different; a mono view is never duplicated')
        self.pending = None
        try:
            for side, device in devices.items():
                if isinstance(device, str) and device.startswith('/dev/'):
                    device = str(Path(device).resolve())
                capture = self.capture_factory(device)
                self.captures[side] = capture
                if not capture.isOpened():
                    raise RuntimeError(f'Cannot open {side} OpenCV device {device!r}')
                for key, prop in (('width', cv2.CAP_PROP_FRAME_WIDTH), ('height', cv2.CAP_PROP_FRAME_HEIGHT),
                                  ('fps', cv2.CAP_PROP_FPS)):
                    value = self.settings.get(key)
                    if value is not None:
                        if not np.isfinite(value) or value <= 0:
                            raise ValueError(f'camera.{key} must be positive')
                        capture.set(prop, float(value))
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best effort, not a latency guarantee
            first = self._read(0)
            self.info = dict(backend='opencv', config=self.cfg, views=list(devices),
                             shutter='unknown', pairing='host-sequential-unsynchronized' if len(devices) == 2 else 'single',
                             exposure_units='driver-specific', buffering='driver-dependent', frame_number_origin='software_read_index', devices=devices)
            for side, image in first.images.items():
                height, width = image.shape
                if self.settings.get('strict_profile', True):
                    for key, actual in (('width', width), ('height', height)):
                        if self.settings.get(key) is not None and int(self.settings[key]) != actual:
                            raise ValueError(f'{side} camera did not activate requested {key}; actual={actual}')
                    requested_fps = self.settings.get('fps')
                    actual_fps = self.captures[side].get(cv2.CAP_PROP_FPS)
                    if requested_fps and actual_fps > 0 and abs(actual_fps-requested_fps) > 1:
                        raise ValueError(f'{side} FPS differs: requested={requested_fps}, driver={actual_fps}')
                K, dist = _calibration(self.cfg, side, image.shape)
                self.info.update({f'{side}_K': K, f'{side}_dist': dist,
                                  f'{side}_size': [width, height],
                                  f'{side}_driver_fps': self.captures[side].get(cv2.CAP_PROP_FPS)})
            self.pending = first
            return self
        except Exception:
            self.close()
            raise

    def _read(self, index):
        images, received, origins = {}, [], {}
        for side, capture in self.captures.items():
            ok, image = capture.read()
            received.append(time.perf_counter_ns())
            if not ok or image is None:
                raise RuntimeError(f'{side} camera stopped returning frames; pause recording and check connection')
            if image.dtype != np.uint8:
                raise ValueError('OpenCV adapter expects 8-bit images; raw 10/12/16-bit devices need an SDK adapter')
            origins[side] = 'native_gray' if image.ndim == 2 else 'color_to_gray'
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
            images[side] = image.copy()
        return CameraFrame(images, dict(frame_number=index, host_receive_ns=max(received),
                          timestamp_ms=None, timestamp_domain='host_receive_only',
                          sync_delta_ms=None, host_view_read_delta_ms=(max(received)-min(received))/1e6,
                          actual_exposure_us=None, skipped_frames=None, pixel_origin=origins))

    def frames(self):
        index = 0
        if self.pending is not None:
            yield self.pending
            self.pending = None
            index += 1
        while True:
            frame = self._read(index)
            for side, image in frame.images.items():
                if self.info[f'{side}_size'] != [image.shape[1], image.shape[0]]:
                    raise RuntimeError('Camera image size changed; reload source and calibration')
            yield frame
            index += 1

    def describe(self):
        return self.info

    def adjust_exposure(self, _factor):
        raise ValueError('OpenCV exposure units depend on the device; use the vendor control tool or an SDK adapter')

    def close(self):
        for capture in self.captures.values():
            capture.release()
        self.captures.clear()

    def __exit__(self, *_exc):
        self.close()


def open_camera(config):
    """Factory for built-ins or an explicitly selected trusted SDK plugin."""
    backend = config.get('camera', {}).get('backend', 'realsense')
    if backend == 'realsense':
        return RealSenseSource(config)
    if backend == 'mindvision':
        from lib.mindvision_camera import MindVisionSource
        return MindVisionSource(config)
    if backend == 'opencv':
        return OpenCVSource(config)
    if backend == 'python':
        adapter = config['camera'].get('adapter', '')
        module, separator, name = adapter.partition(':')
        if not separator or not module or not name:
            raise ValueError('camera.adapter must be a trusted installed module:factory')
        factory = getattr(importlib.import_module(module), name)
        if not callable(factory):
            raise ValueError('Camera adapter factory is not callable')
        return factory(config)
    raise ValueError(f'Unknown camera backend {backend!r}; select realsense, mindvision, opencv, or python')
