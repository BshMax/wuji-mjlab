#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Install the supplied vendor SDK locally, enumerate cameras, inspect capabilities."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.camera_config import load_camera_config
from lib.mindvision_camera import (
    DEFAULT_SDK_ROOT,
    capabilities,
    device_info,
    enumerate_devices,
    load_sdk,
    sdk_call,
    select_device,
)

# Keep original vendor files outside Git; never execute the bundled installer.
SDK_FILES = ('demo/python_demo/mvsdk.py', 'lib/x64/libMVSDK.so', 'lib/arm64/libMVSDK.so')


def install(archive, destination):
    destination = Path(destination).expanduser().resolve()
    with zipfile.ZipFile(archive) as bundle:
        payloads = {}
        for name in SDK_FILES:
            matches = [entry for entry in bundle.infolist() if entry.filename == f'CameraSDK/{name}']
            if len(matches) != 1:
                raise ValueError(f'SDK ZIP must contain exactly one CameraSDK/{name}')
            entry = matches[0]
            if (entry.external_attr >> 16) & 0o170000 == 0o120000 or entry.file_size > 100*1024*1024:
                raise ValueError(f'Invalid SDK entry: {entry.filename}')
            payloads[name] = bundle.read(entry)
        # Check every destination before writing; never replace a different SDK.
        for name, data in payloads.items():
            path = destination/name
            if not path.resolve().is_relative_to(destination):
                raise ValueError('SDK destination contains an escaping symlink')
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f'Different SDK exists: {path}; choose another --sdk-root')
        for name, data in payloads.items():
            path = destination/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    manifest = dict(archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                    files={name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()})
    (destination/'pen_spin_sdk_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return dict(installed=str(destination), files=list(payloads), system_changes=False)


def inspect_camera(sdk, settings):
    device = select_device(enumerate_devices(sdk), settings)
    handle = sdk_call(sdk, 'CameraInit', device, -1, -1)
    try:
        cap = sdk_call(sdk, 'CameraGetCapability', handle)
        resolution = sdk_call(sdk, 'CameraGetImageResolution', handle)
        return dict(**device_info(device), **capabilities(cap), shutter='unknown',
                    current_size=[resolution.iWidth, resolution.iHeight],
                    current_frame_speed_mode=sdk_call(sdk, 'CameraGetFrameSpeed', handle),
                    exposure_range_us=sdk_call(sdk, 'CameraGetExposureTimeRange', handle),
                    exposure_us=sdk_call(sdk, 'CameraGetExposureTime', handle),
                    analog_gain=sdk_call(sdk, 'CameraGetAnalogGain', handle),
                    analog_gain_range=[cap.sExposeDesc.uiAnalogGainMin, cap.sExposeDesc.uiAnalogGainMax],
                    analog_gain_step=cap.sExposeDesc.fAnalogGainStep,
                    sdk=sdk._provenance)
    finally:
        sdk_call(sdk, 'CameraUnInit', handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['install', 'list', 'inspect'])
    parser.add_argument('archive', nargs='?', help='CameraSDK.zip, required for install')
    parser.add_argument('--sdk-root', default=None, help=f'Default: {DEFAULT_SDK_ROOT} or MINDVISION_SDK_ROOT')
    parser.add_argument('--camera-config', help='Read serial/device_index and SDK root from YAML')
    parser.add_argument('--serial', help='Choose a camera by its reported serial')
    parser.add_argument('--output', type=Path, help='Save inspection JSON (SDK also prints its own diagnostic messages)')
    args = parser.parse_args(argv)
    settings = load_camera_config(args.camera_config).get('camera', {}) if args.camera_config else {}
    root = args.sdk_root or settings.get('sdk_root')
    if args.serial is not None:
        settings = dict(settings, serial=args.serial, device_index=None)
    try:
        if args.action == 'install':
            if not args.archive:
                parser.error('install requires CameraSDK.zip')
            result = install(args.archive, root or DEFAULT_SDK_ROOT)
        else:
            sdk = load_sdk(root)
            result = (dict(devices=[device_info(d) for d in enumerate_devices(sdk)]) if args.action == 'list'
                      else inspect_camera(sdk, settings))
        serialized = json.dumps(result, ensure_ascii=False, indent=2)+'\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding='utf-8')
        print(serialized, flush=True)
        return 0
    except (RuntimeError, ValueError, OSError, zipfile.BadZipFile) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
