#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Guided camera calibration: staged prompts, one keypress per capture.

Run in a terminal. A live grayscale preview window opens immediately and stays
open. Each stage prints instructions; every press of c/Space/Enter captures one
frame. A capture that fails the R/D/S/Q/N checks is blocked — press again
within 3 s to force it (T is display-only, its estimate uses approximate
intrinsics). After the last stage it runs cv2.calibrateCamera, saves
camera_calibration.npz and updates camera.yaml.

Detection runs at ~8 Hz on a half-resolution image and corners are refined at
full resolution, so the preview stays at camera rate.
"""
import argparse
import glob
import os
import select
import sys
import time
from ctypes import *
from pathlib import Path

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = DEPLOY_ROOT / "config"

sys.path.insert(0, str(SCRIPT_DIR))

from wuji_reorient_deploy.camera_config import load_camera_config, setup_camera_roi
from wuji_reorient_deploy.mvs_sdk import ensure_mvs_importable

ensure_mvs_importable()

from MvImport.MvCameraControl_class import *  # noqa: E402,F401,F403

from camera_calibrate import (  # noqa: E402
    CHESSBOARD_SIZE,
    SQUARE_SIZE,
    calculate_image_quality,
    check_region,
    get_chessboard_metrics,
)

N_EXPECTED = CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1]
DETECT_SCALE = 0.5          # board search runs on a half-size image
DETECT_INTERVAL = 0.12      # seconds between detection attempts (~8 Hz)
STABLE_NEED = 4             # consecutive stable detections required
STABLE_MOVE_PX = 3.0        # mean corner drift still counted as "holding still"
DIVERSE_MIN_PX = 4.0        # min mean corner distance vs this stage's captures
QUALITY_MIN = 45            # quality floor; manual grabs below it get a warning
FORCE_WINDOW = 3.0          # seconds to re-press and override a blocked grab

# Estimated board distance shown next to `size`; assumes SQUARE_SIZE is true.
REF_FX = 1700.0             # overwritten with camera.yaml fx in main()
BOARD_CORNER_DIAG = SQUARE_SIZE * float(
    np.hypot(CHESSBOARD_SIZE[0] - 1, CHESSBOARD_SIZE[1] - 1))

STAGES = [
    dict(desc="Stage 1/7 · 中心 · 正面 · 中距",
         help="2 张: 板子正对镜头放画面中心 ~45cm; 第1张正面, 第2张转180°或平移换个摆法",
         regions=('center',), size=(0.45, 0.80), tilt=(0, 20), need=2),
    dict(desc="Stage 2/7 · 中心 · 倾斜 · 中距",
         help="3 张: 同一位置分别向左、向右、向上(或下)倾斜 15-45°",
         regions=('center',), size=(0.45, 0.80), tilt=(15, 50), need=3),
    dict(desc="Stage 3/7 · 左/右边缘 · 正面 · 中距",
         help="2 张: 左边缘正面 1 张 + 右边缘正面 1 张",
         regions=('left', 'right'), size=(0.45, 0.80), tilt=(0, 25), need=2),
    dict(desc="Stage 4/7 · 上/下边缘 · 正面 · 中距",
         help="2 张: 上边缘正面 1 张 + 下边缘正面 1 张",
         regions=('top', 'bottom'), size=(0.45, 0.80), tilt=(0, 25), need=2),
    dict(desc="Stage 5/7 · 任意边缘 · 倾斜 · 中距",
         help="2 张: 两个不同边缘位置, 各倾斜 10-45°",
         regions=('left', 'right', 'top', 'bottom'), size=(0.45, 0.80), tilt=(12, 45), need=2),
    dict(desc="Stage 6/7 · 中心 · 正面 · 近距",
         help="1 张: 拉近到 ~30cm, 几乎占满画面, 88 个角点必须全可见",
         regions=('center',), size=(0.65, 0.98), tilt=(0, 30), need=1),
    dict(desc="Stage 7/7 · 中心 · 正面 · 远距",
         help="1 张: 拉远到 ~60cm, 正面",
         regions=('center',), size=(0.30, 0.60), tilt=(0, 30), need=1),
]

ADVICE = {
    'R': '板子中心进黄框',
    'D': '距离（size 大=太近，小=太远）',
    'T': '倾斜角（近似内参估算，仅供参考）',
    'S': '稳定（按 c 瞬间停稳即可）',
    'Q': '清晰度（检查对焦/光线）',
    'N': '与已采姿态不重复',
}

# Normalized target regions, must match camera_calibrate.check_region.
REGION_RECTS = {
    'center': (0.20, 0.80, 0.20, 0.80),
    'left':   (0.00, 0.40, 0.20, 0.80),
    'right':  (0.60, 1.00, 0.20, 0.80),
    'top':    (0.20, 0.80, 0.00, 0.40),
    'bottom': (0.20, 0.80, 0.60, 1.00),
}


def enable_display():
    if not os.environ.get('DISPLAY'):
        sockets = sorted(glob.glob('/tmp/.X11-unix/X*'))
        if not sockets:
            return False
        os.environ['DISPLAY'] = ':' + sockets[0].rsplit('X', 1)[1]
    return True


def find_board(gray):
    found, corners = False, None
    if hasattr(cv2, 'findChessboardCornersSB'):
        try:
            found, corners = cv2.findChessboardCornersSB(
                gray, CHESSBOARD_SIZE,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE
                      + cv2.CALIB_CB_EXHAUSTIVE
                      + cv2.CALIB_CB_LARGER)
        except cv2.error:
            found = False
        if not (found and corners is not None and corners.shape[0] == N_EXPECTED):
            found, corners = False, None
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not (found and corners is not None and corners.shape[0] == N_EXPECTED):
            found, corners = False, None
    return found, corners


def read_line_nonblocking():
    """Return a stripped lowercase line if the user typed one, else None."""
    if not sys.stdin.isatty():
        return None
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if line == '':
                return None
            return line.strip().lower()
    except (OSError, ValueError):
        pass
    return None


def run_calibration(obj_points, img_points, img_shape):
    print("\n" + "=" * 60)
    print(f"Starting calibration... (using {len(img_points)} images)")
    print("=" * 60)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape[::-1], None, None)
    print(f"\nReprojection error (RMS): {ret:.4f} pixels")
    if ret > 1.0:
        print("WARNING: RMS > 1.0, recommend recalibrating")
    elif ret > 0.5:
        print("Acceptable: RMS in 0.5-1.0")
    else:
        print("Excellent: RMS < 0.5")

    per_view = []
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        proj, _ = cv2.projectPoints(obj_points[i], rvec, tvec, K, dist)
        err = np.abs(proj.reshape(-1, 2) - img_points[i].reshape(-1, 2)).mean()
        per_view.append((err, i))
    worst = sorted(per_view, reverse=True)[:5]
    print("Worst views (err px, index): "
          + ", ".join(f"#{i}:{e:.2f}" for e, i in worst))
    print("\nCamera intrinsics matrix K:")
    print(K)
    print("\nDistortion coefficients dist:")
    print(dist.flatten())
    return ret, K, dist


def write_yaml(path, K, dist):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg['intrinsics'] = {
        'fx': float(K[0, 0]), 'fy': float(K[1, 1]),
        'cx': float(K[0, 2]), 'cy': float(K[1, 2]),
    }
    d = [float(x) for x in np.asarray(dist).flatten()[:5]]
    cfg['distortion'] = {'k1': d[0], 'k2': d[1], 'p1': d[2], 'p2': d[3], 'k3': d[4]}
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"Wrote intrinsics+distortion to {path}")


def draw_overlay(vis, stage, armed, got, found, metrics, quality,
                 stable_count, checks, fps, total):
    nW, nH = vis.shape[1], vis.shape[0]
    for r in stage['regions']:
        x1n, x2n, y1n, y2n = REGION_RECTS[r]
        cv2.rectangle(vis, (int(x1n * nW), int(y1n * nH)),
                      (int(x2n * nW), int(y2n * nH)), (0, 255, 255), 1)
    if found and metrics is not None:
        x = 10
        for key, ok in checks:
            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.putText(vis, f"{key}{'OK' if ok else 'X'}", (x, nH - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            x += 62
        est_cm = 100.0 * REF_FX * BOARD_CORNER_DIAG / max(
            metrics['size'] * np.hypot(nW, nH), 1.0)
        info = (f"tilt={metrics['tilt']:.0f}deg(need {stage['tilt'][0]}-{stage['tilt'][1]},参考) "
                f"size={metrics['size']:.2f}(need {stage['size'][0]}-{stage['size'][1]}, ~{est_cm:.0f}cm) "
                f"q={quality['total']:.0f} stable={stable_count}/{STABLE_NEED}")
        cv2.putText(vis, info, (10, nH - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        if armed and all(ok for k, ok in checks if k != 'N') and not checks[-1][1]:
            cv2.putText(vis, "MOVE BOARD for next capture", (10, nH - 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        cv2.putText(vis, "board NOT found - show the whole 11x8 board",
                    (10, nH - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    head = (f"{stage['desc']}  待拍 {got}/{stage['need']}  c/空格/回车=拍 d=撤销 s=跳过" if armed
            else f"{stage['desc']}  c/空格/回车=拍第1张  s=跳过  q=退出")
    cv2.putText(vis, head, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)
    cv2.putText(vis, f"{fps:.0f}fps total={total}", (10, nH - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--yaml', default=str(CONFIG_DIR / 'camera.yaml'),
                    help='camera.yaml to update with the result')
    ap.add_argument('--no-write', action='store_true', help='do not modify camera.yaml')
    ap.add_argument('--no-preview', action='store_true', help='terminal status only')
    args = ap.parse_args()

    preview = not args.no_preview and enable_display()
    if not preview:
        print("[guided] no preview window; watch this terminal")

    print("=" * 64)
    print(" 分阶段相机标定 —— 终端提示 + 实时预览窗")
    print(f" 标定板: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} 内角点, "
          f"{SQUARE_SIZE * 1000:.0f}mm 方格")
    print(" 预览窗常开: 黄框=目标区域; 底部 R/D/T/S/Q/N 实时读数(仅供参考)")
    print(" 判定项: R=区域 D=距离 S=稳定 Q=质量 N=姿态不重复 —— 不达标会拦截拍摄")
    print("         T=倾斜 仅参考(近似估算); 被拦截后 3 秒内再按一次 = 强制拍")
    print(" 手动采集: 每按一次 c/空格/回车 拍一张; 拍满自动进入下一阶段")
    print(" 窗口或终端按键等效: c/空格/回车=拍, d=撤销上一张, s=跳过, q=退出")
    print("=" * 64)

    print("\nInitializing camera...")
    MvCamera.MV_CC_Initialize()
    deviceList = MV_CC_DEVICE_INFO_LIST()
    MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if deviceList.nDeviceNum == 0:
        print("No camera found!")
        return 1

    cam = MvCamera()
    stDevice = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    cam.MV_CC_CreateHandle(stDevice)
    cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    cam.MV_CC_SetFloatValue("ExposureTime", 15000)
    cam.MV_CC_SetFloatValue("Gain", 8.0)
    cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerGB8)

    _cam_cfg = load_camera_config()
    global REF_FX
    REF_FX = float(_cam_cfg['intrinsics']['fx'])
    width, height = setup_camera_roi(cam, _cam_cfg)
    cam.MV_CC_StartGrabbing()
    print(f"Camera ready! Resolution: {width}x{height}")
    if preview:
        cv2.namedWindow('Guided Calibration', cv2.WINDOW_NORMAL)
    print()

    objp = np.zeros((N_EXPECTED, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_points, img_points = [], []
    stOutFrame = MV_FRAME_OUT()
    gray = None
    si, armed = 0, False
    got, stage_corners = 0, []
    stable_count = 0
    last_corners = None
    last_detect = 0.0
    last_status = 0.0
    found, metrics, quality, corners = False, None, None, None
    checks = [('R', False), ('D', False), ('T', False),
              ('S', False), ('Q', False), ('N', False)]
    t_start = time.time()
    frames_total = 0
    stage = STAGES[si]
    quit_all = False
    force_until = 0.0

    def print_stage(idx):
        st = STAGES[idx]
        print("-" * 64)
        print(st['desc'])
        print(f"  怎么做: {st['help']}")
        print(f"  需要 {st['need']} 张 | 区域={'+'.join(st['regions'])} "
              f"占比={st['size']} 倾斜={st['tilt']}°")
        print("  >>> 摆好第 1 个姿态后按 c / 空格 / 回车 拍 (s 跳过, d 撤销, q 退出): ")

    print_stage(0)

    try:
        while si < len(STAGES) and not quit_all:
            ret = cam.MV_CC_GetImageBuffer(stOutFrame, 60)
            now = time.time()
            if ret == 0:
                nH, nW = stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth
                data = string_at(stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
                bayer = np.frombuffer(data, dtype=np.uint8).reshape(nH, nW)
                gray = cv2.cvtColor(bayer, cv2.COLOR_BayerGB2GRAY)
                cam.MV_CC_FreeImageBuffer(stOutFrame)
                frames_total += 1

                if now - last_detect >= DETECT_INTERVAL:
                    last_detect = now
                    small = cv2.resize(gray, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
                    f, c = find_board(small)
                    if f:
                        c_full = cv2.cornerSubPix(
                            gray, c.astype(np.float32) / DETECT_SCALE,
                            (11, 11), (-1, -1), criteria)
                        found, corners = True, c_full
                        metrics = get_chessboard_metrics(c_full, gray.shape, objp)
                        quality = calculate_image_quality(gray, c_full)
                        if last_corners is not None:
                            movement = np.mean(np.abs(c_full - last_corners))
                            stable_count = stable_count + 1 if movement < STABLE_MOVE_PX else 0
                        last_corners = c_full.copy()
                    else:
                        found, metrics, quality = False, None, None
                        last_corners = None
                        stable_count = 0

            region_ok = size_ok = tilt_ok = diverse_ok = False
            if found and metrics is not None:
                region_ok = any(check_region(metrics['center'], r)
                                for r in stage['regions'])
                size_ok = stage['size'][0] <= metrics['size'] <= stage['size'][1]
                tilt_ok = stage['tilt'][0] <= metrics['tilt'] <= stage['tilt'][1]
                diverse_ok = all(np.mean(np.abs(corners - prev)) > DIVERSE_MIN_PX
                                 for prev in stage_corners)
            stable_ok = stable_count >= STABLE_NEED
            quality_ok = bool(quality) and quality['total'] >= QUALITY_MIN
            checks = [('R', region_ok), ('D', size_ok), ('T', tilt_ok),
                      ('S', stable_ok), ('Q', quality_ok), ('N', diverse_ok)]

            want_skip = want_grab = want_undo = False
            if preview and gray is not None:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if found and corners is not None:
                    cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners, True)
                fps = frames_total / max(now - t_start, 1e-6)
                draw_overlay(vis, stage, armed, got, found, metrics, quality,
                             stable_count, checks, fps, len(img_points))
                try:
                    cv2.imshow('Guided Calibration', vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        quit_all = True
                    elif key == ord('s'):
                        want_skip = True
                    elif key == ord('d'):
                        want_undo = True
                    elif key in (13, 32, ord('c')):
                        want_grab = True
                except cv2.error:
                    print("[guided] preview closed/failed, continuing headless")
                    preview = False

            line = read_line_nonblocking()
            if line == 'q':
                quit_all = True
            elif line == 's':
                want_skip = True
            elif line == 'd':
                want_undo = True
            elif line is not None:
                want_grab = True

            if quit_all:
                break
            if want_skip:
                print(f"  {'放弃' if armed else '跳过'} {stage['desc']} (已采 {got})")
                armed = False
                si += 1
                got, stage_corners = 0, []
                if si < len(STAGES):
                    stage = STAGES[si]
                    print_stage(si)
                continue
            if want_undo:
                if got > 0 and img_points:
                    obj_points.pop()
                    img_points.pop()
                    stage_corners.pop()
                    got -= 1
                    print(f"  [撤销] 删除最近一张 -> 本阶段 {got}/{stage['need']} "
                          f"(总 {len(img_points)})")
                else:
                    print("  [撤销] 当前阶段还没有采集")
            if want_grab:
                if not armed:
                    armed = True
                    print(f"  进入采集 —— 每次按键拍一张 (本阶段需 {stage['need']} 张)")
                if not (found and metrics is not None):
                    print("  [采] 未检出棋盘格，把整块板子放进画面再按")
                else:
                    fails = [k for k, ok in checks if not ok and k != 'T']
                    if fails and now > force_until:
                        force_until = now + FORCE_WINDOW
                        msgs = []
                        for k in fails:
                            if k == 'D':
                                lo, hi = stage['size']
                                cm = 100.0 * REF_FX * BOARD_CORNER_DIAG / (
                                    metrics['size'] * np.hypot(gray.shape[1],
                                                               gray.shape[0]))
                                way = '太远, 拿近' if metrics['size'] < lo else '太近, 拿远'
                                msgs.append(f"D: size={metrics['size']:.2f} 需 {lo:.2f}-{hi:.2f}"
                                            f" (~{cm:.0f}cm, {way})")
                            elif k == 'N':
                                msgs.append("N: 与已采姿态太像, 挪动/换个角度")
                            elif k == 'S':
                                msgs.append(f"S: 未停稳 ({stable_count}/{STABLE_NEED}), 停稳再按")
                            elif k == 'Q':
                                msgs.append(f"Q: 质量 {quality['total']:.0f}<{QUALITY_MIN},"
                                            " 可能虚焦/光线不足")
                            elif k == 'R':
                                msgs.append("R: 板子中心不在黄框内")
                        print("  [拦截] " + " | ".join(msgs))
                        print("         姿态确认没问题就 3 秒内再按一次 = 强制拍")
                    else:
                        forced = bool(fails)
                        force_until = 0.0
                        obj_points.append(objp)
                        img_points.append(corners)
                        stage_corners.append(corners.copy())
                        got += 1
                        warn = ""
                        if quality and quality['total'] < QUALITY_MIN:
                            warn = " (q 偏低，可能虚焦/模糊)"
                        print(f"  [采{'·强制' if forced else ''}] {got}/{stage['need']} "
                              f"tilt={metrics['tilt']:.0f}° size={metrics['size']:.2f} "
                              f"q={quality['total']:.0f}{warn}")
                        if got >= stage['need']:
                            armed = False
                            si += 1
                            got, stage_corners = 0, []
                            if si < len(STAGES):
                                stage = STAGES[si]
                                print("  Stage 完成 ✔")
                                print_stage(si)

            if now - last_status >= 1.5:
                last_status = now
                marks = ' '.join(f"{k}{'✓' if ok else '✗'}" for k, ok in checks)
                if armed:
                    note = "" if found else " <- 未检出棋盘格"
                    print(f"    待拍 {got}/{stage['need']} [{marks}] "
                          f"按 c/空格/回车 拍当前帧{note}", flush=True)
                else:
                    print(f"    待开始 {stage['desc']} [{marks}] "
                          f"按 c/空格/回车 拍第 1 张", flush=True)
    except KeyboardInterrupt:
        print("\n[guided] interrupted")
    finally:
        if preview:
            cv2.destroyAllWindows()
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()

    n = len(img_points)
    print("\n" + "=" * 64)
    print(f" 采集结束: 共 {n} 张")
    if n < 12:
        print(f" 不足 12 张 ({n})，无法拟合。重跑补足后再试。")
        return 1

    rms, K, dist = run_calibration(obj_points, img_points, gray.shape)
    out = CONFIG_DIR / "camera_calibration.npz"
    np.savez(str(out), K=K, dist=dist, rms=rms)
    print(f"\nSaved to: {out}")
    if args.no_write:
        print("--no-write: 请手动把上面数值填入 camera.yaml")
    else:
        write_yaml(args.yaml, K, dist)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    raise SystemExit(main())
