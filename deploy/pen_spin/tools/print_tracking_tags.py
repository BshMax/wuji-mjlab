# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Print actual-size wrist markers and six complete source-pen sticker sets.

Run with reportlab, Pillow, numpy and opencv-contrib-python installed. No camera
or robot connection is made. Artwork is copied from the verified source textures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

REPO = Path(__file__).resolve().parents[3]
LAYOUT = REPO / 'deploy/pen_spin/config/pen_tags.json'
TEXTURES = REPO / 'src/wuji_mjlab/assets/objects/aruco_pen250/textures'
MM = 72 / 25.4
PAGE_H = A4[1] / MM


def prepare_faces():
    layout = json.loads(LAYOUT.read_text())
    patterns = defaultdict(list)
    for pattern in layout['custom_dictionary']['patterns']:
        patterns[pattern['texture']].append(pattern)
    faces = []
    for name, items in patterns.items():
        path = TEXTURES / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != layout['source_texture_sha256'][name]:
            raise ValueError(f'Source artwork does not match the tracked layout: {name}')
        width = 18 * (1 + max(p['column'] for p in items))
        height = 18 * (1 + max(p['row'] for p in items))
        # Restore the physical aspect ratio before any printing rotation.
        image = Image.open(path).convert('L').resize(
            (width * 24, height * 24), Image.Resampling.NEAREST)
        tag = layout['tags'][str(items[0]['id'])]
        corners = np.asarray(tag['corners_m'])
        right, up = corners[1] - corners[0], corners[0] - corners[3]
        if width > height:
            image = image.transpose(Image.Transpose.ROTATE_270)
            width, height = height, width
            right, up = up, -right
        # All printed side stickers have their page-top edge toward A_top (+Z).
        if height == 36 and up[2] < 0:
            image = image.transpose(Image.Transpose.ROTATE_180)
            right, up = -right, -up
        faces.append(dict(name=name, image=image, width=width, height=height,
                          ids=[p['id'] for p in items], head=tag['box'],
                          normal=tag['face_normal'], right=right, up=up))
    return faces


def axis(vector):
    index = int(np.argmax(np.abs(vector)))
    return ('+' if vector[index] > 0 else '-') + 'XYZ'[index]


def generate(output, font_path):
    pdfmetrics.registerFont(TTFont('CN', str(font_path)))
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
    pdf.setTitle('Wuji Hand - A4 wrist AprilTag and source pen stickers - actual size')
    pdf.setAuthor('Wuji tracking print utility')

    def text(x, y, value, size=8):
        pdf.setFillColorRGB(0, 0, 0)
        pdf.setFont('CN', size)
        pdf.drawString(x * MM, (PAGE_H - y) * MM, value)

    def image(im, x, y, w, h):
        pdf.drawImage(ImageReader(im), x * MM, (PAGE_H - y - h) * MM,
                      width=w * MM, height=h * MM, preserveAspectRatio=False)

    def cut(x, y, w, h):
        # Keep the line outside the physical sticker edge and outside all code cells.
        pdf.setStrokeColorRGB(.55, .55, .55)
        pdf.setLineWidth(.2)
        pdf.setDash(1.2, 1.2)
        pdf.rect(x * MM, (PAGE_H - y - h) * MM, w * MM, h * MM)
        pdf.setDash()

    def ruler(y):
        pdf.setStrokeColorRGB(0, 0, 0)
        pdf.setLineWidth(.5)
        pdf.line(10 * MM, (PAGE_H - y) * MM, 110 * MM, (PAGE_H - y) * MM)
        for x in (10, 110):
            pdf.line(x * MM, (PAGE_H - y - 1) * MM, x * MM, (PAGE_H - y + 1) * MM)
        text(114, y + 1, '校验线 100 mm', 7)

    text(10, 14, '腕部定位 | AprilTag 36h11 / ID 0', 15)
    text(10, 21, '黑色方框边长 50.4 mm；白边不计入。打印选“实际大小 / 100%”，关闭适合页面。', 8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    # Vector cells: no raster interpolation can blur the wrist tag in the PDF.
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 8)
    for row in range(4):
        for col in range(3):
            x, y = 11 + col * 63.5, 25 + row * 63
            cut(x, y, 60.4, 60.4)
            for r in range(8):
                for c in range(8):
                    if marker[r, c] == 0:
                        pdf.setFillColorRGB(0, 0, 0)
                        pdf.rect((x + 5 + c * 6.3) * MM,
                                 (PAGE_H - y - 5 - (r + 1) * 6.3) * MM,
                                 6.3 * MM, 6.3 * MM, stroke=0, fill=1)
            text(x, y + 62.7, '36h11 ID 0 | 黑框 50.4 mm', 6.5)
    ruler(281)
    text(10, 287, '仅安装 1 个；其余剪下收好，不要让多个 ID 0 备用标签同时入镜。', 7)
    pdf.showPage()

    text(10, 14, '笔上贴纸 | SOURCE_1X2_18 | 6 套备用', 15)
    text(10, 21, '每套 10 张面贴 / 18 个码。实线图案按外缘裁切；虚线在外侧，仅作裁切参考。', 8)
    text(10, 26, '侧贴的页面上边统一朝 +Z（A_top / ID 17 端）；端面另按标注的 up / right 方向贴。', 7)
    faces = prepare_faces()
    sides = [face for face in faces if face['height'] == 36]
    caps = [face for face in faces if face['height'] == 18]
    for copy in range(6):
        left, top = 11 + (copy % 3) * 63.5, 31 + (copy // 3) * 122
        text(left, top, f'第 {copy + 1} 套 | Top=17 / Bottom=8', 7)
        for index, face in enumerate(sides):
            x, y = left + (index % 3) * 20, top + 3 + (index // 3) * 39
            image(face['image'], x, y, 18, 36)
            cut(x - .2, y - .2, 18.4, 36.4)
            head = 'T' if face['head'] == 'A_top' else 'B'
            text(x, y + 38, f"{head} {face['normal']} {','.join(map(str, face['ids']))}", 5.3)
        for index, face in enumerate(caps):
            x, y = left + 40, top + 3 + 78 + index * 20
            image(face['image'], x, y, 18, 18)
            cut(x - .2, y - .2, 18.4, 18.4)
            text(x, y + 19.5, f"{face['ids'][0]} up{axis(face['up'])} rt{axis(face['right'])}", 4.8)
    text(10, 277, 'T=A_top：侧面 9-16，端面 17；B=A_bottom：侧面 0-7，端面 8。', 7)
    ruler(282)
    text(10, 287, '侧贴 18×36 mm；端贴 18×18 mm。先量校验线，再粘贴；建议哑光不干胶。', 7)
    pdf.save()
    print(f'Created {output}: 2 A4 pages, 12 wrist tags, 6 complete pen sticker sets')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path, default=Path(
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'))
    args = parser.parse_args()
    generate(args.output, args.font)
