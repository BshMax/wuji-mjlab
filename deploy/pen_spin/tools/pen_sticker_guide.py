# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""One-page A4 guide showing which sticker goes on each pen-head face.

Draws an unfolded net for each of the two pen heads (A_bottom / A_top) with the
actual sticker thumbnails in place, +Z up-edge arrows for the side stickers and
rotated end-cap thumbnails with face-adjacency labels. Companion to
print_tracking_tags.py — the sticker artwork itself comes from that PDF.

Run with reportlab, Pillow, numpy and opencv-contrib-python installed:

    python deploy/pen_spin/tools/pen_sticker_guide.py \
        --output /tmp/pen_sticker_guide_A4.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import PIL.Image as _I

# Pillow < 9.1 compatibility shims (prepare_faces uses the new enum names).
if not hasattr(_I, 'Resampling'):
    class _R: NEAREST = _I.NEAREST; BILINEAR = _I.BILINEAR
    _I.Resampling = _R
if not hasattr(_I, 'Transpose'):
    class _T: ROTATE_90 = _I.ROTATE_90; ROTATE_180 = _I.ROTATE_180; ROTATE_270 = _I.ROTATE_270
    _I.Transpose = _T

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from print_tracking_tags import prepare_faces, axis  # noqa: E402

MM = 72 / 25.4
PAGE_W, PAGE_H = A4[0] / MM, A4[1] / MM
LAYOUT = HERE.parent / 'config' / 'pen_tags.json'


def tag_centers():
    """tag id -> center in pen coords, for ordering codes within a face."""
    import json
    d = json.loads(LAYOUT.read_text())
    return {int(k): np.asarray(v['corners_m']).mean(0)
            for k, v in d['tags'].items()}


def face_key(head, normal):
    return head, normal


def build_face_map():
    centers = tag_centers()
    out = {}
    for f in prepare_faces():
        # Order ids top-to-bottom within the sticker using the face's up axis:
        # the tag projecting farther along `up` sits in the upper image half.
        # (+Y faces are inverted relative to numeric order — do not assume.)
        if len(f['ids']) == 2:
            f['ids'] = sorted(
                f['ids'], key=lambda i: -float(np.dot(centers[i], f['up'])))
        out[face_key(f['head'], f['normal'])] = f
    return out


def generate(output, font_path):
    faces = build_face_map()
    pdfmetrics.registerFont(TTFont('CN', str(font_path)))
    pdf = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
    pdf.setTitle('Wuji Hand - pen sticker placement guide')
    pdf.setAuthor('Wuji tracking print utility')

    def text(x, y, value, size=8, gray=0.0, center=False):
        pdf.setFillColorRGB(gray, gray, gray)
        pdf.setFont('CN', size)
        if center:
            pdf.drawCentredString(x * MM, (PAGE_H - y) * MM, value)
        else:
            pdf.drawString(x * MM, (PAGE_H - y) * MM, value)

    def image(im, x, y, w, h, rot=0):
        if rot:
            im = im.transpose(rot)
        pdf.drawImage(ImageReader(im), x * MM, (PAGE_H - y - h) * MM,
                      width=w * MM, height=h * MM, preserveAspectRatio=False)

    def dashed_rect(x, y, w, h, gray=.55):
        pdf.setStrokeColorRGB(gray, gray, gray)
        pdf.setLineWidth(.25)
        pdf.setDash(1.2, 1.2)
        pdf.rect(x * MM, (PAGE_H - y - h) * MM, w * MM, h * MM)
        pdf.setDash()

    def solid_rect(x, y, w, h, fill=.92):
        pdf.setFillColorRGB(fill, fill, fill)
        pdf.rect(x * MM, (PAGE_H - y - h) * MM, w * MM, h * MM, stroke=0, fill=1)

    def up_arrow(x, y0, y1, label='+Z'):
        pdf.setStrokeColorRGB(0.1, 0.35, 0.9)
        pdf.setLineWidth(.7)
        pdf.line(x * MM, (PAGE_H - y0) * MM, x * MM, (PAGE_H - y1) * MM)
        ax, ay = x * MM, (PAGE_H - y1) * MM
        pdf.setFillColorRGB(0.1, 0.35, 0.9)
        p = pdf.beginPath()
        p.moveTo(ax, ay + 2.2 * MM)
        p.lineTo(ax - 1.4 * MM, ay - .6 * MM)
        p.lineTo(ax + 1.4 * MM, ay - .6 * MM)
        p.close()
        pdf.drawPath(p, stroke=0, fill=1)
        pdf.setFont('CN', 5.5)
        pdf.drawString((x + 1.6) * MM, (PAGE_H - y1 - 3.2) * MM, label)

    CW, SIDE_H, CAP_W, CAP_H = 42.0, 54.0, 40.0, 34.0
    COLS = [('A', '+X'), ('B', '+Y'), ('C', '-X'), ('D', '-Y')]
    ROW_X = (PAGE_W - 4 * CW) / 2          # center the 4-column grid
    CX = PAGE_W / 2                         # grid horizontal center

    def side_cell(x, y_top, head, letter, label):
        f = faces[face_key(head, label)]
        dashed_rect(x, y_top, CW, SIDE_H)
        mark = 'T' if head == 'A_top' else 'B'
        text(x + 1.5, y_top + 4.4, f"{mark} {label} 面", 7)
        # The 18x36 sticker holds two 18x18 codes; upper half = ids[0].
        im, iw, ih = f['image'], f['image'].size[0], f['image'].size[1]
        halves = (im.crop((0, 0, iw, ih // 2)), im.crop((0, ih // 2, iw, ih)))
        ts = 20.0
        ix = x + 9.5
        for k, (half, cid) in enumerate(zip(halves, f['ids'])):
            iy = y_top + 6.5 + k * (ts + 2.5)
            image(half, ix, iy, ts, ts)
            text(ix + ts + 1.8, iy + ts / 2 + 1.5, str(cid), 8)
            if k == 0:
                pdf.setStrokeColorRGB(.6, .6, .6)
                pdf.setLineWidth(.2)
                pdf.line((ix - 1) * MM, (PAGE_H - iy - ts - 1.2) * MM,
                         (ix + ts + 8) * MM, (PAGE_H - iy - ts - 1.2) * MM)
        up_arrow(x + 5, y_top + SIDE_H - 4, y_top + 8)

    def cap_cell(cx, y_top, head, rot, edges, label_above):
        """Centered cap cell; edges=(top,right,bottom,left) adjoining face letters."""
        f = faces[face_key(head, '-Z' if head == 'A_bottom' else '+Z')]
        x = cx - CAP_W / 2
        dashed_rect(x, y_top, CAP_W, CAP_H)
        t, r, b, l = edges
        text(cx, y_top + 3.4, f"{t}面", 4.5, gray=.35, center=True)
        text(cx, y_top + CAP_H - .8, f"{b}面", 4.5, gray=.35, center=True)
        text(x + CAP_W - 1.2, y_top + CAP_H / 2 + 1.5, f"{r}面", 4.5, gray=.35)
        text(x + 1.2, y_top + CAP_H / 2 + 1.5, f"{l}面", 4.5, gray=.35)
        image(f['image'], x + (CAP_W - 21) / 2, y_top + (CAP_H - 21) / 2 + 1.5,
              21, 21, rot=rot)
        ly = y_top - 1.6 if label_above else y_top + CAP_H + 4.6
        text(cx, ly, f"{'顶面' if head == 'A_top' else '底面'} 码 {f['ids'][0]}",
             6.5, center=True)

    # ---- header ----
    text(12, 11, '笔贴纸安装图 | SOURCE_1X2_18 | 每支笔用 1 套', 13)
    text(12, 17, 'A/B/C/D = 笔头四个侧面（绕笔一圈）。每列上下两格为同一方向的面，一一对应。'
               '格内两个码按实贴时的上下位置画出（B/C 列大号在上），蓝箭头 = 贴纸顶边朝 +Z。', 6.5)
    text(12, 21.5, '缩略图仅示意朝向与对应关系，实贴用 tracking_tags_A4.pdf 第 2 页原件；'
               '沿图案外缘裁切，虚线和标注文字不属于图案。', 6.5)

    # ---- top cap (above the grid) ----
    cap_cell(CX, 30.5, 'A_top', _I.ROTATE_90, ('A', 'D', 'C', 'B'), label_above=True)

    # ---- column headers ----
    hy = 69
    for i, (letter, lab) in enumerate(COLS):
        text(ROW_X + i * CW + CW / 2, hy, f"{letter} = {lab} 面", 7, center=True)

    # ---- upper head row ----
    sy = hy + 5
    for i, (letter, lab) in enumerate(COLS):
        side_cell(ROW_X + i * CW, sy, 'A_top', letter, lab)

    # ---- shaft gap ----
    gy = sy + SIDE_H + 5
    text(CX, gy, '—— 笔杆连接段（内端面不贴码）——', 6.5, gray=.35, center=True)

    # ---- lower head row ----
    sy2 = gy + 6
    for i, (letter, lab) in enumerate(COLS):
        side_cell(ROW_X + i * CW, sy2, 'A_bottom', letter, lab)

    # ---- bottom cap (below the grid) ----
    cap_cell(CX, sy2 + SIDE_H + 4, 'A_bottom', _I.ROTATE_270,
             ('A', 'B', 'C', 'D'), label_above=True)
    fy = sy2 + SIDE_H + 4 + CAP_H + 8

    # ---- footer checklist ----
    text(12, fy, '粘贴步骤', 9)
    steps = [
        '1. 裁切后先贴 A 面：下笔头贴 "B +X 0,1"，上笔头贴 "T +X 9,10"，两面朝同一方向。',
        '2. 侧贴顶边一律朝 +Z（蓝箭头）：下笔头朝笔杆中部，上笔头朝 17 端。',
        '3. 端贴按图中旋转后贴：顶面 17 与底面 8 的图案顶边都朝 B(+Y) 面一侧。',
        '4. 自检：笔竖直放 17 朝上，0/1 与 9/10 面朝自己 → 右侧 +Y(B)、左侧 -Y(D)、背面 -X(C)。',
        '5. 贴完用 pen-observer --preview 核对：线框贴合实物、笔轴和头尾方向正确。',
    ]
    for i, s in enumerate(steps):
        text(12, fy + 4.8 + i * 4.4, s, 7)

    pdf.save()
    print(f'Created {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', type=Path, default=Path(
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'))
    args = p.parse_args()
    generate(args.output, args.font)
