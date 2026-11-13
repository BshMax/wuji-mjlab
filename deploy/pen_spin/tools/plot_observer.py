# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Plot pen-observer --diagnostics JSONL: pose curves, jitter vs trained bounds.

The pen-spin policy was trained with an explicit camera-observation model
(pen_spin_env_cfg.py "object_pose" term): per-tick jitter uniform +-2 mm /
+-0.015 rad, constant per-episode bias +-8 mm / +-0.05 rad, 1-tick latency and
dropout bursts (mean ~92 ms, tail ~0.8 s at the 50 Hz control rate). This tool
plots the recorded stream against those bounds and prints a PASS/WARN table.

Run with any python that has numpy + matplotlib (the deploy env does not):

    /usr/bin/python3 deploy/pen_spin/tools/plot_observer.py \
        /tmp/pen-static.jsonl --output /tmp/pen-static.png

    /usr/bin/python3 deploy/pen_spin/tools/plot_observer.py \
        /tmp/pen-move.jsonl --output /tmp/pen-move.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless: write PNG only, no display needed
import matplotlib.pyplot as plt  # noqa: E402

# Trained observation bounds, pen_spin_env_cfg.py object_pose params.
TRAIN_POS_JITTER_MM = 2.0            # uniform +-2 mm per axis, per tick
TRAIN_ROT_JITTER_DEG = float(np.degrees(0.015))  # uniform +-0.015 rad per tick
TRAIN_BIAS_POS_MM = 8.0              # uniform +-8 mm per episode (constant)
TRAIN_BIAS_ROT_DEG = float(np.degrees(0.05))     # uniform +-0.05 rad per episode
TRAIN_DROPOUT_MEAN_MS = 92.0         # ~4.6 control ticks at 50 Hz
TRAIN_DROPOUT_MAX_MS = 800.0         # 40 control ticks at 50 Hz
CONTROL_HZ = 50.0
MAX_REPROJ_PX = 4.0                  # estimator reject threshold


def load(path: Path):
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not records:
        raise SystemExit(f"{path}: empty file")
    if records[0].get("mode") in ("stereo-detection", "detection-only"):
        raise SystemExit(
            f"{path}: '{records[0]['mode']}' records carry no 3D pose; "
            "record with the pose observer (no --detect-only/--stereo)")
    return records


def quat_angle_deg(q, q_ref):
    """Angle between each wxyz quat and a reference, degrees, sign-insensitive."""
    dot = np.abs(np.clip(np.sum(q * q_ref, axis=1), -1.0, 1.0))
    return np.degrees(2.0 * np.arccos(dot))


def dropout_runs(t, valid):
    """Contiguous invalid stretches (seconds): gaps between valid neighbours."""
    runs = []
    start = None
    for ti, ok in zip(t, valid):
        if not ok and start is None:
            start = ti
        elif ok and start is not None:
            runs.append((start, ti))
            start = None
    if start is not None:
        runs.append((start, t[-1]))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("diagnostics", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--static", action="store_true",
                    help="pen held still: jitter panels/histograms vs trained bounds")
    args = ap.parse_args()

    recs = load(args.diagnostics)
    ready = [r for r in recs if r["world_fixed"]]
    if len(ready) < 2:
        raise SystemExit("No world-fixed frames; show the wrist tag first.")

    t = np.asarray([r["t_monotonic_s"] for r in ready])
    t = t - t[0]
    valid = np.asarray([r["position"] is not None for r in ready])
    n_tags = np.asarray([len(r["tag_ids"]) for r in ready])
    reproj = np.asarray([r["reproj_px"] if r["reproj_px"] is not None
                         else np.nan for r in ready])
    proc_ms = np.asarray([r["processing_ms"] for r in ready])
    dur = t[-1] - t[0]

    pos = np.asarray([r["position"] if r["position"] is not None
                      else [np.nan] * 3 for r in ready]) * 1000.0  # mm
    quat = np.asarray([r["quat_wxyz"] if r["quat_wxyz"] is not None
                       else [np.nan] * 4 for r in ready])

    pv = pos[valid]
    qv = quat[valid]
    med_pos = np.median(pv, axis=0) if len(pv) else np.full(3, np.nan)
    med_q = None
    if len(qv) > 1:
        aligned = qv * np.sign(qv @ qv[0])[:, None]
        med_q = aligned.mean(0)
        med_q /= np.linalg.norm(med_q)

    dev_mm = np.linalg.norm(pos - med_pos, axis=1)           # 3D deviation
    dev_ax = pos - med_pos                                    # per-axis
    ang_deg = (quat_angle_deg(quat, med_q) if med_q is not None
               else np.full(len(t), np.nan))
    rng = np.nanmax(pv, axis=0) - np.nanmin(pv, axis=0) if len(pv) else [0] * 3
    stationary = args.static or float(np.max(rng)) < 30.0     # <3 cm range

    gaps = dropout_runs(t, valid)
    gap_ms = np.asarray([(b - a) * 1000.0 for a, b in gaps])
    rate = float(np.count_nonzero(valid) / max(dur, 1e-9))
    frac = float(np.mean(valid))

    # ---------- figure ----------
    fig, axes = plt.subplots(6, 1, figsize=(13, 15), sharex=False,
                             gridspec_kw=dict(hspace=0.5))
    fig.suptitle(f"{args.diagnostics.name}  ({'static' if stationary else 'moving'} "
                 f"interpretation)  {dur:.1f}s  valid {rate:.1f} Hz ({frac:.0%})")

    ax = axes[0]
    for i, c in enumerate("xyz"):
        ax.plot(t, pos[:, i], lw=.7, label=f"{c}")
    ax.axhline(med_pos[0], color="C0", lw=.4, alpha=.5)
    ax.set_ylabel("pos (mm)")
    ax.set_title("pen position in wrist-tag frame")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.grid(alpha=.3)

    ax = axes[1]
    ax.plot(t, dev_ax[:, 0], lw=.6, label="dx")
    ax.plot(t, dev_ax[:, 1], lw=.6, label="dy")
    ax.plot(t, dev_ax[:, 2], lw=.6, label="dz")
    ax.axhline(TRAIN_POS_JITTER_MM, color="r", ls="--", lw=1,
               label="trained jitter +-2mm")
    ax.axhline(-TRAIN_POS_JITTER_MM, color="r", ls="--", lw=1)
    if stationary:
        ax.set_ylim(-max(3, np.nanmax(np.abs(dev_ax)) * 1.2),
                    max(3, np.nanmax(np.abs(dev_ax)) * 1.2))
    ax.set_ylabel("dev (mm)")
    ax.set_title("per-axis deviation from run median" +
                 ("" if stationary else "  (pen moved — band is for reference)"))
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.grid(alpha=.3)

    ax = axes[2]
    ax.plot(t, ang_deg, lw=.7, color="C3")
    ax.axhline(TRAIN_ROT_JITTER_DEG, color="r", ls="--", lw=1,
               label=f"trained jitter +-{TRAIN_ROT_JITTER_DEG:.2f} deg")
    if stationary:
        ax.set_ylim(0, max(1.5, np.nanmax(ang_deg) * 1.2))
    ax.set_ylabel("angle dev (deg)")
    ax.set_title("orientation deviation from run median quat" +
                 ("" if stationary else "  (pen moved — band is for reference)"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=.3)

    ax = axes[3]
    ax.plot(t, reproj, lw=.7, color="C1", label="reproj px")
    ax.axhline(MAX_REPROJ_PX, color="r", ls="--", lw=1, label="reject 4px")
    ax.set_ylabel("px")
    ax2 = ax.twinx()
    ax2.step(t, n_tags, where="post", lw=.6, color="0.6", alpha=.7)
    ax2.set_ylabel("tags", color="0.4")
    ax.set_title("reprojection error / detected tag count")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=.3)

    ax = axes[4]
    ax.fill_between(t, 0, valid.astype(float), step="post", color="C2",
                    alpha=.7, label="valid pose")
    for a, b in gaps:
        ax.axvspan(a, b, color="r", alpha=.35)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("valid")
    ax3 = ax.twinx()
    ax3.plot(t, proc_ms, lw=.5, color="teal", alpha=.8)
    ax3.set_ylabel("proc ms", color="teal")
    ax.set_title(f"pose validity — {len(gaps)} dropout runs "
                 f"(trained model: mean ~{TRAIN_DROPOUT_MEAN_MS:.0f}ms, "
                 f"tail {TRAIN_DROPOUT_MAX_MS:.0f}ms); teal = per-frame proc ms")
    ax.set_xlabel("time (s)")
    ax.grid(alpha=.3)

    ax = axes[5]
    if stationary and len(pv) > 10:
        bins = 40
        ax.hist(dev_mm[valid], bins=bins, alpha=.6, color="C0",
                label="|pos dev| mm")
        ax.hist(ang_deg[valid], bins=bins, alpha=.6, color="C3",
                label="angle dev deg")
        ax.axvline(TRAIN_POS_JITTER_MM, color="C0", ls="--", lw=1)
        ax.axvline(TRAIN_ROT_JITTER_DEG, color="C3", ls="--", lw=1)
        ax.set_title("jitter histogram (dashed = trained uniform bound)")
        ax.legend(fontsize=8)
    elif len(gap_ms):
        ax.hist(gap_ms, bins=30, alpha=.7, color="C4")
        ax.axvline(TRAIN_DROPOUT_MEAN_MS, color="r", ls="--", lw=1,
                   label=f"trained mean {TRAIN_DROPOUT_MEAN_MS:.0f}ms")
        ax.axvline(TRAIN_DROPOUT_MAX_MS, color="r", ls=":", lw=1,
                   label=f"trained tail {TRAIN_DROPOUT_MAX_MS:.0f}ms")
        ax.set_title("dropout run durations (ms)")
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "no dropouts", ha="center", transform=ax.transAxes)
        ax.set_title("dropout run durations")
    ax.grid(alpha=.3)

    fig.savefig(args.output, dpi=110)
    print(f"wrote {args.output}")

    # ---------- verdict table ----------
    def pct(x, p):
        x = np.asarray(x)[np.isfinite(x)]
        return float(np.percentile(x, p)) if len(x) else float("nan")

    print(f"\nframes={len(ready)}  dur={dur:.1f}s  valid={np.count_nonzero(valid)}"
          f" ({frac:.1%})  rate={rate:.1f} Hz")
    print(f"processing ms: p50={np.median(proc_ms):.2f} p95={pct(proc_ms, 95):.2f}")
    print(f"reproj px: median={np.nanmedian(reproj):.2f} "
          f"p95={pct(reproj, 95):.2f}  (reject at {MAX_REPROJ_PX:g}px)")

    rows = []

    def verdict(name, measured, bound, unit, better_low=True):
        ok = measured <= bound if better_low else measured >= bound
        rows.append((name, f"{measured:.3g} {unit}", f"{bound:.3g} {unit}",
                     "PASS" if ok else "WARN"))

    if len(pv) > 10:
        sig_ax = np.nanstd(dev_ax[valid], axis=0)
        sig_ang = float(np.nanstd(ang_deg[valid]))
        print(f"\nstatic jitter: pos sigma/axis={sig_ax.round(3)} mm, "
              f"|dev| p95={pct(dev_mm[valid], 95):.2f} mm; "
              f"angle sigma={sig_ang:.3f} deg, p95={pct(ang_deg[valid], 95):.2f} deg")
        if stationary:
            verdict("pos jitter p95 (per-axis)", pct(np.abs(dev_ax[valid]), 95),
                    TRAIN_POS_JITTER_MM, "mm")
            verdict("rot jitter p95", pct(ang_deg[valid], 95),
                    TRAIN_ROT_JITTER_DEG, "deg")
            bias_mm = float(np.linalg.norm(med_pos - np.median(pv, 0)))
            print(f"run bias check: median pos={med_pos.round(1)} mm "
                  f"(compare against a ruler; trained bias bound {TRAIN_BIAS_POS_MM:g} mm)")
    else:
        print("\ntoo few valid poses for jitter stats")

    verdict("valid pose rate", rate, CONTROL_HZ, "Hz", better_low=False)
    if len(gap_ms):
        print(f"\ndropouts: {len(gap_ms)} runs, p50={np.median(gap_ms):.0f}ms "
              f"p95={pct(gap_ms, 95):.0f}ms max={gap_ms.max():.0f}ms "
              f"(trained: mean ~{TRAIN_DROPOUT_MEAN_MS:.0f}ms, "
              f"tail {TRAIN_DROPOUT_MAX_MS:.0f}ms)")
        verdict("longest dropout", float(gap_ms.max()), TRAIN_DROPOUT_MAX_MS, "ms")
    else:
        print("\ndropouts: none")

    print("\n=== alignment vs trained observation model ===")
    for name, got, bound, res in rows:
        print(f"  {res:4s}  {name:28s} measured={got:>12s}  bound={bound}")


if __name__ == "__main__":
    main()
