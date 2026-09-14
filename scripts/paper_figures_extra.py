#!/usr/bin/env python3
"""Extra diagnostic figures for the door-localisation evaluation.

Focus: two hypotheses about *what the model has learned*.

  (Q1) Does the model reason through neighbouring-wall context (embeddings)?
       -> error rises monotonically as neighbour walls are removed, and the
          left/right-absent asymmetry at *equal training frequency* argues the
          effect is geometric, not data-scarcity.

  (Q2) Has the model learned the coordinate frame / rotation?
       -> the 9 augmentations are pure rotations about the origin (constant
          radius). We check error vs rotation angle, the fidelity of the
          predicted door-offset *direction*, and whether predictions stay on
          the true ring.

Reuses loading + style from paper_figures.py. Run after paper_figures.py:

    python scripts/paper_figures_extra.py                       # newest run
    python scripts/paper_figures_extra.py data/runs/2026JUL122211
"""
from __future__ import annotations
# --- repo-root bootstrap (added when this entry point moved into scripts/) ---
# Every data path in this file is relative to the repository root, so run it from there:
#     cd ~/Projects/SDF_3 && python scripts/paper_figures_extra.py
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# --- end bootstrap ---


import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from paper_figures import load, OI, INK, MUTED, _finish, _mean_ci


def derive(d):
    """Reconstruct offset geometry and polar coordinates from the report."""
    # Residual-anchored head: pred = wall_center + pred_offset.
    wcx = d["px"] - d["ox"]
    wcy = d["py"] - d["oy"]
    tox = d["tx"] - wcx           # true offset (wall centre -> true door)
    toy = d["ty"] - wcy
    ang_pred = np.arctan2(d["oy"], d["ox"])
    ang_true = np.arctan2(toy, tox)
    ang_err = np.abs((np.degrees(ang_pred - ang_true) + 180) % 360 - 180)
    theta_true = np.degrees(np.arctan2(d["ty"], d["tx"]))   # global door bearing
    r_true = np.hypot(d["tx"], d["ty"])
    r_pred = np.hypot(d["px"], d["py"])
    n_walls = (d["L"] != -1).astype(int) + (d["R"] != -1).astype(int)
    return dict(ang_pred=ang_pred, ang_true=ang_true, ang_err=ang_err,
                theta_true=theta_true, r_true=r_true, r_pred=r_pred, n_walls=n_walls)


# --------------------------------------------------------------------------- #
# E1 — rotation coverage: error around the full 360 deg of door bearings       #
# --------------------------------------------------------------------------- #
def figE1_rotation(d, g, base):
    fig = plt.figure(figsize=(7.2, 3.4))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    th = np.radians(g["theta_true"])
    sc = ax1.scatter(th, g["r_true"], c=d["err"], s=8, cmap="viridis",
                     alpha=0.7, edgecolors="none", vmax=np.percentile(d["err"], 99))
    ax1.set_title("(a) True door bearing\n(colour = error)", pad=14)
    ax1.set_rlabel_position(100)
    cb = fig.colorbar(sc, ax=ax1, shrink=0.7, pad=0.10)
    cb.set_label("Euclidean error (m)", fontsize=8)

    # mean error binned by bearing
    ax2 = fig.add_subplot(1, 2, 2)
    bins = np.linspace(-180, 180, 25)
    idx = np.digitize(g["theta_true"], bins)
    xs, ys, es = [], [], []
    for b in range(1, len(bins)):
        sel = d["err"][idx == b]
        if len(sel) >= 5:
            m, ci = _mean_ci(sel)
            xs.append((bins[b - 1] + bins[b]) / 2); ys.append(m); es.append(ci)
    ax2.errorbar(xs, ys, yerr=es, fmt="o-", color=OI["blue"], ms=4, lw=1.3,
                 capsize=2, ecolor=MUTED)
    ax2.axhline(d["err"].mean(), color=OI["vermillion"], ls="--", lw=1.0,
                label=f"overall mean = {d['err'].mean():.3f} m")
    ax2.set_xlabel("True door bearing (deg)")
    ax2.set_ylabel("Mean Euclidean error (m)")
    ax2.set_title("(b) Error vs. rotation angle")
    ax2.set_xlim(-180, 180)
    ax2.set_xticks(range(-180, 181, 90))
    ax2.legend(loc="upper right")
    _finish(fig, base)


# --------------------------------------------------------------------------- #
# E2 — offset-direction fidelity: has the frame been learned?                  #
# --------------------------------------------------------------------------- #
def figE2_offset_fidelity(d, g, base):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.3))
    ax1.plot([-180, 180], [-180, 180], color=MUTED, ls="--", lw=1.0)
    ax1.scatter(np.degrees(g["ang_true"]), np.degrees(g["ang_pred"]),
                s=6, color=OI["blue"], alpha=0.25, edgecolors="none")
    ax1.set_xlabel("True offset direction (deg)")
    ax1.set_ylabel("Predicted offset direction (deg)")
    ax1.set_title("(a) Door-offset bearing")
    ax1.set_aspect("equal", "box")
    ax1.set_xlim(-180, 180); ax1.set_ylim(-180, 180)
    ax1.set_xticks(range(-180, 181, 90)); ax1.set_yticks(range(-180, 181, 90))

    ae = g["ang_err"]
    ax2.hist(ae, bins=np.linspace(0, np.percentile(ae, 99.5), 50),
             color=OI["green"], alpha=0.85, edgecolor="white", linewidth=0.3)
    med, p90 = np.median(ae), np.percentile(ae, 90)
    ax2.axvline(med, color=OI["vermillion"], lw=1.5, label=f"median = {med:.2f}$\\degree$")
    ax2.axvline(p90, color=OI["orange"], lw=1.5, ls="--", label=f"p90 = {p90:.2f}$\\degree$")
    ax2.set_xlabel("Offset-direction error (deg)")
    ax2.set_ylabel("Test doors")
    ax2.set_title("(b) Direction error " f"({100*(ae<5).mean():.0f}% < 5$\\degree$)")
    ax2.legend(loc="upper right")
    _finish(fig, base)


# --------------------------------------------------------------------------- #
# E3 — coordinate consistency: do predictions stay on the true ring?          #
# --------------------------------------------------------------------------- #
def figE3_ring(d, g, base):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4))
    # (a) predicted points vs the true ring
    order = np.argsort(d["err"])
    ax1.scatter(d["px"][order], d["py"][order], c=d["err"][order], s=7,
                cmap="viridis", alpha=0.6, edgecolors="none",
                vmax=np.percentile(d["err"], 99))
    ring = plt.Circle((0, 0), np.median(g["r_true"]), fill=False,
                      color=OI["vermillion"], lw=1.4, ls="--",
                      label=f"true ring r={np.median(g['r_true']):.2f} m")
    ax1.add_patch(ring)
    ax1.set_aspect("equal", "box")
    ax1.set_xlabel("Predicted x (m)"); ax1.set_ylabel("Predicted y (m)")
    ax1.set_title("(a) Predictions vs. true ring")
    ax1.legend(loc="upper right", fontsize=7)

    # (b) radial error distribution (dist-from-origin)
    dr = g["r_pred"] - g["r_true"]
    ax2.hist(dr, bins=60, color=OI["sky"], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color=INK, lw=1.0)
    ax2.axvline(dr.mean(), color=OI["vermillion"], lw=1.5,
                label=f"mean = {dr.mean():+.3f} m")
    ax2.set_xlabel(r"Radial error $r_{pred}-r_{true}$ (m)")
    ax2.set_ylabel("Test doors")
    ax2.set_title(f"(b) Radius preservation (MAE={np.mean(np.abs(dr)):.3f} m)")
    ax2.legend(loc="upper right")
    _finish(fig, base)


# --------------------------------------------------------------------------- #
# E4 — wall-context evidence: #walls present, and L/R asymmetry               #
# --------------------------------------------------------------------------- #
def figE4_wall_context(d, g, base):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    # (a) error vs number of neighbour walls present
    labels, means, cis, ns = [], [], [], []
    for k in (2, 1, 0):
        sel = d["err"][g["n_walls"] == k]
        if len(sel):
            m, ci = _mean_ci(sel)
            labels.append(f"{k} wall{'s' if k != 1 else ''}")
            means.append(m); cis.append(ci); ns.append(len(sel))
    xp = np.arange(len(labels))
    ax1.bar(xp, means, yerr=cis, capsize=3, width=0.6,
            color=[OI["blue"], OI["orange"], OI["vermillion"]][:len(labels)],
            edgecolor="white", linewidth=0.5)
    for x, m, nn in zip(xp, means, ns):
        ax1.text(x, m + 0.006, f"n={nn}", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax1.set_xticks(xp); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Mean Euclidean error (m)")
    ax1.set_title("(a) Error vs. #neighbour walls")
    ax1.grid(axis="x", visible=False); ax1.set_axisbelow(True)

    # (b) L-absent vs R-absent at equal training frequency
    la = (d["L"] == -1) & (d["R"] != -1)
    ra = (d["L"] != -1) & (d["R"] == -1)
    groups = [("Left\nabsent", la, OI["orange"]), ("Right\nabsent", ra, OI["vermillion"])]
    xp = np.arange(len(groups))
    for i, (lab, sel, col) in enumerate(groups):
        m, ci = _mean_ci(d["err"][sel])
        ax2.bar(i, m, yerr=ci, capsize=3, width=0.55, color=col,
                edgecolor="white", linewidth=0.5)
        ax2.text(i, m + 0.006, f"{m:.3f}\nn={sel.sum()}", ha="center", va="bottom",
                 fontsize=7, color=MUTED)
    ax2.set_xticks(xp); ax2.set_xticklabels([g_[0] for g_ in groups])
    ax2.set_ylabel("Mean Euclidean error (m)")
    ax2.set_title("(b) Same frequency, different error")
    ax2.grid(axis="x", visible=False); ax2.set_axisbelow(True)
    _finish(fig, base)


# --------------------------------------------------------------------------- #
# E5 — difficulty factors: per-door and offset magnitude                       #
# --------------------------------------------------------------------------- #
def figE5_difficulty(d, g, base):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    doors = sorted(set(d["door"]))
    parts = ax1.violinplot([d["err"][d["door"] == dd] for dd in doors],
                           showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], [OI["blue"], OI["orange"], OI["green"], OI["purple"]]):
        pc.set_facecolor(col); pc.set_alpha(0.7); pc.set_edgecolor(INK)
    parts["cmedians"].set_color(INK)
    ax1.set_xticks(range(1, len(doors) + 1))
    ax1.set_xticklabels([dd[:6] for dd in doors], rotation=0)
    ax1.set_ylabel("Euclidean error (m)")
    ax1.set_title("(a) Error by door instance")
    ax1.grid(axis="x", visible=False)

    # error vs offset magnitude (distance door sits from its wall centre)
    ax2.scatter(d["offmag"], d["err"], s=5, color=OI["blue"], alpha=0.15, edgecolors="none")
    bins = np.linspace(d["offmag"].min(), d["offmag"].max(), 16)
    idx = np.digitize(d["offmag"], bins)
    bx, by = [], []
    for b in range(1, len(bins)):
        sel = d["err"][idx == b]
        if len(sel) >= 10:
            bx.append((bins[b - 1] + bins[b]) / 2); by.append(sel.mean())
    ax2.plot(bx, by, "-", color=OI["vermillion"], lw=2, label="binned mean")
    ax2.set_xlabel("Predicted offset magnitude (m)")
    ax2.set_ylabel("Euclidean error (m)")
    ax2.set_title("(b) Error vs. door offset")
    ax2.legend(loc="upper left")
    _finish(fig, base)


def main():
    if len(sys.argv) > 1:
        run_dir = sys.argv[1].rstrip("/")
    else:
        runs = sorted(glob.glob("data/runs/*/output/evaluation_results/test_evaluation_report.json"))
        if not runs:
            sys.exit("No evaluation reports found under data/runs/*/output/")
        run_dir = os.path.dirname(os.path.dirname(os.path.dirname(runs[-1])))
    which = sys.argv[2] if len(sys.argv) > 2 else "test"
    report_name = "holdout_evaluation_report.json" if which == "holdout" else "test_evaluation_report.json"
    print(f"Run: {run_dir}  |  report: {report_name}")

    _, d = load(run_dir, report_name)
    g = derive(d)
    sub = "paper_figures_holdout" if which == "holdout" else "paper_figures"
    out_dir = os.path.join(run_dir, "output", sub)
    os.makedirs(out_dir, exist_ok=True)

    figE1_rotation(d, g, os.path.join(out_dir, "figE1_rotation_coverage"))
    figE2_offset_fidelity(d, g, os.path.join(out_dir, "figE2_offset_fidelity"))
    figE3_ring(d, g, os.path.join(out_dir, "figE3_ring_consistency"))
    figE4_wall_context(d, g, os.path.join(out_dir, "figE4_wall_context"))
    figE5_difficulty(d, g, os.path.join(out_dir, "figE5_difficulty"))

    # console summary of the two hypotheses
    print("\n=== Q1  wall-context (embedding) evidence ===")
    for k in (2, 1, 0):
        sel = d["err"][g["n_walls"] == k]
        print(f"  {k} neighbour wall(s): mean {sel.mean():.4f} m  (n={len(sel)})")
    la = d["err"][(d["L"] == -1) & (d["R"] != -1)]
    ra = d["err"][(d["L"] != -1) & (d["R"] == -1)]
    print(f"  L-absent {la.mean():.4f} m (n={len(la)}) vs R-absent {ra.mean():.4f} m (n={len(ra)})"
          f"  -> {ra.mean()/la.mean():.2f}x at equal frequency")
    print("\n=== Q2  coordinate-frame / rotation evidence ===")
    print(f"  offset-direction error: median {np.median(g['ang_err']):.2f} deg, "
          f"{100*(g['ang_err']<5).mean():.1f}% < 5 deg")
    print(f"  radius preservation MAE: {np.mean(np.abs(g['r_pred']-g['r_true'])):.4f} m "
          f"(true ring r={np.median(g['r_true']):.2f} m)")
    print(f"  bearings covered: [{g['theta_true'].min():.0f}, {g['theta_true'].max():.0f}] deg")
    print(f"\nAll extra figures written to: {out_dir}")


if __name__ == "__main__":
    main()
