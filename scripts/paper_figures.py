#!/usr/bin/env python3
"""Publication-quality figures for the Graphormer door-localisation evaluation.

Reads a run's test-evaluation report plus the parametric filenames
(``graph_<room>_aug<k>_L<left>_R<right>_graph.json``) and renders a set of
paper-ready figures (PDF + PNG, 300 dpi) into ``<run>/output/paper_figures/``.

The graph_batch_idx recorded in the report is the index into the *sorted*
``graph_room*.json`` file list (see Graphormer._run_test_evaluation), so every
test door can be tied back to its (room, aug, L, R) parameters.

Usage
-----
    python scripts/paper_figures.py                       # newest run under data/runs
    python scripts/paper_figures.py data/runs/2026JUL122211
"""
from __future__ import annotations
# --- repo-root bootstrap (added when this entry point moved into scripts/) ---
# Every data path in this file is relative to the repository root, so run it from there:
#     cd ~/Projects/SDF_3 && python scripts/paper_figures.py
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# --- end bootstrap ---


import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# --------------------------------------------------------------------------- #
# Style — colourblind-safe, serif, print-ready                                #
# --------------------------------------------------------------------------- #
# Okabe-Ito categorical palette (assigned in fixed order, never cycled).
OI = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#111111",
}
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "STIXGeneral"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.constrained_layout.use": True,
})

# L/R groups exclude '_' so the optional _W{i} window-state token can't be swallowed.
FNAME_RE = re.compile(r"graph_(room\d+)_aug(\d+)_L([A-Za-z0-9]+)_R([A-Za-z0-9]+)(?:_W(\d+))?_graph\.json")


def _lval(s: str) -> int:
    """Wall-segment length in mm; -1 sentinel for an absent neighbour wall."""
    return -1 if s == "Absent" else int(s)


def load(run_dir: str, report_name: str = "test_evaluation_report.json"):
    out = os.path.join(run_dir, "output")
    report = os.path.join(out, "evaluation_results", report_name)
    with open(report, "r", encoding="utf-8") as f:
        rep = json.load(f)

    # Held-out graphs live under output/holdout/; test-split graphs under output/.
    glob_root = os.path.join(out, "holdout") if "holdout" in report_name else out
    files = sorted(glob.glob(os.path.join(glob_root, "graph_room*.json")))
    meta = []
    for f in files:
        m = FNAME_RE.match(os.path.basename(f))
        meta.append(m.groups()[:4] if m else ("?", "?", "Absent", "Absent"))

    rows = rep["per_door_results"]
    n = len(rows)
    d = {
        "err": np.empty(n), "dx": np.empty(n), "dy": np.empty(n),
        "px": np.empty(n), "py": np.empty(n), "tx": np.empty(n), "ty": np.empty(n),
        "ox": np.empty(n), "oy": np.empty(n), "offmag": np.empty(n),
        "room": np.empty(n, dtype=object), "door": np.empty(n, dtype=object),
        "L": np.empty(n, dtype=int), "R": np.empty(n, dtype=int),
    }
    for i, x in enumerate(rows):
        room, _aug, L, R = meta[x["graph_batch_idx"]]
        d["err"][i] = x["euclidean_error"]
        d["dx"][i], d["dy"][i] = x["dx"], x["dy"]
        d["px"][i], d["py"][i] = x["pred"]
        d["tx"][i], d["ty"][i] = x["true"]
        d["ox"][i], d["oy"][i] = x["pred_offset"]
        d["offmag"][i] = x["pred_offset_mag"]
        d["room"][i] = room
        d["door"][i] = x["door_id"]
        d["L"][i], d["R"][i] = _lval(L), _lval(R)
    return rep, d


def _finish(fig, path_base):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_base}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.basename(path_base)}.pdf / .png")


# --------------------------------------------------------------------------- #
# Fig 1 — error distribution: histogram + empirical CDF                        #
# --------------------------------------------------------------------------- #
def fig_error_distribution(d, base):
    err = d["err"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax1.hist(err, bins=60, color=OI["blue"], alpha=0.85, edgecolor="white", linewidth=0.3)
    mean, med = err.mean(), np.median(err)
    ax1.axvline(med, color=OI["vermillion"], lw=1.6, label=f"median = {med:.3f} m")
    ax1.axvline(mean, color=OI["orange"], lw=1.6, ls="--", label=f"mean = {mean:.3f} m")
    ax1.set_xlabel("Euclidean error (m)")
    ax1.set_ylabel("Test doors")
    ax1.set_title("(a) Error distribution")
    ax1.legend(loc="upper right")
    ax1.set_xlim(0, err.max() * 1.02)

    xs = np.sort(err)
    cdf = np.arange(1, len(xs) + 1) / len(xs)
    ax2.plot(xs, cdf, color=OI["blue"], lw=2)
    for p, c in [(50, OI["vermillion"]), (90, OI["green"]), (95, OI["purple"])]:
        v = np.percentile(err, p)
        ax2.plot([v, v, 0], [0, p / 100, p / 100], color=c, lw=1.0, ls=":")
        ax2.scatter([v], [p / 100], color=c, s=18, zorder=5,
                    label=f"p{p} = {v:.3f} m")
    ax2.set_xlabel("Euclidean error (m)")
    ax2.set_ylabel("Cumulative fraction")
    ax2.set_title("(b) Empirical CDF")
    ax2.set_xlim(0, err.max() * 1.02)
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="lower right")
    _finish(fig, base)


# --------------------------------------------------------------------------- #
# Fig 2 — prediction calibration: predicted vs true, per axis                  #
# --------------------------------------------------------------------------- #
def fig_calibration(d, base):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, t, p, name in [(axes[0], d["tx"], d["px"], "x"),
                           (axes[1], d["ty"], d["py"], "y")]:
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, ls="--", zorder=1)
        ax.scatter(t, p, s=6, color=OI["blue"], alpha=0.25, edgecolors="none", zorder=2)
        # R^2 of prediction vs truth
        ss_res = np.sum((p - t) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        mae = np.mean(np.abs(p - t))
        ax.set_title(f"({'a' if name=='x' else 'b'}) {name}-coordinate  "
                     f"($R^2$={r2:.3f}, MAE={mae:.3f} m)")
        ax.set_xlabel(f"True {name} (m)")
        ax.set_ylabel(f"Predicted {name} (m)")
        ax.set_aspect("equal", "box")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    _finish(fig, base)


# --------------------------------------------------------------------------- #
# Fig 3 — residual bias: signed (dx, dy) density                               #
# --------------------------------------------------------------------------- #
def fig_bias(d, base):
    dx, dy = d["dx"], d["dy"]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    lim = np.percentile(np.abs(np.concatenate([dx, dy])), 99.5)
    hb = ax.hexbin(dx, dy, gridsize=45, cmap="viridis", mincnt=1,
                   extent=(-lim, lim, -lim, lim), linewidths=0.0)
    cb = fig.colorbar(hb, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Test doors", fontsize=8)
    ax.axhline(0, color="white", lw=0.8, alpha=0.7)
    ax.axvline(0, color="white", lw=0.8, alpha=0.7)
    mx, my = dx.mean(), dy.mean()
    ax.scatter([mx], [my], marker="+", s=120, color=OI["vermillion"], lw=2,
               zorder=5, label=f"mean = ({mx:+.3f}, {my:+.3f}) m")
    ax.set_xlabel(r"$\Delta x$ = pred $-$ true (m)")
    ax.set_ylabel(r"$\Delta y$ = pred $-$ true (m)")
    ax.set_title("Residual bias check")
    ax.set_aspect("equal", "box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _finish(fig, base)


# --------------------------------------------------------------------------- #
# Fig 4 — spatial error map: true door positions coloured by error            #
# --------------------------------------------------------------------------- #
def fig_spatial(d, base):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    order = np.argsort(d["err"])  # draw large errors on top
    sc = ax.scatter(d["tx"][order], d["ty"][order], c=d["err"][order],
                    s=10, cmap="viridis", alpha=0.8, edgecolors="none",
                    vmax=np.percentile(d["err"], 99))
    cb = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("Euclidean error (m)", fontsize=8)
    ax.set_xlabel("True door x (m)")
    ax.set_ylabel("True door y (m)")
    ax.set_title("Spatial distribution of error")
    ax.set_aspect("equal", "box")
    _finish(fig, base)


# --------------------------------------------------------------------------- #
# Fig 5 — parametric L x R mean-error heatmap (both walls present)            #
# --------------------------------------------------------------------------- #
def fig_LR_heatmap(d, base):
    lengths = np.arange(100, 3001, 100)
    idx = {v: i for i, v in enumerate(lengths)}
    grid_sum = np.zeros((len(lengths), len(lengths)))
    grid_cnt = np.zeros((len(lengths), len(lengths)))
    for L, R, e in zip(d["L"], d["R"], d["err"]):
        if L in idx and R in idx:
            grid_sum[idx[L], idx[R]] += e
            grid_cnt[idx[L], idx[R]] += 1
    with np.errstate(invalid="ignore"):
        grid = np.where(grid_cnt > 0, grid_sum / np.maximum(grid_cnt, 1), np.nan)

    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(grid, origin="lower", cmap=cmap, aspect="equal",
                   extent=(50, 3050, 50, 3050),
                   vmin=np.nanpercentile(grid, 2), vmax=np.nanpercentile(grid, 98))
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("Mean Euclidean error (m)", fontsize=8)
    ax.set_xlabel("Right wall length R (mm)")
    ax.set_ylabel("Left wall length L (mm)")
    ax.set_title("Error vs. neighbouring wall lengths")
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_major_locator(MultipleLocator(500))
    ax.grid(False)
    _finish(fig, base)


# --------------------------------------------------------------------------- #
# Fig 6 — error by categorical condition: wall-absence + room                 #
# --------------------------------------------------------------------------- #
def _mean_ci(vals):
    vals = np.asarray(vals)
    m = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return m, 1.96 * se


def fig_conditions(d, base):
    # wall-absence condition
    cond = np.empty(len(d["err"]), dtype=object)
    la, ra = d["L"] == -1, d["R"] == -1
    cond[~la & ~ra] = "Both\npresent"
    cond[la & ~ra] = "Left\nabsent"
    cond[~la & ra] = "Right\nabsent"
    cond[la & ra] = "Both\nabsent"
    cond_order = ["Both\npresent", "Left\nabsent", "Right\nabsent", "Both\nabsent"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    labels, means, cis, ns = [], [], [], []
    for c in cond_order:
        sel = d["err"][cond == c]
        if len(sel) == 0:
            continue
        m, ci = _mean_ci(sel)
        labels.append(c); means.append(m); cis.append(ci); ns.append(len(sel))
    xpos = np.arange(len(labels))
    bars = ax1.bar(xpos, means, yerr=cis, capsize=3, width=0.62,
                   color=[OI["blue"], OI["orange"], OI["vermillion"], OI["purple"]][:len(labels)],
                   edgecolor="white", linewidth=0.5)
    for x, m, nn in zip(xpos, means, ns):
        ax1.text(x, m + 0.006, f"n={nn}", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Mean Euclidean error (m)")
    ax1.set_title("(a) By wall-absence condition")
    ax1.set_axisbelow(True)
    ax1.grid(axis="x", visible=False)

    # room boxplot
    rooms = sorted(set(d["room"]))
    data = [d["err"][d["room"] == r] for r in rooms]
    bp = ax2.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                     medianprops=dict(color=INK, lw=1.2))
    for patch in bp["boxes"]:
        patch.set_facecolor(OI["sky"]); patch.set_alpha(0.75); patch.set_edgecolor(INK)
    ax2.set_xticklabels(rooms)
    ax2.set_ylabel("Euclidean error (m)")
    ax2.set_title("(b) By room type")
    ax2.set_axisbelow(True)
    ax2.grid(axis="x", visible=False)
    _finish(fig, base)


# --------------------------------------------------------------------------- #
# Summary table (text) + worst-case listing                                   #
# --------------------------------------------------------------------------- #
def write_summary(rep, d, out_dir):
    gm = rep["global_metrics"]
    lines = ["=== Global metrics ==="]
    for k in ("total_doors_evaluated", "mean_euclidean_error", "median_euclidean_error",
              "rmse_euclidean", "std_euclidean_error", "max_euclidean_error",
              "mean_signed_dx", "mean_signed_dy"):
        lines.append(f"  {k:24s}: {gm[k]:.4f}")
    for thr in (0.2, 0.3, 0.4, 0.5):
        c = int((d["err"] > thr).sum())
        lines.append(f"  frac > {thr:.1f} m           : {c} ({100*c/len(d['err']):.2f}%)")

    # worst cases
    order = np.argsort(-d["err"])[:12]
    lines += ["", "=== Worst 12 test doors ==="]
    lines.append(f"  {'err(m)':>7}  {'door':<9} {'L':>5} {'R':>5}  pred            true")
    for i in order:
        L = "Abs" if d["L"][i] == -1 else int(d["L"][i])
        R = "Abs" if d["R"][i] == -1 else int(d["R"][i])
        lines.append(f"  {d['err'][i]:7.3f}  {str(d['door'][i])[:8]:<9} {str(L):>5} {str(R):>5}  "
                     f"({d['px'][i]:6.2f},{d['py'][i]:6.2f})  ({d['tx'][i]:6.2f},{d['ty'][i]:6.2f})")
    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


def main():
    # Usage: paper_figures.py [run_dir] [test|holdout]
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

    rep, d = load(run_dir, report_name)
    sub = "paper_figures_holdout" if which == "holdout" else "paper_figures"
    out_dir = os.path.join(run_dir, "output", sub)
    os.makedirs(out_dir, exist_ok=True)

    fig_error_distribution(d, os.path.join(out_dir, "fig1_error_distribution"))
    fig_calibration(d, os.path.join(out_dir, "fig2_calibration"))
    fig_bias(d, os.path.join(out_dir, "fig3_residual_bias"))
    fig_spatial(d, os.path.join(out_dir, "fig4_spatial_error"))
    fig_LR_heatmap(d, os.path.join(out_dir, "fig5_LR_heatmap"))
    fig_conditions(d, os.path.join(out_dir, "fig6_conditions"))
    write_summary(rep, d, out_dir)
    print(f"\nAll figures written to: {out_dir}")


if __name__ == "__main__":
    main()
