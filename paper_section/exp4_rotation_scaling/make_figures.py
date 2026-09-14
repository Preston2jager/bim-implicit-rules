"""Figure: per-holdout-copy side accuracy at 81 vs 161 copies (w=0.3).

Reads data/scaling.json, writes figures/fig_scaling.{png,pdf}.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SEC, "ytick.color": SEC,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 300,
})


def main() -> None:
    data = json.loads((HERE / "data/scaling.json").read_text())
    groups = [("copies81_w0.3", "81 copies\n6 holdout transf."),
              ("copies161_w0.3", "161 copies\n12 holdout transf.")]

    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    rng = np.random.default_rng(0)
    for gi, (key, label) in enumerate(groups):
        d = data[key]
        accs = np.array(list(d["per_copy_side_acc"].values())) * 100
        jitter = rng.uniform(-0.09, 0.09, len(accs))
        ax.scatter(gi + jitter, accs, s=26, color=BLUE, alpha=0.85,
                   edgecolor="white", linewidth=0.8, zorder=3)
        overall = d["overall_side_acc"] * 100
        ax.hlines(overall, gi - 0.2, gi + 0.2, color=INK, lw=2, zorder=4)
        ax.annotate(f"overall {overall:.1f}", (gi + 0.24, overall),
                    fontsize=8, color=INK, va="center")
        ax.annotate(f">0.5 m: {d['frac_gt_0.5m'] * 100:.1f}%", (gi, 60.5),
                    fontsize=8, color=SEC, ha="center")

    ax.set_xlim(-0.5, 1.75)
    ax.set_ylim(58, 103)
    ax.set_xticks([0, 1], [g[1] for g in groups], fontsize=8)
    ax.set_ylabel("Per-copy holdout side accuracy (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/fig_scaling.{ext}", bbox_inches="tight")
    print("wrote figures/fig_scaling.{png,pdf}")


if __name__ == "__main__":
    main()
