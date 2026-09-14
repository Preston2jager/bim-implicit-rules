"""Figure: seen-vs-holdout error per prediction head (log scale).

Reads data/summary.json (see extract_data.py), writes
figures/fig_head_gap.{png,pdf}.
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
BLUE, AQUA = "#2a78d6", "#1baf7a"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SEC, "ytick.color": SEC,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 300,
})

LABELS = {
    "absolute": "absolute\nworld = pred",
    "offset": "offset\nwc + pred",
    "scalar_s (axis-product)": "scalar$_s$ (axis-product)\nwc + (pred·axis)·axis",
}


def main() -> None:
    data = json.loads((HERE / "data/summary.json").read_text())
    heads = list(LABELS)
    seen = [data[h]["seen"]["mean_euclidean_error"] for h in heads]
    hold = [data[h]["holdout"]["mean_euclidean_error"] for h in heads]

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = np.arange(len(heads))
    w = 0.34
    b1 = ax.bar(x - w / 2 - 0.01, seen, w, color=BLUE, label="Seen test split")
    b2 = ax.bar(x + w / 2 + 0.01, hold, w, color=AQUA, label="Unseen-transform holdout")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=SEC)
    for xi, (s, h) in enumerate(zip(seen, hold)):
        ax.annotate(f"{h / s:.0f}×", (xi, h), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=9, color=INK, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylim(0.02, 30)
    ax.set_xticks(x, [LABELS[h] for h in heads], fontsize=7.5)
    ax.set_ylabel("Mean Euclidean error (m, log)")
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, 1.16), ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/fig_head_gap.{ext}", bbox_inches="tight")
    print("wrote figures/fig_head_gap.{png,pdf}")


if __name__ == "__main__":
    main()
