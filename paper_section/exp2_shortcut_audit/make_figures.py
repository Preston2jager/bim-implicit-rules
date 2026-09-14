"""Figure: per-rule holdout side accuracy, Name leak vs de-identified.

Reads data/per_rule.json (see extract_data.py), writes
figures/fig_rule_breakdown.{png,pdf}.
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

RULES = ["window", "wall", "semantic"]
RULE_LABELS = ["Six-window rule\n(categorical lookup?)", "Single-wall rule",
               "Semantic rule\n(toilet side)"]


def main() -> None:
    data = json.loads((HERE / "data/per_rule.json").read_text())
    leak = [data["leak_scalar_s"][r]["side_acc"] * 100 for r in RULES]
    deid = [data["deidentified_round5"][r]["side_acc"] * 100 for r in RULES]

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    x = np.arange(len(RULES))
    w = 0.34
    b1 = ax.bar(x - w / 2 - 0.01, leak, w, color=BLUE, label="Identity in Name (leak)")
    b2 = ax.bar(x + w / 2 + 0.01, deid, w, color=AQUA, label="Name de-identified")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=SEC)

    ax.axhline(100, color=MUT, lw=0.9, ls=(0, (4, 3)))
    ax.annotate("provable ceiling (probe + topology pipeline) = 100", (1.5, 98.5),
                ha="center", va="top", fontsize=7.5, color=MUT)
    ax.axhline(50, color=MUT, lw=0.9, ls=(0, (1, 2)))
    ax.annotate("chance = 50", (0.5, 51), ha="center", va="bottom",
                fontsize=7.5, color=MUT)

    ax.set_ylim(0, 112)
    ax.set_xticks(x, RULE_LABELS, fontsize=8)
    ax.set_ylabel("Holdout side accuracy (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, 1.18), ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/fig_rule_breakdown.{ext}", bbox_inches="tight")
    print("wrote figures/fig_rule_breakdown.{png,pdf}")


if __name__ == "__main__":
    main()
