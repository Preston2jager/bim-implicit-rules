"""Figures: GINE message-passing baseline vs Graphormer (identical everything else).

Reads data/gin_baseline.json, writes:
  figures/fig_gin_curve.{png,pdf}  - val-loss curves: saddle escape dynamics
  figures/fig_gin_rules.{png,pdf}  - holdout side accuracy per rule
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
BLUE, YELLOW = "#2a78d6", "#eda100"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SEC, "ytick.color": SEC,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 300,
})

PREDICT_MEAN = 4.20  # E[s^2] of the slot distribution = loss of the constant predictor


def fig_curve(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    curves = data["curves"]

    for key, color, label in (("gin", YELLOW, "GINE (depth 8)"),
                              ("graphormer", BLUE, "Graphormer")):
        c = curves[key]
        ax.plot(c["epoch"], c["val_mse"], color=color, lw=1.6, label=label, zorder=3)

    ax.axhline(PREDICT_MEAN, color=MUT, lw=0.9, ls=(0, (4, 3)))
    ax.annotate("predict-mean saddle (4.20)", (103, PREDICT_MEAN * 1.35),
                fontsize=7.5, color=MUT, ha="right")

    g = curves["graphormer"]
    ax.scatter([g["epoch"][-1]], [g["val_mse"][-1]], s=18, color=BLUE, zorder=4)
    ax.annotate("early stop ep 45\nval 3.5e-5", (g["epoch"][-1] + 2, g["val_mse"][-1] * 2),
                fontsize=7.5, color=BLUE)
    n = curves["gin"]
    ax.annotate(f"ep 100 cap\nval {n['val_mse'][-1]:.2f}", (n["epoch"][-1] - 1, n["val_mse"][-1] * 2.2),
                fontsize=7.5, color="#a87200", ha="right")
    ax.annotate("saddle: ep 8–50", (26, 2.4), fontsize=7.5, color="#a87200", ha="center")

    ax.set_yscale("log")
    ax.set_xlim(0, 104)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MSE (door loss)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/fig_gin_curve.{ext}", bbox_inches="tight")
    print("wrote figures/fig_gin_curve.{png,pdf}")


def fig_rules(data: dict) -> None:
    cats = [("overall", "Overall"), ("window", "Six-window"),
            ("wall", "Single-wall"), ("semantic", "Semantic")]
    accs = {}
    for arch in ("graphormer", "gin"):
        h = data[arch]["holdout"]
        accs[arch] = [100 * (h["side_acc"] if k == "overall" else h["per_rule"][k]["side_acc"])
                      for k, _ in cats]

    x = np.arange(len(cats))
    w = 0.34
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    for off, arch, color, label in ((-1, "graphormer", BLUE, "Graphormer"),
                                    (1, "gin", YELLOW, "GINE (depth 8)")):
        bars = ax.bar(x + off * (w / 2 + 0.01), accs[arch], w, color=color, label=label)
        for b, v in zip(bars, accs[arch]):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v + 1),
                        ha="center", va="bottom", fontsize=8, color=SEC)

    ax.axhline(50, color=MUT, lw=0.9, ls=(0, (1, 2)))
    ax.annotate("chance (side)", (-0.48, 45.5), ha="left", va="top", fontsize=7.5, color=MUT)
    ax.set_ylim(0, 118)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xticks(x, [c[1] for c in cats], fontsize=8.5)
    ax.set_ylabel("Holdout side accuracy (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/fig_gin_rules.{ext}", bbox_inches="tight")
    print("wrote figures/fig_gin_rules.{png,pdf}")


def main() -> None:
    data = json.loads((HERE / "data/gin_baseline.json").read_text())
    (HERE / "figures").mkdir(exist_ok=True)
    fig_curve(data)
    fig_rules(data)


if __name__ == "__main__":
    main()
