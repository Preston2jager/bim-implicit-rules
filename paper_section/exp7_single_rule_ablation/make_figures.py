"""Figures: single-rule ablation — rule isolation as a causal test.

Reads data/single_rule.json, writes:
  figures/fig_rule_isolation.{png,pdf}     - per-rule side / exact-slot accuracy,
      joint model vs isolated single-rule models (24% of the data each)
  figures/fig_isolation_threshold.{png,pdf} - slot-error rate vs decisive wall
      length per rule: the joint model's threshold gradient vanishes (window,
      semantic) or shrinks to the within-rule component (wall) once rule
      selection is removed
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
DARK_YELLOW = "#a87200"  # label ink for yellow marks (text never wears mark color)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SEC, "ytick.color": SEC,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 300,
})

RULES = [("window", "Six-window"), ("wall", "Single-wall"), ("semantic", "Semantic")]


def fig_isolation(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    metrics = [("side_acc", "Holdout side accuracy (%)"),
               ("exact_slot_acc", "Holdout exact-slot accuracy (%)")]
    x = np.arange(len(RULES))
    w = 0.34

    for ax, (key, ylabel) in zip(axes, metrics):
        for off, cond, color, label in ((-1, "joint", BLUE, "joint (all rules, 119.8k graphs)"),
                                        (1, "isolated", YELLOW, "isolated (one rule, 29.2k graphs)")):
            vals = [100 * data["rules"][r][cond][key] for r, _ in RULES]
            bars = ax.bar(x + off * (w / 2 + 0.01), vals, w, color=color, label=label)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v + 1.2),
                            ha="center", va="bottom", fontsize=8, color=SEC)
        if key == "side_acc":
            ax.axhline(50, color=MUT, lw=0.9, ls=(0, (1, 2)))
            ax.annotate("chance (side)", (-0.47, 46), ha="left", va="top",
                        fontsize=7.5, color=MUT)
            labels = [f"{lbl}\nmirrors {data['rules'][r]['joint']['mirrors']}"
                      f"→{data['rules'][r]['isolated']['mirrors']}" for r, lbl in RULES]
        else:
            labels = [lbl for _, lbl in RULES]
        ax.set_ylim(0, 108)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xticks(x, labels, fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", visible=False)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, frameon=False, fontsize=8, ncols=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/fig_rule_isolation.{ext}", bbox_inches="tight")
    print("wrote figures/fig_rule_isolation.{png,pdf}")


def fig_threshold(data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True)

    for ax, (rule, title) in zip(axes, RULES):
        curves = data["threshold"][rule]
        keys = list(curves["joint"].keys())
        if rule == "window":
            xs = np.arange(len(keys))
            ax.set_xticks(xs, ["absent", "300", "600", "900"], fontsize=8)
            ax.set_xlabel("Min present wall length (mm)")
        else:
            xs = np.array([int(k) for k in keys])
            ax.set_xticks([1200, 1800, 2400, 3000], ["1200", "1800", "2400", "3000"], fontsize=8)
            ax.set_xlabel("Effective wall length (mm)")
        for cond, color, z in (("joint", BLUE, 3), ("isolated", YELLOW, 4)):
            ys = [100 * curves[cond][k]["bad_rate"] for k in keys]
            ax.plot(xs, ys, "-", color=color, lw=2, marker="o", ms=5,
                    markeredgecolor="white", markeredgewidth=1, zorder=z,
                    label="joint" if cond == "joint" else "isolated")
        # threshold sits at 1000 mm, just outside the wall/semantic x-range;
        # annotations hand-placed per panel to avoid collisions
        jval = {k: 100 * v["bad_rate"] for k, v in curves["joint"].items()}
        ival = {k: 100 * v["bad_rate"] for k, v in curves["isolated"].items()}
        if rule == "window":
            ax.annotate("1 m threshold\n100 mm above →", (2.98, 25), ha="right",
                        fontsize=7.5, color=MUT)
            ax.annotate(f"{jval['900']:.1f}", (xs[-1], jval["900"]), xytext=(-4, 6),
                        textcoords="offset points", ha="right", fontsize=8, color=SEC)
            ax.annotate(f"{ival['900']:.1f}", (xs[-1], ival["900"]), xytext=(0, -14),
                        textcoords="offset points", ha="center", fontsize=8, color=DARK_YELLOW)
        elif rule == "wall":
            ax.annotate("← 1 m threshold", (1560, 36), ha="left",
                        fontsize=7.5, color=MUT)
            ax.annotate(f"{jval['1200']:.1f}", (1200, jval["1200"]), xytext=(4, 6),
                        textcoords="offset points", ha="left", fontsize=8, color=SEC)
            ax.annotate(f"{ival['1200']:.1f}", (1200, ival["1200"]), xytext=(0, -14),
                        textcoords="offset points", ha="center", fontsize=8, color=DARK_YELLOW)
        else:
            ax.annotate("← 1 m threshold", (1560, 36), ha="left",
                        fontsize=7.5, color=MUT)
            ax.annotate(f"{jval['1200']:.1f}", (1200, jval["1200"]), xytext=(4, 6),
                        textcoords="offset points", ha="left", fontsize=8, color=SEC)
            ax.annotate("isolated: 0.0 at every length", (2350, 4.5), ha="center",
                        fontsize=7.5, color=DARK_YELLOW)
        ax.set_title(f"{title} rule", fontsize=9)
        ax.set_ylim(-2, 40)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Holdout slot-error rate (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/fig_isolation_threshold.{ext}", bbox_inches="tight")
    print("wrote figures/fig_isolation_threshold.{png,pdf}")


def main() -> None:
    data = json.loads((HERE / "data/single_rule.json").read_text())
    (HERE / "figures").mkdir(exist_ok=True)
    fig_isolation(data)
    fig_threshold(data)


if __name__ == "__main__":
    main()
