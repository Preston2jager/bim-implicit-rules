"""Figure: pose-distillation weight sweep (holdout, 81 copies).

Panel (a): side accuracy vs pose_distill_weight per rule + overall, with the
run-to-run noise band measured at w=0 (two identical-config runs).
Panel (b): six-window exact-slot accuracy vs weight.

Reads data/sweep.json, writes figures/fig_pose_distill.{png,pdf}.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SEC, "ytick.color": SEC,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 300,
})

SERIES = [  # (key-fn, label, color)
    (lambda d: d["overall_side_acc"], "overall", BLUE),
    (lambda d: d["window"]["side_acc"], "six-window", AQUA),
    (lambda d: d["wall"]["side_acc"], "single-wall", YELLOW),
    (lambda d: d["semantic"]["side_acc"], "semantic", GREEN),
]


def main() -> None:
    data = json.loads((HERE / "data/sweep.json").read_text())
    weights = [0.0, 0.1, 0.3]
    # at w=0 average the two runs for the line; both shown as markers
    runs0 = [data["w0_runA"], data["w0_runB"]]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(6.6, 2.9), gridspec_kw={"width_ratios": [2.1, 1]})

    lo = min(f(r) for f, _, _ in SERIES for r in runs0) * 100
    hi = max(f(r) for f, _, _ in SERIES for r in runs0) * 100
    ax.axhspan(lo, hi, color=GRID, alpha=0.55, lw=0)
    ax.annotate("w=0 run-to-run spread", (0.155, (lo + hi) / 2), fontsize=7.5,
                color=MUT, va="center")

    for fn, label, color in SERIES:
        y0 = [fn(r) * 100 for r in runs0]
        ys = [sum(y0) / 2, fn(data["w0.1"]) * 100, fn(data["w0.3"]) * 100]
        ax.plot(weights, ys, "-", color=color, lw=2, marker="o", ms=5,
                markeredgecolor="white", markeredgewidth=1, zorder=3)
        ax.scatter([0, 0], y0, s=22, color=color, edgecolor="white",
                   linewidth=1, zorder=3)
        ax.annotate(label, (weights[-1], ys[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=color, va="center")

    ax.set_xlim(-0.02, 0.42)
    ax.set_ylim(48, 100)
    ax.set_xticks(weights)
    ax.set_xlabel("pose_distill_weight")
    ax.set_ylabel("Holdout side accuracy (%)")
    ax.set_title("(a) side accuracy by rule", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)

    slot = [sum(r["window"]["exact_slot_acc"] for r in runs0) / 2 * 100,
            data["w0.1"]["window"]["exact_slot_acc"] * 100,
            data["w0.3"]["window"]["exact_slot_acc"] * 100]
    ax2.plot(weights, slot, "-", color=BLUE, lw=2, marker="o", ms=5,
             markeredgecolor="white", markeredgewidth=1)
    for wgt, v in zip(weights, slot):
        dx, dy, ha = (8, -3, "left") if wgt == 0 else (0, 6, "center")
        ax2.annotate(f"{v:.1f}", (wgt, v), xytext=(dx, dy), textcoords="offset points",
                     ha=ha, fontsize=8, color=SEC)
    ax2.set_xlim(-0.04, 0.36)
    ax2.set_ylim(45, 95)
    ax2.set_xticks(weights)
    ax2.set_xlabel("pose_distill_weight")
    ax2.set_ylabel("Exact-slot accuracy (%)")
    ax2.set_title("(b) six-window exact slot", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", visible=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/fig_pose_distill.{ext}", bbox_inches="tight")
    print("wrote figures/fig_pose_distill.{png,pdf}")


if __name__ == "__main__":
    main()
