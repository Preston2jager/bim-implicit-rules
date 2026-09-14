"""Figures: round-8 holdout failure anatomy (run 2026JUL122211).

  fig_threshold.{png,pdf} : slot-error rate vs decisive wall length (distance
                            from the 1 m effectiveness threshold)
  fig_confusion.{png,pdf} : window-rule 6x6 slot confusion (row-normalised)
  fig_angle.{png,pdf}     : per-copy accuracy vs angular distance to the
                            nearest training rotation

Reads data/failure.json (see extract_data.py).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).parent
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SEC, "ytick.color": SEC,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 300,
})


def fig_threshold(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    lengths = [1200, 1500, 1800, 2100, 2400, 2700, 3000]
    for rule, label, color in (("wall", "single-wall rule", BLUE),
                               ("semantic", "semantic rule", AQUA)):
        ys = [data["threshold_curve"][rule][str(x)]["bad_rate"] * 100 for x in lengths]
        ax.plot(lengths, ys, "-", color=color, lw=2, marker="o", ms=5,
                markeredgecolor="white", markeredgewidth=1, zorder=3, label=label)
    w900 = data["threshold_curve"]["window"]["with_900mm"]["bad_rate"] * 100
    w_rest = data["threshold_curve"]["window"]["without_900mm"]["bad_rate"] * 100
    ax.scatter([900], [w900], s=42, color=YELLOW, edgecolor="white",
               linewidth=1, zorder=4, label="window rule, 900 mm wall present")
    ax.annotate(f"{w900:.0f}%", (900, w900), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=8, color=SEC)
    ax.annotate(f"window rule without a 900 mm wall: {w_rest:.1f}%",
                (2550, 1.0), fontsize=7.5, color=MUT, ha="center")

    ax.axvline(1000, color=MUT, lw=0.9, ls=(0, (4, 3)))
    ax.annotate("1 m effectiveness threshold", (985, 21), fontsize=7.5,
                color=MUT, ha="right", va="center", rotation=90)
    ax.set_xlabel("Decisive wall length (mm)")
    ax.set_ylabel("Holdout slot-error rate (%)")
    ax.set_ylim(0, 35)
    ax.set_xticks([900] + lengths)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/fig_threshold.{ext}", bbox_inches="tight")
    print("wrote figures/fig_threshold.{png,pdf}")


def fig_confusion(data: dict) -> None:
    conf = np.array(data["window_confusion"], float)
    rownorm = conf / conf.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(rownorm ** 0.4, cmap=cmap, vmin=0, vmax=1)  # gamma lifts small cells
    for i in range(6):
        for j in range(6):
            v = int(conf[i, j])
            if v == 0:
                continue
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7.5,
                    color="white" if rownorm[i, j] > 0.5 else INK)
    ax.set_xticks(range(6), [f"P{j+1}" for j in range(6)])
    ax.set_yticks(range(6), [f"T{i+1}" for i in range(6)])
    ax.set_xlabel("Predicted slot")
    ax.set_ylabel("True slot")
    ax.grid(visible=False)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/fig_confusion.{ext}", bbox_inches="tight")
    print("wrote figures/fig_confusion.{png,pdf}")


def fig_angle(data: dict) -> None:
    pc = data["per_copy"]
    d = [v["nearest_train_angle_dist_deg"] for v in pc.values()]
    acc = [v["side_acc"] * 100 for v in pc.values()]

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.scatter(d, acc, s=34, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    worst = min(pc.values(), key=lambda v: v["side_acc"])
    ax.annotate(f"worst copy: {worst['side_acc']*100:.1f}% at only\n"
                f"{worst['nearest_train_angle_dist_deg']:.2f}° from a training rotation",
                (worst["nearest_train_angle_dist_deg"], worst["side_acc"] * 100),
                xytext=(10, -2), textcoords="offset points", fontsize=7.5, color=SEC)
    r = np.corrcoef(d, acc)[0, 1]
    ax.annotate(f"r = {r:.2f}  (n = {len(d)} holdout copies,\n"
                f"{data['n_train_rotations']} training rotations)",
                (0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=7.5, color=MUT)
    ax.set_xlabel("Angular distance to nearest training rotation (°)")
    ax.set_ylabel("Holdout side accuracy (%)")
    ax.set_ylim(76, 103)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/fig_angle.{ext}", bbox_inches="tight")
    print("wrote figures/fig_angle.{png,pdf}")


def main() -> None:
    data = json.loads((HERE / "data/failure.json").read_text())
    fig_threshold(data)
    fig_confusion(data)
    fig_angle(data)


if __name__ == "__main__":
    main()
