"""Figure + statistics check for Experiment 11 (second plan topology).

Reads data/plan2.json only (no run archives), writes figures/fig_plan2.{png,pdf}: per-holdout-copy side
accuracy of every arm. Also recomputes the w=0 paired test (exact sign-flip over the 12 copies + paired
cluster bootstrap) from the per-copy numbers and prints it next to the archived values, so the
statistics can be checked from a fresh clone.
"""
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, GREY, ORANGE = "#2a78d6", "#898781", "#d97b2a"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SEC, "ytick.color": SEC,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 300,
})

GROUPS = [
    ("P2_s42", "Plan 2\nseed 42", BLUE),
    ("P2_s1", "Plan 2\nseed 1", BLUE),
    ("P2_s2", "Plan 2\nseed 2", BLUE),
    ("P2_w0_s42", "Plan 2\nw = 0", GREY),
    ("P2_161_s42", "Plan 2\n161 copies", BLUE),
    ("P1_s42_x2_on_plan2", "Plan-1 model\nzero-shot", ORANGE),
    ("P1_s42_x2", "Plan 1\n(own holdout)", GREY),
]


def side(pc: dict) -> dict:
    return {c: v["side_acc"] for c, v in pc.items()}


def paired(a: dict, b: dict, B: int = 20000) -> dict:
    cs = sorted(a)
    d = np.array([a[c] - b[c] for c in cs])
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    p = float((np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())
    rng = np.random.default_rng(0)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff_pp": 100 * d.mean(), "ci_pp": (100 * lo, 100 * hi), "p": p, "better": int((d > 0).sum())}


def main() -> None:
    data = json.loads((HERE / "data/plan2.json").read_text())
    pcs = data["per_copy"]

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    rng = np.random.default_rng(0)
    for gi, (tag, _, color) in enumerate(GROUPS):
        accs = np.array(list(side(pcs[tag]).values())) * 100
        ax.scatter(gi + rng.uniform(-0.1, 0.1, len(accs)), accs, s=20, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.7, zorder=3)
        ax.hlines(accs.mean(), gi - 0.22, gi + 0.22, color=INK, lw=2, zorder=4)
        ax.annotate(f"{accs.mean():.1f}", (gi, 104), fontsize=8, color=INK, ha="center")
    ax.axhline(50, color=MUT, lw=0.8, ls=":")
    ax.set_xlim(-0.5, len(GROUPS) - 0.5)
    ax.set_ylim(40, 108)
    ax.set_xticks(range(len(GROUPS)), [g[1] for g in GROUPS], fontsize=7.5)
    ax.set_ylabel("Per-copy holdout side accuracy (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/fig_plan2.{ext}", bbox_inches="tight")
    print("wrote figures/fig_plan2.{png,pdf}")

    ref = side(pcs["P2_s42"])
    w0 = side(pcs["P2_w0_s42"])
    ref3 = {c: np.mean([side(pcs[t])[c] for t in ("P2_s42", "P2_s1", "P2_s2")]) for c in ref}
    archived = data["summary"]["p2_5_w0"]["tests"]
    for name, a, key in (("seed 42 - w0", ref, "vs_seed42"), ("3-seed mean - w0", ref3, "vs_3seed_mean")):
        r = paired(a, w0)
        arc = archived[key]
        print(f"{name:17s} {r['diff_pp']:+.1f} pp  CI [{r['ci_pp'][0]:.1f}, {r['ci_pp'][1]:.1f}]  "
              f"p {r['p']:.4f}  better {r['better']}/12   (archived: {arc['mean_diff_pp']:+.1f} pp, "
              f"p {arc['p_signflip_exact']:.4f})")


if __name__ == "__main__":
    main()
