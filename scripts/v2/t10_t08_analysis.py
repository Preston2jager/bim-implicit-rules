# --- repo-root bootstrap (this script is symlinked into SDF_3/scripts/v2/) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path.cwd()
if not (_ROOT / "src" / "Graphormer.py").exists():
    raise SystemExit("run from the SDF_3 repository root (cd ~/Projects/SDF_3)")
# --- end bootstrap ---
"""T10-01 + T08-01: error-distribution analysis and cluster-bootstrap CIs from ONE holdout report.

Works on any arm's holdout_evaluation_report.json (v1 final run, v2, later every seed), because it only
needs per_door_results in the same order as sorted(output/holdout/graph_room*.json).

    python scripts/v2/t10_t08_analysis.py --run 2026JUL122211 \
        --report data/runs/2026JUL122211/output/experiments/scalar_s_pose_w0.3/holdout_evaluation_report.json --tag v1_final

Outputs ($SDF3_RESULTS/T10_T08/<tag>/): analysis.json, fig_error_distribution.png/pdf, fig_confusion.png/pdf,
fig_per_copy.png/pdf. Everything is computed from the report; nothing is retrained.
"""
import argparse
import gzip
import json
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAT = re.compile(r"room(\d+)_aug(\d+)_L(\w+)_R(\w+)_W(\d+)")
SLOTS = np.array([-2.375, -1.425, -0.475, 0.475, 1.425, 2.375])
SPACING = 0.95
OUT_ROOT = _Path(os.environ.get("SDF3_RESULTS", "data/results")) / "T10_T08"
RULES = ("window", "wall", "semantic")


def eff(v: str) -> bool:
    return v != "Absent" and int(v) > 1000


def snap(s: float) -> int:
    return int(np.argmin(np.abs(SLOTS - s)))


def rule_of(l: str, r: str) -> str:
    le, re_ = eff(l), eff(r)
    return "window" if not le and not re_ else ("wall" if le != re_ else "semantic")


def ci_boot(per_copy: np.ndarray, B: int, rng) -> dict:
    """cluster bootstrap over copies: resample copies with replacement, equal-size clusters -> mean of means"""
    n = len(per_copy)
    idx = rng.integers(0, n, size=(B, n))
    m = per_copy[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    from scipy import stats
    se = per_copy.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return {"mean": float(per_copy.mean()), "sd_across_copies": float(per_copy.std(ddof=1)),
            "boot95": [float(lo), float(hi)], "t95": [float(per_copy.mean() - t * se), float(per_copy.mean() + t * se)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--report", required=True, help="holdout_evaluation_report.json (or .json.gz)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--slot-spacing", type=float, default=0.95,
                    help="door slot spacing in m; slots are +-0.5,1.5,2.5 x spacing (plan 1: 0.95, plan 2: 0.99)")
    ap.add_argument("--holdout-dir", default=None,
                    help="graph dir matching the report order (default: data/runs/<run>/output/holdout)")
    args = ap.parse_args()
    global SLOTS, SPACING
    SPACING = args.slot_spacing
    SLOTS = SPACING * np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])

    rp = _Path(args.report)
    rep = json.load(gzip.open(rp, "rt")) if rp.suffix == ".gz" else json.load(open(rp))
    res = rep["per_door_results"]
    hdir = _Path(args.holdout_dir) if args.holdout_dir else _ROOT / "data/runs" / args.run / "output/holdout"
    files = sorted(hdir.glob("graph_room*.json"))
    assert len(files) == len(res), (len(files), len(res))

    aug, room, rule, err = [], [], [], []
    t_slot, p_slot, side = [], [], []
    minlen, has900 = [], []
    for f, r in zip(files, res):
        m = PAT.search(f.name)
        L, R = m[3], m[4]
        aug.append(int(m[2])); room.append(int(m[1])); rule.append(rule_of(L, R))
        err.append(r["euclidean_error"]); side.append(bool(r["side_correct"]))
        t_slot.append(snap(r["wall_s_true"])); p_slot.append(snap(r["wall_s_pred"]))
        els = [int(v) for v, e in ((L, eff(L)), (R, eff(R))) if e]
        minlen.append(min(els) if els else 0)
        has900.append("900" in (L, R))
    aug, room, rule = np.array(aug), np.array(room), np.array(rule)
    err, side = np.array(err), np.array(side)
    t_slot, p_slot = np.array(t_slot), np.array(p_slot)
    minlen, has900 = np.array(minlen), np.array(has900)
    exact = t_slot == p_slot
    copies = sorted(set(aug))
    rng = np.random.default_rng(0)
    out = {"tag": args.tag, "n_graphs": int(len(err)), "n_copies": len(copies), "copies": [int(c) for c in copies]}

    # ------------------------------------------------------------------ T08-01: cluster CIs
    def per_copy(mask_fn):
        return np.array([mask_fn(aug == c) for c in copies])

    boot = {"overall_side_acc": ci_boot(per_copy(lambda m: side[m].mean()), args.boot, rng),
            "overall_exact_slot_acc": ci_boot(per_copy(lambda m: exact[m].mean()), args.boot, rng),
            "frac_err_gt_0.5m": ci_boot(per_copy(lambda m: (err[m] > 0.5).mean()), args.boot, rng)}
    for rl in RULES:
        boot[f"{rl}_side_acc"] = ci_boot(per_copy(lambda m, rl=rl: side[m & (rule == rl)].mean()), args.boot, rng)
        boot[f"{rl}_exact_slot_acc"] = ci_boot(per_copy(lambda m, rl=rl: exact[m & (rule == rl)].mean()), args.boot, rng)
    out["cluster_bootstrap"] = boot
    out["per_copy"] = {str(c): {"side_acc": float(side[aug == c].mean()), "exact_slot_acc": float(exact[aug == c].mean()),
                                "frac_err_gt_0.5m": float((err[aug == c] > 0.5).mean()), "n": int((aug == c).sum())}
                       for c in copies}
    out["naive_binomial_95_side_acc_if_graphs_were_independent"] = float(1.96 * np.sqrt(side.mean() * (1 - side.mean()) / len(side)))

    # ------------------------------------------------------------------ T10-01: distribution
    q = [0.5, 0.75, 0.9, 0.95, 0.99]
    out["error_quantiles_m"] = {str(p): float(np.quantile(err, p)) for p in q}
    out["error_fractions"] = {"lt_1cm": float((err < 0.01).mean()), "lt_5cm": float((err < 0.05).mean()),
                              "lt_0.25m": float((err < 0.25).mean()), "ge_0.5m": float((err >= 0.5).mean()),
                              "ge_0.9m": float((err >= 0.9).mean()), "mean_m": float(err.mean()),
                              "median_m": float(np.median(err)), "max_m": float(err.max())}
    bad = err >= 0.5
    wrong = t_slot != p_slot
    out["tail_attribution"] = {
        "n_err_ge_0.5m": int(bad.sum()),
        "share_with_wrong_slot": float(wrong[bad].mean()),
        "share_side_wrong": float((~side[bad]).mean()),
        "adjacent_slot_share_among_wrong": float((np.abs(t_slot - p_slot)[wrong] == 1).mean()) if wrong.any() else None,
        "share_by_rule": {rl: float((rule[bad] == rl).mean()) for rl in RULES},
        "wrong_rate_by_rule": {rl: float(wrong[rule == rl].mean()) for rl in RULES},
        "near_threshold_share": float(np.mean(((rule == "window") & has900)[bad] | ((rule != "window") & (minlen <= 1500))[bad])),
        "window_rule_with_900mm_wall": {"n": int(((rule == "window") & has900).sum()),
                                        "wrong_rate": float(wrong[(rule == "window") & has900].mean()),
                                        "wrong_rate_without": float(wrong[(rule == "window") & ~has900].mean())},
        "top5_copies_by_share_of_tail": sorted(({"copy": int(c), "share": float(bad[aug == c].sum() / bad.sum())} for c in copies),
                                               key=lambda d: -d["share"])[:5],
        "error_in_slot_spacings_hist": {str(k): int(v) for k, v in zip(*np.unique(np.round(err[bad] / SPACING).astype(int), return_counts=True))},
    }
    # confusion matrices
    conf6 = np.zeros((6, 6), int)
    for a, b in zip(t_slot, p_slot):
        conf6[a, b] += 1
    conf_rule = {}
    for rl in RULES:
        m = rule == rl
        tt, pp = t_slot[m] >= 3, p_slot[m] >= 3   # right-hand side?
        conf_rule[rl] = [[int(((~tt) & (~pp)).sum()), int(((~tt) & pp).sum())], [int((tt & (~pp)).sum()), int((tt & pp).sum())]]
    conf_room = {}
    for r_ in sorted(set(room)):
        m = room == r_
        tt, pp = t_slot[m] >= 3, p_slot[m] >= 3
        conf_room[str(r_)] = [[int(((~tt) & (~pp)).sum()), int(((~tt) & pp).sum())], [int((tt & (~pp)).sum()), int((tt & pp).sum())]]
    out["confusion_6x6_all_rules"] = conf6.tolist()
    out["confusion_side_2x2_by_rule"] = conf_rule
    out["confusion_side_2x2_by_room_state"] = conf_room
    out["label_marginals"] = {"true_slot_counts": np.bincount(t_slot, minlength=6).tolist(),
                              "rule_counts": {rl: int((rule == rl).sum()) for rl in RULES}}
    # chance references (majority class per rule, using the true-slot marginal)
    out["chance_slot_acc_majority_by_rule"] = {rl: float(np.bincount(t_slot[rule == rl], minlength=6).max() / (rule == rl).sum())
                                               for rl in RULES}
    out["chance_side_acc_majority_by_rule"] = {rl: float(max((t_slot[rule == rl] < 3).mean(), (t_slot[rule == rl] >= 3).mean()))
                                               for rl in RULES}

    od = OUT_ROOT / args.tag
    od.mkdir(parents=True, exist_ok=True)
    (od / "analysis.json").write_text(json.dumps(out, indent=1))

    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    xs = np.sort(err)
    ax[0].plot(np.maximum(xs, 1e-4), np.arange(1, len(xs) + 1) / len(xs), lw=1.6)
    for k in range(1, 5):
        ax[0].axvline(k * SPACING, color="grey", ls=":", lw=0.8)
    ax[0].axvline(0.5, color="C3", ls="--", lw=1)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Euclidean error (m)"); ax[0].set_ylabel("cumulative share of graphs")
    ax[0].set_title("ECDF of holdout error (dotted: multiples of the 0.95 m slot spacing)", fontsize=8)
    ax[1].hist(np.log10(np.maximum(err, 1e-4)), bins=60, color="C0")
    ax[1].set_xlabel("log10 error (m)"); ax[1].set_ylabel("graphs"); ax[1].set_title("bimodal: correct slot vs wrong slot", fontsize=8)
    fig.tight_layout(); fig.savefig(od / "fig_error_distribution.png", dpi=170); fig.savefig(od / "fig_error_distribution.pdf"); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    im = ax[0].imshow(conf6, cmap="Blues")
    for i in range(6):
        for j in range(6):
            ax[0].text(j, i, conf6[i, j], ha="center", va="center", fontsize=6, color="k" if conf6[i, j] < conf6.max() / 2 else "w")
    ax[0].set_title("slot confusion (all rules)", fontsize=8); ax[0].set_xlabel("predicted slot"); ax[0].set_ylabel("true slot")
    ax[0].set_xticks(range(6), range(1, 7)); ax[0].set_yticks(range(6), range(1, 7))
    for a_, rl in zip(ax[1:], RULES):
        c = np.array(conf_rule[rl])
        a_.imshow(c, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a_.text(j, i, c[i, j], ha="center", va="center", fontsize=8, color="k" if c[i, j] < c.max() / 2 else "w")
        a_.set_xticks([0, 1], ["left", "right"]); a_.set_yticks([0, 1], ["left", "right"])
        a_.set_title(f"{rl}: side (true × pred)", fontsize=8)
    fig.tight_layout(); fig.savefig(od / "fig_confusion.png", dpi=170); fig.savefig(od / "fig_confusion.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    accs = np.array([side[aug == c].mean() for c in copies])
    order = np.argsort(accs)
    ax.plot(range(len(copies)), accs[order] * 100, "o", color="C0")
    b = boot["overall_side_acc"]
    ax.axhspan(b["boot95"][0] * 100, b["boot95"][1] * 100, color="C1", alpha=0.25, label=f"cluster-bootstrap 95% CI of the mean ({b['mean']*100:.1f}%)")
    ax.axhline(b["mean"] * 100, color="C1")
    ax.set_xticks(range(len(copies)), [f"aug{copies[i]}" for i in order], rotation=60, fontsize=7)
    ax.set_ylabel("side accuracy (%)"); ax.set_title("per unseen transform (n = 12 independent units)", fontsize=8); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(od / "fig_per_copy.png", dpi=170); fig.savefig(od / "fig_per_copy.pdf"); plt.close(fig)

    print(json.dumps({"boot_side": boot["overall_side_acc"], "boot_exact": boot["overall_exact_slot_acc"],
                      "boot_tail": boot["frac_err_gt_0.5m"], "tail": out["tail_attribution"]}, indent=1)[:3000])
    print("wrote", od)


if __name__ == "__main__":
    main()
