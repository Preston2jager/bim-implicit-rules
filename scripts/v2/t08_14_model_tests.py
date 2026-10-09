# --- repo-root bootstrap (this script is symlinked into SDF_3/scripts/v2/) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path.cwd()
# --- end bootstrap ---
"""T08-14 (Reviewer 3, "statistical testing between models"): paired tests of the full Graphormer against
every comparison arm, on the 12 unseen-transform copies (the independent unit -- graphs inside a copy are
correlated). No GPU: uses the per-door reports archived by train_arm.py.

For each arm B versus the reference A: per-copy paired differences d_c = acc_A(c) - acc_B(c), then
  * exact sign-flip permutation test on the 12 d_c (2^12 = 4096 sign patterns, two-sided)
  * paired cluster bootstrap over copies, 95% percentile CI of mean(d)
  * Wilcoxon signed-rank (reported for reference only; ties/zeros make it approximate)
Also the reference vs the seed-averaged Graphormer (5 seeds) so the comparison is not hostage to seed 42.

    .venv/bin/python scripts/v2/t08_14_model_tests.py
"""
import gzip
import itertools
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

RES = _Path(os.environ.get("SDF3_RESULTS", "data/results"))
OUT = RES / "T08_14"
PAT = re.compile(r"aug(\d+)_")
HOLD = _ROOT / "data/runs/V2_inductive/output/holdout"
FILES = sorted(os.listdir(HOLD)) if HOLD.exists() else []  # plan-1 holdout; absent on a plan-2-only machine

REF = "V2_s42"
ARMS = ["GINE_v2_pm_s42", "GINE_v2_orig_s42", "GINE_v2_orig_s1", "GINE_v2_orig_s2", "GAT_v2_s42",
        "ablation_geometry_only", "ablation_topology_only", "ablation_categorical_only",
        "V2_no_struct_s42", "V2_w0_s42"]
SEEDS = ["V2_s42", "V2_s1", "V2_s2", "V2_s3", "V2_s4"]
# Mean-of-seeds groups: compares the reference's seed-averaged accuracy against another model's
# own seed-averaged accuracy, so neither side's number is hostage to a single seed's luck (T08-14
# extension after T09-30/31/32: GINE's original config sits at chance on every one of 3 seeds).
MEAN_GROUPS = {"GINE_v2_orig_mean3": ["GINE_v2_orig_s42", "GINE_v2_orig_s1", "GINE_v2_orig_s2"]}


def per_copy(tag):
    p = RES / tag / "holdout_evaluation_report.json.gz"
    if not p.exists():
        return None
    res = json.load(gzip.open(p, "rt"))["per_door_results"]
    assert len(res) == len(FILES), (tag, len(res), len(FILES))
    by = defaultdict(list)
    for f, r in zip(FILES, res):
        by[int(PAT.search(f).group(1))].append(bool(r["side_correct"]))
    return {c: float(np.mean(v)) for c, v in sorted(by.items())}


def paired(a, b, rng, B=20000):
    cs = sorted(a)
    d = np.array([a[c] - b[c] for c in cs])
    obs = d.mean()
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    perm = (flips * d).mean(axis=1)
    p_perm = float((np.abs(perm) >= abs(obs) - 1e-12).mean())
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        w = stats.wilcoxon(d)
        p_w = float(w.pvalue)
    except ValueError:
        p_w = None
    return {"mean_diff_pp": 100 * float(obs), "boot95_ci_pp": [100 * float(lo), 100 * float(hi)],
            "p_signflip_exact": p_perm, "p_wilcoxon": p_w, "n_copies": len(d),
            "copies_A_better": int((d > 0).sum()), "copies_tied": int((d == 0).sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ref = per_copy(REF)
    out = {"reference": REF, "ref_mean_acc": float(np.mean(list(ref.values()))), "vs_seed42": {}, "vs_5seed_mean": {}}
    seed_pc = [per_copy(t) for t in SEEDS if per_copy(t)]
    ref5 = {c: float(np.mean([s[c] for s in seed_pc])) for c in ref}
    out["ref5_mean_acc"] = float(np.mean(list(ref5.values())))
    for tag in ARMS:
        arm = per_copy(tag)
        if arm is None:
            out["vs_seed42"][tag] = "report not archived yet"
            continue
        out["vs_seed42"][tag] = {"arm_mean_acc": float(np.mean(list(arm.values()))), **paired(ref, arm, rng)}
        out["vs_5seed_mean"][tag] = paired(ref5, arm, rng)

    out["vs_5seed_mean_grouped"] = {}
    for name, tags in MEAN_GROUPS.items():
        pcs = [per_copy(t) for t in tags]
        if any(p is None for p in pcs):
            out["vs_5seed_mean_grouped"][name] = "one or more member reports not archived yet"
            continue
        grouped = {c: float(np.mean([p[c] for p in pcs])) for c in ref}
        out["vs_5seed_mean_grouped"][name] = {
            "member_tags": tags, "group_mean_acc": float(np.mean(list(grouped.values()))),
            **paired(ref5, grouped, rng),
        }
    (OUT / "model_tests.json").write_text(json.dumps(out, indent=1))
    for tag, v in out["vs_seed42"].items():
        if isinstance(v, dict):
            print(f"{tag:28s} arm {100*v['arm_mean_acc']:.1f}%  diff {v['mean_diff_pp']:+.1f} pp  "
                  f"CI [{v['boot95_ci_pp'][0]:.1f}, {v['boot95_ci_pp'][1]:.1f}]  p_signflip {v['p_signflip_exact']:.4f}")
        else:
            print(tag, v)
    print(f"\nreference 5-seed mean: {100*out['ref5_mean_acc']:.1f}%")
    for name, v in out["vs_5seed_mean_grouped"].items():
        if isinstance(v, dict):
            print(f"{name:28s} group {100*v['group_mean_acc']:.1f}%  diff {v['mean_diff_pp']:+.1f} pp  "
                  f"CI [{v['boot95_ci_pp'][0]:.1f}, {v['boot95_ci_pp'][1]:.1f}]  p_signflip {v['p_signflip_exact']:.4f}")
        else:
            print(name, v)


if __name__ == "__main__":
    main()
