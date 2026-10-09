"""P2-5 (plan 2, second topology): paired tests of the plan-2 final model against the w=0 control on the
12 plan-2 holdout copies. Same statistics as t08_14_model_tests.py (imported, not copied): per-copy side
accuracy, exact sign-flip test over 2^12 patterns, paired cluster bootstrap CI, Wilcoxon for reference.
Also the 3-seed mean (P2_s42/s1/s2) vs the w=0 arm. No GPU.

    .venv/bin/python scripts/v2/p2_model_tests.py
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("t0814", Path(__file__).with_name("t08_14_model_tests.py"))
t0814 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t0814)

HOLD = Path("data/runs/V2_P2_inductive/output/holdout")
t0814.FILES = sorted(os.listdir(HOLD))  # plan-2 holdout order (train_arm reports follow sorted order)
OUT = t0814.RES / "P2_summary"

REF = "P2_s42"
SEEDS = ["P2_s42", "P2_s1", "P2_s2"]
ARM = "P2_w0_s42"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ref = t0814.per_copy(REF)
    arm = t0814.per_copy(ARM)
    seed_pc = [t0814.per_copy(t) for t in SEEDS]
    ref3 = {c: float(np.mean([s[c] for s in seed_pc])) for c in ref}
    out = {
        "reference": REF, "arm": ARM, "n_holdout_files": len(t0814.FILES),
        "ref_mean_acc": float(np.mean(list(ref.values()))),
        "ref3_mean_acc": float(np.mean(list(ref3.values()))),
        "arm_mean_acc": float(np.mean(list(arm.values()))),
        "per_copy": {"ref": ref, "ref3": ref3, "arm": arm},
        "vs_seed42": t0814.paired(ref, arm, rng),
        "vs_3seed_mean": t0814.paired(ref3, arm, rng),
    }
    (OUT / "p2_w0_test.json").write_text(json.dumps(out, indent=1))
    for k in ["vs_seed42", "vs_3seed_mean"]:
        v = out[k]
        print(f"{k:14s} diff {v['mean_diff_pp']:+.1f} pp  CI [{v['boot95_ci_pp'][0]:.1f}, {v['boot95_ci_pp'][1]:.1f}]  "
              f"p_signflip {v['p_signflip_exact']:.4f}  A better in {v['copies_A_better']}/{v['n_copies']}")
    print(f"ref {100*out['ref_mean_acc']:.1f}%  ref3 {100*out['ref3_mean_acc']:.1f}%  w0 {100*out['arm_mean_acc']:.1f}%")


if __name__ == "__main__":
    main()
