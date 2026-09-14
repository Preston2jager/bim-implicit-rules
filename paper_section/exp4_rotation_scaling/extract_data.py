"""Extract Experiment-4 (rotation-diversity scaling) metrics.

Same config (scalar_s head, pose_distill_weight 0.3, de-identified Names);
only the number of rigid-transform copies changes:

  81 copies / 6 holdout : run 2026JUL112325, experiments/scalar_s_pose_w0.3
  161 copies / 12 holdout: run 2026JUL122211 (round 8)

Writes data/scaling.json: overall metrics + per-holdout-copy side accuracy
(the per-copy distribution is the headline uncertainty of this benchmark).
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "data" / "scaling.json"

PAT = re.compile(r"room(\d+)_aug(\d+)_L(\w+)_R(\w+)_W(\d+)")

CONFIGS = {
    "copies81_w0.3": (
        ROOT / "data/runs/2026JUL112325/output/experiments/scalar_s_pose_w0.3/holdout_evaluation_report.json",
        ROOT / "data/runs/2026JUL112325/output/holdout",
    ),
    "copies161_w0.3": (
        # NOTE: output/evaluation_results/ of this run was later overwritten by the
        # GINE-baseline eval (exp5_gin_baseline); the Graphormer report lives in the archive.
        ROOT / "data/runs/2026JUL122211/output/experiments/scalar_s_pose_w0.3/holdout_evaluation_report.json",
        ROOT / "data/runs/2026JUL122211/output/holdout",
    ),
}


def metrics(report_path: Path, holdout_dir: Path) -> dict:
    rep = json.load(open(report_path))
    results = rep["per_door_results"]
    files = sorted(holdout_dir.glob("graph_room*.json"))
    assert len(files) == len(results)
    per_copy: dict[int, dict[str, list]] = {}
    errs = []
    for f, r in zip(files, results):
        aug = int(PAT.search(f.name)[2])
        c = per_copy.setdefault(aug, {"side": [], "err": []})
        c["side"].append(bool(r["side_correct"]))
        c["err"].append(r["euclidean_error"])
        errs.append(r["euclidean_error"])
    errs = np.array(errs)
    g = rep["global_metrics"]
    return {
        "overall_side_acc": g["left_right_accuracy"],
        "median_err": g["median_euclidean_error"],
        "mean_err": g["mean_euclidean_error"],
        "frac_gt_0.5m": float((errs > 0.5).mean()),
        "n_doors": g["total_doors_evaluated"],
        "per_copy_side_acc": {str(a): float(np.mean(c["side"])) for a, c in sorted(per_copy.items())},
    }


def main() -> None:
    out = {}
    for name, (rep, hold) in CONFIGS.items():
        out[name] = metrics(rep, hold)
        o = out[name]
        pc = list(o["per_copy_side_acc"].values())
        print(f"{name}: overall {o['overall_side_acc']*100:.1f}% | >0.5m {o['frac_gt_0.5m']*100:.1f}% | "
              f"per-copy min/max {min(pc)*100:.0f}/{max(pc)*100:.0f}% (n copies {len(pc)})")
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
