"""Extract Experiment-3 (pose-distillation weight sweep) metrics.

All four variants share run 2026JUL112325's data (81 copies, 6 holdout copies,
4824 holdout graphs, de-identified Names, scalar_s head); only
training.graphormer.pose_distill_weight differs:

  w=0 (run A) : 2026JUL112325_good_1 evaluation_results   (round-5 original)
  w=0 (run B) : experiments/scalar_s_rerun2_nopose        (accidental identical
                rerun -> measures run-to-run noise on identical data/config)
  w=0.1       : experiments/scalar_s_pose_w0.1
  w=0.3       : experiments/scalar_s_pose_w0.3

Writes data/sweep.json: per-variant overall + per-rule side accuracy and
six-window exact-slot accuracy.
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "data" / "sweep.json"

PAT = re.compile(r"room(\d+)_aug(\d+)_L(\w+)_R(\w+)_W(\d+)")
SLOTS = np.array([-2.375, -1.425, -0.475, 0.475, 1.425, 2.375])


def rule_of(L: str, R: str) -> str:
    def eff(v):
        return v != "Absent" and int(v) > 1000
    l, r = eff(L), eff(R)
    if not l and not r:
        return "window"
    if l and r:
        return "semantic"
    return "wall"


def snap(s: float) -> int:
    return int(np.argmin(np.abs(SLOTS - s)))


def metrics(report_path: Path, holdout_dir: Path) -> dict:
    rep = json.load(open(report_path))
    results = rep["per_door_results"]
    files = sorted(holdout_dir.glob("graph_room*.json"))
    assert len(files) == len(results)
    buckets: dict[str, dict[str, list]] = {}
    for f, r in zip(files, results):
        m = PAT.search(f.name)
        b = buckets.setdefault(rule_of(m[3], m[4]), {"side": [], "slot": [], "err": []})
        b["side"].append(bool(r["side_correct"]))
        b["slot"].append(snap(r["wall_s_true"]) == snap(r["wall_s_pred"]))
        b["err"].append(r["euclidean_error"])
    out = {
        "overall_side_acc": rep["global_metrics"]["left_right_accuracy"],
        "overall_median_err": rep["global_metrics"]["median_euclidean_error"],
    }
    for rule, b in buckets.items():
        out[rule] = {
            "n": len(b["side"]),
            "side_acc": float(np.mean(b["side"])),
            "exact_slot_acc": float(np.mean(b["slot"])),
            "median_err": float(np.median(b["err"])),
        }
    return out


R5 = ROOT / "data/runs/2026JUL112325"
CONFIGS = {
    "w0_runA": (ROOT / "data/runs/2026JUL112325_good_1/output/evaluation_results/holdout_evaluation_report.json",
                ROOT / "data/runs/2026JUL112325_good_1/output/holdout"),
    "w0_runB": (R5 / "output/experiments/scalar_s_rerun2_nopose/holdout_evaluation_report.json",
                R5 / "output/holdout"),
    "w0.1": (R5 / "output/experiments/scalar_s_pose_w0.1/holdout_evaluation_report.json",
             R5 / "output/holdout"),
    "w0.3": (R5 / "output/experiments/scalar_s_pose_w0.3/holdout_evaluation_report.json",
             R5 / "output/holdout"),
}


def main() -> None:
    out = {}
    for name, (rep, hold) in CONFIGS.items():
        out[name] = metrics(rep, hold)
        o = out[name]
        print(f"{name:8} overall {o['overall_side_acc']*100:.1f}% | " + " | ".join(
            f"{r} {o[r]['side_acc']*100:.1f}%" for r in ("window", "wall", "semantic")) +
            f" | window exact-slot {o['window']['exact_slot_acc']*100:.1f}%")
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
