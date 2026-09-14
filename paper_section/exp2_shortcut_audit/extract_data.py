"""Extract Experiment-2 (shortcut audit / Name de-identification) metrics.

Per-rule holdout side accuracy before vs after stripping instance identity from
the Name categorical feature:
  - before: run 2026JUL110024 (six-window data, Name still carries
            instance numbers + _L#### suffixes), heads scalar_s & scalar_mag_side
  - after : run 2026JUL112325_good_1 (round 5, de-identified Names, 81 copies)

Rules (mirrors DataPreprocessor.determine_door_position):
  window   = neither changing wall effective (present AND >1000 mm)
  wall     = exactly one effective wall
  semantic = both effective (toilet side decides)

Writes data/per_rule.json.
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "data" / "per_rule.json"

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


def per_rule_metrics(report_path: Path, holdout_dir: Path) -> dict:
    rep = json.load(open(report_path))
    results = rep["per_door_results"]
    files = sorted(holdout_dir.glob("graph_room*.json"))
    assert len(files) == len(results), (report_path, len(files), len(results))

    buckets: dict[str, dict[str, list]] = {}
    for f, r in zip(files, results):
        m = PAT.search(f.name)
        rule = rule_of(m[3], m[4])
        b = buckets.setdefault(rule, {"side": [], "slot": [], "err": []})
        b["side"].append(bool(r["side_correct"]))
        b["slot"].append(snap(r["wall_s_true"]) == snap(r["wall_s_pred"]))
        b["err"].append(r["euclidean_error"])

    out = {"overall": {
        "side_acc": rep["global_metrics"]["left_right_accuracy"],
        "median_err": rep["global_metrics"]["median_euclidean_error"],
        "mean_err": rep["global_metrics"]["mean_euclidean_error"],
        "n": rep["global_metrics"]["total_doors_evaluated"],
    }}
    for rule, b in buckets.items():
        out[rule] = {
            "n": len(b["side"]),
            "side_acc": float(np.mean(b["side"])),
            "exact_slot_acc": float(np.mean(b["slot"])),
            "median_err": float(np.median(b["err"])),
        }
    return out


CONFIGS = {
    "leak_scalar_s": (
        ROOT / "data/runs/2026JUL110024/output/experiments/scalar_s/holdout_evaluation_report.json",
        ROOT / "data/runs/2026JUL110024/output/holdout",
    ),
    "leak_scalar_mag_side": (
        ROOT / "data/runs/2026JUL110024/output/experiments/scalar_mag_side/holdout_evaluation_report.json",
        ROOT / "data/runs/2026JUL110024/output/holdout",
    ),
    "deidentified_round5": (
        ROOT / "data/runs/2026JUL112325_good_1/output/evaluation_results/holdout_evaluation_report.json",
        ROOT / "data/runs/2026JUL112325_good_1/output/holdout",
    ),
}

# Ceiling facts established by the derivability study (scratch scripts
# verify_direction_derivable.py; see experiment.md): a linear ridge probe
# z -> centroid fitted only on training copies, plus topology and the allowed
# categorical features, reproduces all three rules on 100% of holdout graphs.
CEILING = {
    "linear_probe_centroid_R2_holdout": 0.9994,
    "linear_probe_centroid_err_m": 0.113,
    "hand_pipeline_holdout_acc": 1.0,
}


def main() -> None:
    out = {"ceiling": CEILING}
    for name, (rep, hold) in CONFIGS.items():
        out[name] = per_rule_metrics(rep, hold)
        o = out[name]
        print(f"{name}: overall {o['overall']['side_acc']*100:.1f}% | " + " | ".join(
            f"{r} {o[r]['side_acc']*100:.1f}% (n={o[r]['n']})"
            for r in ("window", "wall", "semantic") if r in o))
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
