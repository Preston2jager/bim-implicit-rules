"""Extract Experiment-1 (prediction-head ablation, legacy 2-slot setting) metrics.

Sources:
  - experiments/exp1_absolute/            (absolute head, run 2026JUL100130)
  - experiments/exp2_offset/              (offset head,   run 2026JUL100130)
  - data/runs/2026JUL100130/output/experiments/scalar_s/
        (scalar_s with the pre-fix axis-product projection, run 2026JUL100130)

Writes data/summary.json with seen/holdout global metrics per head plus the
mirror-signature statistics of the trapped scalar_s head.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "data" / "summary.json"

SOURCES = {
    "absolute": ROOT / "experiments/exp1_absolute",
    "offset": ROOT / "experiments/exp2_offset",
    "scalar_s (axis-product)": ROOT / "data/runs/2026JUL100130/output/experiments/scalar_s",
}


def globals_of(path: Path) -> dict:
    g = json.load(open(path))["global_metrics"]
    return {
        "mean_euclidean_error": g["mean_euclidean_error"],
        "median_euclidean_error": g["median_euclidean_error"],
        "left_right_accuracy": g.get("left_right_accuracy"),
        "n_doors": g["total_doors_evaluated"],
    }


def main() -> None:
    summary = {}
    for head, folder in SOURCES.items():
        seen = globals_of(folder / "test_evaluation_report.json")
        hold = globals_of(folder / "holdout_evaluation_report.json")
        summary[head] = {
            "seen": seen,
            "holdout": hold,
            "gap_ratio": hold["mean_euclidean_error"] / seen["mean_euclidean_error"],
        }

    # Mirror signature of the trapped scalar_s head: s_pred vs s_true.
    rep = json.load(open(SOURCES["scalar_s (axis-product)"] / "holdout_evaluation_report.json"))
    s_true = np.array([r["wall_s_true"] for r in rep["per_door_results"]])
    s_pred = np.array([r["wall_s_pred"] for r in rep["per_door_results"]])
    summary["scalar_s (axis-product)"]["mirror_signature"] = {
        "corr_s_true_s_pred": float(np.corrcoef(s_true, s_pred)[0, 1]),
        "mean_abs_s_pred": float(np.abs(s_pred).mean()),
        "mean_abs_s_true": float(np.abs(s_true).mean()),
    }

    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")
    for head, m in summary.items():
        print(f"  {head:26} seen {m['seen']['mean_euclidean_error']:.3f} m | "
              f"holdout {m['holdout']['mean_euclidean_error']:.3f} m | gap {m['gap_ratio']:.0f}x")


if __name__ == "__main__":
    main()
