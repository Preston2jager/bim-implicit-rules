"""Extract Experiment-5 (GINE message-passing baseline) metrics.

Controlled architecture comparison on the final round-8 data (run 2026JUL122211,
161 copies / 12 holdout transforms): ONLY the graph encoder changes, everything
else identical — same HeterogeneousNodeEncoder (TabTransformer + SDF latent),
same scalar_s head, same pose-distillation aux head (w=0.3), same seed/split,
same 100-epoch budget with the same early-stop rule (rolling val < 1e-3).

  graphormer : dense attention + structural encodings, early-stopped 45/100 ep
               (archive output/experiments/scalar_s_pose_w0.3)
  gin        : GINE (GIN with edge features, Hu et al. 2020), depth 8, mean
               aggregation, bidirectional message passing; ran the full 100 ep
               (archive output/experiments/gin_scalar_s_pose_w0.3)

Two earlier GINE attempts (sum aggregation depth 4; mean depth 8 stopped at
12 ep) never escaped the predict-mean saddle (~4.2); their logs are parsed too.

Writes data/gin_baseline.json: seen/holdout global metrics, per-rule holdout
breakdown, per-copy side accuracy, and the val-loss curves from the logs.
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/runs/2026JUL122211"
OUT = Path(__file__).parent / "data" / "gin_baseline.json"

PAT = re.compile(r"room(\d+)_aug(\d+)_L(\w+)_R(\w+)_W(\d+)")
EPOCH_PAT = re.compile(r"Epoch +(\d+) \| Train MSE: ([\d.]+) \| Val MSE: ([\d.]+)")
SLOTS = np.array([-2.375, -1.425, -0.475, 0.475, 1.425, 2.375])

CONFIGS = {
    "graphormer": RUN / "output/experiments/scalar_s_pose_w0.3",
    "gin": RUN / "output/experiments/gin_scalar_s_pose_w0.3",
}

LOGS = {
    "graphormer": RUN / "intermediate/graphormer_retrain.log",
    "gin": RUN / "intermediate/gin_baseline_retrain.log",
    "gin_attempt_sum_d4": RUN / "intermediate/gin_baseline_attempt1_sum_d4_stalled.log",
    "gin_attempt_mean_d8_12ep": RUN / "intermediate/gin_baseline_attempt2_mean_d8_partial.log",
}


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


def global_metrics(report_path: Path) -> dict:
    g = json.load(open(report_path))["global_metrics"]
    return {
        "side_acc": g["left_right_accuracy"],
        "median_err": g["median_euclidean_error"],
        "mean_err": g["mean_euclidean_error"],
        "rmse": g["rmse_euclidean"],
        "n": g["total_doors_evaluated"],
    }


def holdout_breakdown(report_path: Path, holdout_dir: Path) -> dict:
    rep = json.load(open(report_path))
    results = rep["per_door_results"]
    files = sorted(holdout_dir.glob("graph_room*.json"))
    assert len(files) == len(results), (report_path, len(files), len(results))

    buckets: dict[str, dict[str, list]] = {}
    per_copy: dict[int, list] = {}
    errs = []
    for f, r in zip(files, results):
        m = PAT.search(f.name)
        b = buckets.setdefault(rule_of(m[3], m[4]), {"side": [], "slot": [], "err": []})
        b["side"].append(bool(r["side_correct"]))
        b["slot"].append(snap(r["wall_s_true"]) == snap(r["wall_s_pred"]))
        b["err"].append(r["euclidean_error"])
        per_copy.setdefault(int(m[2]), []).append(bool(r["side_correct"]))
        errs.append(r["euclidean_error"])

    out = {"per_rule": {}, "frac_gt_0.5m": float((np.array(errs) > 0.5).mean())}
    for rule, b in buckets.items():
        out["per_rule"][rule] = {
            "n": len(b["side"]),
            "side_acc": float(np.mean(b["side"])),
            "exact_slot_acc": float(np.mean(b["slot"])),
            "median_err": float(np.median(b["err"])),
        }
    out["exact_slot_acc"] = float(np.mean([s for b in buckets.values() for s in b["slot"]]))
    out["per_copy_side_acc"] = {str(a): float(np.mean(v)) for a, v in sorted(per_copy.items())}
    return out


def parse_curve(log_path: Path) -> dict:
    epochs, train, val = [], [], []
    for m in EPOCH_PAT.finditer(log_path.read_text()):
        epochs.append(int(m[1]))
        train.append(float(m[2]))
        val.append(float(m[3]))
    return {"epoch": epochs, "train_mse": train, "val_mse": val}


def main() -> None:
    out = {}
    for name, arch_dir in CONFIGS.items():
        split = json.load(open(arch_dir / "graphormer_split_metrics.json"))["metrics"]
        out[name] = {
            "best_val_mse": split["best_val_mse"],
            "seen_test": global_metrics(arch_dir / "test_evaluation_report.json"),
            "holdout": global_metrics(arch_dir / "holdout_evaluation_report.json"),
        }
        out[name]["holdout"].update(
            holdout_breakdown(arch_dir / "holdout_evaluation_report.json", RUN / "output/holdout"))
    out["curves"] = {name: parse_curve(p) for name, p in LOGS.items() if p.exists()}

    for name in CONFIGS:
        o = out[name]
        h = o["holdout"]
        print(f"{name}: best val {o['best_val_mse']:.2e} | seen side {o['seen_test']['side_acc']*100:.1f}% "
              f"median {o['seen_test']['median_err']*1000:.1f}mm | holdout side {h['side_acc']*100:.1f}% "
              f"exact-slot {h['exact_slot_acc']*100:.1f}% median {h['median_err']*1000:.1f}mm | "
              + " | ".join(f"{r} {h['per_rule'][r]['side_acc']*100:.1f}% (n={h['per_rule'][r]['n']})"
                           for r in ("window", "wall", "semantic")))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
