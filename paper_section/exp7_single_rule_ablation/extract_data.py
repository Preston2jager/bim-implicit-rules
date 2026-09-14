"""Extract Experiment-7 (single-rule ablation — rule-isolation causal test) data.

Menu-8 ablation on run 2026JUL122211: the round-8 training pool is split by
placement rule (via filename), balanced to the smallest rule (29,204 graphs
= 24% of the joint model's 119,796), and one model per rule is retrained with
the exact final configuration (scalar_s head, pose distillation w=0.3, same
seed/split/early-stop rule). Holdout = the same 12 unseen transforms,
restricted to each rule's graphs (window 4608 / wall 2688 / semantic 2352).

The joint model's per-rule numbers come from the round-8 archive
(output/experiments/scalar_s_pose_w0.3), so both conditions are evaluated on
identical holdout graphs. For the semantic rule the isolated model's TRAINING
set is also identical to the joint model's semantic subset (the balance target
is the semantic count), which makes that comparison a clean causal contrast:
the only difference is the presence of the other rules' graphs.

Writes data/single_rule.json:
  rules.{window,wall,semantic}.{joint,isolated} : n, side/exact-slot accuracy,
      median/mean error, frac >0.5 m, exact-mirror count
  threshold.{rule}.{joint,isolated} : slot-error rate vs decisive wall length
      (min effective length for wall/semantic; min PRESENT length for window,
      'absent' = both walls absent) — same convention as exp6's curve
  isolated.{rule} : best_val_mse, seen-test metrics, per-copy side accuracy
  train_sizes : available per rule + balanced size + joint total
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/runs/2026JUL122211"
OUT = Path(__file__).parent / "data" / "single_rule.json"

PAT = re.compile(r"room(\d+)_aug(\d+)_L(\w+)_R(\w+)_W(\d+)")
SLOTS = np.array([-2.375, -1.425, -0.475, 0.475, 1.425, 2.375])

ABLATIONS = {
    "window": RUN / "output/experiments/ablation_six_window",
    "wall": RUN / "output/experiments/ablation_single_wall",
    "semantic": RUN / "output/experiments/ablation_semantic",
}
JOINT = RUN / "output/experiments/scalar_s_pose_w0.3"


def eff(v: str) -> bool:
    return v != "Absent" and int(v) > 1000


def rule_of(L: str, R: str) -> str:
    l, r = eff(L), eff(R)
    if not l and not r:
        return "window"
    if l and r:
        return "semantic"
    return "wall"


def snap(s: float) -> int:
    return int(np.argmin(np.abs(SLOTS - s)))


def rows_from(report_path: Path, filenames: list[str]) -> list[dict]:
    results = json.load(open(report_path))["per_door_results"]
    names = sorted(Path(f).name for f in filenames)
    assert len(names) == len(results), (report_path, len(names), len(results))
    rows = []
    for name, r in zip(names, results):
        m = PAT.search(name)
        rows.append({
            "rule": rule_of(m[3], m[4]), "aug": int(m[2]), "L": m[3], "R": m[4],
            "s_t": r["wall_s_true"], "s_p": r["wall_s_pred"],
            "t": snap(r["wall_s_true"]), "p": snap(r["wall_s_pred"]),
            "side": bool(r["side_correct"]), "err": r["euclidean_error"],
        })
    return rows


def is_mirror(x: dict) -> bool:
    return (not x["side"]) and abs(x["s_t"]) > 0.4 and abs(x["s_p"] + x["s_t"]) < 0.4


def stats(rows: list[dict]) -> dict:
    errs = np.array([x["err"] for x in rows])
    return {
        "n": len(rows),
        "side_acc": float(np.mean([x["side"] for x in rows])),
        "exact_slot_acc": float(np.mean([x["t"] == x["p"] for x in rows])),
        "median_err": float(np.median(errs)),
        "mean_err": float(errs.mean()),
        "frac_gt_0.5m": float((errs > 0.5).mean()),
        "mirrors": int(sum(is_mirror(x) for x in rows)),
    }


def length_key(x: dict) -> str:
    """Decisive wall length, matching exp6's threshold-curve convention."""
    if x["rule"] == "window":
        present = [int(v) for v in (x["L"], x["R"]) if v != "Absent"]
        return "absent" if not present else str(min(present))
    elens = [int(v) for v in (x["L"], x["R"]) if eff(v)]
    return str(min(elens))


def threshold_curve(rows: list[dict]) -> dict:
    buckets: dict[str, list[int]] = {}
    for x in rows:
        b = buckets.setdefault(length_key(x), [0, 0])
        b[0] += 1
        b[1] += int(x["t"] != x["p"])
    def order(k):
        return -1 if k == "absent" else int(k)
    return {k: {"n": v[0], "bad_rate": v[1] / v[0]}
            for k, v in sorted(buckets.items(), key=lambda kv: order(kv[0]))}


def main() -> None:
    joint_files = [str(p) for p in (RUN / "output/holdout").glob("graph_room*.json")]
    joint_rows = rows_from(JOINT / "holdout_evaluation_report.json", joint_files)

    out = {"rules": {}, "threshold": {}, "isolated": {}, "train_sizes": None}
    for rule, arch in ABLATIONS.items():
        man = json.load(open(arch / "ablation_manifest.json"))
        if out["train_sizes"] is None:
            out["train_sizes"] = {
                "available": man["train_counts_available"],
                "balanced": man["train_size_balanced"],
                "joint_total": sum(man["train_counts_available"].values()),
            }
        iso_rows = rows_from(arch / "holdout_evaluation_report.json", man["holdout_files"])
        jr = [x for x in joint_rows if x["rule"] == rule]
        assert len(jr) == len(iso_rows)

        out["rules"][rule] = {"joint": stats(jr), "isolated": stats(iso_rows)}
        out["threshold"][rule] = {"joint": threshold_curve(jr),
                                  "isolated": threshold_curve(iso_rows)}

        split = json.load(open(arch / "graphormer_split_metrics.json"))["metrics"]
        seen = json.load(open(arch / "test_evaluation_report.json"))["global_metrics"]
        per_copy = {}
        for x in iso_rows:
            per_copy.setdefault(x["aug"], []).append(x["side"])
        out["isolated"][rule] = {
            "best_val_mse": split["best_val_mse"],
            "seen_test": {"side_acc": seen["left_right_accuracy"],
                          "median_err": seen["median_euclidean_error"],
                          "mean_err": seen["mean_euclidean_error"]},
            "per_copy_side_acc": {str(a): float(np.mean(v)) for a, v in sorted(per_copy.items())},
        }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")
    for rule, d in out["rules"].items():
        j, i = d["joint"], d["isolated"]
        print(f"  {rule:8s} side {j['side_acc']*100:5.1f}→{i['side_acc']*100:5.1f}% | "
              f"slot {j['exact_slot_acc']*100:5.1f}→{i['exact_slot_acc']*100:5.1f}% | "
              f"mirrors {j['mirrors']:3d}→{i['mirrors']:3d} | "
              f">0.5m {j['frac_gt_0.5m']*100:4.1f}→{i['frac_gt_0.5m']*100:4.1f}%")


if __name__ == "__main__":
    main()
