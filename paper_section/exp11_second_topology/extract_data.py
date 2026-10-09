"""Extract Experiment-11 (second plan topology, Data_2.ifc) numbers into data/plan2.json.

Reads the results directory written by scripts/v2/ (default data/results, override with SDF3_RESULTS):
  P2_summary/summary.json            all headline numbers (from scripts/v2/p2_summary.py)
  T10_T08/<tag>/analysis.json        per-holdout-copy accuracy of every arm

Arms: P2_s42 / P2_s1 / P2_s2 (plan 2, 81 copies, w=0.3), P2_w0_s42 (no distillation),
P2_161_s42 (plan 2, 161 copies), P1_s42_x2 (plan-1 retrain on its own holdout) and
P1_s42_x2_on_plan2 (the same model, zero-shot on the plan-2 holdout).
The per-copy numbers are enough to redo every CI and paired test without the run archives.
"""
import json
import os
from pathlib import Path

RES = Path(os.environ.get("SDF3_RESULTS", "data/results"))
OUT = Path(__file__).parent / "data" / "plan2.json"
ARMS = ["P2_s42", "P2_s1", "P2_s2", "P2_w0_s42", "P2_161_s42", "P1_s42_x2", "P1_s42_x2_on_plan2"]


def main() -> None:
    summary = json.loads((RES / "P2_summary/summary.json").read_text())
    per_copy = {}
    for tag in ARMS:
        pc = json.loads((RES / "T10_T08" / tag / "analysis.json").read_text())["per_copy"]
        per_copy[tag] = {c: {k: v[k] for k in ("side_acc", "exact_slot_acc", "frac_err_gt_0.5m", "n")}
                         for c, v in pc.items()}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "per_copy": per_copy}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
