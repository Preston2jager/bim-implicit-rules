"""Plan 2 (second topology, R3Q1): collect every P2 number into $SDF3_RESULTS/P2_summary/summary.json.
Reads only result files written by the queue (no retraining, no checkpoint loading). The prose version
(summary.md) is written by hand from this file.

    python3 scripts/v2/p2_summary.py
"""
import json
import os
import statistics as st
from pathlib import Path

RES = Path(os.environ.get("SDF3_RESULTS", "data/results"))
OUT = RES / "P2_summary"
METRICS = ["overall_side_acc", "overall_exact_slot_acc", "frac_err_gt_0.5m",
           "window_side_acc", "wall_side_acc", "semantic_side_acc"]


def load(p):
    return json.load(open(RES / p))


def arm(tag, analysis_tag=None):
    s = load(f"{tag}/summary.json")
    a = load(f"T10_T08/{analysis_tag or tag}/analysis.json")
    cb = a["cluster_bootstrap"]
    pc = a["per_copy"]
    pc = list(pc.values()) if isinstance(pc, dict) else pc
    side = [p.get("side_acc", p.get("overall_side_acc")) if isinstance(p, dict) else p for p in pc]
    return {
        "n_train_graphs": s["n_train_graphs"], "seed": s["seed"], "wall_clock_h": s["wall_clock_h"],
        "seen_side_acc": s["seen_side_acc"], "median_err_m": s["holdout_median_err_m"],
        **{k: cb[k]["mean"] for k in METRICS},
        "side_boot95": cb["overall_side_acc"]["boot95"],
        "side_sd_across_copies": cb["overall_side_acc"]["sd_across_copies"],
        "per_copy_side_range": [min(side), max(side)],
    }


def mean_sd(arms, k):
    v = [a[k] for a in arms]
    return {"mean": st.mean(v), "sd": st.stdev(v), "values": v}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = {t: arm(t) for t in ["P2_s42", "P2_s1", "P2_s2"]}
    tr = load("P2_split/transforms.json")
    tr161 = load("P2_split/transforms_161.json")
    fits = {k: load(f"{k}/fit_report.json") for k in ["V2_P2_fit", "V2_P2x1_fit", "V2_P2_161_fit"]}
    p1 = load("T10_T08/table7_v2_per_seed.json")
    p1_side = {}
    for k, v in p1.items():
        n = sum(r["n"] for r in v.values())
        p1_side[k] = sum(r["side"] * r["n"] for r in v.values()) / n
    p1_seeds = [p1_side[t] for t in ["V2_s42", "V2_s1", "V2_s2", "V2_s3", "V2_s4"]]
    p1_w0 = load("T08_14/model_tests.json")["vs_seed42"]["V2_w0_s42"]
    x2_on_p2 = arm("P1_s42_x2", "P1_s42_x2_on_plan2")  # summary.json fields there describe the plan-1 holdout
    x2_on_p2["median_err_m"] = load("P1_s42_x2/summary.json")["extra_holdout"]["plan2_holdout"]["median_err_m"]
    x2_on_p2.pop("seen_side_acc")

    out = {
        "data": {
            "plan2_81": {"copies": tr["n_copies"], "train_copies": tr["n_train"], "holdout_copies": tr["n_holdout"],
                         "train_graphs": 55476, "holdout_graphs": 9648, "nodes_per_graph": [39, 40],
                         "holdout_nearest_train_deg": tr["holdout_nearest_train_deg"],
                         "holdout_copies_detail": [c for c in tr["copies"] if c["holdout"]]},
            "plan2_161": {"copies": tr161["n_copies"], "train_copies": tr161["n_train"],
                          "holdout_copies": tr161["n_holdout"], "train_graphs": 119796, "holdout_graphs": 9648,
                          "holdout_nearest_train_deg": tr161.get("holdout_nearest_train_deg")},
            "slots_m": [-2.475, -1.485, -0.495, 0.495, 1.485, 2.475],
        },
        "sdf": {
            "plan2_81": {"early_stop_epoch": 72, "train_reference_mae": fits["V2_P2_fit"]["train_reference_mae"],
                         "holdout_fit_mae": fits["V2_P2_fit"]["holdout_fit_mae"], "ratio": fits["V2_P2_fit"]["ratio"],
                         "gate_passed": fits["V2_P2_fit"]["gate_passed"]},
            "plan2_161": {"early_stop_epoch": 56, "train_reference_mae": fits["V2_P2_161_fit"]["train_reference_mae"],
                          "holdout_fit_mae": fits["V2_P2_161_fit"]["holdout_fit_mae"],
                          "ratio": fits["V2_P2_161_fit"]["ratio"], "gate_passed": fits["V2_P2_161_fit"]["gate_passed"]},
            "plan2_on_plan1_decoder": {"ratio": fits["V2_P2x1_fit"]["ratio"],
                                       "holdout_fit_mae": fits["V2_P2x1_fit"]["holdout_fit_mae"],
                                       "train_reference_mae": fits["V2_P2x1_fit"]["train_reference_mae"],
                                       "gate_passed": fits["V2_P2x1_fit"]["gate_passed"]},
            "plan1_reference": {"early_stop_epoch": 40, "ratio": 0.60},
        },
        "p2_4_final_model": {"per_seed": seeds,
                             "three_seed": {k: mean_sd(list(seeds.values()), k) for k in METRICS}},
        "p2_5_w0": {"arm": arm("P2_w0_s42"), "tests": load("P2_summary/p2_w0_test.json"),
                    "plan1_reference_vs_seed42": {k: p1_w0[k] for k in ["mean_diff_pp", "boot95_ci_pp",
                                                                       "p_signflip_exact", "copies_A_better"]}},
        "p2_6_cross_plan": {"plan1_retrain_on_plan1_holdout": arm("P1_s42_x2"),
                            "plan1_retrain_on_plan2_holdout": x2_on_p2,
                            "note": "plan-2 Corridor label and new walls/doors are absent from the plan-1 vocabulary (encoded 0)"},
        "p2_7_probe": "not done: t06_derivability.py hard-codes plan-1 IFC Tags and a plan-1 rule engine",
        "p2_8_161_copies": arm("P2_161_s42"),
        "plan1_reference": {"side_5seed": {"mean": st.mean(p1_seeds), "sd": st.stdev(p1_seeds), "values": p1_seeds},
                            "V2_s42": {"side": 0.808, "exact_slot": 0.708, "window": 0.712, "wall": 0.808,
                                       "semantic": 0.997},
                            "copies": 161, "train_graphs": 119796},
        "notes": [
            "plan-2 arms use training.graphormer.max_device_data_gb=4 (streaming, as plan 1) after a GPU OOM",
            "exact-slot accuracy uses 0.99 m slot spacing for plan-2 holdouts (t10_t08_analysis --slot-spacing 0.99)",
            "P2_161 holdout is a different set of 12 random transforms than the 81-copy holdout",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
