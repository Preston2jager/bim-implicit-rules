"""Extract Experiment-6 (round-8 holdout failure anatomy) data.

Source: run 2026JUL122211 (161 copies, 12 holdout copies, scalar_s + pose
distillation w=0.3). Produces data/failure.json with three views:

  1. threshold_curve : slot-error rate as a function of the minimum effective
     wall length involved (wall / semantic rules), plus the window-rule split
     by whether a near-threshold 900 mm wall is present.
  2. confusion       : 6x6 true-vs-predicted slot confusion matrix restricted
     to window-rule graphs.
  3. per_copy        : per-holdout-copy side accuracy, recovered rotation angle
     (from the IfcBuilding placement RefDirection in the augmented IFC) and
     angular distance to the nearest training-copy rotation.
"""
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/runs/2026JUL122211"
OUT = Path(__file__).parent / "data" / "failure.json"

PAT = re.compile(r"room(\d+)_aug(\d+)_L(\w+)_R(\w+)_W(\d+)")
SLOTS = np.array([-2.375, -1.425, -0.475, 0.475, 1.425, 2.375])


def eff(v: str) -> bool:
    return v != "Absent" and int(v) > 1000


def snap(s: float) -> int:
    return int(np.argmin(np.abs(SLOTS - s)))


def ifc_angle_deg(path: Path):
    """Building-level rotation angle from the non-axis-aligned unit IFCDIRECTION."""
    txt = path.read_text(errors="ignore")
    dirs = re.findall(r"IFCDIRECTION\(\(([-0-9.Ee]+),([-0-9.Ee]+),0\.?\)\)", txt)
    for a, b in dirs:
        x, y = float(a), float(b)
        if abs(x * x + y * y - 1) < 1e-6 and (x, y) not in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            return math.degrees(math.atan2(y, x)) % 360
    return None


def main() -> None:
    # NOTE: output/evaluation_results/ of this run was later overwritten by the
    # GINE-baseline eval (exp5_gin_baseline); the Graphormer report lives in the archive.
    rep = json.load(open(RUN / "output/experiments/scalar_s_pose_w0.3/holdout_evaluation_report.json"))
    results = rep["per_door_results"]
    files = sorted((RUN / "output/holdout").glob("graph_room*.json"))
    assert len(files) == len(results)

    rows = []
    for f, r in zip(files, results):
        m = PAT.search(f.name)
        rows.append({
            "aug": int(m[2]), "L": m[3], "R": m[4],
            "t": snap(r["wall_s_true"]), "p": snap(r["wall_s_pred"]),
            "side": bool(r["side_correct"]), "err": r["euclidean_error"],
        })

    # 1. threshold curve -----------------------------------------------------
    curve = {"wall": {}, "semantic": {}}
    for x in rows:
        l, rr = eff(x["L"]), eff(x["R"])
        if not l and not rr:
            continue
        rule = "semantic" if (l and rr) else "wall"
        elens = [int(v) for v, e in ((x["L"], l), (x["R"], rr)) if e]
        key = str(min(elens))
        n_bad = curve[rule].setdefault(key, [0, 0])
        n_bad[0] += 1
        n_bad[1] += int(x["t"] != x["p"])
    window_split = {"with_900mm": [0, 0], "without_900mm": [0, 0]}
    for x in rows:
        if eff(x["L"]) or eff(x["R"]):
            continue
        has900 = "900" in (x["L"], x["R"])
        k = "with_900mm" if has900 else "without_900mm"
        window_split[k][0] += 1
        window_split[k][1] += int(x["t"] != x["p"])
    threshold_curve = {
        rule: {k: {"n": v[0], "bad_rate": v[1] / v[0]} for k, v in sorted(d.items(), key=lambda kv: int(kv[0]))}
        for rule, d in curve.items()
    }
    threshold_curve["window"] = {k: {"n": v[0], "bad_rate": v[1] / v[0]} for k, v in window_split.items()}

    # 2. window-rule slot confusion matrix ----------------------------------
    conf = np.zeros((6, 6), int)
    for x in rows:
        if eff(x["L"]) or eff(x["R"]):
            continue
        conf[x["t"]][x["p"]] += 1

    # 3. per-copy accuracy vs rotation-angle distance ------------------------
    emb = sorted((RUN / "intermediate/embedded_ifcs").glob("*.ifc"))  # aug_idx = 1-based position
    holdout_augs = sorted({x["aug"] for x in rows})
    train_angles = [ifc_angle_deg(emb[i]) for i in range(len(emb)) if (i + 1) not in holdout_augs]
    train_angles = np.array([a for a in train_angles if a is not None])
    per_copy = {}
    for aug in holdout_augs:
        sub = [x for x in rows if x["aug"] == aug]
        ang = ifc_angle_deg(emb[aug - 1])
        d = float(np.abs(((train_angles - ang + 180) % 360) - 180).min()) if ang is not None else None
        per_copy[str(aug)] = {
            "angle_deg": ang,
            "nearest_train_angle_dist_deg": d,
            "side_acc": float(np.mean([x["side"] for x in sub])),
            "frac_gt_0.5m": float(np.mean([x["err"] > 0.5 for x in sub])),
        }

    out = {
        "global": rep["global_metrics"],
        "threshold_curve": threshold_curve,
        "window_confusion": conf.tolist(),
        "per_copy": per_copy,
        "n_train_rotations": int(len(train_angles)),
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}")
    for rule in ("wall", "semantic"):
        print(f"  {rule}: " + " ".join(f"{k}:{v['bad_rate']*100:.1f}%" for k, v in threshold_curve[rule].items()))
    print("  window:", {k: f"{v['bad_rate']*100:.1f}%" for k, v in threshold_curve["window"].items()})


if __name__ == "__main__":
    main()
