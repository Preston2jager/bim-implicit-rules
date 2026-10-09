"""Plan 2 / P2-1: read each copy's IfcBuilding placement (rotation, translation) from data/runs/P2_src and
compute every holdout copy's angular distance to the nearest TRAIN copy (same holdout rule as v2_prepare:
sorted(*.ifc), enumerate from 1, last holdout_copies held out). Writes $SDF3_RESULTS/P2_split/transforms.json."""
import json, math, os, sys
from pathlib import Path
import ifcopenshell, ifcopenshell.util.unit, yaml

src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/runs/P2_src")
out = Path(os.environ.get("SDF3_RESULTS", "data/results")) / "P2_split"
cfg = yaml.safe_load((src / "input/input_config/config.yaml").read_text())
n_hold = int(cfg["extraction"]["holdout_copies"])
files = sorted((src / "intermediate/expanded_ifcs").glob("*.ifc"))
rows = []
for i, f in enumerate(files, 1):
    m = ifcopenshell.open(str(f))
    us = ifcopenshell.util.unit.calculate_unit_scale(m)
    b = (m.by_type("IfcBuilding") or m.by_type("IfcSite"))[0]
    a = b.ObjectPlacement.RelativePlacement
    rd = a.RefDirection.DirectionRatios if a.RefDirection else (1.0, 0.0, 0.0)
    loc = a.Location.Coordinates
    rows.append({"i": i, "file": f.name, "angle_deg": math.degrees(math.atan2(rd[1], rd[0])) % 360,
                 "tx_m": loc[0] * us, "ty_m": loc[1] * us, "holdout": i > len(files) - n_hold})
train = [r["angle_deg"] for r in rows if not r["holdout"]]
d = lambda a, b: min(abs(a - b) % 360, 360 - abs(a - b) % 360)
for r in rows:
    if r["holdout"]:
        r["nearest_train_deg"] = min(d(r["angle_deg"], t) for t in train)
hold = [r for r in rows if r["holdout"]]
nn = sorted(r["nearest_train_deg"] for r in hold)
summary = {"n_copies": len(rows), "n_train": len(train), "n_holdout": len(hold),
           "holdout_nearest_train_deg": {"min": nn[0], "median": nn[len(nn) // 2], "max": nn[-1], "mean": sum(nn) / len(nn)},
           "copies": rows}
out.mkdir(parents=True, exist_ok=True)
(out / (sys.argv[2] if len(sys.argv) > 2 else "transforms.json")).write_text(json.dumps(summary, indent=1))
for r in hold:
    print(f"{r['file']:<28} angle {r['angle_deg']:7.2f}  t=({r['tx_m']:+.2f},{r['ty_m']:+.2f}) m  nearest train {r['nearest_train_deg']:.2f} deg")
print(json.dumps(summary["holdout_nearest_train_deg"]))
