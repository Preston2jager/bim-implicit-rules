# --- repo-root bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path.cwd()
if not (_ROOT / "src" / "Graphormer.py").exists():
    raise SystemExit("run from the SDF_3 repository root (cd ~/Projects/SDF_3)")
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# --- end bootstrap ---
"""Plan 2 / P2-6 step 3: graphs for the 12 plan-2 holdout copies with PLAN-1-decoder latents, assembled as ONE
extra-holdout set for train_arm.py --extra-holdout-root.

generate_combinatorial_graphs always keeps >=1 copy for training (11 -> output/holdout, 1 -> output/); both
are collected here, so the extra set holds all 12 copies x 804 = 9,648 graphs.

    python scripts/v2/p2x1_graphs.py --hold-run V2_P2_holdfit_x1 --extra-root V2_P2x1_extra
"""
import argparse, json, os, shutil
from src.DataPreprocessor import DataPreprocessor

ap = argparse.ArgumentParser(); ap.add_argument("--hold-run", required=True); ap.add_argument("--extra-root", required=True)
ap.add_argument("--expect", type=int, default=9648)
a = ap.parse_args()
RUNS = _ROOT / "data/runs"
hw, ex = RUNS / a.hold_run, RUNS / a.extra_root
assert (hw / "FIT_DONE").exists(), "fit plan-2 holdout on the plan-1 decoder first"
if (ex / "DONE").exists():
    print("already done"); raise SystemExit(0)
DataPreprocessor.generate_combinatorial_graphs(hw / "input/input_config/config.yaml", hw)
graphs = sorted((hw / "output").glob("graph_room*.json")) + sorted((hw / "output/holdout").glob("graph_room*.json"))
assert len(graphs) == a.expect, f"{len(graphs)} graphs != {a.expect}"
if ex.exists():
    shutil.rmtree(ex)
cond = ex / "plan2_holdout"
cond.mkdir(parents=True)
for f in graphs:
    os.symlink(f.resolve(), cond / f.name)
assert len(list(cond.glob("graph_room*.json"))) == a.expect, "graph file names collide"
(ex / "DONE").write_text(json.dumps({"graphs": len(graphs)}))
print("extra holdout set:", cond, len(graphs))
