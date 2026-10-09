# --- repo-root bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path.cwd()
if not (_ROOT / "src" / "Graphormer.py").exists():
    raise SystemExit("run from the SDF_3 repository root (cd ~/Projects/SDF_3)")
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# --- end bootstrap ---
"""Plan 2 / P2-1 (CPU): build data/runs/P2_src with 80 rigid-transform copies of Data_2.ifc (+ the base).
Only wall variants + copy expansion; OBJ export / sampling happen in v2_prepare. See paper_section/exp11_second_topology/README.md."""
import argparse, shutil
import yaml
from src.DataPreprocessor import DataPreprocessor

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="P2_src")
ap.add_argument("--copies", type=int, default=80, help="80 = main plan-2 run; 160 = copy-count control (P2-08)")
args = ap.parse_args()
wf = _ROOT / "data/runs" / args.name
assert not wf.exists(), "P2_src already exists"
(wf / "input/input_ifcs").mkdir(parents=True); (wf / "input/input_config").mkdir(parents=True)
shutil.copy("data/input/plan2/Data_2.ifc", wf / "input/input_ifcs/Data_2.ifc")
cfg = yaml.safe_load(open("data/input/plan2/config_plan2.yaml"))
cfg["extraction"]["copies"] = args.copies
(wf / "input/input_config/config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
DataPreprocessor.generate_wall_variants_ifc(wf)
p = DataPreprocessor.config_load(wf / "input/input_config/config.yaml")
p.config.extraction.data_expand = True
p._stage_expand_data(wf)
(wf / "intermediate/embedded_ifcs").symlink_to("expanded_ifcs")
n = len(list((wf / "intermediate/expanded_ifcs").glob("*.ifc")))
print("copies:", n)
assert n == args.copies + 1, n
(wf / "SRC_DONE").write_text(str(n))
