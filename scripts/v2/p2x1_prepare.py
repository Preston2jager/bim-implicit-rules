# --- repo-root bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path.cwd()
if not (_ROOT / "src" / "Graphormer.py").exists():
    raise SystemExit("run from the SDF_3 repository root (cd ~/Projects/SDF_3)")
# --- end bootstrap ---
"""Plan 2 / P2-6 step 1 prep: V2_P2_holdfit_x1 = the 12 plan-2 holdout copies WITHOUT their own latents,
to be fitted on plan 1's FROZEN decoder (v2_fit_holdout --run V2_inductive). intermediate/ is a REAL folder
of per-file symlinks (samples, manifest, expanded IFCs), so the fit writes embedded_ifcs/, holdout_latents.npy
and FIT_DONE only into V2_P2_holdfit_x1, never into V2_P2_holdfit or V2_inductive. Config = plan-2 config.

    python scripts/v2/p2x1_prepare.py --src-hold V2_P2_holdfit --dst V2_P2_holdfit_x1
"""
import argparse, os, shutil

ap = argparse.ArgumentParser(); ap.add_argument("--src-hold", required=True); ap.add_argument("--dst", required=True)
a = ap.parse_args()
RUNS = _ROOT / "data/runs"
s, d = RUNS / a.src_hold, RUNS / a.dst
assert (s / "PREP_DONE").exists(), f"{s} not prepared"
assert d.name.startswith("V2_P2"), d
if (d / "PREP_DONE").exists():
    print("already prepared"); raise SystemExit(0)
if d.exists():
    shutil.rmtree(d)
(d / "input/input_config").mkdir(parents=True)
shutil.copy2(s / "input/input_config/config.yaml", d / "input/input_config/config.yaml")
shutil.copy2(s / "split_manifest.json", d / "split_manifest.json")
(d / "intermediate/expanded_ifcs").mkdir(parents=True)
for f in (s / "intermediate/expanded_ifcs").iterdir():
    os.symlink(f.resolve(), d / "intermediate/expanded_ifcs" / f.name)
for f in (s / "intermediate").iterdir():
    if f.name in ("expanded_ifcs", "embedded_ifcs"):
        continue
    os.symlink(f.resolve(), d / "intermediate" / f.name)
shutil.copy2(s / "PREP_DONE", d / "PREP_DONE")
print("prepared", d, sorted(p.name for p in (d / "intermediate").iterdir()))
