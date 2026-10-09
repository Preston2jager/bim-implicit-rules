# --- repo-root bootstrap (this script is symlinked into SDF_3/scripts/v2/) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path.cwd()
if not (_ROOT / "src" / "Graphormer.py").exists():
    raise SystemExit("run from the SDF_3 repository root (cd ~/Projects/SDF_3)")
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# --- end bootstrap ---
"""v2 step 1 (CPU): split the 161 transformed IFCs into TRAIN and HOLDOUT copies, build two run
folders and sample SDF points for each.

  data/runs/<run>/       149 train copies  -> the SDF decoder is trained here and only sees these
  data/runs/<hold-run>/   12 holdout copies -> sampled separately, latents fitted later with a FROZEN decoder

The holdout rule mirrors DataPreprocessor._run_combinatorial_graphs exactly: sorted(*.ifc) of the
source run's embedded_ifcs, enumerate from 1, the last `holdout_copies` are held out. The same 161 IFC
files (same rigid transforms) are reused so v1 and v2 are directly comparable. Geometry is read from the
source run's embedded_ifcs (identical to the expanded copies; OBJs/expanded_ifcs were pruned).

    python scripts/v2/v2_prepare.py --src 2026JUL122211 --run V2_inductive --hold-run V2_holdfit
    (dry run) ... --run V2_TEST --hold-run V2_TESTHOLD --limit-train 3 --limit-hold 2 --set training.hash.epochs=2
"""
import argparse
import json
import os
import shutil

import yaml
from loguru import logger

from src.DataPreprocessor import DataPreprocessor

RUNS = _ROOT / "data" / "runs"
OBJECTS_PER_COPY = int(os.environ.get("OBJECTS_PER_COPY", 52))  # plan 2 (Data_2.ifc): 56


def set_dotted(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def build_run(name: str, ifcs: list, src_cfg: dict, overrides: list) -> _Path:
    wf = RUNS / name
    if wf.exists():
        assert (wf / "PREP_DONE").exists() is False, f"{wf} already prepared"
        assert name.startswith("V2_"), "refusing to delete a non-V2 run folder"
        shutil.rmtree(wf)
    (wf / "input/input_config").mkdir(parents=True)
    cfg = json.loads(json.dumps(src_cfg))
    for kv in overrides:
        k, v = kv.split("=", 1)
        set_dotted(cfg, k, yaml.safe_load(v))
    (wf / "input/input_config/config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    exp = wf / "intermediate/expanded_ifcs"
    exp.mkdir(parents=True)
    for f in ifcs:
        os.symlink(f.resolve(), exp / f.name)
    return wf


def prepare(wf: _Path, n_copies: int) -> None:
    cfg_path = wf / "input/input_config/config.yaml"
    proc = DataPreprocessor.config_load(cfg_path)
    logger.info(f"[{wf.name}] exporting OBJs for {n_copies} copies")
    proc._stage_export_objs(wf)
    logger.info(f"[{wf.name}] sampling SDF points")
    proc._stage_sample_sdf(wf)
    man = json.loads((wf / "intermediate/samples_manifest.json").read_text())
    n_inst = len(man)
    exp_inst = n_copies * OBJECTS_PER_COPY
    logger.info(f"[{wf.name}] instances {n_inst} (expected {exp_inst})")
    assert n_inst == exp_inst, f"instance count {n_inst} != {exp_inst}"
    srcs = {r["source_ifc"] for r in man.values()}
    assert len(srcs) == n_copies, f"{len(srcs)} source ifcs != {n_copies}"
    (wf / "PREP_DONE").write_text(json.dumps({"copies": n_copies, "instances": n_inst}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--hold-run", required=True)
    ap.add_argument("--limit-train", type=int, default=0, help="dry run: only the first N train copies")
    ap.add_argument("--limit-hold", type=int, default=0, help="dry run: only the first N holdout copies")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    logger.remove()
    logger.add(_sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}", level="INFO", colorize=False)

    if (RUNS / args.run / "PREP_DONE").exists() and (RUNS / args.hold_run / "PREP_DONE").exists():
        logger.info("both run folders already prepared; nothing to do.")
        return

    src = RUNS / args.src
    src_cfg = yaml.safe_load((src / "input/input_config/config.yaml").read_text())
    n_hold = int(src_cfg["extraction"]["holdout_copies"])
    files = sorted((src / "intermediate/embedded_ifcs").glob("*.ifc"))
    n = len(files)
    hold_ids = set(range(n - n_hold + 1, n + 1))
    train = [f for i, f in enumerate(files, 1) if i not in hold_ids]
    hold = [f for i, f in enumerate(files, 1) if i in hold_ids]
    logger.info(f"{n} copies in {args.src}: {len(train)} train, {len(hold)} holdout (aug ids {min(hold_ids)}..{max(hold_ids)})")
    assert len(hold) == n_hold and len(train) == n - n_hold

    split = {"src": args.src, "n": n,
             "train": [f.name for f in train], "holdout": [f.name for f in hold],
             "holdout_aug_ids": sorted(hold_ids),
             "rule": "sorted(embedded_ifcs/*.ifc) [lexicographic], enumerate from 1, last holdout_copies held out"}
    if args.limit_train:
        train = train[:args.limit_train]
    if args.limit_hold:
        hold = hold[:args.limit_hold]

    for name, files in ((args.run, train), (args.hold_run, hold)):
        if (RUNS / name / "PREP_DONE").exists():
            logger.info(f"{name} already prepared; skipping.")
            continue
        wf = build_run(name, files, src_cfg, args.set)
        (wf / "split_manifest.json").write_text(json.dumps(split, indent=1))
        prepare(wf, len(files))
    logger.success(f"v2 prep done: {args.run} ({len(train)} copies), {args.hold_run} ({len(hold)} copies)")


if __name__ == "__main__":
    main()
