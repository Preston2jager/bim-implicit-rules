# --- repo-root bootstrap (this script is symlinked into SDF_3/scripts/v2/) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path.cwd()
if not (_ROOT / "src" / "Graphormer.py").exists():
    raise SystemExit("run from the SDF_3 repository root (cd ~/Projects/SDF_3)")
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# --- end bootstrap ---
"""v2 step 4 (CPU): merge train-copy and holdout-copy embedded IFCs and generate the combinatorial graphs.

The holdout copies' embedded IFCs (frozen-decoder latents) are symlinked into the train run's
intermediate/embedded_ifcs/, so it holds all 161 files under their original names. Because the
holdout rule is "last N of sorted(*.ifc)" the same 12 transforms end up in output/holdout/ as in v1.
Then the standard generate_combinatorial_graphs runs unchanged.

    python scripts/v2/v2_finalize.py --src 2026JUL122211 --run V2_inductive --hold-run V2_holdfit
"""
import argparse
import json
import os

from loguru import logger

from src.DataPreprocessor import DataPreprocessor

RUNS = _ROOT / "data" / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--hold-run", required=True)
    ap.add_argument("--expect-train", type=int, default=119796)
    ap.add_argument("--expect-holdout", type=int, default=9648)
    args = ap.parse_args()
    logger.remove()
    logger.add(_sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}", level="INFO", colorize=False)

    wf, hw, sw = RUNS / args.run, RUNS / args.hold_run, RUNS / args.src
    assert (hw / "FIT_DONE").exists(), "run v2_fit_holdout.py first"
    if (wf / "GRAPHS_DONE").exists():
        logger.info("graphs already generated; nothing to do.")
        return

    emb = wf / "intermediate/embedded_ifcs"
    for f in sorted((hw / "intermediate/embedded_ifcs").glob("*.ifc")):
        link = emb / f.name
        if not link.exists():
            os.symlink(f.resolve(), link)
    names = sorted(p.name for p in emb.glob("*.ifc"))
    src_names = sorted(p.name for p in (sw / "intermediate/embedded_ifcs").glob("*.ifc"))
    assert names == src_names, "v2 embedded_ifcs file names differ from the source run (would change the holdout copies)"
    logger.info(f"{len(names)} embedded IFCs (same names as {args.src})")

    DataPreprocessor.generate_combinatorial_graphs(wf / "input/input_config/config.yaml", wf)

    n_tr = len(list((wf / "output").glob("graph_room*.json")))
    n_ho = len(list((wf / "output/holdout").glob("graph_room*.json")))
    logger.info(f"graphs: {n_tr} train, {n_ho} holdout (expected {args.expect_train} / {args.expect_holdout})")
    assert n_tr == args.expect_train and n_ho == args.expect_holdout, "graph counts differ from v1"
    (wf / "GRAPHS_DONE").write_text(json.dumps({"train": n_tr, "holdout": n_ho}))
    logger.success("v2 graphs generated.")


if __name__ == "__main__":
    main()
