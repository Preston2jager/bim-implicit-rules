# --- repo-root bootstrap (this script is symlinked into SDF_3/scripts/v2/) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path.cwd()
if not (_ROOT / "src" / "Graphormer.py").exists():
    raise SystemExit("run from the SDF_3 repository root (cd ~/Projects/SDF_3)")
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# --- end bootstrap ---
"""v2 step 2 (GPU): train the hash-grid SDF auto-decoder on the TRAIN copies only.

The run folder built by v2_prepare.py contains samples for the 149 training copies and nothing else, so
the existing SDFHashEmbeddingTrainer needs no changes: the bbox, the pose-probe centroid table and every
latent row involve training instances only. Latents of the training copies are written back into
<run>/intermediate/embedded_ifcs/.

    python scripts/v2/v2_sdf_train.py --run V2_inductive
"""
import argparse
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".95")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json  # noqa: E402

from loguru import logger  # noqa: E402

RUNS = _ROOT / "data" / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    logger.remove()
    logger.add(_sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}", level="INFO", colorize=False)

    wf = RUNS / args.run
    assert (wf / "PREP_DONE").exists(), f"run v2_prepare.py first ({wf})"
    prep = json.loads((wf / "PREP_DONE").read_text())
    done = wf / "SDF_TRAIN_DONE"
    if done.exists():
        logger.info("SDF decoder already trained; nothing to do.")
        return

    from src.EmbeddingHash import SDFHashEmbeddingTrainer

    SDFHashEmbeddingTrainer.execute(wf / "input/input_config/config.yaml", wf)

    n_emb = len(list((wf / "intermediate/embedded_ifcs").glob("*.ifc")))
    assert n_emb == prep["copies"], f"{n_emb} embedded IFCs written, expected {prep['copies']}"
    done.write_text(json.dumps({"embedded_ifcs": n_emb}))
    logger.success(f"v2 SDF decoder trained on {n_emb} train copies; latents written back.")


if __name__ == "__main__":
    main()
