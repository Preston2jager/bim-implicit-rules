# --- repo-root bootstrap (this script is symlinked into SDF_3/scripts/v2/) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path.cwd()
if not (_ROOT / "src" / "Graphormer.py").exists():
    raise SystemExit("run from the SDF_3 repository root (cd ~/Projects/SDF_3)")
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# --- end bootstrap ---
"""Isolated, non-interactive train + evaluate "arm" on an existing run's graphs.

Creates data/runs/REV_<tag>/ whose output/ holds symlinks to the source run's graph files, so the
paper's archives in the source run are never touched. Trains (60/20/20 split from --seed), evaluates
the seen test split and the unseen-transform holdout, archives the reports and writes summary.json.

    python scripts/v2/train_arm.py --src 2026JUL122211 --tag V1_s1 --seed 1
    python scripts/v2/train_arm.py --src 2026JUL122211 --tag GINE_lr5e-4_d4 --arch gin \
        --set training.graphormer.lr=0.0005 --set training.graphormer.gin_graph_depth=4

Idempotent: if the tag already finished (DONE marker) it exits 0 immediately. A partial REV_<tag>
folder is deleted and the arm restarts from scratch (training has no resume).
"""
import argparse
import gzip
import json
import os
import re
import shutil
from datetime import datetime

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".95")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import yaml  # noqa: E402
from loguru import logger  # noqa: E402

RUNS = _ROOT / "data" / "runs"
REV_RESULTS = _Path(os.environ.get("SDF3_RESULTS", "data/results"))


def set_dotted(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def link_graphs(src_dir: _Path, dst_dir: _Path, include, exclude, max_n=None, sample_seed=0) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    matched = []
    for f in sorted(src_dir.glob("graph_room*.json")):
        if include and not re.search(include, f.name):
            continue
        if exclude and re.search(exclude, f.name):
            continue
        matched.append(f)
    if max_n is not None and len(matched) > max_n:
        import random
        rng = random.Random(sample_seed)
        matched = rng.sample(matched, max_n)
    for f in matched:
        os.symlink(f.resolve(), dst_dir / f.name)
    return len(matched)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source run folder name under data/runs/")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--arch", default=None, choices=[None, "graphormer", "gin", "gat"])
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value config override (YAML value)")
    ap.add_argument("--include", default=None, help="regex: keep only matching TRAIN graph filenames")
    ap.add_argument("--exclude", default=None, help="regex: drop matching TRAIN graph filenames")
    ap.add_argument("--max-train", type=int, default=None, help="randomly subsample TRAIN graphs (after include/exclude) down to this count, for data-volume-matched comparisons (e.g. Experiment 7 rule isolation)")
    ap.add_argument("--extra-holdout-root", default=None, help="folder whose sub-folders are extra holdout sets (e.g. perturbed copies); each is evaluated IN THIS PROCESS right after the main holdout, because re-loading a checkpoint in a new process does not reproduce archived numbers (FINDINGS G9/G13)")
    args = ap.parse_args()

    logger.remove()
    logger.add(_sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}", level="INFO", colorize=False)

    src = RUNS / args.src
    dst = RUNS / f"REV_{args.tag}"
    exp_dir = dst / "output" / "experiments" / args.tag
    if (exp_dir / "DONE").exists():
        logger.info(f"{args.tag} already finished ({exp_dir / 'DONE'}); nothing to do.")
        return
    if dst.exists():
        assert dst.name.startswith("REV_")
        logger.warning(f"Removing partial run folder {dst}")
        shutil.rmtree(dst)

    cfg = yaml.safe_load((src / "input/input_config/config.yaml").read_text())
    if args.seed is not None:
        set_dotted(cfg, "training.reproducibility.seed", args.seed)
    for kv in args.set:
        k, v = kv.split("=", 1)
        set_dotted(cfg, k, yaml.safe_load(v))
    (dst / "input/input_config").mkdir(parents=True)
    cfg_path = dst / "input/input_config/config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    (dst / "intermediate").mkdir()
    (dst / "output/evaluation_results").mkdir(parents=True)

    n_train = link_graphs(src / "output", dst / "output", args.include, args.exclude,
                          max_n=args.max_train, sample_seed=cfg["training"]["reproducibility"]["seed"])
    n_hold = link_graphs(src / "output/holdout", dst / "output/holdout", None, None)
    logger.info(f"arm {args.tag}: {n_train} train graphs, {n_hold} holdout graphs linked from {args.src}; "
                f"seed={cfg['training']['reproducibility']['seed']} arch={args.arch or cfg['training']['graphormer'].get('model_arch')}")

    from src.Graphormer import SDFGraphormerTrainer  # imported late: pulls in JAX

    t0 = datetime.now()
    logger.info("=== [1/3] train ===")
    SDFGraphormerTrainer.execute_with_split(cfg_path, dst, model_arch=args.arch)
    logger.info("=== [2/3] seen test evaluation ===")
    SDFGraphormerTrainer.evaluate_test_set(cfg_path, dst, model_arch=args.arch)
    logger.info("=== [3/3] unseen-transform holdout evaluation ===")
    SDFGraphormerTrainer.evaluate_holdout_set(cfg_path, dst, model_arch=args.arch)

    exp_dir.mkdir(parents=True, exist_ok=True)
    ev = dst / "output/evaluation_results"
    for name in ("test_evaluation_report.json", "holdout_evaluation_report.json"):
        if (ev / name).exists():
            shutil.copy2(ev / name, exp_dir / name)
    if (dst / "output/graphormer_split_metrics.json").exists():
        shutil.copy2(dst / "output/graphormer_split_metrics.json", exp_dir / "graphormer_split_metrics.json")

    hold = json.loads((exp_dir / "holdout_evaluation_report.json").read_text())
    seen = json.loads((exp_dir / "test_evaluation_report.json").read_text())
    gh, gs = hold["global_metrics"], seen["global_metrics"]
    split = json.loads((exp_dir / "graphormer_split_metrics.json").read_text()) if (exp_dir / "graphormer_split_metrics.json").exists() else {}
    summary = {
        "tag": args.tag, "src": args.src, "seed": cfg["training"]["reproducibility"]["seed"],
        "arch": args.arch or cfg["training"]["graphormer"].get("model_arch", "graphormer"),
        "overrides": args.set, "include": args.include, "exclude": args.exclude,
        "n_train_graphs": n_train, "n_holdout_graphs": n_hold,
        "holdout_side_acc": gh["left_right_accuracy"], "holdout_median_err_m": gh["median_euclidean_error"],
        "holdout_mean_err_m": gh["mean_euclidean_error"], "seen_side_acc": gs["left_right_accuracy"],
        "split_metrics": split, "wall_clock_h": round((datetime.now() - t0).total_seconds() / 3600, 2),
        "finished": datetime.now().isoformat(timespec="minutes"),
    }
    out = REV_RESULTS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    shutil.copy2(cfg_path, out / "config.yaml")
    with gzip.open(out / "holdout_evaluation_report.json.gz", "wt") as f:
        json.dump(hold, f)
    if args.extra_holdout_root:
        # Same process as training. output/holdout is swapped for a symlink to each extra set in turn;
        # a failure here must not lose the main result, which is already archived above.
        hold_dir, main_hold = dst / "output" / "holdout", dst / "output" / "holdout_main"
        hold_dir.rename(main_hold)
        extra = {}
        try:
            for cond in sorted(p for p in _Path(args.extra_holdout_root).iterdir() if p.is_dir()):
                logger.info(f"=== extra holdout: {cond.name} ===")
                hold_dir.symlink_to(cond.resolve(), target_is_directory=True)
                try:
                    SDFGraphormerTrainer.evaluate_holdout_set(cfg_path, dst, model_arch=args.arch)
                finally:
                    hold_dir.unlink()
                rep = json.loads((ev / "holdout_evaluation_report.json").read_text())
                with gzip.open(out / f"extra_{cond.name}.json.gz", "wt") as f:
                    json.dump(rep, f)
                m = rep["global_metrics"]
                extra[cond.name] = {"side_acc": m["left_right_accuracy"], "median_err_m": m["median_euclidean_error"]}
                logger.info(f"  {cond.name}: side {extra[cond.name]['side_acc']:.4f}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"extra holdout evaluation failed: {e!r}")
        finally:
            if hold_dir.is_symlink():
                hold_dir.unlink()
            main_hold.rename(hold_dir)
        summary["extra_holdout"] = extra
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (exp_dir / "DONE").write_text(json.dumps(summary))
    logger.success(f"arm {args.tag} done: holdout side {summary['holdout_side_acc']:.4f}, "
                   f"median {summary['holdout_median_err_m']*1000:.1f} mm, {summary['wall_clock_h']} h")


if __name__ == "__main__":
    main()
