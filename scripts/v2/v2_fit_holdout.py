# --- repo-root bootstrap (this script is symlinked into SDF_3/scripts/v2/) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path.cwd()
if not (_ROOT / "src" / "Graphormer.py").exists():
    raise SystemExit("run from the SDF_3 repository root (cd ~/Projects/SDF_3)")
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# --- end bootstrap ---
"""v2 step 3 (GPU): fit latents of UNSEEN copies with a FROZEN decoder (auto-decoder inference).

Loads the decoder trained on the train copies (<run>/checkpoints_hash), freezes the hash grid, the MLP and
the grid bbox, and optimises ONLY a fresh latent table for the instances of <hold-run>. No pose-probe /
centroid loss is used (coordinates never touch these latents), only L1 SDF + the same latent L2 as training.

Quality gate: the held-out mean L1 must be <= GATE x the same decoder's L1 on random training points.
If the gate fails the script exits 3 and does NOT write the IFCs, so the queue stops before graph
generation. The report goes to $SDF3_RESULTS/V2_fit/fit_report.json.

    python scripts/v2/v2_fit_holdout.py --run V2_inductive --hold-run V2_holdfit
"""
import argparse
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".95")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax.training import checkpoints  # noqa: E402
from loguru import logger  # noqa: E402

from src.Embedding import DeviceBatcher, load_flattened_samples  # noqa: E402
from src.EmbeddingHash import SDFHashEmbeddingTrainer, SpatiaSDFHashNet  # noqa: E402

RUNS = _ROOT / "data" / "runs"
REPORT_DIR = _Path(os.environ.get("SDF3_RESULTS", "data/results")) / "V2_fit"


def build_model(p: dict, n_inst: int, use_probe: bool, bmin, bmax) -> SpatiaSDFHashNet:
    return SpatiaSDFHashNet(
        num_instances=n_inst, latent_size=p["latent_size"], levels=p["levels"],
        features_per_level=p["features_per_level"], log2_table_size=p["log2_table_size"],
        base_resolution=p["base_resolution"], max_resolution=p["max_resolution"],
        inner_dim=p["inner_dim"], num_layers=p["num_layers"], use_pose_probe=use_probe,
        bbox_min=tuple(float(v) for v in bmin), bbox_max=tuple(float(v) for v in bmax))


def train_reference_mae(run_wf, p, tparams, bmin, bmax, n_train_inst, n_chunks=2000, chunk=2048, seed=0) -> float:
    """Same decoder + its own trained latents, on random TRAIN points (the number the holdout must match)."""
    idx, pts, sdf = load_flattened_samples(run_wf / "intermediate")
    rng = np.random.default_rng(seed)
    n = idx.shape[0]
    starts = rng.integers(0, n - chunk, size=n_chunks)
    sel = np.concatenate([np.arange(s, s + chunk) for s in np.sort(starts)])
    ii, pp, ss = np.asarray(idx[sel]), np.asarray(pts[sel]), np.asarray(sdf[sel])
    model = build_model(p, n_train_inst, True, bmin, bmax)
    params = jax.tree_util.tree_map(jnp.asarray, tparams)
    apply = jax.jit(lambda i, x: model.apply({"params": params}, i, x)[0])
    errs = []
    for a in range(0, len(sel), 524288):
        pred = apply(jnp.asarray(ii[a:a + 524288]), jnp.asarray(pp[a:a + 524288]))
        errs.append(np.abs(np.asarray(jnp.squeeze(pred)) - ss[a:a + 524288]))
    return float(np.mean(np.concatenate(errs)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="train run folder (decoder checkpoint lives here)")
    ap.add_argument("--hold-run", required=True, help="folder with the held-out copies' samples")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--gate", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-dir", default=str(REPORT_DIR), help="where fit_report.json goes (plan 2: $SDF3_RESULTS/V2_P2_fit)")
    args = ap.parse_args()
    logger.remove()
    logger.add(_sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}", level="INFO", colorize=False)

    wf, hw = RUNS / args.run, RUNS / args.hold_run
    assert (wf / "SDF_TRAIN_DONE").exists(), "train the decoder first (v2_sdf_train.py)"
    if (hw / "FIT_DONE").exists():
        logger.info("holdout latents already fitted; nothing to do.")
        return
    cfg_path = wf / "input/input_config/config.yaml"
    trainer = SDFHashEmbeddingTrainer.config_load(cfg_path)
    p = trainer._hash_profile()

    best = checkpoints.restore_checkpoint(ckpt_dir=str((wf / "checkpoints_hash").resolve()), target=None, prefix="best_")
    assert best is not None, "no decoder checkpoint"
    tparams = best["params"]
    n_train_inst = int(np.asarray(tparams["latent_embedding"]["embedding"]).shape[0])
    frozen = {k: jax.tree_util.tree_map(jnp.asarray, v) for k, v in tparams.items()
              if k not in ("latent_embedding", "pose_probe")}
    bbox = json.loads((wf / "intermediate/hash_grid_bbox.json").read_text())
    bmin, bmax = np.array(bbox["bbox_min"], np.float32), np.array(bbox["bbox_max"], np.float32)
    logger.info(f"decoder: {n_train_inst} train latents; frozen keys {sorted(frozen)}; bbox {bmin.tolist()}..{bmax.tolist()}")

    h_idx, h_pts, h_sdf = load_flattened_samples(hw / "intermediate")
    manifest = json.loads((hw / "intermediate/samples_manifest.json").read_text())
    n_hold = len(manifest)
    logger.info(f"holdout: {n_hold} instances, {h_pts.shape[0]} points")

    # fraction of holdout points outside the (train-only) grid bbox: those get clipped by the encoder
    out = 0
    for a in range(0, h_pts.shape[0], 16_777_216):
        blk = np.asarray(h_pts[a:a + 16_777_216])
        out += int(np.sum(np.any((blk < bmin) | (blk > bmax), axis=1)))
    frac_out = out / h_pts.shape[0]
    logger.info(f"holdout points outside the train bbox: {frac_out:.4%}")

    model = build_model(p, n_hold, False, bmin, bmax)
    rng = jax.random.PRNGKey(args.seed)
    init = model.init(rng, jnp.zeros((1,), jnp.int32), jnp.zeros((1, 3), jnp.float32))["params"]
    emb = jnp.asarray(init["latent_embedding"]["embedding"])
    reg_weight = float(p["reg_weight"])

    batcher = DeviceBatcher(h_idx, h_pts, h_sdf, int(p["batch_size"]), args.seed, 10.0)
    spe = batcher.steps_per_epoch
    sched = optax.exponential_decay(args.lr, transition_steps=spe * 5, decay_rate=0.9)
    tx = optax.adam(sched)
    opt_state = tx.init(emb)

    @jax.jit
    def step(emb, opt_state, frozen, idx, xyz, sdf):
        def loss_fn(e):
            params = {**frozen, "latent_embedding": {"embedding": e}}
            pred, lat = model.apply({"params": params}, idx, xyz)
            l1 = jnp.mean(jnp.abs(jnp.squeeze(pred) - jnp.squeeze(sdf)))
            return l1 + reg_weight * jnp.mean(jnp.linalg.norm(lat, axis=-1) ** 2), l1
        (_, l1), g = jax.value_and_grad(loss_fn, has_aux=True)(emb)
        upd, opt_state = tx.update(g, opt_state)
        return optax.apply_updates(emb, upd), opt_state, l1

    hist, best_mae, best_emb, since = [], float("inf"), emb, 0
    es_thr, es_win = float(p["early_stop_mae"]), int(p["early_stop_window"])
    for epoch in range(args.epochs):
        t0 = time.time()
        ls = []
        for idx, xyz, sdf in batcher.epoch_batches(epoch):
            emb, opt_state, l1 = step(emb, opt_state, frozen, idx, xyz, sdf)
            ls.append(l1)
        mae = float(jnp.mean(jnp.stack(ls)))
        hist.append(mae)
        logger.info(f"Epoch {epoch + 1:04d} | holdout-fit MAE {mae:.6f} | {time.time() - t0:.1f}s")
        if mae < best_mae * 0.99:
            best_mae, best_emb, since = mae, emb, 0
        else:
            since += 1
        if es_thr > 0 and len(hist) >= es_win and sum(hist[-es_win:]) / es_win < es_thr:
            logger.success(f"early stop: mean MAE of last {es_win} epochs < {es_thr:g}")
            break
        if since >= args.patience:
            logger.info(f"plateau: no >1% improvement for {args.patience} epochs")
            break
    emb = best_emb

    # per-instance MAE with the final latents
    n = h_pts.shape[0]
    sums, cnts = jnp.zeros(n_hold), jnp.zeros(n_hold)
    eval_params = lambda e: {**frozen, "latent_embedding": {"embedding": e}}  # noqa: E731
    ev = jax.jit(lambda e, idx, xyz: jnp.squeeze(model.apply({"params": eval_params(e)}, idx, xyz)[0]))
    for idx, xyz, sdf in batcher.epoch_batches(10_000):
        err = jnp.abs(ev(emb, idx, xyz) - jnp.squeeze(sdf))
        sums, cnts = sums.at[idx].add(err), cnts.at[idx].add(1.0)
    per_inst = np.asarray(sums / jnp.maximum(cnts, 1.0))
    hold_mae = float(np.mean(per_inst))

    train_mae = train_reference_mae(wf, p, tparams, bmin, bmax, n_train_inst)
    ratio = hold_mae / train_mae
    passed = ratio <= args.gate
    by_class = {}
    for k, rec in manifest.items():
        by_class.setdefault(rec["ifc_class"], []).append(per_inst[int(k)])
    report = {
        "run": args.run, "hold_run": args.hold_run, "n_hold_instances": n_hold, "n_hold_points": int(n),
        "holdout_fit_mae": hold_mae, "train_reference_mae": train_mae, "ratio": ratio, "gate": args.gate,
        "gate_passed": bool(passed), "epochs_run": len(hist), "final_epoch_mae": hist[-1],
        "per_instance_mae": {"median": float(np.median(per_inst)), "p95": float(np.percentile(per_inst, 95)),
                             "max": float(per_inst.max())},
        "per_class_mean_mae": {c: float(np.mean(v)) for c, v in sorted(by_class.items())},
        "frac_points_outside_train_bbox": frac_out, "lr": args.lr, "seed": args.seed,
        "mae_history": [round(h, 6) for h in hist],
    }
    report_dir = _Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "fit_report.json").write_text(json.dumps(report, indent=1))
    np.save(hw / "holdout_latents.npy", np.asarray(emb))
    logger.info(f"holdout-fit MAE {hold_mae:.6f} vs train reference {train_mae:.6f} (ratio {ratio:.2f}, gate {args.gate}) "
                f"-> {'PASS' if passed else 'FAIL'}")
    if not passed:
        logger.error("quality gate FAILED: latents not written to IFCs. Inspect fit_report.json.")
        raise SystemExit(3)

    trainer._write_latents_to_ifcs(hw, np.asarray(emb))
    n_emb = len(list((hw / "intermediate/embedded_ifcs").glob("*.ifc")))
    (hw / "FIT_DONE").write_text(json.dumps({"embedded_ifcs": n_emb, "ratio": ratio}))
    logger.success(f"holdout latents fitted (frozen decoder) and written into {n_emb} IFCs.")


if __name__ == "__main__":
    main()
