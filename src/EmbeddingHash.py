"""Hash-grid (Instant-NGP style) variant of the pose-conditioned DeepSDF auto-decoder.

Design intent (see experiment.md / instructions.md):
  The per-instance latent MUST keep encoding shape+size+POSE, exactly like the
  plain-MLP DeepSDF in Embedding.py. That property comes from the auto-decoder
  setup, not from the coordinate encoding:

    - sample points stay in (scaled) WORLD coordinates  -> pose is in the data
    - ONE hash grid + ONE small MLP are SHARED by all instances
    - only the per-instance latent z differs

  For the same query x, different instances have different SDFs, so z is still
  the only channel that can disambiguate them -- it is forced to encode the
  instance's full posed geometry, same as before. The multi-resolution hash
  grid merely replaces "raw xyz into a 6x512 MLP" with "interpolated grid
  features into a small MLP", which is where the ~10-20x per-point compute
  saving comes from.

  This is deliberately NOT the native Instant-NGP usage (one grid per object):
  that stores the geometry in the grid and leaves no meaningful per-object
  latent to feed the Graphormer.

Drop-in compatibility:
  - latent table parameter is named 'latent_embedding' so the existing
    _write_latents_to_ifcs / write_embeddings_only extraction path
    (params['latent_embedding']['embedding']) works unchanged.
  - reads the same intermediate sample data (flat samples_points/sdf/obj_idx.npy,
    with legacy samples_dict.npy fallback) + idx_str2int_dict.npy.
  - checkpoints go to work_folder/checkpoints_hash (separate from the MLP
    version so the two backends never clobber each other).
  - normalization bbox is saved to intermediate/hash_grid_bbox.json so any
    later reconstruction uses the identical mapping.

Config block (training.hash), all optional with defaults:
    training:
        hash:
            epochs: 100
            batch_size: 524288
            lr_model: 0.001
            lr_decay_steps: 50
            lr_decay_rate: 0.95
            reg_weight: 0.0001
            table_reg: 0.000001   # L2 on hash-table entries; pulls sample-free cells to 0
            network:
                latent_size: 128
                levels: 16
                features_per_level: 2
                log2_table_size: 19
                base_resolution: 16
                max_resolution: 512
                inner_dim: 128
                num_layers: 3
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state, checkpoints

from src.Embedding import (
    SDFEmbeddingTrainer, load_flattened_samples, compute_object_centroids, DeviceBatcher,
)


# =====================================================================
# 1. Multi-resolution hash grid encoding
# =====================================================================

# Spatial-hash primes from Instant-NGP (Mueller et al. 2022).
_HASH_PRIMES = np.array([1, 2654435761, 805459861], dtype=np.uint32)

# The 8 corner offsets of a unit cube, fixed order.
_CORNERS = np.array(
    [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int32
)  # [8, 3]


class HashGridEncoding(nn.Module):
    """Shared multi-resolution hash grid: R^3 (normalized to [0,1]^3) -> R^{L*F}.

    One table per level, all levels stored in a single [L, T, F] parameter.
    Kept in float32 -- grid features are sensitive to quantization noise
    (a common cause of poor reconstructions when run in bf16).
    """
    levels: int = 16
    features_per_level: int = 2
    log2_table_size: int = 19
    base_resolution: int = 16
    max_resolution: int = 1024

    @nn.compact
    def __call__(self, x01):
        """x01: [..., 3] coordinates already normalized into [0, 1]^3."""
        L = self.levels
        T = 1 << self.log2_table_size
        F = self.features_per_level

        table = self.param(
            "table",
            nn.initializers.uniform(scale=1e-4),
            (L, T, F),
            jnp.float32,
        )

        # Per-level resolutions with geometric growth (Instant-NGP eq. 2-3).
        if L > 1:
            b = float(np.exp((np.log(self.max_resolution) - np.log(self.base_resolution)) / (L - 1)))
        else:
            b = 1.0
        resolutions = [int(np.floor(self.base_resolution * (b ** l))) for l in range(L)]

        primes = jnp.asarray(_HASH_PRIMES)          # [3] uint32
        corners = jnp.asarray(_CORNERS)             # [8, 3] int32

        feats = []
        for lvl in range(L):
            res = resolutions[lvl]
            xl = x01 * res                                        # [..., 3]
            x0 = jnp.floor(xl).astype(jnp.int32)                  # [..., 3]
            w = xl - x0                                           # [..., 3] in [0,1)

            corner_coords = x0[..., None, :] + corners            # [..., 8, 3]
            # Spatial hash: xor of coordinate*prime, modulo table size.
            h = (corner_coords.astype(jnp.uint32) * primes[None, :])
            idx = (h[..., 0] ^ h[..., 1] ^ h[..., 2]) % jnp.uint32(T)  # [..., 8]
            corner_feats = table[lvl][idx]                        # [..., 8, F]

            # Trilinear interpolation weights per corner.
            cw = jnp.where(corners[..., :] == 1, w[..., None, :], 1.0 - w[..., None, :])
            cw = jnp.prod(cw, axis=-1)                            # [..., 8]
            feats.append(jnp.sum(corner_feats * cw[..., None], axis=-2))  # [..., F]

        return jnp.concatenate(feats, axis=-1)                    # [..., L*F]


class SpatiaSDFHashNet(nn.Module):
    """Auto-decoder SDF net: shared hash encoding + shared small MLP + per-instance latent.

    Input to the MLP is [z, gamma(x), x]: raw x is appended so the network keeps
    a smooth global signal alongside the discrete grid features.
    """
    num_instances: int
    latent_size: int = 128
    levels: int = 16
    features_per_level: int = 2
    log2_table_size: int = 19
    base_resolution: int = 16
    max_resolution: int = 1024
    inner_dim: int = 128
    num_layers: int = 3
    output_dim: int = 1
    use_pose_probe: bool = False
    # Normalization bbox (set from data before model creation; static fields).
    bbox_min: tuple = (-1.0, -1.0, -1.0)
    bbox_max: tuple = (1.0, 1.0, 1.0)

    @nn.compact
    def __call__(self, obj_indices, xyz, external_latent=None):
        # 1. Latent handling -- parameter name matches Embedding.py so the
        #    write-back path (params['latent_embedding']['embedding']) is identical.
        if external_latent is not None:
            latents = external_latent
        else:
            obj_indices = jnp.asarray(obj_indices)
            latents = nn.Embed(
                num_embeddings=self.num_instances,
                features=self.latent_size,
                embedding_init=nn.initializers.normal(stddev=0.01),
                name="latent_embedding",
            )(obj_indices)

        if latents.ndim < xyz.ndim:
            latents = jnp.expand_dims(latents, axis=tuple(range(xyz.ndim - latents.ndim)))
        latents = jnp.broadcast_to(latents, xyz.shape[:-1] + (self.latent_size,))

        # 2. Normalize world coords into the grid domain [0,1]^3 (clipped).
        bmin = jnp.asarray(self.bbox_min, dtype=jnp.float32)
        bext = jnp.asarray(self.bbox_max, dtype=jnp.float32) - bmin
        x01 = jnp.clip((xyz - bmin) / jnp.maximum(bext, 1e-6), 0.0, 1.0)

        gamma = HashGridEncoding(
            levels=self.levels,
            features_per_level=self.features_per_level,
            log2_table_size=self.log2_table_size,
            base_resolution=self.base_resolution,
            max_resolution=self.max_resolution,
            name="hash_grid",
        )(x01)

        # 3. Small shared decoder MLP.
        h = jnp.concatenate([latents, gamma, xyz], axis=-1)
        for i in range(self.num_layers - 1):
            h = nn.Dense(self.inner_dim, name=f"net_{i}")(h)
            h = nn.relu(h)
        sdf = nn.Dense(self.output_dim, name="final_layer_0")(h)

        # Pose probe: linear read-out z -> object centroid (see Embedding.py rationale).
        if self.use_pose_probe:
            pose_pred = nn.Dense(3, name="pose_probe")(latents.astype(jnp.float32))
            return sdf.astype(jnp.float32), latents, pose_pred

        return sdf.astype(jnp.float32), latents


# =====================================================================
# 2. Loss / step (same objective as Embedding.py)
# =====================================================================

def _hash_table_l2(params, table_reg):
    """L2 on the hash-grid table entries (sum, not mean: with ~16M entries a mean
    makes the per-entry gradient vanish). Entries with no sample coverage get no
    data gradient at all, so this is what pulls them to zero instead of leaving
    them as frozen noise that fragments the zero level set between samples."""
    return table_reg * jnp.sum(jnp.square(params["hash_grid"]["table"]))


def hash_sdf_loss_fn(params, apply_fn, obj_indices, xyz, true_sdf, reg_weight=1e-4, table_reg=1e-6):
    pred_sdf, latents = apply_fn({"params": params}, obj_indices, xyz)
    pred_sdf = jnp.squeeze(pred_sdf)
    true_sdf = jnp.squeeze(true_sdf)
    l1_loss = jnp.mean(jnp.abs(pred_sdf - true_sdf))
    l2_reg = reg_weight * jnp.mean(jnp.linalg.norm(latents, axis=-1) ** 2)
    table_l2 = _hash_table_l2(params, table_reg)
    return l1_loss + l2_reg + table_l2, {"l1_loss": l1_loss, "l2_reg": l2_reg, "table_l2": table_l2}


@jax.jit
def hash_train_step(state, obj_indices, xyz, true_sdf, reg_weight, table_reg):
    grad_fn = jax.value_and_grad(hash_sdf_loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(
        state.params, state.apply_fn, obj_indices, xyz, true_sdf,
        reg_weight=reg_weight, table_reg=table_reg
    )
    return state.apply_gradients(grads=grads), loss, metrics


def hash_sdf_loss_probe_fn(params, apply_fn, obj_indices, xyz, true_sdf, centers_table,
                           reg_weight=1e-4, probe_weight=0.1, table_reg=1e-6):
    pred_sdf, latents, pose_pred = apply_fn({"params": params}, obj_indices, xyz)
    pred_sdf = jnp.squeeze(pred_sdf)
    true_sdf = jnp.squeeze(true_sdf)
    l1_loss = jnp.mean(jnp.abs(pred_sdf - true_sdf))
    l2_reg = reg_weight * jnp.mean(jnp.linalg.norm(latents, axis=-1) ** 2)
    gt_centers = centers_table[obj_indices]
    probe_loss = jnp.mean(jnp.sum((pose_pred - gt_centers) ** 2, axis=-1))
    table_l2 = _hash_table_l2(params, table_reg)
    total = l1_loss + l2_reg + probe_weight * probe_loss + table_l2
    return total, {"l1_loss": l1_loss, "l2_reg": l2_reg, "probe_loss": probe_loss, "table_l2": table_l2}


@jax.jit
def hash_train_step_probe(state, obj_indices, xyz, true_sdf, centers_table, reg_weight, probe_weight, table_reg):
    grad_fn = jax.value_and_grad(hash_sdf_loss_probe_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(
        state.params, state.apply_fn, obj_indices, xyz, true_sdf, centers_table,
        reg_weight=reg_weight, probe_weight=probe_weight, table_reg=table_reg
    )
    return state.apply_gradients(grads=grads), loss, metrics


# =====================================================================
# 3. Trainer (reuses config loading + IFC write-back from Embedding.py)
# =====================================================================

class SDFHashEmbeddingTrainer(SDFEmbeddingTrainer):
    """Hash-grid variant of the embedding trainer. Same data in, same latents out."""

    CKPT_DIRNAME = "checkpoints_hash"

    @classmethod
    def write_embeddings_only(cls, yaml_path: Path, work_folder: Path) -> None:
        self = cls.config_load(yaml_path)
        work_folder = Path(work_folder)
        checkpoint_dir = (work_folder / cls.CKPT_DIRNAME).resolve()
        logger.info(f"Restoring best hash checkpoint from {checkpoint_dir}...")
        best_state = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint_dir), target=None, prefix="best_")
        if best_state is None:
            best_state = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint_dir), target=None)
        if best_state is None:
            logger.error("No hash checkpoint file found on disk.")
            return
        try:
            embeddings = np.array(best_state["params"]["latent_embedding"]["embedding"])
            logger.success(f"Extracted hash-trained latent embeddings. Shape: {embeddings.shape}")
            self._write_latents_to_ifcs(work_folder, embeddings)
        except KeyError:
            logger.error("Latent embeddings not found in the restored hash checkpoint.")

    def _hash_profile(self):
        """training.hash block with defaults; falls back to training.traditional values where sensible."""
        prof = getattr(self.config.training, "hash", None)
        trad = getattr(self.config.training, "traditional", None)

        def g(obj, name, default):
            return getattr(obj, name, default) if obj is not None else default

        net = g(prof, "network", None)
        trad_net = g(trad, "network", None)
        return {
            "epochs": int(g(prof, "epochs", 100)),
            # Loss-based early stop: epochs above is a hard cap; stop when the mean
            # train MAE of the last `early_stop_window` epochs < `early_stop_mae`
            # (0 disables).
            "early_stop_mae": float(g(prof, "early_stop_mae", 0.0)),
            "early_stop_window": int(g(prof, "early_stop_window", 5)),
            "batch_size": int(g(prof, "batch_size", g(trad, "batch_size", 524288))),
            "lr_model": float(g(prof, "lr_model", 1e-3)),
            "lr_decay_steps": int(g(prof, "lr_decay_steps", 50)),
            "lr_decay_rate": float(g(prof, "lr_decay_rate", 0.95)),
            "reg_weight": float(g(prof, "reg_weight", g(trad, "reg_weight", 1e-4))),
            "table_reg": float(g(prof, "table_reg", 1e-6)),
            "pose_probe_weight": float(g(prof, "pose_probe_weight", 0.0) or 0.0),
            "latent_size": int(g(net, "latent_size", g(trad_net, "latent_size", 128))),
            "levels": int(g(net, "levels", 16)),
            "features_per_level": int(g(net, "features_per_level", 2)),
            "log2_table_size": int(g(net, "log2_table_size", 19)),
            "base_resolution": int(g(net, "base_resolution", 16)),
            "max_resolution": int(g(net, "max_resolution", 1024)),
            "inner_dim": int(g(net, "inner_dim", 128)),
            "num_layers": int(g(net, "num_layers", 3)),
        }

    def _run(self, work_folder: Path) -> None:
        work_folder = Path(work_folder)
        logger.info(f"=== Starting JAX Hash-Grid SDF Training in: {work_folder} ===")

        str2int_path = work_folder / "intermediate" / "idx_str2int_dict.npy"
        if not str2int_path.exists():
            logger.error(f"Required data files not found in intermediate path of: {work_folder}")
            return

        str2int_map = np.load(str2int_path, allow_pickle=True).item()
        num_instances = len(str2int_map.keys())
        logger.info(f"Loaded {num_instances} instances for hash-grid SDF training.")

        try:
            all_indices, all_points, all_sdfs = load_flattened_samples(work_folder / "intermediate")
        except FileNotFoundError as e:
            logger.error(str(e))
            return
        dataset_size = all_points.shape[0]
        logger.info(f"Total dataset points parsed: {dataset_size}")

        p = self._hash_profile()

        # Normalization bbox over ALL sample points (world/scaled space), padded 5%.
        # Blocked reduction so a memmapped full dataset is streamed, not materialized.
        block = 16_777_216
        bmin = np.full(3, np.inf, dtype=np.float64)
        bmax = np.full(3, -np.inf, dtype=np.float64)
        for start in range(0, dataset_size, block):
            blk = np.asarray(all_points[start:start + block])
            bmin = np.minimum(bmin, blk.min(axis=0))
            bmax = np.maximum(bmax, blk.max(axis=0))
        pad = 0.05 * (bmax - bmin)
        bmin = (bmin - pad).astype(np.float32)
        bmax = (bmax + pad).astype(np.float32)
        bbox_path = work_folder / "intermediate" / "hash_grid_bbox.json"
        with open(bbox_path, "w", encoding="utf-8") as f:
            json.dump({"bbox_min": bmin.tolist(), "bbox_max": bmax.tolist()}, f)
        logger.info(f"Grid domain bbox {bmin.tolist()} .. {bmax.tolist()} (saved to {bbox_path.name})")

        use_probe = p["pose_probe_weight"] > 0.0
        if use_probe:
            logger.info(f"Pose-probe auxiliary loss enabled (weight {p['pose_probe_weight']}): z -> object centroid.")

        model = SpatiaSDFHashNet(
            num_instances=num_instances,
            latent_size=p["latent_size"],
            levels=p["levels"],
            features_per_level=p["features_per_level"],
            log2_table_size=p["log2_table_size"],
            base_resolution=p["base_resolution"],
            max_resolution=p["max_resolution"],
            inner_dim=p["inner_dim"],
            num_layers=p["num_layers"],
            use_pose_probe=use_probe,
            bbox_min=tuple(float(v) for v in bmin),
            bbox_max=tuple(float(v) for v in bmax),
        )
        logger.info(
            f"HashNet: L={p['levels']} F={p['features_per_level']} T=2^{p['log2_table_size']} "
            f"res {p['base_resolution']}..{p['max_resolution']} | MLP {p['num_layers']}x{p['inner_dim']} "
            f"| latent {p['latent_size']}"
        )

        seed = getattr(self.config.training.reproducibility, "seed", 42)
        rng = jax.random.PRNGKey(seed)
        params = model.init(rng, jnp.zeros((1,), dtype=jnp.int32), jnp.zeros((1, 3), dtype=jnp.float32))["params"]

        batch_size = p["batch_size"]
        # VRAM-aware batcher: pins small datasets to the device, streams large ones
        # from the memmapped host arrays (chunk-shuffled).
        max_device_gb = float(getattr(self.config.training, "max_device_data_gb", 4.0))
        batcher = DeviceBatcher(all_indices, all_points, all_sdfs, batch_size, seed, max_device_gb)
        steps_per_epoch = batcher.steps_per_epoch
        if p["lr_decay_rate"] < 1.0:
            lr_actual = optax.exponential_decay(
                init_value=p["lr_model"],
                transition_steps=steps_per_epoch * p["lr_decay_steps"],
                decay_rate=p["lr_decay_rate"],
            )
        else:
            lr_actual = p["lr_model"]
        optimizer = optax.adam(learning_rate=lr_actual)
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

        d_centers = None
        if use_probe:
            centers_np = compute_object_centroids(all_indices, all_points, num_instances)
            d_centers = jax.device_put(centers_np)
            logger.info(f"Object centroid table computed for pose probe: {centers_np.shape}")

        reg_weight = p["reg_weight"]
        epochs = p["epochs"]
        logger.info(f"Starting hash-grid SDF training: {epochs} epochs, {steps_per_epoch} steps/epoch")

        best_mae = float("inf")
        mae_history: list = []
        checkpoint_dir = (work_folder / self.CKPT_DIRNAME).resolve()
        checkpoint_dir.mkdir(exist_ok=True)

        for epoch in range(epochs):
            epoch_start_time = time.time()
            batch_maes, batch_regs, batch_probes = [], [], []

            pbar = tqdm(batcher.epoch_batches(epoch), total=steps_per_epoch,
                        desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for step, (b_indices, b_points, b_sdfs) in enumerate(pbar):
                if use_probe:
                    state, loss, metrics = hash_train_step_probe(
                        state, b_indices, b_points, b_sdfs,
                        d_centers, reg_weight, p["pose_probe_weight"], p["table_reg"]
                    )
                    batch_probes.append(metrics["probe_loss"])
                else:
                    state, loss, metrics = hash_train_step(
                        state, b_indices, b_points, b_sdfs,
                        reg_weight, p["table_reg"]
                    )
                batch_maes.append(metrics["l1_loss"])
                batch_regs.append(metrics["l2_reg"])
                if step % 20 == 0:
                    pbar.set_postfix({"MAE": f"{metrics['l1_loss']:.6f}"})

            avg_mae = jnp.mean(jnp.stack(batch_maes)).item()
            avg_reg = jnp.mean(jnp.stack(batch_regs)).item()
            probe_str = ""
            if use_probe and batch_probes:
                probe_str = f" | Probe: {jnp.mean(jnp.stack(batch_probes)).item():.6f}"
            logger.info(
                f"Epoch {epoch+1:04d} | MAE: {avg_mae:.6f} | L2Reg: {avg_reg:.6f}{probe_str} | "
                f"Time: {time.time()-epoch_start_time:.1f}s"
            )

            if avg_mae < best_mae:
                best_mae = avg_mae
                checkpoints.save_checkpoint(
                    ckpt_dir=str(checkpoint_dir), target=state, step=epoch + 1, prefix="best_", overwrite=True
                )
                logger.success(f"New best hash model saved (MAE: {best_mae:.6f})")

            mae_history.append(avg_mae)
            es_thr, es_win = p["early_stop_mae"], p["early_stop_window"]
            if es_thr > 0 and len(mae_history) >= es_win:
                recent_mean = sum(mae_history[-es_win:]) / es_win
                if recent_mean < es_thr:
                    logger.success(
                        f"Early stop at epoch {epoch+1}/{epochs}: mean MAE of last "
                        f"{es_win} epochs = {recent_mean:.6f} < {es_thr:g}"
                    )
                    break

        logger.success(f"Hash-grid training completed. Best MAE achieved: {best_mae:.6f}")

        logger.info("Restoring best hash checkpoint to extract learned latent codes...")
        best_state = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint_dir), target=None, prefix="best_")
        if best_state is None:
            best_state = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint_dir), target=None)

        if best_state is not None:
            try:
                embeddings = np.array(best_state["params"]["latent_embedding"]["embedding"])
                logger.success(f"Extracted hash-trained latent embeddings. Shape: {embeddings.shape}")
                self._write_latents_to_ifcs(work_folder, embeddings)
            except KeyError:
                logger.error("Latent embeddings not found in the restored hash checkpoint parameters.")
        else:
            logger.warning("No hash checkpoint found on disk. Extracting from in-memory parameters...")
            try:
                embeddings = np.array(state.params["latent_embedding"]["embedding"])
                self._write_latents_to_ifcs(work_folder, embeddings)
            except KeyError:
                logger.error("Latent embeddings not found in the in-memory parameters.")

        logger.success("=== Hash-Grid SDF Training & Integration Pipeline Completed ===")

    # =====================================================================
    # Single-IFC quality test: train on ONE source IFC's objects, then
    # reconstruct meshes (hierarchical marching cubes, ported from SDF_2's
    # Reconstructer) so reconstruction quality can be inspected by eye.
    # =====================================================================

    @classmethod
    def run_single_ifc_test(cls, config_path: Path, work_folder: Path) -> None:
        trainer = cls.config_load(config_path)
        try:
            trainer._run_single_ifc_test(Path(work_folder))
        finally:
            del trainer
            logger.info("SDFHashEmbeddingTrainer (single-IFC test) released.")

    def _run_single_ifc_test(self, work_folder: Path) -> None:
        work_folder = Path(work_folder)
        logger.info(f"=== Hash-Grid single-IFC embedding TEST in: {work_folder} ===")

        tcfg = getattr(self.config.training, "hash_test", None)
        source_ifc = str(getattr(tcfg, "source_ifc", "wall_variants.ifc"))
        epochs = int(getattr(tcfg, "epochs", 30))
        recon_objects = int(getattr(tcfg, "recon_objects", 12))
        recon_res = int(getattr(tcfg, "recon_res", 256))
        recon_coarse = int(getattr(tcfg, "recon_coarse_res", 48))

        inter = work_folder / "intermediate"

        # From-scratch data: sample the test IFC's exported OBJs with the CURRENT
        # config.sampling settings, so density experiments only need a config edit +
        # rerun of this menu item (no full preprocessing rerun). Results are cached in
        # intermediate/hash_test_samples.npz keyed by the sampling parameters; any
        # change to them (density, levels, scale, source_ifc) triggers a resample.
        seed = getattr(self.config.training.reproducibility, "seed", 42)
        scfg = self.config.sampling
        fingerprint = {
            # Bump when the sampler implementation changes semantics (v2: thickness
            # clamp no longer collapses the coarse jitter levels).
            "sampler_version": 2,
            "backend": str(getattr(scfg, "backend", "rust")).lower(),
            "source_ifc": source_ifc,
            "density_per_m2": int(scfg.density_per_m2),
            "min_per_surface": int(scfg.min_per_surface),
            "min_wall_thickness": float(getattr(scfg, "min_wall_thickness", 0.1)),
            "scale": float(self.config.extraction.scale),
            "levels": [[float(lv.radius), int(lv.count)] for lv in scfg.levels],
            "seed": int(seed),
        }
        cache_npz = inter / "hash_test_samples.npz"
        cache_meta = inter / "hash_test_samples.json"

        meta = None
        if cache_npz.exists() and cache_meta.exists():
            try:
                with open(cache_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("fingerprint") != fingerprint:
                    logger.info("Sampling config changed -> resampling test IFC.")
                    meta = None
            except Exception as e:
                logger.warning(f"Could not read sample cache meta ({e}); resampling.")
                meta = None

        if meta is not None:
            logger.info(f"Reusing cached test samples: {cache_npz.name}")
            cached = np.load(cache_npz)
            all_indices, all_points, all_sdfs = cached["indices"], cached["points"], cached["sdfs"]
            int2str_map = {int(k): v for k, v in meta["int2str"].items()}
            num_instances = int(meta["num_instances"])
        else:
            from src.DataPreprocessor import DataPreprocessor
            np.random.seed(seed)
            try:
                all_indices, all_points, all_sdfs, int2str_map, num_instances = (
                    DataPreprocessor(self.config).sample_single_ifc_subset(work_folder, source_ifc)
                )
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Test sampling failed: {e}")
                return
            np.savez(cache_npz, indices=all_indices, points=all_points, sdfs=all_sdfs)
            with open(cache_meta, "w", encoding="utf-8") as f:
                json.dump({
                    "fingerprint": fingerprint,
                    "int2str": {str(k): v for k, v in int2str_map.items()},
                    "num_instances": num_instances,
                }, f, ensure_ascii=False, indent=2)
            logger.success(f"Test samples cached to {cache_npz.name}")

        subset_ids = sorted(int2str_map.keys())
        dataset_size = all_points.shape[0]
        logger.info(
            f"Test IFC '{source_ifc}': {len(subset_ids)} objects (of {num_instances} total) "
            f"| {dataset_size} points."
        )

        p = self._hash_profile()
        use_probe = p["pose_probe_weight"] > 0.0

        bmin = (all_points.min(axis=0) - 0.05 * np.ptp(all_points, axis=0)).astype(np.float32)
        bmax = (all_points.max(axis=0) + 0.05 * np.ptp(all_points, axis=0)).astype(np.float32)

        model = SpatiaSDFHashNet(
            num_instances=num_instances,
            latent_size=p["latent_size"],
            levels=p["levels"],
            features_per_level=p["features_per_level"],
            log2_table_size=p["log2_table_size"],
            base_resolution=p["base_resolution"],
            max_resolution=p["max_resolution"],
            inner_dim=p["inner_dim"],
            num_layers=p["num_layers"],
            use_pose_probe=use_probe,
            bbox_min=tuple(float(v) for v in bmin),
            bbox_max=tuple(float(v) for v in bmax),
        )
        logger.info(
            f"HashNet TEST: L={p['levels']} T=2^{p['log2_table_size']} res {p['base_resolution']}..{p['max_resolution']} "
            f"| MLP {p['num_layers']}x{p['inner_dim']} | {epochs} epochs | probe={'on' if use_probe else 'off'}"
        )

        rng = jax.random.PRNGKey(seed)
        params = model.init(rng, jnp.zeros((1,), dtype=jnp.int32), jnp.zeros((1, 3), dtype=jnp.float32))["params"]
        max_device_gb = float(getattr(self.config.training, "max_device_data_gb", 4.0))
        batcher = DeviceBatcher(all_indices, all_points, all_sdfs, p["batch_size"], seed, max_device_gb)
        steps_per_epoch = batcher.steps_per_epoch
        # Hash tables need the lr to actually come down or they keep jittering and
        # the zero level set stays noisy (fragmented reconstructions). The configured
        # exponential decay is far too slow over a short test run, so decay to 1%
        # over the whole test with a cosine schedule instead.
        lr_schedule = optax.cosine_decay_schedule(
            init_value=p["lr_model"],
            decay_steps=max(1, steps_per_epoch * epochs),
            alpha=0.01,
        )
        optimizer = optax.adam(learning_rate=lr_schedule)
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

        d_centers = None
        if use_probe:
            d_centers = jax.device_put(compute_object_centroids(all_indices, all_points, num_instances))

        for epoch in range(epochs):
            t0 = time.time()
            maes = []
            for b_indices, b_points, b_sdfs in batcher.epoch_batches(epoch):
                if use_probe:
                    state, loss, met = hash_train_step_probe(
                        state, b_indices, b_points, b_sdfs, d_centers,
                        p["reg_weight"], p["pose_probe_weight"], p["table_reg"])
                else:
                    state, loss, met = hash_train_step(
                        state, b_indices, b_points, b_sdfs,
                        p["reg_weight"], p["table_reg"])
                maes.append(met["l1_loss"])
            logger.info(f"[test] Epoch {epoch+1:03d}/{epochs} | MAE: {jnp.mean(jnp.stack(maes)).item():.6f} | {time.time()-t0:.1f}s")

        ckpt_dir = (work_folder / "checkpoints_hash_test").resolve()
        ckpt_dir.mkdir(exist_ok=True)
        checkpoints.save_checkpoint(ckpt_dir=str(ckpt_dir), target=state, step=epochs, prefix="best_", overwrite=True)
        logger.success(f"Test checkpoint saved to {ckpt_dir}")

        # Per-object near-surface bboxes + reconstruction target list.
        recon_ids = subset_ids if recon_objects <= 0 else subset_ids[:recon_objects]
        obj_bboxes = {}
        for oid in recon_ids:
            pts_o = all_points[all_indices == oid]
            sdf_o = all_sdfs[all_indices == oid]
            near = pts_o[np.abs(sdf_o) < 0.05]
            src = near if len(near) >= 100 else pts_o
            lo, hi = src.min(axis=0), src.max(axis=0)
            pad = 0.15 * np.maximum(hi - lo, 0.05)
            obj_bboxes[oid] = (lo - pad, hi + pad)

        out_dir = work_folder / "reconstruct_hash_test"
        out_dir.mkdir(exist_ok=True)
        self._reconstruct_hash_objects(model, state.params, recon_ids, int2str_map,
                                       obj_bboxes, out_dir, recon_res, recon_coarse)

    def _reconstruct_hash_objects(self, model, params, obj_ids, int2str_map,
                                  obj_bboxes, out_dir: Path, res: int, coarse_res: int) -> None:
        """Hierarchical marching-cubes reconstruction (port of SDF_2 Reconstructer with
        per-object bounding boxes). Writes one OBJ per object for visual inspection."""
        import trimesh
        from skimage import measure

        @jax.jit
        def predict_sdf_batch(params, obj_idx, points):
            obj_idx_arr = jnp.full((points.shape[0],), obj_idx, dtype=jnp.int32)
            out = model.apply({"params": params}, obj_idx_arr, points)
            return jnp.squeeze(out[0])

        batch = 262144
        ok = 0
        for oid in tqdm(obj_ids, desc="Reconstructing (hash test)"):
            name = int2str_map.get(oid, f"obj{oid}")
            try:
                lo, hi = obj_bboxes[oid]
                axes_c = [np.linspace(lo[d], hi[d], coarse_res) for d in range(3)]
                axes_f = [np.linspace(lo[d], hi[d], res) for d in range(3)]

                # Coarse pass over the object's own bbox.
                CX, CY, CZ = np.meshgrid(*axes_c, indexing="ij")
                c_pts = np.stack([CX.ravel(), CY.ravel(), CZ.ravel()], axis=-1).astype(np.float32)
                c_vals = []
                for i in range(0, len(c_pts), batch):
                    c_vals.append(np.asarray(predict_sdf_batch(params, oid, jnp.array(c_pts[i:i + batch]))))
                c_grid = np.concatenate(c_vals).reshape(coarse_res, coarse_res, coarse_res)

                # Refinement mask: fine cells whose interpolated |sdf| is near the surface.
                diag = float(np.linalg.norm((hi - lo) / max(coarse_res - 1, 1)))
                threshold = diag * 1.5
                FX, FY, FZ = np.meshgrid(*axes_f, indexing="ij")
                f_pts = np.stack([FX.ravel(), FY.ravel(), FZ.ravel()], axis=-1).astype(np.float32)
                # Trilinear interp of coarse grid at fine points (jax.scipy, matches SDF_2).
                coords = jnp.stack([
                    (jnp.asarray(f_pts[:, d]) - lo[d]) / max(hi[d] - lo[d], 1e-9) * (coarse_res - 1)
                    for d in range(3)
                ], axis=0)
                estimates = np.asarray(jax.scipy.ndimage.map_coordinates(
                    jnp.asarray(c_grid), coords, order=1, mode="constant", cval=1.0))
                active = np.abs(estimates) < (threshold * 2.0)

                f_flat = estimates.copy()
                act_pts = f_pts[active]
                refined = []
                for i in range(0, len(act_pts), batch):
                    refined.append(np.asarray(predict_sdf_batch(params, oid, jnp.array(act_pts[i:i + batch]))))
                if refined:
                    f_flat[active] = np.concatenate(refined)
                f_grid = f_flat.reshape(res, res, res)

                gmin, gmax = float(f_grid.min()), float(f_grid.max())
                if gmin > 0:
                    if gmin < 0.05:
                        level = gmin + 1e-3
                        logger.warning(f"{name}: weak surface (min SDF {gmin:.4f}), using epsilon level.")
                    else:
                        logger.warning(f"{name}: no surface found (min SDF {gmin:.4f}).")
                        continue
                elif gmax < 0:
                    logger.warning(f"{name}: grid fully inside (max SDF {gmax:.4f}).")
                    continue
                else:
                    level = 0.0

                verts, faces, _, _ = measure.marching_cubes(f_grid, level=level)
                for d in range(3):
                    verts[:, d] = verts[:, d] * (hi[d] - lo[d]) / (res - 1) + lo[d]
                trimesh.Trimesh(vertices=verts, faces=faces).export(str(out_dir / f"{name}_recon.obj"))
                ok += 1
            except Exception as e:
                logger.error(f"Failed to reconstruct {name}: {e}")

        logger.success(f"Reconstructed {ok}/{len(obj_ids)} objects into {out_dir}")
