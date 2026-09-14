import os
import time
import yaml
import json
import math
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from loguru import logger
from tqdm import tqdm

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state, checkpoints


# =====================================================================
# 1. Flax Model Architecture (ported from SDF_2)
# =====================================================================

class SpatiaSDFNetBlock(nn.Module):
    """
    Inner MLP block to enable clean rematerialization (gradient checkpointing).
    """
    inner_dim: int
    output_dim: int
    num_layers: int
    skip_connections: bool
    latent_size: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, input_data):
        x = input_data
        skip_tensor_dim = self.latent_size + 3

        # 3. Main Network with Skip Connections
        if self.skip_connections and self.num_layers >= 5:
            # First 3 layers (Indices 0, 1, 2)
            for i in range(3):
                x = nn.WeightNorm(nn.Dense(self.inner_dim, dtype=self.dtype), name=f'net_{i}')(x)
                x = nn.relu(x)
            
            # Skip layer (reduces dimension so concat result is inner_dim)
            x = nn.Dense(self.inner_dim - skip_tensor_dim, dtype=self.dtype, name='skip_layer_0')(x)
            x = nn.relu(x)
            x = jnp.concatenate([x, input_data], axis=-1)
            
            # Remaining layers
            for i in range(self.num_layers - 5):
                x = nn.WeightNorm(nn.Dense(self.inner_dim, dtype=self.dtype), name=f'net_{3+i}')(x)
                x = nn.relu(x)
                
            # Final output: raw linear scalar for metric SDF space
            sdf = nn.Dense(self.output_dim, dtype=self.dtype, name='final_layer_0')(x)
        else:
            # Standard sequential MLP
            for i in range(self.num_layers - 1):
                x = nn.WeightNorm(nn.Dense(self.inner_dim, dtype=self.dtype), name=f'net_{i}')(x)
                x = nn.relu(x)

            # Final output: raw linear scalar for metric SDF space
            sdf = nn.Dense(self.output_dim, dtype=self.dtype, name='final_layer_0')(x)

        return sdf


class SpatiaSDFNet(nn.Module):
    """
    JAX/Flax version of the reference PyTorch SDFModel.
    Enhanced with Rematerialization and Mixed Precision support.
    """
    num_instances: int
    latent_size: int = 128
    inner_dim: int = 512
    output_dim: int = 1
    num_layers: int = 8
    skip_connections: bool = True
    use_remat: bool = False
    use_pose_probe: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, obj_indices, xyz, external_latent=None):
        """
        Forward pass matching the reference PyTorch implementation.
        """
        # 1. Latent Handling (Embeddings always in float32 for stability)
        if external_latent is not None:
            latents = external_latent
        else:
            # Ensure obj_indices is a jax array for nn.Embed
            obj_indices = jnp.asarray(obj_indices)
            latents = nn.Embed(
                num_embeddings=self.num_instances, 
                features=self.latent_size,
                embedding_init=nn.initializers.normal(stddev=0.01),
                name='latent_embedding'
            )(obj_indices)

        # 2. Alignment & Concatenation [Latent, XYZ]
        if latents.ndim < xyz.ndim:
            latents = jnp.expand_dims(latents, axis=tuple(range(xyz.ndim - latents.ndim)))
        latents = jnp.broadcast_to(latents, xyz.shape[:-1] + (self.latent_size,))

        # Cast to target dtype for the MLP part
        input_data = jnp.concatenate([latents, xyz], axis=-1)
        input_data = input_data.astype(self.dtype)

        # 3. Network Execution (Optional Remat)
        if self.use_remat:
            # Use nn.remat to save activation memory
            remat_block = nn.remat(SpatiaSDFNetBlock)
            sdf = remat_block(
                inner_dim=self.inner_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers,
                skip_connections=self.skip_connections,
                latent_size=self.latent_size,
                dtype=self.dtype,
                name='mlp_block'
            )(input_data)
        else:
            block = SpatiaSDFNetBlock(
                inner_dim=self.inner_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers,
                skip_connections=self.skip_connections,
                latent_size=self.latent_size,
                dtype=self.dtype,
                name='mlp_block'
            )
            sdf = block(input_data)

        # Pose probe: a LINEAR read-out z -> object centroid. Its gradient flows into the
        # latents, forcing them to organize position linearly/smoothly so the downstream
        # Graphormer can interpolate unseen rigid transforms instead of memorizing copies.
        if self.use_pose_probe:
            pose_pred = nn.Dense(3, name='pose_probe')(latents.astype(jnp.float32))
            return sdf.astype(jnp.float32), latents, pose_pred

        # Always return float32 for loss stability
        return sdf.astype(jnp.float32), latents


# =====================================================================
# 2. Training Helper Functions
# =====================================================================

def extract_flattened_data(samples_dict: Dict[int, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and concatenate all numpy arrays from the sample dictionary.
    """
    points_list, sdfs_list, indices_list = [], [], []
    
    for obj_idx, data in samples_dict.items():
        if "sdf" in data and "samples_latent_class" in data:
            sdfs = data["sdf"]
            lcc = data["samples_latent_class"]  # Format: [class_idx, x, y, z, is_edge]
            
            if sdfs is not None and sdfs.size > 0:
                indices = lcc[:, 0].astype(np.int32)
                points = lcc[:, 1:4].astype(np.float32)
                
                if points.shape[0] == sdfs.shape[0]:
                    points_list.append(points)
                    sdfs_list.append(sdfs.astype(np.float32))
                    indices_list.append(indices)
                else:
                    logger.warning(f"Object {obj_idx} mismatch: coords {points.shape[0]}, SDF {sdfs.shape[0]}")

    if not points_list:
        raise ValueError("No data collected from samples_dict.")

    all_indices = np.concatenate(indices_list, axis=0)
    all_points = np.concatenate(points_list, axis=0)
    all_sdfs = np.concatenate(sdfs_list, axis=0)

    return all_indices, all_points, all_sdfs


def load_flattened_samples(intermediate_folder: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the flattened training samples (indices, points, sdfs).

    Prefers the flat float32 arrays written by the parallel sampler
    (samples_obj_idx/points/sdf.npy, opened memory-mapped so the full dataset is
    never forced into RAM); falls back to the legacy pickled samples_dict.npy.
    """
    inter = Path(intermediate_folder)
    flat = {
        "indices": inter / "samples_obj_idx.npy",
        "points": inter / "samples_points.npy",
        "sdfs": inter / "samples_sdf.npy",
    }
    if all(p.exists() for p in flat.values()):
        all_indices = np.load(flat["indices"], mmap_mode="r")
        all_points = np.load(flat["points"], mmap_mode="r")
        all_sdfs = np.load(flat["sdfs"], mmap_mode="r")
        logger.info(f"Loaded flat sample arrays (memory-mapped): {all_points.shape[0]} points.")
        return all_indices, all_points, all_sdfs

    samples_path = inter / "samples_dict.npy"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"No sample data in {inter}: neither flat arrays (samples_points.npy ...) "
            f"nor legacy samples_dict.npy. Run preprocessing (menu 1.1/1.2) first."
        )
    logger.info("Flat sample arrays not found; loading legacy samples_dict.npy.")
    samples_dict = np.load(samples_path, allow_pickle=True).item()
    return extract_flattened_data(samples_dict)


def deepsdf_loss_fn(params, apply_fn, obj_indices, xyz, true_sdf, reg_weight=1e-4):
    """
    Standard DeepSDF loss with Full-range L1 and L2 latent regularization.
    Uses raw L1 distance to ensure metric SDF properties and prevent ghosting.
    """
    pred_sdf, latents = apply_fn({'params': params}, obj_indices, xyz)
    pred_sdf = jnp.squeeze(pred_sdf)
    true_sdf = jnp.squeeze(true_sdf)

    # 1. Full-range L1 Loss
    l1_loss = jnp.mean(jnp.abs(pred_sdf - true_sdf))
    
    # 2. Latent Regularization
    l2_reg = reg_weight * jnp.mean(jnp.linalg.norm(latents, axis=-1)**2)

    return l1_loss + l2_reg, {"l1_loss": l1_loss, "l2_reg": l2_reg}


@jax.jit
def train_step(state, obj_indices, xyz, true_sdf, reg_weight):
    """
    Single step JIT compilation for parameter and embedding optimization.
    """
    grad_fn = jax.value_and_grad(deepsdf_loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params, state.apply_fn, obj_indices, xyz, true_sdf, reg_weight=reg_weight)
    return state.apply_gradients(grads=grads), loss, metrics


def deepsdf_loss_probe_fn(params, apply_fn, obj_indices, xyz, true_sdf, centers_table,
                          reg_weight=1e-4, probe_weight=0.1):
    """
    DeepSDF loss + pose-probe auxiliary loss (model must be built with use_pose_probe=True).
    centers_table: [num_instances, 3] ground-truth object centroids (sample-point means).
    """
    pred_sdf, latents, pose_pred = apply_fn({'params': params}, obj_indices, xyz)
    pred_sdf = jnp.squeeze(pred_sdf)
    true_sdf = jnp.squeeze(true_sdf)

    l1_loss = jnp.mean(jnp.abs(pred_sdf - true_sdf))
    l2_reg = reg_weight * jnp.mean(jnp.linalg.norm(latents, axis=-1) ** 2)
    gt_centers = centers_table[obj_indices]
    probe_loss = jnp.mean(jnp.sum((pose_pred - gt_centers) ** 2, axis=-1))

    total = l1_loss + l2_reg + probe_weight * probe_loss
    return total, {"l1_loss": l1_loss, "l2_reg": l2_reg, "probe_loss": probe_loss}


@jax.jit
def train_step_probe(state, obj_indices, xyz, true_sdf, centers_table, reg_weight, probe_weight):
    grad_fn = jax.value_and_grad(deepsdf_loss_probe_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(
        state.params, state.apply_fn, obj_indices, xyz, true_sdf, centers_table,
        reg_weight=reg_weight, probe_weight=probe_weight
    )
    return state.apply_gradients(grads=grads), loss, metrics


def compute_object_centroids(all_indices: np.ndarray, all_points: np.ndarray, num_instances: int) -> np.ndarray:
    """Per-object mean of sample points (jitter is symmetric around the surface, so the
    mean approximates the posed surface centroid). Rows with no samples stay zero.

    Accumulates in fixed-size blocks so memory-mapped full-dataset arrays are
    streamed instead of materialized (np.bincount with weights copies its input)."""
    counts = np.zeros(num_instances, dtype=np.float64)
    centers = np.zeros((num_instances, 3), dtype=np.float64)
    block = 16_777_216
    for start in range(0, all_indices.shape[0], block):
        idx = np.asarray(all_indices[start:start + block])
        pts = np.asarray(all_points[start:start + block])
        counts += np.bincount(idx, minlength=num_instances)
        for d in range(3):
            centers[:, d] += np.bincount(idx, weights=pts[:, d], minlength=num_instances)
    centers /= np.maximum(counts, 1.0)[:, None]
    return centers.astype(np.float32)


class DeviceBatcher:
    """
    Serves training batches as device arrays under a VRAM budget.

    - Small datasets (indices+points+sdfs below `max_device_gb`) are pinned to the
      device once and sliced there per step -- the original fast path.
    - Larger datasets stay in the (memory-mapped) host arrays and are STREAMED:
      the dataset is viewed as contiguous chunks of `chunk_size` points; every
      epoch shuffles the chunk order, and each batch gathers `batch_size /
      chunk_size` random chunks. Chunk-level shuffling keeps host reads
      sequential-ish (a few dozen 100KB+ reads per step) instead of ~500k random
      12-byte reads, which a memmap over a multi-GB file cannot sustain.
      Each epoch drops at most the last partial chunk-group so every jitted step
      sees a constant batch shape (no recompilation).
    """

    def __init__(self, all_indices, all_points, all_sdfs,
                 batch_size: int, seed: int,
                 max_device_gb: float = 4.0, chunk_size: int = 8192):
        self.n = int(all_points.shape[0])
        data_gb = self.n * (all_indices.dtype.itemsize
                            + all_points.dtype.itemsize * 3
                            + all_sdfs.dtype.itemsize) / 1e9
        self.batch_size = int(min(batch_size, self.n))
        self.seed = int(seed)
        self.streaming = data_gb > max_device_gb

        if not self.streaming:
            self.d_indices = jax.device_put(np.ascontiguousarray(all_indices))
            self.d_points = jax.device_put(np.ascontiguousarray(all_points))
            self.d_sdfs = jax.device_put(np.ascontiguousarray(all_sdfs))
            self.steps_per_epoch = max(1, self.n // self.batch_size)
            logger.info(f"Dataset ({data_gb:.2f} GB) pinned to device memory.")
        else:
            self.h_indices, self.h_points, self.h_sdfs = all_indices, all_points, all_sdfs
            self.chunk = int(min(chunk_size, self.batch_size))
            self.chunks_per_batch = max(1, self.batch_size // self.chunk)
            self.batch_size = self.chunks_per_batch * self.chunk
            self.n_chunks = self.n // self.chunk
            self.steps_per_epoch = max(1, self.n_chunks // self.chunks_per_batch)
            logger.info(
                f"Dataset ({data_gb:.2f} GB) exceeds device budget ({max_device_gb} GB): "
                f"streaming host->device, {self.chunks_per_batch} x {self.chunk}-point "
                f"chunks per batch, {self.steps_per_epoch} steps/epoch."
            )

    def epoch_batches(self, epoch: int):
        """Yields (d_indices, d_points, d_sdfs) device arrays for one epoch."""
        if not self.streaming:
            key = jax.random.PRNGKey(self.seed * 1_000_003 + epoch)
            perm = jax.random.permutation(key, self.n)
            for step in range(self.steps_per_epoch):
                b = perm[step * self.batch_size:(step + 1) * self.batch_size]
                yield self.d_indices[b], self.d_points[b], self.d_sdfs[b]
            return

        rng = np.random.default_rng(self.seed * 1_000_003 + epoch)
        chunk_order = rng.permutation(self.n_chunks)
        c = self.chunk
        for step in range(self.steps_per_epoch):
            ids = np.sort(chunk_order[step * self.chunks_per_batch:(step + 1) * self.chunks_per_batch])
            sl = [slice(cid * c, (cid + 1) * c) for cid in ids]
            b_idx = np.concatenate([self.h_indices[s] for s in sl])
            b_pts = np.concatenate([self.h_points[s] for s in sl])
            b_sdf = np.concatenate([self.h_sdfs[s] for s in sl])
            yield jax.device_put(b_idx), jax.device_put(b_pts), jax.device_put(b_sdf)


# =====================================================================
# 3. Main Standalone Class
# =====================================================================

class SDFEmbeddingTrainer:
    """
    A standalone class to execute training of DeepSDF models and write 
    learned latent embeddings back to the exported topology graph JSONs.
    """
    
    def __init__(self, config: SimpleNamespace):
        self.config = config

    @classmethod
    def config_load(cls, yaml_path: Path) -> "SDFEmbeddingTrainer":
        """
        Loads the YAML configuration file and returns an SDFEmbeddingTrainer instance.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        def dict_to_obj(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
            if isinstance(d, list):
                return [dict_to_obj(i) for i in d]
            return d
            
        config_obj = dict_to_obj(data)
        logger.success(f"Config loaded from: {yaml_path}")
        return cls(config_obj)

    @classmethod
    def execute(cls, config_path: Path, work_folder: Path) -> None:
        """
        Executes the configuration loading, instantiation, training pipeline,
        and back-writing of latent codes.
        """
        trainer = cls.config_load(config_path)
        try:
            trainer._run(work_folder)
        finally:
            del trainer
            logger.info("SDFEmbeddingTrainer instance has been released.")

    @classmethod
    def write_embeddings_only(cls, yaml_path: Path, work_folder: Path) -> None:
        """
        Loads the pre-trained checkpoint and writes the embeddings directly to IFC files
        in intermediate/embedded_ifcs/ without training.
        """
        self = cls.config_load(yaml_path)
        work_folder = Path(work_folder)
        checkpoint_dir = (work_folder / "checkpoints").resolve()
        
        logger.info(f"Restoring best checkpoint from {checkpoint_dir} to extract latent codes...")
        best_state = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint_dir), target=None, prefix="best_")
        if best_state is None:
            best_state = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint_dir), target=None)
            
        if best_state is not None:
            try:
                embeddings = np.array(best_state['params']['latent_embedding']['embedding'])
                logger.success(f"Successfully extracted trained latent embeddings. Shape: {embeddings.shape}")
                self._write_latents_to_ifcs(work_folder, embeddings)
            except KeyError:
                logger.error("Latent embeddings not found in the restored checkpoint parameters.")
        else:
            logger.error("No checkpoint file found on disk.")

    def _run(self, work_folder: Path) -> None:
        """
        Executes DeepSDF training and saves the best embeddings.
        """
        work_folder = Path(work_folder)
        logger.info(f"=== Starting JAX/Flax SDF Training in: {work_folder} ===")

        # 1. Load preprocessed datasets
        str2int_path = work_folder / "intermediate" / "idx_str2int_dict.npy"
        if not str2int_path.exists():
            logger.error(f"Required data files not found in intermediate path of: {work_folder}")
            return

        str2int_map = np.load(str2int_path, allow_pickle=True).item()
        num_instances = len(str2int_map.keys())

        logger.info(f"Loaded {num_instances} instances for DeepSDF training.")

        # 2. Load flattened data (memory-mapped flat arrays, or legacy dict)
        try:
            all_indices, all_points, all_sdfs = load_flattened_samples(work_folder / "intermediate")
        except FileNotFoundError as e:
            logger.error(str(e))
            return
        dataset_size = all_points.shape[0]
        logger.info(f"Total dataset points parsed: {dataset_size}")

        # 3. Model Initialization
        profile = getattr(self.config.training, 'traditional', None)
        if profile is None:
            logger.error("Configuration section 'training.traditional' is missing.")
            return
            
        net_cfg = getattr(profile, 'network', None)
        if net_cfg is None:
            logger.error("Configuration section 'training.traditional.network' is missing.")
            return

        use_remat = getattr(net_cfg, 'use_remat', False)
        use_bfloat16 = getattr(net_cfg, 'use_bfloat16', False)
        dtype = jnp.bfloat16 if use_bfloat16 else jnp.float32
        
        if use_bfloat16:
            logger.info("Mixed precision (bfloat16) enabled for training.")
        if use_remat:
            logger.info("Gradient rematerialization (checkpointing) enabled.")

        probe_weight = float(getattr(profile, 'pose_probe_weight', 0.0) or 0.0)
        use_probe = probe_weight > 0.0
        if use_probe:
            logger.info(f"Pose-probe auxiliary loss enabled (weight {probe_weight}): z -> object centroid.")

        model = SpatiaSDFNet(
            num_instances=num_instances,
            latent_size=net_cfg.latent_size,
            inner_dim=net_cfg.inner_dim,
            num_layers=net_cfg.num_layers,
            skip_connections=net_cfg.skip_connections,
            use_remat=use_remat,
            use_pose_probe=use_probe,
            dtype=dtype
        )

        # 4. Setup State & Optimizer
        seed = getattr(self.config.training.reproducibility, 'seed', 42)
        rng = jax.random.PRNGKey(seed)
        params = model.init(rng, jnp.zeros((1,), dtype=jnp.int32), jnp.zeros((1, 3), dtype=jnp.float32))['params']
        
        lr_init = float(profile.lr_model)
        decay_rate = float(profile.lr_decay_rate)
        decay_steps = int(profile.lr_decay_steps)
        epochs = int(profile.epochs)
        batch_size = int(profile.batch_size)
        # Batches come from a VRAM-aware batcher: small datasets are pinned to the
        # device, large ones are streamed from the memmapped host arrays.
        max_device_gb = float(getattr(self.config.training, "max_device_data_gb", 4.0))
        batcher = DeviceBatcher(all_indices, all_points, all_sdfs, batch_size, seed, max_device_gb)
        steps_per_epoch = batcher.steps_per_epoch

        if decay_rate < 1.0:
            lr_schedule = optax.exponential_decay(
                init_value=lr_init,
                transition_steps=steps_per_epoch * decay_steps,
                decay_rate=decay_rate
            )
            lr_actual = lr_schedule
        else:
            lr_actual = lr_init

        optimizer = optax.chain(
            optax.adam(learning_rate=lr_actual)
        )

        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

        d_centers = None
        if use_probe:
            centers_np = compute_object_centroids(all_indices, all_points, num_instances)
            d_centers = jax.device_put(centers_np)
            logger.info(f"Object centroid table computed for pose probe: {centers_np.shape}")

        # 5. Training Loop
        reg_weight = float(profile.reg_weight)
        logger.info(f"Starting DeepSDF training: {epochs} epochs, {steps_per_epoch} steps/epoch")
        
        best_mae = float('inf')
        mae_history: list = []
        # Loss-based early stop: `epochs` is a hard cap; stop when the mean train MAE
        # of the last `early_stop_window` epochs < `early_stop_mae` (0 disables).
        es_thr = float(getattr(profile, 'early_stop_mae', 0.0) or 0.0)
        es_win = int(getattr(profile, 'early_stop_window', 5))
        checkpoint_dir = (work_folder / "checkpoints").resolve()
        checkpoint_dir.mkdir(exist_ok=True)

        for epoch in range(epochs):
            epoch_start_time = time.time()
            batch_losses, batch_maes, batch_regs, batch_probes = [], [], [], []

            pbar = tqdm(batcher.epoch_batches(epoch), total=steps_per_epoch,
                        desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for step, (b_indices, b_points, b_sdfs) in enumerate(pbar):
                if use_probe:
                    state, loss, metrics = train_step_probe(
                        state, b_indices, b_points, b_sdfs,
                        d_centers, reg_weight, probe_weight
                    )
                else:
                    state, loss, metrics = train_step(
                        state, b_indices, b_points, b_sdfs, reg_weight
                    )

                batch_losses.append(loss)
                batch_maes.append(metrics['l1_loss'])
                batch_regs.append(metrics['l2_reg'])
                if use_probe:
                    batch_probes.append(metrics['probe_loss'])
                
                if step % 20 == 0:
                    pbar.set_postfix({"MAE": f"{metrics['l1_loss']:.6f}"})

            # End of epoch: Synchronize and calculate averages
            avg_mae = jnp.mean(jnp.stack(batch_maes)).item()
            avg_met = jnp.mean(jnp.stack(batch_regs)).item()
            probe_str = ""
            if use_probe and batch_probes:
                probe_str = f" | Probe: {jnp.mean(jnp.stack(batch_probes)).item():.6f}"

            logger.info(f"Epoch {epoch+1:04d} | MAE: {avg_mae:.6f} | L2Reg: {avg_met:.6f}{probe_str} | Time: {time.time()-epoch_start_time:.1f}s")

            # Checkpoint management
            if avg_mae < best_mae:
                best_mae = avg_mae
                checkpoints.save_checkpoint(ckpt_dir=str(checkpoint_dir), target=state, step=epoch+1, prefix="best_", overwrite=True)
                logger.success(f"New best model saved (MAE: {best_mae:.6f})")

            # Early stopping check (rolling mean of recent epochs vs config threshold)
            mae_history.append(avg_mae)
            if es_thr > 0 and len(mae_history) >= es_win:
                recent_mean = sum(mae_history[-es_win:]) / es_win
                if recent_mean < es_thr:
                    logger.success(
                        f"Early stop at epoch {epoch+1}/{epochs}: mean MAE of last "
                        f"{es_win} epochs = {recent_mean:.6f} < {es_thr:g}"
                    )
                    break

        logger.success(f"Training completed. Best MAE achieved: {best_mae:.6f}")

        # 6. Retrieve best embeddings and write back to IFC files
        logger.info("Restoring best checkpoint to extract learned latent codes...")
        best_state = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint_dir), target=None, prefix="best_")
        if best_state is None:
            best_state = checkpoints.restore_checkpoint(ckpt_dir=str(checkpoint_dir), target=None)
            
        if best_state is not None:
            try:
                embeddings = np.array(best_state['params']['latent_embedding']['embedding'])
                logger.success(f"Successfully extracted trained latent embeddings. Shape: {embeddings.shape}")
                self._write_latents_to_ifcs(work_folder, embeddings)
            except KeyError:
                logger.error("Latent embeddings not found in the restored checkpoint parameters.")
        else:
            logger.warning("No checkpoint file found on disk. Extracting from current in-memory parameters...")
            try:
                embeddings = np.array(state.params['latent_embedding']['embedding'])
                logger.success(f"Extracted in-memory latent embeddings. Shape: {embeddings.shape}")
                self._write_latents_to_ifcs(work_folder, embeddings)
            except KeyError:
                logger.error("Latent embeddings not found in the current in-memory parameters.")

        logger.success("=== SDF Training & Integration Pipeline Completed ===")

    def _write_latents_to_ifcs(self, work_folder: Path, embeddings: np.ndarray) -> None:
        """
        Writes the trained latent codes back into the IFC files in expanded_ifcs,
        saving the updated IFCs to intermediate/embedded_ifcs/.

        Precise mode: uses intermediate/samples_manifest.json (obj_idx -> {guid,
        source_ifc}) to write each embedding row into the ONE IFC file / GUID it was
        exported from. This replaces the previous "try every GUID against every IFC"
        blind matching. Falls back to guid2int global matching if the manifest is
        absent (backward compatibility with older runs).
        """
        intermediate_folder = work_folder / "intermediate"
        expanded_folder = intermediate_folder / "expanded_ifcs"
        embedded_folder = intermediate_folder / "embedded_ifcs"
        embedded_folder.mkdir(parents=True, exist_ok=True)

        import ifcopenshell
        import ifcopenshell.api
        import ifcopenshell.util.element

        ifc_files = list(expanded_folder.glob("*.ifc"))
        if not ifc_files:
            logger.warning(f"No IFC files found in {expanded_folder}")
            return

        def _write_one(model, guid, obj_idx) -> bool:
            if obj_idx >= len(embeddings):
                return False
            element = model.by_guid(guid)
            if element is None:
                return False
            pset = None
            for rel in element.IsDefinedBy:
                if rel.is_a("IfcRelDefinesByProperties") and \
                   rel.RelatingPropertyDefinition.is_a("IfcPropertySet") and \
                   rel.RelatingPropertyDefinition.Name == "NeuralSDF":
                    pset = rel.RelatingPropertyDefinition
                    break
            if pset is None:
                pset = ifcopenshell.api.run("pset.add_pset", model, product=element, name="NeuralSDF")
            latent_str = ",".join(f"{x:.6f}" for x in embeddings[obj_idx])
            ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties={"LatentCode": latent_str})
            return True

        # --- Preferred path: precise manifest-based routing ---
        manifest_path = intermediate_folder / "samples_manifest.json"
        ifc_to_objects = defaultdict(list)   # ifc_name -> [(guid, obj_idx)]
        unrouted = []                        # objects with no known source_ifc
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            for obj_idx_str, rec in manifest.items():
                obj_idx = int(obj_idx_str)
                guid = rec.get("guid")
                src = rec.get("source_ifc")
                if not guid:
                    continue
                if src:
                    ifc_to_objects[src].append((guid, obj_idx))
                else:
                    unrouted.append((guid, obj_idx))
            logger.info(f"Loaded manifest with {sum(len(v) for v in ifc_to_objects.values())} routed objects "
                        f"({len(unrouted)} unrouted) across {len(ifc_to_objects)} source IFCs.")
        else:
            logger.warning("samples_manifest.json not found; falling back to guid2int global matching.")
            guid2int_path = intermediate_folder / "idx_guid2int_dict.npy"
            if not guid2int_path.exists():
                logger.error(f"Neither manifest nor idx_guid2int_dict.npy found in {intermediate_folder}")
                return
            guid2int_map = np.load(guid2int_path, allow_pickle=True).item()
            unrouted = [(g, i) for g, i in guid2int_map.items()]

        logger.info(f"Writing latent codes back to {len(ifc_files)} IFC files...")
        for ifc_path in ifc_files:
            try:
                logger.info(f"Updating IFC file: {ifc_path.name}")
                model = ifcopenshell.open(str(ifc_path))
                updated_count = 0
                # Objects known to belong to this IFC + any unrouted (tried opportunistically).
                candidates = list(ifc_to_objects.get(ifc_path.name, [])) + unrouted
                for guid, obj_idx in candidates:
                    try:
                        if _write_one(model, guid, obj_idx):
                            updated_count += 1
                    except Exception as e:
                        logger.error(f"Error writing latent to element {guid}: {e}")
                output_path = embedded_folder / ifc_path.name
                model.write(str(output_path))
                logger.success(f"Successfully wrote {updated_count} latent codes to {output_path.name}")
            except Exception as e:
                logger.error(f"Failed to process IFC file {ifc_path.name}: {e}")
