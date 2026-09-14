#!/usr/bin/env python3
"""
res.py - Graphormer GNN Test Dataset Evaluation, Numerical Attribution, and Visualization.
This script performs a full evaluation of the best-trained Graphormer checkpoint on the test split.
It calculates numerical coordinate prediction errors, performs JAX-native Integrated Gradients attribution,
and generates beautiful visualized charts.
"""
# --- repo-root bootstrap (added when this entry point moved into scripts/) ---
# Every data path in this file is relative to the repository root, so run it from there:
#     cd ~/Projects/SDF_3 && python scripts/res.py
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# --- end bootstrap ---


import os
import sys
import json
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

# Ensure JAX client does not preallocate all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.Graphormer import (
    SDFGraphormerTrainer,
    LatentCodePredictionModel,
    LatentCodePredictionAnalyzer,
    analyze_door_influences_jax
)
from loguru import logger
from tqdm import tqdm

def find_latest_run() -> Path:
    runs_dir = Path("data/runs")
    if not runs_dir.exists():
        logger.error("No 'data/runs' directory found. Please run preprocessing and training first.")
        sys.exit(1)
    
    folders = [f for f in runs_dir.iterdir() if f.is_dir()]
    if not folders:
        logger.error("No run folders found inside 'data/runs'.")
        sys.exit(1)
        
    # Prefer active runs over backups
    active_folders = [f for f in folders if "backup" not in f.name.lower() and "temp" not in f.name.lower()]
    if active_folders:
        active_folders.sort(key=lambda x: x.name, reverse=True)
        return active_folders[0]
        
    # Fallback to any folder if only backups exist
    folders.sort(key=lambda x: x.name, reverse=True)
    return folders[0]

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Graphormer model on the test dataset split.")
    parser.add_argument("--run", type=str, default=None, help="Name of the run folder to evaluate. Defaults to the latest.")
    parser.add_argument("--steps", type=int, default=32, help="Integrated Gradients interpolation steps.")
    parser.add_argument("--attribution_sample_size", type=int, default=20, help="Number of random test doors to sample for global attribution.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default=None, 
                        help="Force evaluation on a specific split. Defaults to 'test' for split-trained runs, and 'all' otherwise.")
    args = parser.parse_args()

    # 1. Determine run folder
    if args.run:
        work_folder = Path("data/runs") / args.run
        if not work_folder.exists():
            logger.error(f"Specified run folder not found: {work_folder}")
            sys.exit(1)
    else:
        work_folder = find_latest_run()
        
    logger.info(f"🚀 Using run folder: {work_folder.resolve()}")

    config_path = work_folder / "input" / "input_config" / "config.yaml"
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # 2. Instantiate and load trainer configurations
    logger.info("Loading trainer configurations...")
    trainer = SDFGraphormerTrainer.config_load(config_path)

    # 3. Load node schemas & dataset
    schema_path = work_folder / "intermediate" / "graphormer_node_type_schemas.json"
    if schema_path.exists():
        logger.info(f"Loading existing node type schemas from {schema_path}...")
        with open(schema_path, "r", encoding="utf-8") as sf:
            trainer.node_type_schemas = json.load(sf)
    else:
        logger.warning(f"Schema file not found at {schema_path}. It will be inferred from the graphs.")

    output_dir = work_folder / "output"
    graph_files = sorted(list(output_dir.glob("graph_room*.json")))
    if not graph_files:
        graph_files = sorted(list(output_dir.glob("*_graph.json")))
        
    if not graph_files:
        logger.error(f"No spatial topology graphs found in: {output_dir}")
        sys.exit(1)

    logger.info(f"Loading {len(graph_files)} spatial graph files...")
    all_graphs = []
    for gf in graph_files:
        try:
            with open(gf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "graphs" in data:
                for g in data["graphs"]:
                    all_graphs.append((g.get("nodes", []), g.get("links", [])))
            else:
                all_graphs.append((data.get("nodes", []), data.get("links", [])))
        except Exception as e:
            logger.error(f"Failed to load graph file {gf.name}: {e}")

    if not all_graphs:
        logger.error("No valid graph files loaded.")
        sys.exit(1)

    merged_nodes = []
    for nodes, _ in all_graphs:
        merged_nodes.extend(nodes)
    trainer.node_type_schemas = trainer._infer_schema_from_nodes(merged_nodes)

    # Prepare batches
    logger.info("Preparing GraphBatch instances...")
    batches = []
    for i, (nodes, links) in enumerate(all_graphs):
        B = trainer._prepare_graph(nodes, links)
        if B.door_indices:
            batches.append(B)

    if not batches:
        logger.error("No batches containing doors prepared.")
        sys.exit(1)

    # 4. Recover dataset splits
    is_split_run = (work_folder / "intermediate" / "graphormer_jax_split_checkpoints").exists()
    
    # Determine which indices we evaluate on
    if is_split_run:
        seed = getattr(trainer.config.training.reproducibility, 'seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        
        shuffled_indices = list(range(len(batches)))
        random.shuffle(shuffled_indices)
        
        num_total = len(batches)
        num_train = int(0.6 * num_total)
        num_val = int(0.2 * num_total)
        
        train_indices = shuffled_indices[:num_train]
        val_indices = shuffled_indices[num_train:num_train + num_val]
        test_indices = shuffled_indices[num_train + num_val:]
        
        selected_split = args.split or "test"
        if selected_split == "test":
            eval_indices = test_indices
        elif selected_split == "val":
            eval_indices = val_indices
        elif selected_split == "train":
            eval_indices = train_indices
        else:
            eval_indices = list(range(len(batches)))
            
        logger.info(f"Dataset split recovered | Total: {num_total} | Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")
        logger.info(f"Evaluating on split: '{selected_split}' ({len(eval_indices)} graphs)")
    else:
        logger.info("No split-training detected. Evaluating on the entire dataset...")
        eval_indices = list(range(len(batches)))
        logger.info(f"Evaluating on all {len(eval_indices)} graphs")

    # Helper function to place batches on GPU and pre-mask target door attributes
    def to_gpu_batch(B):
        type_feats = trainer._get_jax_type_features(B)
        masked_type_features = {}
        for n_type, tf in type_feats.items():
            masked_tf = dict(tf)
            geom = tf['geometry']
            coord = tf['coordinates']
            for node_idx in B.door_indices:
                indices_np = np.array(tf['indices'])
                locs = np.where(indices_np == node_idx)[0]
                if len(locs) > 0:
                    geom = geom.at[locs[0]].set(0.0)
                    coord = coord.at[locs[0]].set(jnp.array([0.0, 0.0], dtype=jnp.float32))
            masked_tf['geometry'] = jax.device_put(geom)
            masked_tf['coordinates'] = jax.device_put(coord)
            if tf['numeric'] is not None:
                masked_tf['numeric'] = jax.device_put(tf['numeric'])
            masked_tf['categorical'] = jax.device_put(tf['categorical'])
            masked_tf['indices'] = jax.device_put(tf['indices'])
            masked_type_features[n_type] = masked_tf
        
        return {
            'B_ref': B,
            'type_features': masked_type_features,
            'num_nodes': len(B.node_types),
            'edge_index': jax.device_put(B.edge_index.numpy()),
            'edge_types': jax.device_put(B.edge_types.numpy()),
            'spatial_indices': jax.device_put(B.spatial_indices),
            'edge_path_indices': jax.device_put(B.edge_path_indices),
            'edge_path_mask': jax.device_put(B.edge_path_mask),
            'door_indices': B.door_indices,
            'door_target_coords': jax.device_put(B.door_target_coords.numpy())
        }

    logger.info("Formatting and transferring split batches to JAX device memory...")
    eval_gpu = [to_gpu_batch(batches[i]) for i in eval_indices]

    # 5. Initialize the Graphormer Model Architecture
    graphormer_cfg = getattr(trainer.config.training, 'graphormer', None)
    epochs = 10
    lr = 2e-4
    accumulation_steps = 16
    edge_dim = 64
    d_model = 256
    n_heads = 8
    node_depth = 2
    graph_depth = 4
    ff_dim = 512
    dropout = 0.1
    max_in_degree = 50
    max_out_degree = 50
    max_path_distance = 10

    if graphormer_cfg is not None:
        edge_dim = int(getattr(graphormer_cfg, 'edge_dim', edge_dim))
        d_model = int(getattr(graphormer_cfg, 'd_model', d_model))
        n_heads = int(getattr(graphormer_cfg, 'n_heads', n_heads))
        node_depth = int(getattr(graphormer_cfg, 'node_depth', node_depth))
        graph_depth = int(getattr(graphormer_cfg, 'graph_depth', graph_depth))
        ff_dim = int(getattr(graphormer_cfg, 'ff_dim', ff_dim))
        dropout = float(getattr(graphormer_cfg, 'dropout', dropout))
        max_in_degree = int(getattr(graphormer_cfg, 'max_in_degree', max_in_degree))
        max_out_degree = int(getattr(graphormer_cfg, 'max_out_degree', max_out_degree))
        max_path_distance = int(getattr(graphormer_cfg, 'max_path_distance', max_path_distance))

    # Must mirror training: checkpoints from runs with pose_distill_weight > 0 contain
    # a pose_head param that only restores into a model built with use_pose_head=True.
    pose_distill_weight = float(getattr(graphormer_cfg, 'pose_distill_weight', 0.0) or 0.0) if graphormer_cfg is not None else 0.0

    model = LatentCodePredictionModel(
        node_type_schemas=trainer.node_type_schemas,
        edge_dim=edge_dim,
        d_model=d_model,
        n_heads=n_heads,
        node_depth=node_depth,
        graph_depth=graph_depth,
        ff_dim=ff_dim,
        dropout=dropout,
        max_in_degree=max_in_degree,
        max_out_degree=max_out_degree,
        max_path_distance=max_path_distance,
        use_pose_head=(pose_distill_weight > 0.0)
    )

    # Dry run for model parameters initialization
    logger.info("Initializing Graphormer model structure...")
    rng = jax.random.PRNGKey(seed if is_split_run else 42)
    dummy_batch = batches[0]
    dummy_type_feats = trainer._get_jax_type_features(dummy_batch)
    dummy_edge_index = jnp.zeros((2, 1), dtype=jnp.int32)
    dummy_edge_types = jnp.zeros((1,), dtype=jnp.int32)
    dummy_spatial = jnp.zeros((len(dummy_batch.node_types), len(dummy_batch.node_types)), dtype=jnp.int32)
    dummy_path_indices = jnp.zeros((len(dummy_batch.node_types), len(dummy_batch.node_types), max_path_distance), dtype=jnp.int32)
    dummy_path_mask = jnp.zeros((len(dummy_batch.node_types), len(dummy_batch.node_types), max_path_distance), dtype=jnp.float32)
    
    params = model.init(
        rng, 
        type_features=dummy_type_feats,
        num_nodes=len(dummy_batch.node_types),
        edge_index=dummy_edge_index,
        edge_types=dummy_edge_types,
        spatial_indices=dummy_spatial,
        edge_path_indices=dummy_path_indices,
        edge_path_mask=dummy_path_mask
    )['params']

    optimizer = optax.adam(learning_rate=lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # 6. Load Best Checkpoint from file (supporting split or traditional checkpoints)
    ckpt_dir = work_folder / "intermediate" / "graphormer_jax_split_checkpoints"
    if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
        ckpt_dir = work_folder / "intermediate" / "graphormer_jax_checkpoints"
        
    if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
        logger.error(f"No checkpoint directories found under intermediate/ in {work_folder}")
        sys.exit(1)
        
    ckpt_dir = ckpt_dir.resolve()
    logger.info(f"Loading best checkpoint from: {ckpt_dir}")
    best_state = checkpoints.restore_checkpoint(ckpt_dir=str(ckpt_dir), target=state, prefix="best_")
    if best_state is None:
        logger.warning(f"Could not restore checkpoint with prefix 'best_'. Checking standard prefix...")
        best_state = checkpoints.restore_checkpoint(ckpt_dir=str(ckpt_dir), target=state)
    
    if best_state is None:
        logger.error("Failed to restore any checkpoint. Please ensure training has finished and checkpoints are saved.")
        sys.exit(1)
        
    state = best_state
    logger.success("Model checkpoint successfully restored!")

    # 7. Evaluate on target split
    @jax.jit(static_argnums=(2,))
    def predict_coords_fn(params, type_features, num_nodes, edge_index, edge_types, spatial_indices, edge_path_indices, edge_path_mask, target_indices):
        pred_coords, _, _, _ = model.apply(
            {'params': params},
            type_features=type_features,
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_types=edge_types,
            spatial_indices=spatial_indices,
            edge_path_indices=edge_path_indices,
            edge_path_mask=edge_path_mask,
            target_indices=target_indices
        )
        return pred_coords

    results_list = []
    logger.info("Evaluating coordinates prediction...")
    for idx_in_eval, gb in enumerate(tqdm(eval_gpu, desc="Evaluating dataset")):
        pred = predict_coords_fn(
            state.params,
            gb['type_features'],
            gb['num_nodes'],
            gb['edge_index'],
            gb['edge_types'],
            gb['spatial_indices'],
            gb['edge_path_indices'],
            gb['edge_path_mask'],
            jnp.array(gb['door_indices'], dtype=jnp.int32)
        )
        
        true = gb['door_target_coords']
        pred_np = np.array(pred)
        true_np = np.array(true)
        
        scale = getattr(trainer.config.extraction, 'scale', 1.0)
        for door_idx_local, node_idx_global in enumerate(gb['door_indices']):
            p_coord = pred_np[door_idx_local] / scale
            t_coord = true_np[door_idx_local] / scale
            
            dx = p_coord[0] - t_coord[0]
            dy = p_coord[1] - t_coord[1]
            sq_err = dx**2 + dy**2
            abs_err = abs(dx) + abs(dy)
            euc_dist = math.sqrt(sq_err)
            
            door_id = gb['B_ref'].idx2id[node_idx_global] if node_idx_global < len(gb['B_ref'].idx2id) else f"unknown_{node_idx_global}"
            
            results_list.append({
                'eval_batch_idx': idx_in_eval,
                'door_node_idx': int(node_idx_global),
                'door_id': door_id,
                'pred': p_coord.tolist(),
                'true': t_coord.tolist(),
                'dx': float(dx),
                'dy': float(dy),
                'squared_error': float(sq_err),
                'absolute_error': float(abs_err),
                'euclidean_error': float(euc_dist)
            })

    if not results_list:
        logger.error("No door predictions computed. Check target door definitions.")
        sys.exit(1)

    # 8. Compute numerical error metrics
    squared_errors = [r['squared_error'] for r in results_list]
    absolute_errors = [r['absolute_error'] for r in results_list]
    euclidean_errors = [r['euclidean_error'] for r in results_list]
    
    mean_mse = np.mean(squared_errors)
    mean_mae = np.mean(absolute_errors)
    mean_euc = np.mean(euclidean_errors)
    median_euc = np.median(euclidean_errors)
    max_euc = np.max(euclidean_errors)
    min_euc = np.min(euclidean_errors)
    std_euc = np.std(euclidean_errors)
    
    logger.info("=========================================")
    logger.success(f"Evaluation complete for {len(results_list)} doors.")
    logger.info(f"MSE Loss (Coordinates): {mean_mse:.6f}")
    logger.info(f"MAE Loss (Coordinates): {mean_mae:.6f}")
    logger.info(f"Mean Euclidean Distance Error: {mean_euc:.6f} meters")
    logger.info(f"Median Euclidean Distance Error: {median_euc:.6f} meters")
    logger.info(f"Max Euclidean Distance Error: {max_euc:.6f} meters")
    logger.info(f"Min Euclidean Distance Error: {min_euc:.6f} meters")
    logger.info(f"Std Dev of Euclidean Error: {std_euc:.6f} meters")
    logger.info("=========================================")

    # Write numerical metrics to JSON file
    eval_output_dir = work_folder / "output" / "evaluation_results"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_metrics = {
        "global_metrics": {
            "mse": float(mean_mse),
            "mae": float(mean_mae),
            "mean_euclidean_error": float(mean_euc),
            "median_euclidean_error": float(median_euc),
            "max_euclidean_error": float(max_euc),
            "min_euclidean_error": float(min_euc),
            "std_euclidean_error": float(std_euc),
            "total_doors_evaluated": len(results_list),
            "total_graphs_evaluated": len(eval_gpu)
        },
        "per_door_results": results_list
    }
    
    report_json_path = eval_output_dir / "test_evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, ensure_ascii=False)
    logger.success(f"Saved evaluation report JSON to: {report_json_path}")

    logger.info("=========================================")
    logger.success("Evaluation pipeline completed successfully!")
    logger.info(f"Find the raw JSON report under: {eval_output_dir}")
    logger.info("=========================================")

if __name__ == "__main__":
    main()
