import os
import sys
import time
import json
import math
import random
import warnings
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict, defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union
from functools import partial

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state, checkpoints

# For offline graph dataset parsing and conversion
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


# =====================================================================
# 1. Shortest Path CPU Helper Utilities
# =====================================================================

def floyd_warshall_source_to_all(G, source, cutoff=None):
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))
    edges = {edge: i for i, edge in enumerate(G.edges())}
    level = 0
    nextlevel = {source: 1}
    node_paths = {source: [source]}
    edge_paths = {source: []}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G[v]:
                if w not in node_paths:
                    node_paths[w] = node_paths[v] + [w]
                    path_pair = tuple(node_paths[w][-2:])
                    if path_pair in edges:
                        edge_paths[w] = edge_paths[v] + [edges[path_pair]]
                    elif path_pair[::-1] in edges:
                        edge_paths[w] = edge_paths[v] + [edges[path_pair[::-1]]]
                    else:
                        edge_paths[w] = edge_paths[v] + [0]
                    nextlevel[w] = 1
        level = level + 1
        if (cutoff is not None and cutoff <= level):
            break
    return node_paths, edge_paths

def all_pairs_shortest_path(G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
    node_paths = {n: paths[n][0] for n in paths}
    edge_paths = {n: paths[n][1] for n in paths}
    return node_paths, edge_paths

def shortest_path_distance(data: Data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    G = to_networkx(data)
    node_paths, edge_paths = all_pairs_shortest_path(G)
    return node_paths, edge_paths


# =====================================================================
# 2. Flax Linen Layers for Graphormer
# =====================================================================

class CentralityEncoding(nn.Module):
    max_in_degree: int
    max_out_degree: int
    node_dim: int

    @nn.compact
    def __call__(self, x, edge_index):
        num_nodes = x.shape[0]
        in_deg = jnp.zeros(num_nodes, dtype=jnp.int32)
        if edge_index.shape[1] > 0:
            in_deg = in_deg.at[edge_index[1]].add(1)
        out_deg = jnp.zeros(num_nodes, dtype=jnp.int32)
        if edge_index.shape[1] > 0:
            out_deg = out_deg.at[edge_index[0]].add(1)

        in_deg = jnp.clip(in_deg, 0, self.max_in_degree - 1)
        out_deg = jnp.clip(out_deg, 0, self.max_out_degree - 1)

        z_in = self.param('z_in', nn.initializers.normal(), (self.max_in_degree, self.node_dim))
        z_out = self.param('z_out', nn.initializers.normal(), (self.max_out_degree, self.node_dim))

        x = x + z_in[in_deg] + z_out[out_deg]
        return x


class SpatialEncoding(nn.Module):
    max_path_distance: int

    @nn.compact
    def __call__(self, spatial_indices):
        b = self.param('b', nn.initializers.normal(), (self.max_path_distance + 1,))
        spatial_matrix = b[spatial_indices]
        return spatial_matrix


class EdgeEncoding(nn.Module):
    edge_dim: int
    max_path_distance: int

    @nn.compact
    def __call__(self, edge_attr, edge_path_indices, edge_path_mask):
        edge_vector = self.param('edge_vector', nn.initializers.normal(), (self.max_path_distance, self.edge_dim))
        dummy_row = jnp.zeros((1, self.edge_dim))
        safe_edge_attr = jnp.concatenate([edge_attr, dummy_row], axis=0)
        path_edge_attrs = safe_edge_attr[edge_path_indices]  # [N, N, max_path, edge_dim]
        prod = path_edge_attrs * edge_vector
        summed = prod.sum(axis=-1)  # [N, N, max_path]
        cij = (summed * edge_path_mask).sum(axis=-1) / jnp.maximum(edge_path_mask.sum(axis=-1), 1.0)
        return cij


class GraphormerAttentionHead(nn.Module):
    dim_in: int
    dim_q: int
    dim_k: int
    edge_dim: int
    max_path_distance: int

    @nn.compact
    def __call__(self, x, edge_attr, b, edge_path_indices, edge_path_mask, relative_offsets=None):
        cij = EdgeEncoding(
            edge_dim=self.edge_dim,
            max_path_distance=self.max_path_distance,
            name='edge_encoding'
        )(edge_attr, edge_path_indices, edge_path_mask)
        
        query = nn.Dense(self.dim_q, name='q')(x)
        key = nn.Dense(self.dim_k, name='k')(x)
        value = nn.Dense(self.dim_k, name='v')(x)
        
        a = jnp.matmul(query, key.T) / (self.dim_k ** 0.5)
        a = a + b + cij
        
        if relative_offsets is not None:
            rel_bias = nn.Dense(16, name='rel_bias_1')(relative_offsets)
            rel_bias = nn.gelu(rel_bias)
            rel_bias = nn.Dense(1, name='rel_bias_2')(rel_bias)
            rel_bias = jnp.squeeze(rel_bias, axis=-1)
            a = a + rel_bias
            
        softmax = jax.nn.softmax(a, axis=-1)
        out = jnp.matmul(softmax, value)
        return out


class GraphormerMultiHeadAttention(nn.Module):
    num_heads: int
    dim_in: int
    dim_q: int
    dim_k: int
    edge_dim: int
    max_path_distance: int

    @nn.compact
    def __call__(self, x, edge_attr, b, edge_path_indices, edge_path_mask, relative_offsets=None):
        heads = [
            GraphormerAttentionHead(
                dim_in=self.dim_in,
                dim_q=self.dim_q,
                dim_k=self.dim_k,
                edge_dim=self.edge_dim,
                max_path_distance=self.max_path_distance,
                name=f'head_{i}'
            ) for i in range(self.num_heads)
        ]
        
        head_outs = [head(x, edge_attr, b, edge_path_indices, edge_path_mask, relative_offsets=relative_offsets) for head in heads]
        concat = jnp.concatenate(head_outs, axis=-1)
        out = nn.Dense(self.dim_in, name='proj')(concat)
        return out


class GraphormerEncoderLayer(nn.Module):
    node_dim: int
    edge_dim: int
    n_heads: int
    ff_dim: int
    max_path_distance: int

    @nn.compact
    def __call__(self, x, edge_attr, b, edge_path_indices, edge_path_mask, relative_offsets=None):
        norm_x = nn.LayerNorm(name='ln_1')(x)
        attn_out = GraphormerMultiHeadAttention(
            num_heads=self.n_heads,
            dim_in=self.node_dim,
            dim_q=self.node_dim,
            dim_k=self.node_dim,
            edge_dim=self.edge_dim,
            max_path_distance=self.max_path_distance,
            name='attention'
        )(norm_x, edge_attr, b, edge_path_indices, edge_path_mask, relative_offsets=relative_offsets)
        x = x + attn_out
        
        norm_x = nn.LayerNorm(name='ln_2')(x)
        ff_out = nn.Sequential([
            nn.Dense(self.ff_dim),
            nn.gelu,
            nn.Dense(self.node_dim)
        ], name='ff')(norm_x)
        x = x + ff_out
        return x


# =====================================================================
# 3. TabTransformer Node Embedder in Flax
# =====================================================================

class NodeTabTransformer(nn.Module):
    num_categories_per_col: List[int]
    num_numeric_cols: int
    d_model: int
    n_heads: int = 8
    depth: int = 2
    ff_dim: int = 256
    dropout: float = 0.1

    @nn.compact
    def __call__(self, cats, nums):
        C = len(self.num_categories_per_col)
        M = self.num_numeric_cols
        
        seq_len = 1 + C + M
        col_pos = self.param('col_pos', nn.initializers.normal(stddev=0.01), (seq_len, self.d_model))
        cls_tok = self.param('cls', nn.initializers.normal(stddev=0.01), (1, 1, self.d_model))
        
        tokens = []
        N = cats.shape[0] if cats is not None else (nums.shape[0] if nums is not None else 1)
        cls_tok = jnp.broadcast_to(cls_tok, (N, 1, self.d_model))
        tokens.append(cls_tok)
        
        if C > 0 and cats is not None:
            cat_toks = []
            for i, num_classes in enumerate(self.num_categories_per_col):
                emb = nn.Embed(num_embeddings=num_classes + 1, features=self.d_model, name=f'cat_embed_{i}')(cats[:, i])
                cat_toks.append(jnp.expand_dims(emb, axis=1))
            tokens.append(jnp.concatenate(cat_toks, axis=1))
            
        if M > 0 and nums is not None:
            num_toks = []
            for j in range(M):
                col = nums[:, j:j+1]
                proj = nn.Dense(self.d_model, name=f'num_proj_{j}')(col)
                proj = nn.LayerNorm(name=f'num_ln_{j}')(proj)
                num_toks.append(jnp.expand_dims(proj, axis=1))
            tokens.append(jnp.concatenate(num_toks, axis=1))
            
        x = jnp.concatenate(tokens, axis=1)
        x = x + col_pos[:x.shape[1]]
        
        for d in range(self.depth):
            norm_x = nn.LayerNorm(name=f'enc_ln1_{d}')(x)
            attn_out = nn.SelfAttention(num_heads=self.n_heads, qkv_features=self.d_model, name=f'enc_attn_{d}')(norm_x)
            x = x + attn_out
            
            norm_x = nn.LayerNorm(name=f'enc_ln2_{d}')(x)
            ff_out = nn.Sequential([
                nn.Dense(self.ff_dim),
                nn.gelu,
                nn.Dense(self.d_model)
            ], name=f'enc_ff_{d}')(norm_x)
            x = x + ff_out
            
        z = nn.LayerNorm(name='out_ln')(x[:, 0, :])
        return z


class HeterogeneousNodeEncoder(nn.Module):
    node_type_schemas: Dict[str, Dict[str, Any]]
    d_model: int = 256
    n_heads: int = 8
    depth: int = 2
    dropout: float = 0.1

    @nn.compact
    def __call__(self, type_features, num_nodes):
        all_embeddings = jnp.zeros((num_nodes, self.d_model))
        
        for n_type, schema in self.node_type_schemas.items():
            cat_cols = schema.get('categorical_dims', [])
            num_cols = len(schema.get('numeric', []))
            gdim = int(schema.get("geometry_dim", 128))
            
            # Instantiating modules
            encoder = NodeTabTransformer(
                num_categories_per_col=cat_cols,
                num_numeric_cols=num_cols,
                d_model=self.d_model,
                n_heads=self.n_heads,
                depth=self.depth,
                name=f'node_tab_{n_type}'
            )
            
            geom_proj = nn.Sequential([
                nn.Dense(self.d_model),
                nn.LayerNorm(),
                nn.gelu
            ], name=f'geometry_proj_{n_type}')

            # NOTE (pure-embedding mode): node coordinates are NOT used as an input
            # feature. Spatial information must flow into the model ONLY through the
            # DeepSDF geometry embedding, which encodes the object's world pose.
            # `coordinates` is retained in type_features purely as the training target
            # (door_target_coords) and is never consumed here.

            # Check if this node type is active in the current batch
            is_active = (n_type in type_features) and (type_features[n_type]['indices'].shape[0] > 0)
            
            if is_active:
                tf = type_features[n_type]
                cats = tf['categorical']
                nums = tf.get('numeric')
                geoms = tf['geometry']
                coords = tf['coordinates']
                indices = tf['indices']
            else:
                # Prepare dummy inputs (batch size 1) just to trigger parameter initialization
                cats = jnp.zeros((1, len(cat_cols)), dtype=jnp.int32)
                nums = jnp.zeros((1, num_cols), dtype=jnp.float32) if num_cols > 0 else None
                geoms = jnp.zeros((1, gdim), dtype=jnp.float32)
                coords = jnp.zeros((1, 2), dtype=jnp.float32)
                
            node_emb = encoder(cats, nums)
            geom_emb = geom_proj(geoms)

            if is_active:
                total_emb = node_emb + geom_emb
                all_embeddings = all_embeddings.at[indices].set(total_emb)
                
        return all_embeddings


class LatentCodePredictionModel(nn.Module):
    node_type_schemas: Dict[str, Dict[str, Any]]
    edge_dim: int = 64
    d_model: int = 256
    n_heads: int = 8
    node_depth: int = 2
    graph_depth: int = 4
    ff_dim: int = 512
    max_in_degree: int = 50
    max_out_degree: int = 50
    max_path_distance: int = 10
    dropout: float = 0.1
    # Pose-distillation aux head (config: training.graphormer.pose_distill_weight > 0).
    # Kept optional so checkpoints from runs without it still restore.
    use_pose_head: bool = False

    @nn.compact
    def __call__(self, type_features, num_nodes, edge_index, edge_types, spatial_indices, edge_path_indices, edge_path_mask, target_indices=None, external_node_embeddings=None):
        if external_node_embeddings is not None:
            node_embeddings = external_node_embeddings
        else:
            node_embeddings = HeterogeneousNodeEncoder(
                node_type_schemas=self.node_type_schemas,
                d_model=self.d_model,
                n_heads=self.n_heads,
                depth=self.node_depth,
                name='node_encoder'
            )(type_features, num_nodes)
            
        num_edge_types = 10
        edge_emb_layer = nn.Embed(num_embeddings=num_edge_types, features=self.edge_dim, name='edge_type_embedding')
        edge_attr = edge_emb_layer(edge_types)
        
        spatial_matrix = SpatialEncoding(
            max_path_distance=self.max_path_distance,
            name='spatial_encoding'
        )(spatial_indices)
        
        x = node_embeddings
        x = nn.Dense(self.d_model, name='node_in_lin')(x)
        edge_attr = nn.Dense(self.edge_dim, name='edge_in_lin')(edge_attr)
        
        x = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.d_model,
            name='centrality_encoding'
        )(x, edge_index)
        
        # Pure-embedding mode: no coordinate-derived relative-offset bias is injected.
        # Node absolute coordinates are never used as an input, so the model must
        # recover the spatial layout from the geometry embeddings and graph topology
        # alone. `relative_offsets=None` disables the coordinate attention bias.
        for i in range(self.graph_depth):
            x = GraphormerEncoderLayer(
                node_dim=self.d_model,
                edge_dim=self.edge_dim,
                n_heads=self.n_heads,
                ff_dim=self.ff_dim,
                max_path_distance=self.max_path_distance,
                name=f'encoder_layer_{i}'
            )(x, edge_attr, spatial_matrix, edge_path_indices, edge_path_mask, relative_offsets=None)
            
        x = nn.Dense(self.d_model, name='node_out_lin')(x)
        
        # Residual-anchored head: outputs a 2D offset per node. The world coordinate is
        # composed downstream as: wall_center + offset, where wall_center is the PUBLIC
        # host-wall centerline midpoint (never fed as an input). Anchoring on the public
        # wall center frees the model from decoding the coarse frame position, so its
        # capacity focuses on the along-wall placement (fine same-side position).
        coordinate_head = nn.Sequential([
            nn.Dense(self.d_model),
            nn.gelu,
            nn.Dense(2)
        ], name='coordinate_head')

        if target_indices is not None:
            target_embeddings = x[target_indices]
            pred_coordinates = coordinate_head(target_embeddings)
        else:
            pred_coordinates = coordinate_head(x)

        # Pose-distillation aux head: a LINEAR readout of every node's centroid from
        # the final node states. A linear probe on the raw latents provably recovers
        # centroids across unseen rigid transforms (R^2 0.9994 incl. holdout), yet the
        # end-to-end door loss alone does not make the network internalize that
        # readout (it interpolates near training poses instead). Supervising this head
        # (coordinates as aux TARGET only - never an input) forces the representations
        # to carry pose, making relative geometry/chirality available to attention.
        pose_pred = None
        if self.use_pose_head:
            pose_pred = nn.Dense(2, name='pose_head')(x)

        return pred_coordinates, node_embeddings, x, pose_pred


# =====================================================================
# 3b. Message-passing GNN baseline (GINE)
# =====================================================================

class GINEBaselineLayer(nn.Module):
    """
    One pre-norm residual GINE block (GIN with edge features; Hu et al. 2020,
    "Strategies for Pre-training Graph Neural Networks").

    Messages flow along stored edges in BOTH directions: links are stored
    one-directional (see _make_edge_tensors), and the Graphormer sees every
    node pair through dense attention, so an MPNN restricted to the stored
    direction would be artificially crippled rather than a fair baseline.

    Padded edges are self-loops on the last PAD node (see _to_host_batch),
    so they never inject messages into real nodes.

    Aggregation is MEAN (degree-normalized), not GIN's canonical sum: attempt 1
    (sum, depth 4, run 2026JUL122211 gin_baseline_attempt1_sum_d4_stalled.log)
    parked at the predict-mean saddle (Val ~4.2-4.5 for 7 epochs, ~13.5k updates)
    while the Graphormer broke through by epoch 7 on the same data - hub nodes
    here reach degree ~50, so summed messages swamp the update MLP's input scale.
    """
    node_dim: int
    edge_dim: int
    ff_dim: int

    @nn.compact
    def __call__(self, x, edge_attr, edge_index):
        norm_x = nn.LayerNorm(name='ln')(x)
        e = nn.Dense(self.node_dim, name='edge_proj')(edge_attr)
        src, dst = edge_index[0], edge_index[1]
        msg_fwd = nn.gelu(norm_x[src] + e)
        msg_bwd = nn.gelu(norm_x[dst] + e)
        agg = jnp.zeros_like(norm_x).at[dst].add(msg_fwd)
        agg = agg.at[src].add(msg_bwd)
        deg = jnp.zeros((norm_x.shape[0],), dtype=norm_x.dtype).at[dst].add(1.0)
        deg = deg.at[src].add(1.0)
        agg = agg / jnp.maximum(deg, 1.0)[:, None]
        eps = self.param('eps', nn.initializers.zeros, ())
        h = nn.Sequential([
            nn.Dense(self.ff_dim),
            nn.gelu,
            nn.Dense(self.node_dim)
        ], name='update_mlp')((1.0 + eps) * norm_x + agg)
        return x + h


class GNNBaselineModel(nn.Module):
    """
    Architecture baseline for the paper: a standard local message-passing GNN
    (GINE) swapped in for the Graphormer encoder stack. Everything else is
    IDENTICAL to LatentCodePredictionModel - same HeterogeneousNodeEncoder over
    the same pure-embedding inputs, same coordinate/pose heads, same losses,
    batching, split and evaluation - so the comparison isolates the graph
    encoder: global attention + centrality/spatial/edge-path structural
    encodings vs. plain neighborhood message passing at equal depth/width.

    Selected via training.graphormer.model_arch: gin (default: graphormer).
    The Graphormer-only inputs (spatial_indices, edge_path_*) are accepted and
    ignored; max_* fields exist only for constructor parity with the config.
    """
    node_type_schemas: Dict[str, Dict[str, Any]]
    edge_dim: int = 64
    d_model: int = 256
    n_heads: int = 8
    node_depth: int = 2
    graph_depth: int = 4
    ff_dim: int = 512
    max_in_degree: int = 50      # unused (Graphormer centrality encoding only)
    max_out_degree: int = 50     # unused
    max_path_distance: int = 10  # unused
    dropout: float = 0.1
    use_pose_head: bool = False

    @nn.compact
    def __call__(self, type_features, num_nodes, edge_index, edge_types, spatial_indices, edge_path_indices, edge_path_mask, target_indices=None, external_node_embeddings=None):
        if external_node_embeddings is not None:
            node_embeddings = external_node_embeddings
        else:
            node_embeddings = HeterogeneousNodeEncoder(
                node_type_schemas=self.node_type_schemas,
                d_model=self.d_model,
                n_heads=self.n_heads,
                depth=self.node_depth,
                name='node_encoder'
            )(type_features, num_nodes)

        num_edge_types = 10
        edge_emb_layer = nn.Embed(num_embeddings=num_edge_types, features=self.edge_dim, name='edge_type_embedding')
        edge_attr = edge_emb_layer(edge_types)

        x = nn.Dense(self.d_model, name='node_in_lin')(node_embeddings)
        edge_attr = nn.Dense(self.edge_dim, name='edge_in_lin')(edge_attr)

        for i in range(self.graph_depth):
            x = GINEBaselineLayer(
                node_dim=self.d_model,
                edge_dim=self.edge_dim,
                ff_dim=self.ff_dim,
                name=f'gnn_layer_{i}'
            )(x, edge_attr, edge_index)

        x = nn.Dense(self.d_model, name='node_out_lin')(x)

        coordinate_head = nn.Sequential([
            nn.Dense(self.d_model),
            nn.gelu,
            nn.Dense(2)
        ], name='coordinate_head')

        if target_indices is not None:
            target_embeddings = x[target_indices]
            pred_coordinates = coordinate_head(target_embeddings)
        else:
            pred_coordinates = coordinate_head(x)

        pose_pred = None
        if self.use_pose_head:
            pose_pred = nn.Dense(2, name='pose_head')(x)

        return pred_coordinates, node_embeddings, x, pose_pred


# =====================================================================
# 4. Custom Differentiable JAX Integrated Gradients
# =====================================================================

def analyze_door_influences_jax(
    params, apply_fn, B, door_idx_global, door_target_latent, node_type_schemas, feature_splits_by_type, max_dim, num_steps=32
):
    X = jnp.array(B.X)
    baseline = jnp.zeros_like(X)
    edge_index = jnp.array(B.edge_index.numpy())
    edge_types = jnp.array(B.edge_types.numpy())
    spatial_indices = jnp.array(B.spatial_indices)
    edge_path_indices = jnp.array(B.edge_path_indices)
    edge_path_mask = jnp.array(B.edge_path_mask)
    N = len(B.node_types)
    
    type_to_indices = defaultdict(list)
    for i, t in enumerate(B.node_types):
        type_to_indices[t].append(i)
        
    type_to_coords = {}
    for n_type, indices in type_to_indices.items():
        coord_list = []
        for idx in indices:
            node_id = B.idx2id[idx]
            if idx == door_idx_global:
                coord_list.append(np.array([0.0, 0.0], dtype=np.float32))
            else:
                coord_list.append(B.node_features[node_id]['coordinates'].numpy())
        type_to_coords[n_type] = jnp.array(np.stack(coord_list), dtype=jnp.float32)
    
    def forward_x(X_val, alpha):
        X_interp = baseline + alpha * (X_val - baseline)
        
        type_features = {}
        for n_type, schema in node_type_schemas.items():
            indices = type_to_indices[n_type]
            if not indices:
                continue
            rows = X_interp[jnp.array(indices, dtype=jnp.int32)]
            
            splits = feature_splits_by_type[n_type]
            cats_list = []
            nums_list = []
            geom_val = None
            
            for feat_name, (start, end) in splits:
                val = rows[:, start:end]
                if feat_name in schema.get('categorical_fields', ['label', 'type']):
                    cats_list.append(val)
                elif feat_name in schema.get('numeric', []):
                    nums_list.append(val)
                elif feat_name == "geometry":
                    geom_val = val
            
            coord_val = type_to_coords[n_type]
            
            cats_stacked = jnp.concatenate(cats_list, axis=-1).astype(jnp.int32) if cats_list else jnp.zeros((len(indices), 0), dtype=jnp.int32)
            nums_stacked = jnp.concatenate(nums_list, axis=-1) if nums_list else None
            
            type_features[n_type] = {
                'categorical': cats_stacked,
                'numeric': nums_stacked,
                'geometry': geom_val,
                'coordinates': coord_val,
                'indices': jnp.array(indices, dtype=jnp.int32)
            }
            
        latent_codes, _, _, _ = apply_fn(
            {'params': params},
            type_features=type_features,
            num_nodes=N,
            edge_index=edge_index,
            edge_types=edge_types,
            spatial_indices=spatial_indices,
            edge_path_indices=edge_path_indices,
            edge_path_mask=edge_path_mask,
            target_indices=jnp.array([int(door_idx_global)], dtype=jnp.int32)
        )
        door_pred = latent_codes[0]
        return -jnp.sum((door_pred - door_target_latent) ** 2)

    grad_fn = jax.grad(forward_x, argnums=0)
    
    alphas = jnp.linspace(0.0, 1.0, num_steps)
    grads = jax.vmap(lambda a: grad_fn(X, a))(alphas)
    
    mean_grads = jnp.mean(grads, axis=0)
    attributions = mean_grads * (X - baseline)
    return attributions


# =====================================================================
# 5. Captum-like Attribution Analyzer in JAX
# =====================================================================

class LatentCodePredictionAnalyzer:
    def __init__(
        self,
        params,
        apply_fn,
        node_type_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
        node_names: Optional[List[str]] = None,
    ):
        self.params = params
        self.apply_fn = apply_fn
        self.node_type_schemas = node_type_schemas or {}
        self.node_names = node_names
        self.influence_cache = defaultdict(list)

    @staticmethod
    def _is_door_type(nt: str) -> bool:
        if not isinstance(nt, str):
            return False
        s = nt.strip().lower()
        return ("door" in s) or (s == "ifcdoor")

    def _get_feature_info(self, node_type: str) -> List[Tuple[str, int]]:
        schema = self.node_type_schemas.get(node_type, {})
        features = []
        
        cat_fields = schema.get('categorical_fields', ['label', 'type'])
        for field in cat_fields:
            features.append((field, 1))
        
        numeric_list = schema.get('numeric', [])
        for name in numeric_list:
            features.append((name, 1))
        
        geom_dim = schema.get('geometry_dim', 128)
        features.append(('geometry', geom_dim))
        return features

    def _resolve_door_key(self, results, door_idx):
        if int(door_idx) in results: return int(door_idx)
        for k in results.keys():
            if str(k) == str(door_idx): return k
        return door_idx

    def _door_display_ids(self, results, key):
        entry = results[key]
        return entry.get("ord", 0) + 1, self.node_names[key] if self.node_names and key < len(self.node_names) else f"unknown_{key}"

    def plot_top_nodes_bar(self, results: dict, door_idx: int, savepath: str = None):
        if not results: return
        key = self._resolve_door_key(results, door_idx)
        entry = results[key]
        order, global_key = self._door_display_ids(results, key)
        tops = entry["top_nodes"]
        if not tops: return
        labels = [f'{t["node_id"]} ({t["node_type"]})' for t in tops]
        values = [t["node_influence"] for t in tops]

        plt.figure(figsize=(6, 3.5))
        y = range(len(labels))[::-1]
        plt.barh(y, values)
        plt.yticks(y, labels)
        plt.xlabel("Node Influence")
        plt.title(f"Door {order} (global {global_key}) - Top Influences")
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=160)
            plt.close()
        else:
            plt.show()

    def plot_node_feature_heatmap(self, results: dict, door_idx: int, topk_feats: int = 5, savepath: str = None):
        if not results: return
        key = self._resolve_door_key(results, door_idx)
        entry = results[key]
        order, global_key = self._door_display_ids(results, key)
        tops = entry["top_nodes"]
        if not tops: return

        feat_names = []
        for t in tops:
            for (name, _) in t["top_features"][:topk_feats]:
                if name not in feat_names:
                    feat_names.append(name)

        M = len(tops)
        F = len(feat_names)
        mat = np.zeros((M, F), dtype=float)
        row_labels = []
        for i, t in enumerate(tops):
            row_labels.append(f'{t["node_id"]} ({t["node_type"]})')
            d = dict(t["top_features"])
            for j, fn in enumerate(feat_names):
                if fn in d:
                    mat[i, j] = float(d[fn])

        plt.figure(figsize=(max(6, F * 0.7), max(3.5, M * 0.4)))
        im = plt.imshow(mat, aspect="auto")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(F), feat_names, rotation=45, ha="right")
        plt.yticks(range(M), row_labels)
        plt.title(f"Door {order} (global {global_key}) - Feature Influence Breakdown")
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=160)
            plt.close()
        else:
            plt.show()

    def analyze_door_influences(
        self,
        B,
        door_indices: List[int],
        door_latents: np.ndarray,
        num_steps: int = 32,
        topk_nodes: int = 5,
        topk_feats: int = 5
    ) -> Dict:
        logger.info(f"🔍 Starting JAX-native Integrated Gradients Attribution ({num_steps} steps)")
        results = {}
        
        feature_splits_by_type = {}
        for nt in set(B.node_types):
            feat_info = self._get_feature_info(nt)
            splits = []
            offset = 0
            for name, dim in feat_info:
                splits.append((name, (offset, offset + dim)))
                offset += dim
            feature_splits_by_type[nt] = splits

        for i, door_idx in enumerate(door_indices):
            door_target_latent = jnp.array(door_latents[i])
            logger.info(f"  🚪 Analyzing Door Node Index: {door_idx}...")
            
            attributions = analyze_door_influences_jax(
                self.params, self.apply_fn, B, door_idx, door_target_latent, 
                self.node_type_schemas, feature_splits_by_type, B.max_dim, num_steps
            )
            
            attributions_np = np.array(attributions)
            node_influences = np.sum(np.abs(attributions_np), axis=-1)
            
            sorted_nodes = []
            for node_idx in range(len(B.node_types)):
                if node_idx == door_idx:
                    continue
                sorted_nodes.append((node_idx, node_influences[node_idx]))
            sorted_nodes.sort(key=lambda x: x[1], reverse=True)
            
            top_nodes_list = []
            for rank_idx, (node_idx, influence) in enumerate(sorted_nodes[:topk_nodes]):
                nt = B.node_types[node_idx]
                splits = feature_splits_by_type[nt]
                
                feat_attrs = []
                for feat_name, (start, end) in splits:
                    feat_attr_sum = float(np.abs(attributions_np[node_idx, start:end]).sum())
                    feat_attrs.append((feat_name, feat_attr_sum))
                feat_attrs.sort(key=lambda x: x[1], reverse=True)
                
                top_nodes_list.append({
                    "node_id": int(node_idx),
                    "node_type": nt,
                    "node_influence": float(influence),
                    "top_features": feat_attrs[:topk_feats]
                })
                
            results[int(door_idx)] = {
                "ord": i,
                "top_nodes": top_nodes_list,
                "attributions": torch.tensor(attributions_np)
            }
            
        return results


# =====================================================================
# 6. Graph Batching & Trainer Orchestrator Class
# =====================================================================

@dataclass
class GraphBatch:
    node_features: "OrderedDict[str, Dict[str, torch.Tensor]]"
    node_types: List[str]
    edge_index: torch.LongTensor
    edge_types: torch.LongTensor
    door_indices: List[int]
    door_latent_codes: torch.Tensor
    door_target_coords: torch.Tensor
    # Plan-B wall-relative targets (one row per target door), aligned with door_indices:
    #   door_target_coords == door_wall_centers + door_s[:,None] * door_wall_axes
    door_wall_centers: torch.Tensor   # [D, 2] anchor (scaled world). Zero in absolute mode; host-wall center otherwise.
    door_wall_proj: torch.Tensor      # [D, 2, 2] linear map applied to the model output before adding the anchor:
                                      #   absolute/offset -> identity ; scalar_s -> axis (x) axis^T (projects onto wall axis)
    door_wall_axes: torch.Tensor      # [D, 2] unit world direction of wall local +x (stable sign anchor)
    door_wall_centers_true: torch.Tensor  # [D, 2] ALWAYS the real host-wall center (even in absolute mode); metrics only
    door_s: torch.Tensor              # [D]    signed along-wall offset (scaled world meters)
    idx2id: List[str]
    # JAX path arrays
    spatial_indices: np.ndarray
    edge_path_indices: np.ndarray
    edge_path_mask: np.ndarray
    # Flattened feature matrices for attribution
    X: np.ndarray
    feature_splits: Dict[str, Tuple[List[Tuple[str, Tuple[int, int]]], int]]
    max_dim: int


# Differentiable JAX loss function for optimization
def compose_world_coords(pred_offset, wall_centers, wall_proj=None):
    """
    Generic composition: world = anchor + proj @ pred. The head mode is fully encoded in
    (anchor, proj), precomputed per door in _prepare_graph:
        absolute : anchor = 0 ,          proj = I            -> world = pred
        offset   : anchor = wall_center, proj = I            -> world = wall_center + pred
        scalar_s : anchor = wall_center, proj = axis e_x^T   -> world = wall_center + pred_x axis
    (scalar_mag_side does NOT use this path: its sign comes from a logit, so the loss and
    eval compose world = wall_center + sign(pred_1)*pred_0*axis explicitly.)
    `pred_offset` : [..., 2] model output ; `wall_centers` : [..., 2] ; `wall_proj` : [..., 2, 2] (I if None).
    """
    if wall_proj is None:
        return wall_centers + pred_offset
    return wall_centers + jnp.einsum('...ij,...j->...i', wall_proj, pred_offset)


def graphormer_loss_fn(
    params, apply_fn, type_features, num_nodes, edge_index, edge_types,
    spatial_indices, edge_path_indices, edge_path_mask, target_indices, target_coords, target_mask,
    target_wall_centers, target_wall_proj, target_s_abs, target_side, mag_side_flag,
    node_coords, node_pose_mask, pose_weight
):
    pred_offset, _, _, pose_pred = apply_fn(
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
    pred_world = compose_world_coords(pred_offset, target_wall_centers, target_wall_proj)

    # Coordinate MSE in composed world space (100x for a stronger gradient signal).
    denom = jnp.maximum(jnp.sum(target_mask), 1.0)
    squared_errors = jnp.sum((pred_world - target_coords) ** 2, axis=-1) * target_mask
    world_loss = jnp.sum(squared_errors) / denom * 100.0

    # scalar_mag_side head: output ch0 regresses |s| (scaled), ch1 is a side logit
    # (P(s > 0) along the wall axis). Decouples the well-learned magnitude from the
    # side decision that flips on unseen transforms (see experiment.md, exp 3).
    pred_mag = pred_offset[..., 0]
    side_logit = pred_offset[..., 1]
    mag_sq = ((pred_mag - target_s_abs) ** 2) * target_mask
    mag_loss = jnp.sum(mag_sq) / denom * 100.0
    side_bce = optax.sigmoid_binary_cross_entropy(side_logit, target_side) * target_mask
    side_loss = jnp.sum(side_bce) / denom
    mag_side_loss = mag_loss + side_loss

    # mag_side_flag is 1.0 for head_mode scalar_mag_side, 0.0 otherwise; both branches
    # are cheap elementwise ops so blending avoids a static-arg recompile.
    door_loss = world_loss * (1.0 - mag_side_flag) + mag_side_loss * mag_side_flag

    # Pose-distillation aux term (see pose_head in the model). node_pose_mask zeroes
    # padded nodes and the synthetic target door (its coordinate IS the answer -
    # supervising it there would just re-add the failed absolute-head objective).
    # `pose_pred is None` is static (model attribute), so this branch never recompiles.
    if pose_pred is None:
        return door_loss
    pose_denom = jnp.maximum(jnp.sum(node_pose_mask), 1.0)
    pose_sq = jnp.sum((pose_pred - node_coords) ** 2, axis=-1) * node_pose_mask
    pose_loss = jnp.sum(pose_sq) / pose_denom * 100.0
    return door_loss + pose_weight * pose_loss



# The loss function for a batch of graphs, vectorized using vmap
def batch_loss_fn(
    params, apply_fn, type_features, num_nodes, edge_index, edge_types,
    spatial_indices, edge_path_indices, edge_path_mask, target_indices, target_coords, target_mask,
    target_wall_centers, target_wall_proj, target_s_abs, target_side, mag_side_flag,
    node_coords, node_pose_mask, pose_weight
):
    type_features_in_axes = jax.tree_util.tree_map(lambda x: 0, type_features)
    vmapped_loss = jax.vmap(
        lambda tf, ei, et, si, epi, epm, ti, tc, tm, twc, twp, tsa, tsd, nc, npm: graphormer_loss_fn(
            params, apply_fn, tf, num_nodes, ei, et, si, epi, epm, ti, tc, tm, twc, twp, tsa, tsd, mag_side_flag,
            nc, npm, pose_weight
        ),
        in_axes=(type_features_in_axes, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )
    losses = vmapped_loss(
        type_features, edge_index, edge_types, spatial_indices,
        edge_path_indices, edge_path_mask, target_indices, target_coords, target_mask,
        target_wall_centers, target_wall_proj, target_s_abs, target_side,
        node_coords, node_pose_mask
    )
    return jnp.mean(losses)


@partial(jax.jit, static_argnums=(2,))
def train_step(
    state, type_features, num_nodes, edge_index, edge_types,
    spatial_indices, edge_path_indices, edge_path_mask, target_indices, target_coords, target_mask,
    target_wall_centers, target_wall_proj, target_s_abs, target_side, mag_side_flag,
    node_coords, node_pose_mask, pose_weight
):
    grad_fn = jax.value_and_grad(batch_loss_fn)
    loss, grads = grad_fn(
        state.params, state.apply_fn, type_features, num_nodes, edge_index, edge_types,
        spatial_indices, edge_path_indices, edge_path_mask, target_indices, target_coords, target_mask,
        target_wall_centers, target_wall_proj, target_s_abs, target_side, mag_side_flag,
        node_coords, node_pose_mask, pose_weight
    )
    state = state.apply_gradients(grads=grads)
    return state, loss


@partial(jax.jit, static_argnums=(2,))
def compute_loss_and_grads(
    state, type_features, num_nodes, edge_index, edge_types,
    spatial_indices, edge_path_indices, edge_path_mask, target_indices, target_coords, target_mask,
    target_wall_centers, target_wall_proj, target_s_abs, target_side, mag_side_flag,
    node_coords, node_pose_mask, pose_weight
):
    grad_fn = jax.value_and_grad(batch_loss_fn)
    loss, grads = grad_fn(
        state.params, state.apply_fn, type_features, num_nodes, edge_index, edge_types,
        spatial_indices, edge_path_indices, edge_path_mask, target_indices, target_coords, target_mask,
        target_wall_centers, target_wall_proj, target_s_abs, target_side, mag_side_flag,
        node_coords, node_pose_mask, pose_weight
    )
    return loss, grads


@jax.jit
def apply_gradients(state, grads):
    return state.apply_gradients(grads=grads)


@partial(jax.jit, static_argnums=(2,))
def eval_step(
    state, type_features, num_nodes, edge_index, edge_types,
    spatial_indices, edge_path_indices, edge_path_mask, target_indices, target_coords, target_mask,
    target_wall_centers, target_wall_proj, target_s_abs, target_side, mag_side_flag,
    node_coords, node_pose_mask, pose_weight
):
    loss = batch_loss_fn(
        state.params, state.apply_fn, type_features, num_nodes, edge_index, edge_types,
        spatial_indices, edge_path_indices, edge_path_mask, target_indices, target_coords, target_mask,
        target_wall_centers, target_wall_proj, target_s_abs, target_side, mag_side_flag,
        node_coords, node_pose_mask, pose_weight
    )
    return loss


class SDFGraphormerTrainer:
    def __init__(self, config: SimpleNamespace):
        self.config = config
        self.node_type_schemas = None
        self._label_vocab = {}
        self._type_vocab = defaultdict(dict)
        self._idx2id = []

    @classmethod
    def config_load(cls, yaml_path: Path) -> "SDFGraphormerTrainer":
        import yaml
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML config not found: {yaml_path}")
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

    def _apply_arch_override(self, model_arch: Optional[str]) -> None:
        """
        Forces training.graphormer.model_arch in the LOADED config (file untouched).
        Lets menu entries pin an architecture (e.g. 6.2 -> gin) regardless of what
        the run's config snapshot says.
        """
        if not model_arch:
            return
        g = getattr(self.config.training, 'graphormer', None)
        if g is None:
            return
        setattr(g, 'model_arch', model_arch)
        logger.info(f"Model architecture overridden to '{model_arch}' for this invocation.")

    @classmethod
    def execute(cls, config_path: Path, work_folder: Path) -> None:
        trainer = cls.config_load(config_path)
        try:
            trainer._run(work_folder)
        finally:
            del trainer
            logger.info("SDFGraphormerTrainer released.")

    def _infer_schema_from_nodes(self, nodes: List[Dict[str, Any]]):
        by_label = defaultdict(list)
        for n in nodes:
            lab = n.get("label")
            if lab:
                by_label[lab].append(n)

        schema = {}
        all_labels = set(by_label.keys())
        self._label_vocab = {lab: i for i, lab in enumerate(sorted(all_labels))}

        # Define unified categorical and numerical fields for all objects.
        # NOTE: `pos_index` (a door/window's fractional position along its host wall)
        # is deliberately EXCLUDED. For a target door it equals the answer's along-wall
        # component and would let the model place the door analytically from the host
        # wall pose, bypassing genuine embedding/topology reasoning. Dropping it here
        # removes it as a feature for every node (graph JSONs may still contain the
        # field; it is simply never read).
        cat_fields = ["label", "IsExternal", "space_name", "ObjectType", "OperationType", "PredefinedType", "Name", "type"]
        num_fields = ["Height", "Width", "OverallHeight", "OverallWidth"]

        # Build global vocabularies by scanning all nodes in the dataset
        global_vocabs = {}
        for field in ["space_name", "ObjectType", "OperationType", "PredefinedType", "Name", "type"]:
            global_vocabs[field] = {}

        for n in nodes:
            lab = n.get("label")
            if not lab:
                continue

            # space_name (only populated for spaces, otherwise None)
            space_name = n.get("Name") if lab in ("IfcSpace", "Space") else None
            if space_name is not None:
                val_str = str(space_name)
                if val_str not in global_vocabs["space_name"]:
                    global_vocabs["space_name"][val_str] = len(global_vocabs["space_name"]) + 1

            # ObjectType, OperationType, PredefinedType, Name, type
            for field in ["ObjectType", "OperationType", "PredefinedType", "Name", "type"]:
                val = n.get(field)
                if val is not None and val != "NONE":
                    val_str = str(val)
                    if val_str not in global_vocabs[field]:
                        global_vocabs[field][val_str] = len(global_vocabs[field]) + 1

        # We set type_vocab[lab] to point to the shared global_vocabs so _get_cat_index can read/write them
        self._type_vocab = defaultdict(dict)
        for lab in all_labels:
            self._type_vocab[lab] = global_vocabs

        # Calculate cat_dims based on vocabulary sizes
        cat_dims = [
            len(self._label_vocab), # label
            3, # IsExternal (0, 1, 2)
            len(global_vocabs["space_name"]) + 1,
            len(global_vocabs["ObjectType"]) + 1,
            len(global_vocabs["OperationType"]) + 1,
            len(global_vocabs["PredefinedType"]) + 1,
            len(global_vocabs["Name"]) + 1,
            len(global_vocabs["type"]) + 1
        ]

        for lab in all_labels:
            schema[lab] = {
                "categorical_dims": cat_dims,
                "categorical_fields": cat_fields,
                "numeric": num_fields,
                "geometry_dim": 128,
                "coordinates_dim": 2,
            }
        return schema

    def _label_index(self, lab: str) -> int:
        return self._label_vocab.get(lab, 0)

    def _get_cat_index(self, node_label: str, field_name: str, val: Any) -> int:
        if field_name == "IsExternal":
            if val is None:
                return 2
            try:
                return int(val)
            except Exception:
                return 2
        
        if val is None or val == "NONE":
            return 0
        val_str = str(val)
        vocab = self._type_vocab[node_label]
        if field_name not in vocab:
            vocab[field_name] = {}
        sub_vocab = vocab[field_name]
        return sub_vocab.get(val_str, 0)

    def _filter_and_index_nodes(self, nodes_all: List[Dict[str, Any]]):
        target_classes = getattr(self.config.extraction, 'target_classes', [])
        nodes = []
        for n in nodes_all:
            lab = n.get("label")
            if lab in target_classes:
                nodes.append(n)

        nodes.sort(key=lambda n: n.get("id", ""))
        id2idx = {str(n["id"]): i for i, n in enumerate(nodes)}
        idx2id = [str(n["id"]) for n in nodes]
        return nodes, id2idx, idx2id

    def _make_edge_tensors(self, links: List[Dict[str, Any]], id2idx: Dict[str, int]):
        edges = []
        etypes = []
        
        # NOTE: keys must cover the link types actually emitted by the graph
        # generators. The combinatorial generator emits SEPARATES / CONNECTS /
        # HAS_DOOR / HAS_WINDOW / INTERNAL_TO; before these were added, every
        # edge fell through to the generic 3 and all edge-type information was
        # lost. Indices must stay < num_edge_types (10) in the model.
        type_map = {
            "contained": 1,
            "boundary": 2,
            "connected": 3,
            "voids": 4,
            "fills": 5,
            "connects": 3,
            "separates": 6,
            "has_door": 7,
            "has_window": 8,
            "internal_to": 9,
        }
        
        for l in links:
            s_str = str(l.get("source") or l.get("start"))
            t_str = str(l.get("target") or l.get("end"))
            if s_str in id2idx and t_str in id2idx:
                s_idx = id2idx[s_str]
                t_idx = id2idx[t_str]
                edges.append((s_idx, t_idx))
                
                type_str = l.get("type", "connected").lower()
                etypes.append(type_map.get(type_str, 3))

        if not edges:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_types = torch.zeros(0, dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_types = torch.tensor(etypes, dtype=torch.long)
            
        return edge_index, edge_types

    def _get_feature_info(self, node_type: str) -> List[Tuple[str, int]]:
        schema = self.node_type_schemas.get(node_type, {})
        features = []
        
        cat_fields = schema.get('categorical_fields', ['label', 'type'])
        for field in cat_fields:
            features.append((field, 1))
        
        numeric_list = schema.get('numeric', [])
        for name in numeric_list:
            features.append((name, 1))
        
        geom_dim = schema.get('geometry_dim', 128)
        features.append(('geometry', geom_dim))
        return features

    def _model_arch(self) -> str:
        """
        Graph encoder architecture: 'graphormer' (default) | 'gin' (GINE
        message-passing baseline; see GNNBaselineModel).
        """
        g = getattr(self.config.training, 'graphormer', None)
        arch = getattr(g, 'model_arch', 'graphormer') if g is not None else 'graphormer'
        return str(arch or 'graphormer').strip().lower()

    def _split_ckpt_dir(self, work_folder: Path) -> Path:
        """
        Arch-specific split-checkpoint dir, so retraining a baseline arch on the
        same run folder cannot clobber the Graphormer's best checkpoint.
        """
        arch = self._model_arch()
        name = "graphormer_jax_split_checkpoints" if arch == "graphormer" else f"{arch}_jax_split_checkpoints"
        return (Path(work_folder) / "intermediate" / name).resolve()

    def _head_mode(self) -> str:
        """Prediction head mode: 'absolute' | 'offset' | 'scalar_s' | 'scalar_mag_side'
        (see experiment.md). Falls back to the legacy `residual_anchored` bool if
        `head_mode` is unset."""
        g = getattr(self.config.training, "graphormer", None)
        hm = getattr(g, "head_mode", None)
        if hm in ("absolute", "offset", "scalar_s", "scalar_mag_side"):
            return hm
        return "offset" if bool(getattr(g, "residual_anchored", True)) else "absolute"

    def _prepare_graph(self, nodes_all: List[Dict[str, Any]], links: List[Dict[str, Any]]) -> GraphBatch:
        nodes, id2idx, idx2id = self._filter_and_index_nodes(nodes_all)

        node_features = OrderedDict()
        node_types = []
        door_indices = []
        door_latents = []
        door_coords = []
        door_wall_centers = []
        door_wall_axes = []
        door_s_list = []

        for idx, n in enumerate(nodes):
            lab = n["label"]
            node_types.append(lab)
            sch = self.node_type_schemas[lab]

            cat_indices = []
            for field in sch["categorical_fields"]:
                if field == "label":
                    cat_indices.append(self._label_index(lab))
                elif field == "space_name":
                    val = n.get("Name") if lab in ("IfcSpace", "Space") else None
                    cat_indices.append(self._get_cat_index(lab, field, val))
                else:
                    cat_indices.append(self._get_cat_index(lab, field, n.get(field)))
            cat_tensor = torch.tensor(cat_indices, dtype=torch.long)

            num_list = []
            for fname in sch["numeric"]:
                val = n.get(fname, 0.0)
                if val is None: val = 0.0
                num_list.append(float(val))

            geo_vec = n.get("geometry") or []
            gdim = int(sch.get("geometry_dim", 128))
            if len(geo_vec) < gdim:
                geo_vec = list(geo_vec) + [0.0] * (gdim - len(geo_vec))
            else:
                geo_vec = list(geo_vec[:gdim])

            coord_val = n.get("coordinates") or [0.0, 0.0]
            scale = getattr(self.config.extraction, 'scale', 1.0)
            coord_val_scaled = [coord_val[0] * scale, coord_val[1] * scale]
            node_id = str(n["id"])
            node_features[node_id] = {
                "categorical": cat_tensor,
                "numeric": torch.tensor(num_list, dtype=torch.float32) if num_list else torch.empty(0),
                "geometry": torch.tensor(geo_vec, dtype=torch.float32),
                "coordinates": torch.tensor(coord_val_scaled, dtype=torch.float32),
            }

            # Target-door ids: the two legacy GUIDs (old graph data) and the synthetic
            # placeholder door emitted by the six-window combinatorial generator.
            if lab in ("IfcDoor", "Door") and node_id in (
                "0XVTLR_9H3Ox9MAKGyWioT", "0i$VJnJOTDM92a$t1VH2bb", "TARGET_DOOR_SYNTH"
            ):
                door_indices.append(idx)
                door_latents.append(geo_vec)
                door_coords.append(coord_val_scaled)

                # Plan-B wall-relative target: anchor (scaled), unit axis, signed offset (scaled).
                wc = n.get("host_wall_center") or coord_val  # fall back to door coord if no host wall
                wc_scaled = [wc[0] * scale, wc[1] * scale]
                wa = n.get("host_wall_axis") or [1.0, 0.0]
                wa_norm = math.hypot(wa[0], wa[1]) or 1.0
                wa_unit = [wa[0] / wa_norm, wa[1] / wa_norm]
                # s in raw world meters -> scale to match scaled coordinates.
                s_raw = n.get("wall_s")
                if s_raw is None:
                    # recover from geometry: s = (coord - wall_center) . axis  (in scaled space)
                    s_scaled = (coord_val_scaled[0] - wc_scaled[0]) * wa_unit[0] + (coord_val_scaled[1] - wc_scaled[1]) * wa_unit[1]
                else:
                    s_scaled = float(s_raw) * scale
                door_wall_centers.append(wc_scaled)
                door_wall_axes.append(wa_unit)
                door_s_list.append(s_scaled)

        edge_index, edge_types = self._make_edge_tensors(links, id2idx)
        door_latent_codes = (
            torch.tensor(door_latents, dtype=torch.float32)
            if door_latents
            else torch.empty((0, 128), dtype=torch.float32)
        )
        door_target_coords = (
            torch.tensor(door_coords, dtype=torch.float32)
            if door_coords
            else torch.empty((0, 2), dtype=torch.float32)
        )
        # Prediction anchor + projection, encoding the head mode (see experiment.md):
        #   absolute : anchor 0 , proj I           -> world = pred (regress full coordinate)
        #   offset   : anchor wc, proj I           -> world = wc + pred (2D residual)
        #   scalar_s : anchor wc, proj axis e_x^T  -> world = wc + pred_x axis (1-DOF along wall,
        #              transform-invariant: the model never needs the world axis direction)
        #   scalar_mag_side : anchor wc, proj as scalar_s (composition happens outside the
        #              generic path: pred_0 = |s|, pred_1 = side logit -> world = wc +/- |s| axis;
        #              the loss and eval branch on this mode explicitly)
        head_mode = self._head_mode()
        D = len(door_wall_centers)
        if D:
            centers = torch.tensor(door_wall_centers, dtype=torch.float32)   # [D, 2]
            axes = torch.tensor(door_wall_axes, dtype=torch.float32)         # [D, 2]
            if head_mode == "absolute":
                door_wall_centers_t = torch.zeros((D, 2), dtype=torch.float32)
                door_wall_proj_t = torch.eye(2, dtype=torch.float32).expand(D, 2, 2).contiguous()
            elif head_mode in ("scalar_s", "scalar_mag_side"):
                door_wall_centers_t = centers
                # world = wc + pred_x * axis: the model emits the along-wall scalar in
                # its FIRST output component; the external axis supplies the world
                # orientation. (axis outer axis was wrong: it made the effective output
                # axis.pred, which requires the model to know the copy's rotation --
                # the same product trap as the offset head.)
                e_x = torch.tensor([1.0, 0.0], dtype=torch.float32)
                door_wall_proj_t = torch.einsum("di,j->dij", axes, e_x)       # axis outer e_x
            else:  # offset
                door_wall_centers_t = centers
                door_wall_proj_t = torch.eye(2, dtype=torch.float32).expand(D, 2, 2).contiguous()
            door_wall_centers_true_t = centers  # real center regardless of head mode (metrics only)
        else:
            door_wall_centers_t = torch.empty((0, 2), dtype=torch.float32)
            door_wall_proj_t = torch.empty((0, 2, 2), dtype=torch.float32)
            door_wall_centers_true_t = torch.empty((0, 2), dtype=torch.float32)
        door_wall_axes_t = (
            torch.tensor(door_wall_axes, dtype=torch.float32)
            if door_wall_axes
            else torch.empty((0, 2), dtype=torch.float32)
        )
        door_s_t = (
            torch.tensor(door_s_list, dtype=torch.float32)
            if door_s_list
            else torch.empty((0,), dtype=torch.float32)
        )

        # Precompute shortest paths
        temp_data = Data(edge_index=edge_index, num_nodes=len(node_types))
        try:
            node_paths, edge_paths = shortest_path_distance(temp_data)
        except Exception as e:
            logger.warning(f"Failed to precompute shortest paths in _prepare_graph: {e}")
            node_paths, edge_paths = {}, {}
            
        N = len(node_types)
        max_path_distance = 10
        if hasattr(self.config.training, 'graphormer'):
            max_path_distance = int(getattr(self.config.training.graphormer, 'max_path_distance', 10))
            
        spatial_indices = np.full((N, N), max_path_distance, dtype=np.int32)
        for src in node_paths:
            for dst in node_paths[src]:
                spatial_indices[src][dst] = min(len(node_paths[src][dst]), max_path_distance) - 1
                
        num_edges = int(edge_index.shape[1])
        edge_path_indices = np.full((N, N, max_path_distance), num_edges, dtype=np.int32)
        edge_path_mask = np.zeros((N, N, max_path_distance), dtype=np.float32)
        for src in edge_paths:
            for dst in edge_paths[src]:
                path_ij = edge_paths[src][dst][:max_path_distance]
                for k, edge_idx in enumerate(path_ij):
                    edge_path_indices[src][dst][k] = edge_idx
                    edge_path_mask[src][dst][k] = 1.0

        # Construct flat features X for attribution
        feature_splits_by_type = {}
        for nt in set(node_types):
            feat_info = self._get_feature_info(nt)
            splits = []
            offset = 0
            for name, dim in feat_info:
                splits.append((name, (offset, offset + dim)))
                offset += dim
            feature_splits_by_type[nt] = (splits, offset)

        max_dim = max(total for _, total in feature_splits_by_type.values())
        X_list = []
        for i, node_id in enumerate(idx2id):
            nt = node_types[i]
            feat_dict = node_features[node_id]
            splits, total_dim = feature_splits_by_type[nt]
            
            row = np.zeros(max_dim, dtype=np.float32)
            for feat_name, (start, end) in splits:
                if feat_name in feat_dict:
                    feat_val = feat_dict[feat_name].numpy()
                    row[start:start+len(feat_val.flatten())] = feat_val.flatten()
            X_list.append(row)
        X = np.stack(X_list, axis=0)

        return GraphBatch(
            node_features=node_features,
            node_types=node_types,
            edge_index=edge_index,
            edge_types=edge_types,
            door_indices=door_indices,
            door_latent_codes=door_latent_codes,
            door_target_coords=door_target_coords,
            door_wall_centers=door_wall_centers_t,
            door_wall_proj=door_wall_proj_t,
            door_wall_axes=door_wall_axes_t,
            door_wall_centers_true=door_wall_centers_true_t,
            door_s=door_s_t,
            idx2id=idx2id,
            spatial_indices=spatial_indices,
            edge_path_indices=edge_path_indices,
            edge_path_mask=edge_path_mask,
            X=X,
            feature_splits=feature_splits_by_type,
            max_dim=max_dim
        )

    def _get_jax_type_features(self, B: GraphBatch):
        type_features = {}
        type_to_indices = defaultdict(list)
        for i, t in enumerate(B.node_types):
            type_to_indices[t].append(i)
            
        for n_type, indices in type_to_indices.items():
            if n_type not in self.node_type_schemas:
                continue
            schema = self.node_type_schemas[n_type]
            cat_list = []
            num_list = []
            geom_list = []
            coord_list = []
            
            for idx in indices:
                node_id = B.idx2id[idx]
                feats = B.node_features[node_id]
                cat_list.append(feats['categorical'].numpy())
                if feats['numeric'].numel() > 0:
                    num_list.append(feats['numeric'].numpy())
                geom_list.append(feats['geometry'].numpy())
                coord_list.append(feats['coordinates'].numpy())
                
            type_features[n_type] = {
                'categorical': jnp.array(np.stack(cat_list), dtype=jnp.int32),
                'numeric': jnp.array(np.stack(num_list), dtype=jnp.float32) if num_list else None,
                'geometry': jnp.array(np.stack(geom_list), dtype=jnp.float32),
                'coordinates': jnp.array(np.stack(coord_list), dtype=jnp.float32),
                'indices': jnp.array(indices, dtype=jnp.int32)
            }
        return type_features

    # ===== Shared strict-padding helpers (used by split training AND evaluation) =====

    def _compute_padding_specs(self, batches: List[GraphBatch], max_path_distance: int) -> Dict[str, Any]:
        """
        Computes the global maximum node/edge/door counts across all batches so every
        graph can be padded to identical shapes (required for jnp.stack collation and
        a single JIT compilation). Also registers the synthetic 'PAD' node type schema.
        """
        max_nodes_per_type = defaultdict(int)
        for B_item in batches:
            type_counts = defaultdict(int)
            for t in B_item.node_types:
                type_counts[t] += 1
            for t, count in type_counts.items():
                max_nodes_per_type[t] = max(max_nodes_per_type[t], count)

        max_nodes = max(len(B_item.node_types) for B_item in batches)
        max_nodes_per_type["PAD"] = max_nodes
        max_edges = max(B_item.edge_index.shape[1] for B_item in batches)
        max_doors = max(len(B_item.door_indices) for B_item in batches)

        if "PAD" not in self.node_type_schemas:
            self.node_type_schemas["PAD"] = {
                "categorical_fields": ["label"],
                "categorical_dims": [1],
                "numeric": [],
                "geometry_dim": 128,
            }

        return {
            "max_nodes_per_type": max_nodes_per_type,
            "max_nodes": max_nodes,
            "max_edges": max_edges,
            "max_doors": max_doors,
            "max_path_distance": max_path_distance,
        }

    def _to_host_batch(self, B: GraphBatch, specs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pads a single GraphBatch to the global maximum shapes described by `specs`,
        keeping every array in host memory (numpy). Target-door geometry AND
        coordinates are zeroed here (self-leakage masking).
        """
        max_nodes_per_type = specs["max_nodes_per_type"]
        max_nodes = specs["max_nodes"]
        max_edges = specs["max_edges"]
        max_doors = specs["max_doors"]
        max_path_distance = specs["max_path_distance"]

        N = len(B.node_types)

        type_to_indices = defaultdict(list)
        for idx, t in enumerate(B.node_types):
            type_to_indices[t].append(idx)
        type_to_indices["PAD"] = list(range(N, max_nodes))

        padded_type_features = {}
        for n_type, max_count in max_nodes_per_type.items():
            indices = list(type_to_indices[n_type])

            sch = self.node_type_schemas[n_type]
            cat_cols = len(sch.get("categorical_fields", []))
            num_cols = len(sch.get("numeric", []))
            gdim = int(sch.get("geometry_dim", 128))

            cat_list, num_list, geom_list, coord_list = [], [], [], []

            for idx in indices:
                if idx < N:
                    node_id = B.idx2id[idx]
                    feats = B.node_features[node_id]
                    cat_list.append(feats['categorical'].numpy())
                    if num_cols > 0:
                        num_list.append(feats['numeric'].numpy())

                    geom_val = feats['geometry'].numpy()
                    coord_val = feats['coordinates'].numpy()

                    if idx in B.door_indices:
                        geom_val = np.zeros(gdim, dtype=np.float32)
                        coord_val = np.zeros(2, dtype=np.float32)

                    geom_list.append(geom_val)
                    coord_list.append(coord_val)
                else:
                    cat_list.append(np.zeros(cat_cols, dtype=np.int32))
                    if num_cols > 0:
                        num_list.append(np.zeros(num_cols, dtype=np.float32))
                    geom_list.append(np.zeros(gdim, dtype=np.float32))
                    coord_list.append(np.zeros(2, dtype=np.float32))

            while len(cat_list) < max_count:
                cat_list.append(np.zeros(cat_cols, dtype=np.int32))
                if num_cols > 0:
                    num_list.append(np.zeros(num_cols, dtype=np.float32))
                geom_list.append(np.zeros(gdim, dtype=np.float32))
                coord_list.append(np.zeros(2, dtype=np.float32))
                indices.append(max_nodes - 1)

            padded_type_features[n_type] = {
                'categorical': np.stack(cat_list).astype(np.int32),
                'numeric': np.stack(num_list).astype(np.float32) if num_cols > 0 else None,
                'geometry': np.stack(geom_list).astype(np.float32),
                'coordinates': np.stack(coord_list).astype(np.float32),
                'indices': np.array(indices, dtype=np.int32),
            }

        edge_idx_np = B.edge_index.numpy()
        edge_types_np = B.edge_types.numpy()
        E = edge_idx_np.shape[1]

        padded_edge_index = np.full((2, max_edges), max_nodes - 1, dtype=np.int32)
        padded_edge_index[:, :E] = edge_idx_np

        padded_edge_types = np.zeros(max_edges, dtype=np.int32)
        padded_edge_types[:E] = edge_types_np

        padded_spatial = np.full((max_nodes, max_nodes), max_path_distance, dtype=np.int32)
        padded_spatial[:N, :N] = B.spatial_indices

        padded_path_indices = np.full((max_nodes, max_nodes, max_path_distance), max_edges - 1, dtype=np.int32)
        padded_path_indices[:N, :N, :] = B.edge_path_indices

        padded_path_mask = np.zeros((max_nodes, max_nodes, max_path_distance), dtype=np.float32)
        padded_path_mask[:N, :N, :] = B.edge_path_mask

        real_doors = len(B.door_indices)
        padded_door_indices = list(B.door_indices)
        while len(padded_door_indices) < max_doors:
            padded_door_indices.append(max_nodes - 1)

        door_target_coords_np = B.door_target_coords.numpy()
        padded_door_coords = np.zeros((max_doors, 2), dtype=np.float32)
        padded_door_coords[:real_doors, :] = door_target_coords_np

        # Pad Plan-B wall anchor / projection / axis (dummy doors get identity proj + unit axis to avoid NaNs)
        padded_wall_centers = np.zeros((max_doors, 2), dtype=np.float32)
        padded_wall_centers_true = np.zeros((max_doors, 2), dtype=np.float32)
        padded_wall_proj = np.tile(np.eye(2, dtype=np.float32), (max_doors, 1, 1))
        padded_wall_axes = np.tile(np.array([1.0, 0.0], dtype=np.float32), (max_doors, 1))
        # scalar_mag_side targets: |s| (scaled) and side class (1 if s > 0 else 0)
        padded_s_abs = np.zeros(max_doors, dtype=np.float32)
        padded_side = np.zeros(max_doors, dtype=np.float32)
        if real_doors > 0:
            padded_wall_centers[:real_doors, :] = B.door_wall_centers.numpy()
            padded_wall_centers_true[:real_doors, :] = B.door_wall_centers_true.numpy()
            padded_wall_proj[:real_doors, :, :] = B.door_wall_proj.numpy()
            padded_wall_axes[:real_doors, :] = B.door_wall_axes.numpy()
            s_np = B.door_s.numpy()
            padded_s_abs[:real_doors] = np.abs(s_np)
            padded_side[:real_doors] = (s_np > 0.0).astype(np.float32)

        door_loss_mask = np.zeros(max_doors, dtype=np.float32)
        door_loss_mask[:real_doors] = 1.0

        # Pose-distillation targets in GLOBAL node-index space (aligned with the
        # model's per-node output). Mask covers real nodes only, minus the target
        # door(s), whose coordinate is the prediction target itself.
        node_coords = np.zeros((max_nodes, 2), dtype=np.float32)
        node_pose_mask = np.zeros(max_nodes, dtype=np.float32)
        for idx in range(N):
            node_coords[idx] = B.node_features[B.idx2id[idx]]['coordinates'].numpy()
            node_pose_mask[idx] = 1.0
        for d_idx in B.door_indices:
            node_coords[d_idx] = 0.0
            node_pose_mask[d_idx] = 0.0

        return {
            'B_ref': B,
            'type_features': padded_type_features,
            'num_nodes': max_nodes,
            'edge_index': padded_edge_index,
            'edge_types': padded_edge_types,
            'spatial_indices': padded_spatial,
            'edge_path_indices': padded_path_indices,
            'edge_path_mask': padded_path_mask,
            'door_indices': padded_door_indices,
            'door_target_coords': padded_door_coords,
            'door_wall_centers': padded_wall_centers,
            'door_wall_centers_true': padded_wall_centers_true,  # metrics only, never on device
            'door_wall_proj': padded_wall_proj,
            'door_wall_axes': padded_wall_axes,
            'door_s_abs': padded_s_abs,
            'door_side': padded_side,
            'door_loss_mask': door_loss_mask,
            'node_coords': node_coords,
            'node_pose_mask': node_pose_mask,
        }

    # Fields that must stay on host: python ints / lists (static jit args, index
    # lookups) and metrics-only arrays.
    _HOST_ONLY_FIELDS = ('B_ref', 'type_features', 'num_nodes', 'door_indices',
                         'door_wall_centers_true')

    def _device_put_batch(self, hb: Dict[str, Any]) -> Dict[str, Any]:
        """Pushes a host batch's tensor fields to the JAX device."""
        out = dict(hb)
        out['type_features'] = {
            n_type: {k: (jax.device_put(v) if isinstance(v, np.ndarray) else v)
                     for k, v in fields.items()}
            for n_type, fields in hb['type_features'].items()
        }
        for k, v in hb.items():
            if k not in self._HOST_ONLY_FIELDS and isinstance(v, np.ndarray):
                out[k] = jax.device_put(v)
        return out

    def _to_gpu_batch(self, B: GraphBatch, specs: Dict[str, Any]) -> Dict[str, Any]:
        """Pads a single GraphBatch and pushes all arrays to the JAX device."""
        return self._device_put_batch(self._to_host_batch(B, specs))

    def _build_model_from_config(self):
        """Builds a LatentCodePredictionModel using the Graphormer config block (with defaults)."""
        graphormer_cfg = getattr(self.config.training, 'graphormer', None)
        defaults = dict(edge_dim=64, d_model=256, n_heads=8, node_depth=2, graph_depth=4,
                        ff_dim=512, dropout=0.1, max_in_degree=50, max_out_degree=50, max_path_distance=10)
        vals = dict(defaults)
        if graphormer_cfg is not None:
            for k in defaults:
                if hasattr(graphormer_cfg, k):
                    v = getattr(graphormer_cfg, k)
                    vals[k] = float(v) if k == 'dropout' else int(v)
        pose_w = float(getattr(graphormer_cfg, 'pose_distill_weight', 0.0) or 0.0) if graphormer_cfg is not None else 0.0
        model_cls = GNNBaselineModel if self._model_arch() != 'graphormer' else LatentCodePredictionModel
        if model_cls is GNNBaselineModel and graphormer_cfg is not None:
            # Local message passing gets its own (typically deeper) depth so its
            # receptive field can span door-wall -> determining-node paths that the
            # Graphormer covers with dense attention. Falls back to graph_depth.
            vals['graph_depth'] = int(getattr(graphormer_cfg, 'gin_graph_depth', vals['graph_depth']))
        model = model_cls(node_type_schemas=self.node_type_schemas,
                          use_pose_head=(pose_w > 0.0), **vals)
        return model, vals['max_path_distance']

    # ===== Full test-set evaluation (Step 6) =====

    @classmethod
    def evaluate_test_set(cls, config_path: Path, work_folder: Path, model_arch: Optional[str] = None) -> None:
        """
        Loads the best split checkpoint, recovers the exact 60/20/20 split, runs the
        model over the entire test split and writes a detailed per-door metrics report
        to output/evaluation_results/test_evaluation_report.json.
        """
        trainer = cls.config_load(config_path)
        trainer._apply_arch_override(model_arch)
        try:
            trainer._run_test_evaluation(Path(work_folder))
        finally:
            del trainer

    @classmethod
    def evaluate_holdout_set(cls, config_path: Path, work_folder: Path, model_arch: Optional[str] = None) -> None:
        """
        Evaluates the trained model on the entirely unseen held-out copies under
        output/holdout/ (an unseen rigid transform of the plan), writing
        output/evaluation_results/holdout_evaluation_report.json for comparison
        against the in-distribution test split. No-op with a warning if no holdout set exists.
        """
        holdout_dir = Path(work_folder) / "output" / "holdout"
        if not holdout_dir.exists() or not any(holdout_dir.glob("graph_room*.json")):
            logger.warning(f"No held-out graphs under {holdout_dir}; skipping holdout evaluation.")
            return
        trainer = cls.config_load(config_path)
        trainer._apply_arch_override(model_arch)
        try:
            trainer._run_test_evaluation(Path(work_folder), holdout=True)
        finally:
            del trainer

    def _run_test_evaluation(self, work_folder: Path, holdout: bool = False) -> None:
        import math as _math
        work_folder = Path(work_folder)
        label = "held-out (unseen-transform)" if holdout else "test-split"
        report_name = "holdout_evaluation_report.json" if holdout else "test_evaluation_report.json"
        logger.info(f"=== Running {label} evaluation ===")

        def _load_graphs(files):
            loaded = []
            for gf in files:
                try:
                    with open(gf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "graphs" in data:
                        for g in data["graphs"]:
                            loaded.append((g.get("nodes", []), g.get("links", [])))
                    else:
                        loaded.append((data.get("nodes", []), data.get("links", [])))
                except Exception as e:
                    logger.error(f"Failed to load graph file {gf.name}: {e}")
            return loaded

        # 1. Training graphs (output/, non-recursive so holdout/ is excluded) rebuild the
        #    schema+vocab EXACTLY as during training so categorical encodings match the
        #    checkpoint -- required also for holdout, whose model never saw these copies.
        output_dir = work_folder / "output"
        train_files = sorted(list(output_dir.glob("graph_room*.json"))) or sorted(list(output_dir.glob("*_graph.json")))
        if not train_files:
            logger.error(f"No training graphs found in: {output_dir}")
            return
        logger.info(f"Loading {len(train_files)} training graph files for schema/vocab...")
        train_graphs = _load_graphs(train_files)
        if not train_graphs:
            logger.error("No valid training graphs loaded.")
            return
        merged_nodes = []
        for nodes, _ in train_graphs:
            merged_nodes.extend(nodes)
        self.node_type_schemas = self._infer_schema_from_nodes(merged_nodes)

        # 2. Assemble the evaluation batch set.
        if holdout:
            holdout_dir = output_dir / "holdout"
            eval_files = sorted(list(holdout_dir.glob("graph_room*.json")))
            if not eval_files:
                logger.error(f"No held-out graphs in {holdout_dir}. Run 3.2 with extraction.holdout_copies > 0 first.")
                return
            logger.info(f"Loading {len(eval_files)} held-out graph files (evaluating ALL, no split)...")
            batches = []
            for nodes, links in _load_graphs(eval_files):
                B = self._prepare_graph(nodes, links)
                if B.door_indices:
                    batches.append(B)
            if not batches:
                logger.error("No held-out batches containing doors prepared.")
                return
            test_indices = list(range(len(batches)))
            logger.info(f"Held-out set | Graphs with doors: {len(batches)}")
        else:
            batches = []
            for nodes, links in train_graphs:
                B = self._prepare_graph(nodes, links)
                if B.door_indices:
                    batches.append(B)
            if not batches:
                logger.error("No batches containing doors prepared.")
                return
            # Recover the identical shuffle/split (matches _run_with_split ordering)
            seed = getattr(self.config.training.reproducibility, 'seed', 42)
            random.seed(seed)
            np.random.seed(seed)
            shuffled_indices = list(range(len(batches)))
            random.shuffle(shuffled_indices)
            num_total = len(batches)
            num_train = int(0.6 * num_total)
            num_val = int(0.2 * num_total)
            test_indices = shuffled_indices[num_train + num_val:]
            logger.info(f"Split recovered | Total: {num_total} | Train: {num_train} | Val: {num_val} | Test: {len(test_indices)}")

        # 3. Model + padding specs
        model, max_path_distance = self._build_model_from_config()
        specs = self._compute_padding_specs(batches, max_path_distance)
        seed = getattr(self.config.training.reproducibility, 'seed', 42)

        # 6. Init params, restore best checkpoint
        init_gpu = self._to_gpu_batch(batches[test_indices[0] if test_indices else 0], specs)
        rng = jax.random.PRNGKey(seed)
        params = model.init(
            rng,
            type_features=init_gpu['type_features'],
            num_nodes=init_gpu['num_nodes'],
            edge_index=init_gpu['edge_index'],
            edge_types=init_gpu['edge_types'],
            spatial_indices=init_gpu['spatial_indices'],
            edge_path_indices=init_gpu['edge_path_indices'],
            edge_path_mask=init_gpu['edge_path_mask'],
        )['params']
        optimizer = optax.adam(learning_rate=2e-4)
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

        ckpt_dir = self._split_ckpt_dir(work_folder)
        # Legacy fallback applies only to the graphormer arch; a baseline arch must
        # never silently restore a Graphormer checkpoint (param trees differ anyway).
        if self._model_arch() == "graphormer" and (not ckpt_dir.exists() or not any(ckpt_dir.iterdir())):
            ckpt_dir = (work_folder / "intermediate" / "graphormer_jax_checkpoints").resolve()
        if not ckpt_dir.exists():
            logger.error("No checkpoint directories found to restore for evaluation.")
            return
        logger.info(f"Restoring best checkpoint from: {ckpt_dir}")
        best_state = checkpoints.restore_checkpoint(ckpt_dir=str(ckpt_dir), target=state, prefix="best_")
        if best_state is None:
            best_state = checkpoints.restore_checkpoint(ckpt_dir=str(ckpt_dir), target=state)
        if best_state is None:
            logger.error("Failed to restore any checkpoint.")
            return
        state = best_state
        logger.success("Checkpoint restored for validation.")

        @partial(jax.jit, static_argnums=(2,))
        def predict_coords_fn(params, type_features, num_nodes, edge_index, edge_types,
                              spatial_indices, edge_path_indices, edge_path_mask, target_indices):
            pred_coords, _, _, _ = model.apply(
                {'params': params},
                type_features=type_features,
                num_nodes=num_nodes,
                edge_index=edge_index,
                edge_types=edge_types,
                spatial_indices=spatial_indices,
                edge_path_indices=edge_path_indices,
                edge_path_mask=edge_path_mask,
                target_indices=target_indices,
            )
            return pred_coords

        scale = getattr(self.config.extraction, 'scale', 1.0)
        results_list = []
        for eval_idx, batch_i in enumerate(tqdm(test_indices, desc=f"Evaluating {label}")):
            gb = self._to_gpu_batch(batches[batch_i], specs)
            pred = predict_coords_fn(
                state.params,
                gb['type_features'],
                gb['num_nodes'],
                gb['edge_index'],
                gb['edge_types'],
                gb['spatial_indices'],
                gb['edge_path_indices'],
                gb['edge_path_mask'],
                jnp.array(gb['door_indices'], dtype=jnp.int32),
            )
            # Head-mode composition: world = anchor + proj @ pred (matches training; see _prepare_graph).
            pred_offset_np = np.array(pred)                     # [D, 2]
            wc_np = np.array(gb['door_wall_centers'])           # [D, 2] anchor
            proj_np = np.array(gb['door_wall_proj'])            # [D, 2, 2]
            true_np = np.array(gb['door_target_coords'])
            axis_np = np.array(gb['door_wall_axes'])            # [D, 2] real wall axis
            if self._head_mode() == "scalar_mag_side":
                # ch0 = |s| (scaled), ch1 = side logit: world = wc + sign(logit)*|s|*axis
                s_signed = np.where(pred_offset_np[:, 1] > 0.0, 1.0, -1.0) * pred_offset_np[:, 0]
                pred_np = wc_np + s_signed[:, None] * axis_np   # [D, 2] world (scaled)
            else:
                pred_np = wc_np + np.einsum('dij,dj->di', proj_np, pred_offset_np)  # [D, 2] world (scaled)
            wct_np = np.array(gb['door_wall_centers_true'])     # [D, 2] real wall center (all modes)
            loss_mask_np = np.array(gb['door_loss_mask'])
            for door_idx_local, node_idx_global in enumerate(gb['door_indices']):
                if loss_mask_np[door_idx_local] == 0.0:
                    continue
                p_coord = pred_np[door_idx_local] / scale
                t_coord = true_np[door_idx_local] / scale
                offset_pred = pred_offset_np[door_idx_local] / scale
                dx = float(p_coord[0] - t_coord[0])
                dy = float(p_coord[1] - t_coord[1])
                sq_err = dx ** 2 + dy ** 2
                abs_err = abs(dx) + abs(dy)
                euc = _math.sqrt(sq_err)
                # Side (left/right) = sign of the along-wall offset relative to the real wall center.
                # This isolates the semantically-determined placement from the given wall frame.
                ax = axis_np[door_idx_local]
                wct = wct_np[door_idx_local]
                s_true = float(np.dot(true_np[door_idx_local] - wct, ax))
                s_pred = float(np.dot(pred_np[door_idx_local] - wct, ax))
                side_correct = bool(np.sign(s_true) == np.sign(s_pred))
                door_id = gb['B_ref'].idx2id[node_idx_global] if node_idx_global < len(gb['B_ref'].idx2id) else f"unknown_{node_idx_global}"
                results_list.append({
                    'eval_batch_idx': eval_idx,
                    'graph_batch_idx': int(batch_i),
                    'door_node_idx': int(node_idx_global),
                    'door_id': door_id,
                    'pred': p_coord.tolist(),
                    'true': t_coord.tolist(),
                    'pred_offset': offset_pred.tolist(),
                    'pred_offset_mag': float(np.linalg.norm(offset_pred)),
                    'dx': dx, 'dy': dy,
                    'squared_error': float(sq_err),
                    'absolute_error': float(abs_err),
                    'euclidean_error': float(euc),
                    'wall_s_true': s_true / scale,
                    'wall_s_pred': s_pred / scale,
                    'side_correct': side_correct,
                })

        if not results_list:
            logger.error("No predictions computed.")
            return

        squared_errors = [r['squared_error'] for r in results_list]
        absolute_errors = [r['absolute_error'] for r in results_list]
        euclidean_errors = [r['euclidean_error'] for r in results_list]
        dxs = [r['dx'] for r in results_list]
        dys = [r['dy'] for r in results_list]

        side_correct_all = [1.0 if r['side_correct'] else 0.0 for r in results_list]
        left_right_accuracy = float(np.mean(side_correct_all))

        # Per-door-id breakdown (left vs right target door)
        per_door_id = defaultdict(list)
        per_door_id_side = defaultdict(list)
        for r in results_list:
            per_door_id[r['door_id']].append(r['euclidean_error'])
            per_door_id_side[r['door_id']].append(1.0 if r['side_correct'] else 0.0)

        summary = {
            "global_metrics": {
                "mse": float(np.mean(squared_errors)),
                "mae": float(np.mean(absolute_errors)),
                "mean_euclidean_error": float(np.mean(euclidean_errors)),
                "median_euclidean_error": float(np.median(euclidean_errors)),
                "max_euclidean_error": float(np.max(euclidean_errors)),
                "min_euclidean_error": float(np.min(euclidean_errors)),
                "std_euclidean_error": float(np.std(euclidean_errors)),
                "mean_signed_dx": float(np.mean(dxs)),
                "mean_signed_dy": float(np.mean(dys)),
                "rmse_euclidean": float(np.sqrt(np.mean(squared_errors))),
                "left_right_accuracy": left_right_accuracy,
                "total_doors_evaluated": len(results_list),
                "total_graphs_evaluated": len(test_indices),
            },
            "per_door_id_metrics": {
                k: {
                    "count": len(v),
                    "mean_euclidean_error": float(np.mean(v)),
                    "median_euclidean_error": float(np.median(v)),
                    "left_right_accuracy": float(np.mean(per_door_id_side[k])),
                } for k, v in per_door_id.items()
            },
            "per_door_results": results_list,
        }

        gm = summary["global_metrics"]
        logger.info("=========================================")
        logger.success(f"Evaluation complete for {len(results_list)} doors across {len(test_indices)} {label} graphs.")
        logger.info(f"Mean Euclidean Error : {gm['mean_euclidean_error']:.4f} m")
        logger.info(f"Median Euclidean Err : {gm['median_euclidean_error']:.4f} m")
        logger.info(f"RMSE (Euclidean)     : {gm['rmse_euclidean']:.4f} m")
        logger.info(f"Mean signed (dx, dy) : ({gm['mean_signed_dx']:.4f}, {gm['mean_signed_dy']:.4f}) m  <- bias check")
        logger.info(f"Left/Right accuracy  : {gm['left_right_accuracy']*100:.2f}%  <- correct door side (net semantic capability)")
        logger.info("=========================================")

        eval_output_dir = work_folder / "output" / "evaluation_results"
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = eval_output_dir / report_name
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.success(f"Saved {label} evaluation report to: {report_json_path}")

    def _run(self, work_folder: Path) -> None:
        work_folder = Path(work_folder)
        logger.info(f"=== Starting Graphormer Training in: {work_folder} ===")

        # 1. Scan output directory for topology graphs
        output_dir = work_folder / "output"
        graph_files = sorted(list(output_dir.glob("*_graph.json")))
        if not graph_files:
            logger.error(f"No spatial topology graphs found in: {output_dir}")
            return

        logger.info(f"Scanning {len(graph_files)} spatial graph files...")
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
            return

        # 2. Extract schemas and vocabularies
        merged_nodes = []
        for nodes, _ in all_graphs:
            merged_nodes.extend(nodes)
        
        self.node_type_schemas = self._infer_schema_from_nodes(merged_nodes)
        
        schema_path = work_folder / "intermediate" / "graphormer_node_type_schemas.json"
        with open(schema_path, "w", encoding="utf-8") as sf:
            json.dump(self.node_type_schemas, sf, indent=2)
        logger.success(f"Inferred schema written to: {schema_path}")

        # 3. Build GraphBatch instances
        batches = []
        for i, (nodes, links) in enumerate(all_graphs):
            B = self._prepare_graph(nodes, links)
            if B.door_indices:
                batches.append(B)
                logger.info(f"Graph {i} prepared: {len(B.node_types)} nodes, {len(B.door_indices)} door(s)")
            else:
                logger.warning(f"Graph {i} has no doors, skipping in Graphormer training.")

        if not batches:
            logger.error("No batches containing doors prepared.")
            return

        # 4. Config & Model Setup
        graphormer_cfg = getattr(self.config.training, 'graphormer', None)
        
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
            epochs = int(getattr(graphormer_cfg, 'epochs', epochs))
            lr = float(getattr(graphormer_cfg, 'lr', lr))
            accumulation_steps = int(getattr(graphormer_cfg, 'accumulation_steps', accumulation_steps))
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
            logger.info("Loaded Graphormer configuration from YAML.")
        else:
            logger.warning("Graphormer configuration not found in training block; using defaults.")

        arch = self._model_arch()
        model_cls = GNNBaselineModel if arch != 'graphormer' else LatentCodePredictionModel
        if model_cls is GNNBaselineModel and graphormer_cfg is not None:
            # See _build_model_from_config: deeper depth for local message passing.
            graph_depth = int(getattr(graphormer_cfg, 'gin_graph_depth', graph_depth))
        logger.info(f"Graph encoder architecture: {arch} ({model_cls.__name__}, graph_depth={graph_depth})")
        model = model_cls(
            node_type_schemas=self.node_type_schemas,
            edge_dim=edge_dim,
            d_model=d_model,
            n_heads=n_heads,
            node_depth=node_depth,
            graph_depth=graph_depth,
            ff_dim=ff_dim,
            dropout=dropout,
            max_in_degree=max_in_degree,
            max_out_degree=max_out_degree,
            max_path_distance=max_path_distance
        )

        # Initialize model variables with JAX
        seed = getattr(self.config.training.reproducibility, 'seed', 42)
        rng = jax.random.PRNGKey(seed)
        
        # Build dummy parameters for initialization
        dummy_batch = batches[0]
        dummy_type_feats = self._get_jax_type_features(dummy_batch)
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

        # Setup state & optimizer with learning rate decay
        num_effective_graphs = sum(1 for B in batches if B.door_indices)
        steps_per_epoch = math.ceil((num_effective_graphs * 8) / accumulation_steps)
        total_steps = steps_per_epoch * epochs
        warmup_steps = max(1, min(500, total_steps // 10))
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=lr, warmup_steps=warmup_steps,
            decay_steps=total_steps, end_value=lr * 0.1)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr_schedule))
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

        # Upload static graph arrays to GPU
        gpu_batches = []
        for B in batches:
            type_feats = self._get_jax_type_features(B)
            gpu_batches.append({
                'B_ref': B,
                'type_features': type_feats,
                'num_nodes': len(B.node_types),
                'edge_index': jax.device_put(B.edge_index.numpy()),
                'edge_types': jax.device_put(B.edge_types.numpy()),
                'spatial_indices': jax.device_put(B.spatial_indices),
                'edge_path_indices': jax.device_put(B.edge_path_indices),
                'edge_path_mask': jax.device_put(B.edge_path_mask),
                'door_indices': B.door_indices,
                'door_target_coords': jax.device_put(B.door_target_coords.numpy())
            })

        logger.info(f"Starting Graphormer optimization for {epochs} epochs on device...")

        # 5. Training Loop
        for epoch in range(epochs):
            running_loss = 0.0
            num_effective = 0
            
            accumulated_grads = None
            accumulated_count = 0

            for gb in gpu_batches:
                if not gb['door_indices']:
                    continue
                
                # JAX uses fully parallelized / vectorized evaluations
                # Masking geometry features of selected doors
                mask_size = min(len(gb['door_indices']), 2)
                batch_losses = []
                
                # Perform 8 random masking schemes per epoch iteration
                for _ in range(8):
                    target_indices_in_list = random.sample(range(len(gb['door_indices'])), mask_size)
                    target_node_indices = [gb['door_indices'][i] for i in target_indices_in_list]
                    target_coords = gb['door_target_coords'][jnp.array(target_indices_in_list)]
                    
                    # Create masked type features
                    masked_type_features = {}
                    for n_type, tf in gb['type_features'].items():
                        masked_tf = tf.copy()
                        # Masking geom vector and coordinate input for target nodes
                        geom = tf['geometry']
                        coord = tf['coordinates']
                        for node_idx in target_node_indices:
                            # If this node belongs to the node type, zero out its geometry and coordinates
                            indices_np = np.array(tf['indices'])
                            locs = np.where(indices_np == node_idx)[0]
                            if len(locs) > 0:
                                geom = geom.at[locs[0]].set(0.0)
                                coord = coord.at[locs[0]].set(jnp.array([0.0, 0.0], dtype=jnp.float32))
                        masked_tf['geometry'] = geom
                        masked_tf['coordinates'] = coord
                        masked_type_features[n_type] = masked_tf
                    
                    # Compute JIT compiled train step
                    loss, grads = compute_loss_and_grads(
                        state, 
                        type_features=masked_type_features,
                        num_nodes=gb['num_nodes'],
                        edge_index=gb['edge_index'],
                        edge_types=gb['edge_types'],
                        spatial_indices=gb['spatial_indices'],
                        edge_path_indices=gb['edge_path_indices'],
                        edge_path_mask=gb['edge_path_mask'],
                        target_indices=jnp.array(target_node_indices, dtype=jnp.int32),
                        target_coords=target_coords
                    )
                    batch_losses.append(loss)

                    # Accumulate gradients
                    scaled_grads = jax.tree_util.tree_map(lambda x: x / accumulation_steps, grads)
                    if accumulated_grads is None:
                        accumulated_grads = scaled_grads
                    else:
                        accumulated_grads = jax.tree_util.tree_map(lambda x, y: x + y, accumulated_grads, scaled_grads)
                    
                    accumulated_count += 1
                    if accumulated_count >= accumulation_steps:
                        state = apply_gradients(state, accumulated_grads)
                        accumulated_grads = None
                        accumulated_count = 0

                running_loss += float(jnp.mean(jnp.stack(batch_losses)))
                num_effective += 1

            # Apply remaining gradients at the end of epoch
            if accumulated_count > 0 and accumulated_grads is not None:
                scale_factor = accumulation_steps / accumulated_count
                remaining_grads = jax.tree_util.tree_map(lambda x: x * scale_factor, accumulated_grads)
                state = apply_gradients(state, remaining_grads)
                accumulated_grads = None
                accumulated_count = 0

            logger.info(f"Epoch {epoch+1:03d} | Average Mask Loss: {running_loss / max(1, num_effective):.6f}")

        # Save Flax Model parameters
        ckpt_dir = (work_folder / "intermediate" / "graphormer_jax_checkpoints").resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoints.save_checkpoint(ckpt_dir=str(ckpt_dir), target=state, step=epochs, prefix="best_", overwrite=True)
        logger.success(f"Graphormer JAX model saved to: {ckpt_dir}")

        # Save a reference file path for backward compatibility
        legacy_ckpt_path = work_folder / "intermediate" / "graphormer_model.pth"
        with open(legacy_ckpt_path, "w") as f:
            f.write("JAX/Flax checkpoint saved under graphormer_jax_checkpoints/\n")

        # 6. Evaluation and Attribution Analysis using custom JAX Integrated Gradients
        logger.info("Starting prediction attribution analysis on the first training batch...")
        
        B_eval = batches[0]
        self._idx2id = B_eval.idx2id
        
        analyzer = LatentCodePredictionAnalyzer(
            params=state.params,
            apply_fn=model.apply,
            node_type_schemas=self.node_type_schemas,
            node_names=B_eval.idx2id
        )

        # Perform attribution analysis on up to 3 doors
        analysis_results = analyzer.analyze_door_influences(
            B=B_eval,
            door_indices=B_eval.door_indices[:3],
            door_latents=B_eval.door_target_coords[:3].numpy(),
            num_steps=32
        )

        # 7. Write results to output directory
        json_output = work_folder / "output" / "graphormer_door_analysis.json"
        
        export_data = {
            "metadata": {
                "analysis_method": "integrated_gradients_jax",
                "total_nodes": len(B_eval.node_types),
                "total_doors": len(B_eval.door_indices),
                "node_type_distribution": dict(Counter(B_eval.node_types)),
            },
            "doors": []
        }

        for door_idx in B_eval.door_indices[:3]:
            if door_idx not in analysis_results:
                continue
            result = analysis_results[door_idx]
            
            door_data = {
                "door_index": int(door_idx),
                "door_id": B_eval.idx2id[door_idx] if door_idx < len(B_eval.idx2id) else f"unknown_{door_idx}",
                "door_order": result["ord"],
                "top_nodes": []
            }

            for rank, node_info in enumerate(result["top_nodes"], 1):
                node_data = {
                    "rank": rank,
                    "node_index": node_info["node_id"],
                    "node_id": B_eval.idx2id[node_info["node_id"]] if node_info["node_id"] < len(B_eval.idx2id) else f"unknown_{node_info['node_id']}",
                    "node_type": node_info["node_type"],
                    "total_influence": float(node_info["node_influence"]),
                    "feature_attributions": {feat_name: float(feat_val) for feat_name, feat_val in node_info["top_features"]},
                }
                door_data["top_nodes"].append(node_data)
            
            export_data["doors"].append(door_data)

            # Generate visual attribution plots
            logger.info(f"Generating attribution figures for door index {door_idx}...")
            bar_path = work_folder / "output" / f"graphormer_door_{door_idx}_top_nodes.png"
            heatmap_path = work_folder / "output" / f"graphormer_door_{door_idx}_heatmap.png"
            analyzer.plot_top_nodes_bar(analysis_results, door_idx, savepath=str(bar_path))
            analyzer.plot_node_feature_heatmap(analysis_results, door_idx, savepath=str(heatmap_path))

        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        logger.success(f"Attribution summary exported to: {json_output}")

        # Save full attribution matrix
        full_matrix_path = work_folder / "output" / "graphormer_door_analysis_full_matrix.json"
        full_matrix_data = {
            "metadata": export_data["metadata"].copy(),
            "feature_schema": {},
            "doors": []
        }

        for nt in set(B_eval.node_types):
            feat_info = analyzer._get_feature_info(nt)
            full_matrix_data["feature_schema"][nt] = [
                {"name": name, "dimension": dim} for name, dim in feat_info
            ]

        for door_idx in B_eval.door_indices[:3]:
            if door_idx not in analysis_results:
                continue
            result = analysis_results[door_idx]
            if "attributions" not in result:
                continue
            
            A = result["attributions"].numpy()
            door_matrix_data = {
                "door_index": int(door_idx),
                "door_id": B_eval.idx2id[door_idx] if door_idx < len(B_eval.idx2id) else f"unknown_{door_idx}",
                "door_order": result["ord"],
                "all_nodes": []
            }

            for node_idx in range(len(B_eval.node_types)):
                if node_idx == door_idx:
                    continue
                node_type = B_eval.node_types[node_idx]
                feat_info = analyzer._get_feature_info(node_type)
                node_attr_data = {
                    "node_index": node_idx,
                    "node_id": B_eval.idx2id[node_idx] if node_idx < len(B_eval.idx2id) else f"unknown_{node_idx}",
                    "node_type": node_type,
                    "feature_attributions": {}
                }

                offset = 0
                total_influence = 0.0
                for feat_name, feat_dim in feat_info:
                    feat_attr = A[node_idx, offset : offset + feat_dim]
                    attr_sum = float(feat_attr.sum())
                    node_attr_data["feature_attributions"][feat_name] = {
                        "total": attr_sum,
                        "abs_total": float(abs(attr_sum)),
                        "values": feat_attr.tolist() if feat_dim <= 10 else None,
                        "dimension": feat_dim,
                    }
                    total_influence += abs(attr_sum)
                    offset += feat_dim

                node_attr_data["total_influence"] = total_influence
                door_matrix_data["all_nodes"].append(node_attr_data)

            door_matrix_data["all_nodes"].sort(key=lambda x: x["total_influence"], reverse=True)
            full_matrix_data["doors"].append(door_matrix_data)

        with open(full_matrix_path, "w", encoding="utf-8") as f:
            json.dump(full_matrix_data, f, indent=2, ensure_ascii=False)
        logger.success(f"Full attribution matrix exported to: {full_matrix_path}")

        logger.info("=== Graphormer Training & Integration Pipeline Completed ===")

    @classmethod
    def execute_with_split(cls, config_path: Path, work_folder: Path, model_arch: Optional[str] = None) -> None:
        trainer = cls.config_load(config_path)
        trainer._apply_arch_override(model_arch)
        try:
            trainer._run_with_split(work_folder)
        finally:
            del trainer
            logger.info("SDFGraphormerTrainer (Split Mode) released.")

    def _run_with_split(self, work_folder: Path) -> None:
        work_folder = Path(work_folder)
        logger.info(f"=== Starting Graphormer Split Training (60/20/20) in: {work_folder} ===")

        # 1. Scan output directory for topology graphs
        output_dir = work_folder / "output"
        graph_files = sorted(list(output_dir.glob("graph_room*.json")))
        if not graph_files:
            graph_files = sorted(list(output_dir.glob("*_graph.json")))
            
        if not graph_files:
            logger.error(f"No spatial topology graphs found in: {output_dir}")
            return

        logger.info(f"Scanning {len(graph_files)} spatial graph files...")
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
            return

        # 2. Extract schemas and vocabularies
        merged_nodes = []
        for nodes, _ in all_graphs:
            merged_nodes.extend(nodes)
        
        self.node_type_schemas = self._infer_schema_from_nodes(merged_nodes)
        
        schema_path = work_folder / "intermediate" / "graphormer_node_type_schemas.json"
        with open(schema_path, "w", encoding="utf-8") as sf:
            json.dump(self.node_type_schemas, sf, indent=2)
        logger.success(f"Inferred schema written to: {schema_path}")

        # 3. Build GraphBatch instances
        batches = []
        for i, (nodes, links) in enumerate(all_graphs):
            B = self._prepare_graph(nodes, links)
            if B.door_indices:
                batches.append(B)
            else:
                logger.warning(f"Graph {i} has no doors, skipping in training.")

        if not batches:
            logger.error("No batches containing doors prepared.")
            return

        # 4. Config & Model Setup
        graphormer_cfg = getattr(self.config.training, 'graphormer', None)
        
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
        pose_distill_weight = 0.0

        if graphormer_cfg is not None:
            epochs = int(getattr(graphormer_cfg, 'epochs', epochs))
            lr = float(getattr(graphormer_cfg, 'lr', lr))
            accumulation_steps = int(getattr(graphormer_cfg, 'accumulation_steps', accumulation_steps))
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
            pose_distill_weight = float(getattr(graphormer_cfg, 'pose_distill_weight', 0.0) or 0.0)
            logger.info("Loaded Graphormer configuration from YAML.")
        else:
            logger.warning("Graphormer configuration not found in training block; using defaults.")

        # Loud so a stale run-config snapshot can't silently drop the aux head again
        # (the 2026-07-12 retrain trained without it because the run's config copy
        # predated the pose_distill_weight key).
        if pose_distill_weight > 0.0:
            logger.success(f"Pose-distillation aux head ENABLED (weight={pose_distill_weight}).")
        else:
            logger.warning("Pose-distillation aux head DISABLED (pose_distill_weight missing or 0 in this run's config).")

        arch = self._model_arch()
        model_cls = GNNBaselineModel if arch != 'graphormer' else LatentCodePredictionModel
        if model_cls is GNNBaselineModel and graphormer_cfg is not None:
            # See _build_model_from_config: deeper depth for local message passing.
            graph_depth = int(getattr(graphormer_cfg, 'gin_graph_depth', graph_depth))
        logger.info(f"Graph encoder architecture: {arch} ({model_cls.__name__}, graph_depth={graph_depth})")
        model = model_cls(
            node_type_schemas=self.node_type_schemas,
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

        # Strict padding across all graphs, then push each to device (shared helper).
        specs = self._compute_padding_specs(batches, max_path_distance)

        def to_gpu_batch(B):
            return self._to_gpu_batch(B, specs)

        def to_host_batch(B):
            return self._to_host_batch(B, specs)

        # Initialize model variables with JAX using a padded batch
        seed = getattr(self.config.training.reproducibility, 'seed', 42)
        import random
        random.seed(seed)
        np.random.seed(seed)
        
        init_gpu = to_gpu_batch(batches[0])
        rng = jax.random.PRNGKey(seed)
        
        params = model.init(
            rng, 
            type_features=init_gpu['type_features'],
            num_nodes=init_gpu['num_nodes'],
            edge_index=init_gpu['edge_index'],
            edge_types=init_gpu['edge_types'],
            spatial_indices=init_gpu['spatial_indices'],
            edge_path_indices=init_gpu['edge_path_indices'],
            edge_path_mask=init_gpu['edge_path_mask']
        )['params']

        # Setup state & optimizer with learning rate decay
        batch_size = int(getattr(self.config.training.graphormer, 'batch_size', 64))
        num_train = int(0.6 * len(batches))
        steps_per_epoch = math.ceil((num_train / batch_size) / accumulation_steps)
        total_steps = steps_per_epoch * epochs
        # Warmup + gradient clipping: without them the early phase is violently
        # unstable (loss spikes of 100x) and the model parks at the predict-mean
        # saddle; verified 2026-07-11 that the same model/data only converges
        # with clip(1.0) + warmup (see experiment.md diagnosis).
        warmup_steps = max(1, min(500, total_steps // 10))
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=lr, warmup_steps=warmup_steps,
            decay_steps=total_steps, end_value=lr * 0.1)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=lr_schedule))
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

        # Shuffle and Split dataset (60% Train, 20% Val, 20% Test)
        shuffled_indices = list(range(len(batches)))
        random.shuffle(shuffled_indices)
        
        num_total = len(batches)
        num_train = int(0.6 * num_total)
        num_val = int(0.2 * num_total)
        
        train_indices = shuffled_indices[:num_train]
        val_indices = shuffled_indices[num_train:num_train + num_val]
        test_indices = shuffled_indices[num_train + num_val:]
        
        logger.info(f"Dataset split summary | Total: {num_total} | Train (60%): {len(train_indices)} | Val (20%): {len(val_indices)} | Test (20%): {len(test_indices)}")

        # VRAM guard (2026-07-13): the padded per-graph tensors (dominated by the
        # [N,N,path] edge-path arrays) fit on the 16 GB card at ~60k graphs but
        # OOMed at 120k (round 8). Datasets under `max_device_data_gb` are pinned
        # to the device as before (fast path); larger ones stay in host RAM and
        # each collated batch is streamed to the device on use (~ms per step).
        logger.info("Padding split datasets in host memory...")
        host_train = [to_host_batch(batches[i]) for i in train_indices]
        host_val = [to_host_batch(batches[i]) for i in val_indices]
        host_test = [to_host_batch(batches[i]) for i in test_indices]

        def _batch_nbytes(hb):
            n = sum(v.nbytes for v in hb.values() if isinstance(v, np.ndarray))
            for fields in hb['type_features'].values():
                n += sum(a.nbytes for a in fields.values() if isinstance(a, np.ndarray))
            return n

        total_gb = (_batch_nbytes(host_train[0]) * num_total / 1024**3) if host_train else 0.0
        max_device_gb = float(getattr(graphormer_cfg, 'max_device_data_gb', 10.0))
        stream_batches = total_gb > max_device_gb
        if stream_batches:
            logger.info(f"Padded dataset ~{total_gb:.1f} GB > max_device_data_gb={max_device_gb:g} "
                        f"-> batches stay in host RAM, streamed to device per step.")
            train_gpu, val_gpu, test_gpu = host_train, host_val, host_test
        else:
            logger.info(f"Padded dataset ~{total_gb:.1f} GB <= max_device_data_gb={max_device_gb:g} "
                        f"-> pinning all batches to device memory.")
            train_gpu = [self._device_put_batch(hb) for hb in host_train]
            val_gpu = [self._device_put_batch(hb) for hb in host_val]
            test_gpu = [self._device_put_batch(hb) for hb in host_test]
            host_train = host_val = host_test = None  # free the host copies

        # Batch collation helper. Pinned mode stacks device arrays on device;
        # streaming mode stacks host arrays and does ONE transfer per field.
        if stream_batches:
            def _stack(arrays):
                return jax.device_put(np.stack(arrays))
            def _stack_doors(batch_list):
                return jax.device_put(np.array([gb['door_indices'] for gb in batch_list], dtype=np.int32))
        else:
            _stack = jnp.stack
            def _stack_doors(batch_list):
                return jnp.stack([jnp.array(gb['door_indices'], dtype=jnp.int32) for gb in batch_list])

        def collate_batches(batch_list):
            collated_type_feats = {}
            first_gb = batch_list[0]
            for n_type in first_gb['type_features'].keys():
                collated_type_feats[n_type] = {}
                for field in first_gb['type_features'][n_type].keys():
                    arrays = [gb['type_features'][n_type][field] for gb in batch_list]
                    if arrays[0] is not None:
                        collated_type_feats[n_type][field] = _stack(arrays)
                    else:
                        collated_type_feats[n_type][field] = None

            collated = {
                'type_features': collated_type_feats,
                'num_nodes': first_gb['num_nodes'],
                'edge_index': _stack([gb['edge_index'] for gb in batch_list]),
                'edge_types': _stack([gb['edge_types'] for gb in batch_list]),
                'spatial_indices': _stack([gb['spatial_indices'] for gb in batch_list]),
                'edge_path_indices': _stack([gb['edge_path_indices'] for gb in batch_list]),
                'edge_path_mask': _stack([gb['edge_path_mask'] for gb in batch_list]),
                'door_indices': _stack_doors(batch_list),
                'door_target_coords': _stack([gb['door_target_coords'] for gb in batch_list]),
                'door_wall_centers': _stack([gb['door_wall_centers'] for gb in batch_list]),
                'door_wall_proj': _stack([gb['door_wall_proj'] for gb in batch_list]),
                'door_wall_axes': _stack([gb['door_wall_axes'] for gb in batch_list]),
                'door_s_abs': _stack([gb['door_s_abs'] for gb in batch_list]),
                'door_side': _stack([gb['door_side'] for gb in batch_list]),
                'door_loss_mask': _stack([gb['door_loss_mask'] for gb in batch_list]),
                'node_coords': _stack([gb['node_coords'] for gb in batch_list]),
                'node_pose_mask': _stack([gb['node_pose_mask'] for gb in batch_list])
            }
            return collated

        batch_size = int(getattr(self.config.training.graphormer, 'batch_size', 64))
        logger.info(f"Grouping datasets into batches of size {batch_size}...")
        num_train_batches = math.ceil(len(train_gpu) / batch_size) if train_gpu else 0
        num_val_batches = math.ceil(len(val_gpu) / batch_size) if val_gpu else 0
        num_test_batches = math.ceil(len(test_gpu) / batch_size) if test_gpu else 0

        def make_train_batches():
            # Fresh graph-to-batch assignment every epoch (batches used to be
            # collated once, so composition AND order were fixed for the run).
            order = list(range(len(train_gpu)))
            random.shuffle(order)
            for i in range(0, len(order), batch_size):
                yield collate_batches([train_gpu[j] for j in order[i:i + batch_size]])

        def make_eval_batches(gpu_list):
            for i in range(0, len(gpu_list), batch_size):
                yield collate_batches(gpu_list[i:i + batch_size])

        if stream_batches:
            # Lazy: pre-collating would put the whole split on the device at once.
            val_batches_fn = lambda: make_eval_batches(val_gpu)
            test_batches_fn = lambda: make_eval_batches(test_gpu)
        else:
            _val_cached = list(make_eval_batches(val_gpu))
            _test_cached = list(make_eval_batches(test_gpu))
            val_batches_fn = lambda: _val_cached
            test_batches_fn = lambda: _test_cached

        best_val_loss = float('inf')
        best_params = None

        _hm = self._head_mode()
        _hm_desc = {
            "absolute": "ABSOLUTE | world = pred (wall anchor disabled)",
            "offset": "OFFSET | world = wall_center + pred (2D residual)",
            "scalar_s": "SCALAR-S | world = wall_center + pred_x*axis (1-DOF along wall)",
            "scalar_mag_side": "SCALAR-MAG+SIDE | world = wall_center ± pred_mag*axis (|s| MSE + side BCE)",
        }.get(_hm, _hm)
        # 1.0 routes the loss to the |s|-regression + side-BCE branch (val/test 'MSE'
        # logs then report that combined loss, not a pure coordinate MSE).
        _mag_side_flag = 1.0 if _hm == "scalar_mag_side" else 0.0
        # Pose-distillation weight: applied to the TRAIN loss only. Val/test always
        # use pose_weight=0 so the logged Val MSE (and the early-stop threshold)
        # stays a pure door loss, comparable across runs with/without distillation.
        _pose_w = pose_distill_weight
        if _pose_w > 0:
            logger.info(f"Pose-distillation aux head enabled: weight {_pose_w} (train loss only; "
                        f"logged Train MSE includes it, Val/Test MSE do not)")
        logger.info(f"Coordinate head: {_hm_desc}")
        logger.info(f"Starting Graphormer optimization for {epochs} epochs on device...")

        # Loss-based early stop: `epochs` becomes a hard cap; training ends once the
        # mean Val loss over the last `early_stop_window` epochs drops below
        # `early_stop_val_loss` (0 disables). Threshold units = training loss
        # (~m² for scalar heads at scale 0.1 x loss-factor 100, so 1e-3 = ~3 cm RMS).
        es_window = int(getattr(graphormer_cfg, 'early_stop_window', 5))
        es_threshold = float(getattr(graphormer_cfg, 'early_stop_val_loss', 0.0))
        if es_threshold > 0:
            logger.info(f"Early stop enabled: mean Val loss of last {es_window} epochs < {es_threshold:g}")
        val_history: List[float] = []

        # 5. Training Loop
        for epoch in range(epochs):
            # Training Phase
            train_loss = 0.0
            train_count = 0

            accumulated_grads = None
            accumulated_count = 0

            pbar = tqdm(make_train_batches(), total=num_train_batches, desc=f"Epoch {epoch+1:03d} [Train]")
            for gb in pbar:
                loss, grads = compute_loss_and_grads(
                    state,
                    type_features=gb['type_features'],
                    num_nodes=gb['num_nodes'],
                    edge_index=gb['edge_index'],
                    edge_types=gb['edge_types'],
                    spatial_indices=gb['spatial_indices'],
                    edge_path_indices=gb['edge_path_indices'],
                    edge_path_mask=gb['edge_path_mask'],
                    target_indices=gb['door_indices'],
                    target_coords=gb['door_target_coords'],
                    target_mask=gb['door_loss_mask'],
                    target_wall_centers=gb['door_wall_centers'],
                    target_wall_proj=gb['door_wall_proj'],
                    target_s_abs=gb['door_s_abs'],
                    target_side=gb['door_side'],
                    mag_side_flag=_mag_side_flag,
                    node_coords=gb['node_coords'],
                    node_pose_mask=gb['node_pose_mask'],
                    pose_weight=_pose_w
                )
                train_loss = train_loss + loss
                train_count += 1
                
                if train_count % 10 == 0:
                    current_loss = float(train_loss / train_count)
                    pbar.set_postfix({"Loss": f"{current_loss:.6f}"})
                
                scaled_grads = jax.tree_util.tree_map(lambda x: x / accumulation_steps, grads)
                if accumulated_grads is None:
                    accumulated_grads = scaled_grads
                else:
                    accumulated_grads = jax.tree_util.tree_map(lambda x, y: x + y, accumulated_grads, scaled_grads)
                
                accumulated_count += 1
                if accumulated_count >= accumulation_steps:
                    state = apply_gradients(state, accumulated_grads)
                    accumulated_grads = None
                    accumulated_count = 0
                    
            # Apply remaining gradients at the end of epoch
            if accumulated_count > 0 and accumulated_grads is not None:
                scale_factor = accumulation_steps / accumulated_count
                remaining_grads = jax.tree_util.tree_map(lambda x: x * scale_factor, accumulated_grads)
                state = apply_gradients(state, remaining_grads)
                accumulated_grads = None
                accumulated_count = 0

            # Validation Phase
            val_loss = 0.0
            val_count = 0
            pbar_val = tqdm(val_batches_fn(), total=num_val_batches, desc=f"Epoch {epoch+1:03d} [Val]", leave=False)
            for gb in pbar_val:
                loss = eval_step(
                    state,
                    type_features=gb['type_features'],
                    num_nodes=gb['num_nodes'],
                    edge_index=gb['edge_index'],
                    edge_types=gb['edge_types'],
                    spatial_indices=gb['spatial_indices'],
                    edge_path_indices=gb['edge_path_indices'],
                    edge_path_mask=gb['edge_path_mask'],
                    target_indices=gb['door_indices'],
                    target_coords=gb['door_target_coords'],
                    target_mask=gb['door_loss_mask'],
                    target_wall_centers=gb['door_wall_centers'],
                    target_wall_proj=gb['door_wall_proj'],
                    target_s_abs=gb['door_s_abs'],
                    target_side=gb['door_side'],
                    mag_side_flag=_mag_side_flag,
                    node_coords=gb['node_coords'],
                    node_pose_mask=gb['node_pose_mask'],
                    pose_weight=0.0
                )
                val_loss = val_loss + loss
                val_count += 1
                
                if val_count % 10 == 0:
                    current_val_loss = float(val_loss / val_count)
                    pbar_val.set_postfix({"Val Loss": f"{current_val_loss:.6f}"})

            epoch_train_loss = float(train_loss / max(1, train_count))
            epoch_val_loss = float(val_loss / max(1, val_count))
            
            logger.info(f"Epoch {epoch+1:03d} | Train MSE: {epoch_train_loss:.6f} | Val MSE: {epoch_val_loss:.6f}")

            # Keep track of the best validation params
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_params = state.params
                ckpt_dir = self._split_ckpt_dir(work_folder)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                checkpoints.save_checkpoint(ckpt_dir=str(ckpt_dir), target=state, step=epoch+1, prefix="best_", overwrite=True)

            val_history.append(epoch_val_loss)
            if es_threshold > 0 and len(val_history) >= es_window:
                recent_mean = sum(val_history[-es_window:]) / es_window
                if recent_mean < es_threshold:
                    logger.success(
                        f"Early stop at epoch {epoch+1}/{epochs}: mean Val loss of last "
                        f"{es_window} epochs = {recent_mean:.6g} < {es_threshold:g}"
                    )
                    break

        # 6. Final Evaluation on Test Set using the best validation model
        if best_params is not None:
            state = state.replace(params=best_params)
            logger.success(f"Restored best model from validation with Val MSE: {best_val_loss:.6f}")

        test_loss = 0.0
        test_count = 0
        pbar_test = tqdm(test_batches_fn(), total=num_test_batches, desc="Evaluating on Test Split", leave=False)
        for gb in pbar_test:
            loss = eval_step(
                state,
                type_features=gb['type_features'],
                num_nodes=gb['num_nodes'],
                edge_index=gb['edge_index'],
                edge_types=gb['edge_types'],
                spatial_indices=gb['spatial_indices'],
                edge_path_indices=gb['edge_path_indices'],
                edge_path_mask=gb['edge_path_mask'],
                target_indices=gb['door_indices'],
                target_coords=gb['door_target_coords'],
                target_mask=gb['door_loss_mask'],
                target_wall_centers=gb['door_wall_centers'],
                target_wall_proj=gb['door_wall_proj'],
                target_s_abs=gb['door_s_abs'],
                target_side=gb['door_side'],
                mag_side_flag=_mag_side_flag,
                node_coords=gb['node_coords'],
                node_pose_mask=gb['node_pose_mask'],
                pose_weight=0.0
            )
            test_loss = test_loss + loss
            test_count += 1

        final_test_loss = float(test_loss / max(1, test_count))
        
        logger.info("=========================================")
        logger.success(f"Final Test Evaluation MSE Loss: {final_test_loss:.6f}")
        logger.info("=========================================")

        # Write final metrics to JSON
        metrics_path = work_folder / "output" / "graphormer_split_metrics.json"
        metrics = {
            "dataset_split": {
                "train_pct": 60,
                "val_pct": 20,
                "test_pct": 20,
                "train_count": len(train_indices),
                "val_count": len(val_indices),
                "test_count": len(test_indices),
                "total": num_total
            },
            "metrics": {
                "best_val_mse": float(best_val_loss),
                "final_test_mse": float(final_test_loss)
            }
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            
        logger.success(f"Metrics saved to: {metrics_path}")
        logger.info("=== Graphormer Split Training Pipeline Completed ===")
