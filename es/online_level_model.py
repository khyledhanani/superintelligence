"""Online level model management for co-evolutionary ACCEL.

These utilities are called from Python-level (outside JIT) to retrain
the MazeTaskAwareVAE on evolving replay buffer contents, and to prepare
PCA directions for inside-JIT mutations.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from es.maze_ae import (
    MazeEncoder,
    MazeTaskAwareVAE,
    compute_structural_targets_from_grids,
    maze_level_to_grid,
)

# Import compute_loss from train_maze_ae (reuse for online training)
from vae.train_maze_ae import compute_loss


# ---------------------------------------------------------------------------
# Loss config construction
# ---------------------------------------------------------------------------

def _build_loss_cfg(cfg: dict) -> dict:
    """Map coevo config keys to compute_loss config keys (stage1 only)."""
    return {
        "latent_dim": cfg["level_model_latent_dim"],
        "height": cfg.get("height", 13),
        "width": cfg.get("width", 13),
        "wall_pos_weight": cfg.get("level_model_wall_pos_weight", 7.0),
        "wall_bce_weight": cfg.get("level_model_wall_bce_weight", 1.0),
        "wall_dice_weight": cfg.get("level_model_wall_dice_weight", 1.5),
        "goal_ce_weight": cfg.get("level_model_goal_ce_weight", 2.0),
        "agent_ce_weight": cfg.get("level_model_agent_ce_weight", 2.0),
        "overlap_penalty_weight": cfg.get("level_model_overlap_penalty_weight", 0.5),
        "beta": cfg["level_model_beta"],
        "lambda_static": cfg["level_model_lambda_static"],
        "lambda_curriculum": 0.0,  # always stage1 (task-agnostic)
        "lambda_metric": cfg["level_model_lambda_metric"],
        "lambda_valid": cfg["level_model_lambda_valid"],
        "dynamic_confidence_ref": cfg.get("level_model_dynamic_confidence_ref", 20.0),
        "metric_y_weights": cfg.get(
            "level_model_metric_y_weights", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ),
        "train_stage": "stage1",
    }


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_level_model_state(
    rng: jax.Array,
    cfg: dict,
    height: int,
    width: int,
) -> Tuple[dict, optax.OptState]:
    """Initialize MazeTaskAwareVAE params and optimizer state.

    Returns:
        params: dict of model parameters (unfrozen, plain dict)
        opt_state: optax optimizer state
    """
    latent_dim = cfg["level_model_latent_dim"]
    model = MazeTaskAwareVAE(latent_dim=latent_dim, height=height, width=width)

    dummy_grid = jnp.zeros((1, height, width, 3), dtype=jnp.float32)
    rng_init, rng_z = jax.random.split(rng)
    variables = model.init(rng_init, dummy_grid, z_rng=rng_z)
    params = variables["params"]

    optimizer = optax.adam(cfg["level_model_lr"])
    opt_state = optimizer.init(params)

    return params, opt_state


# ---------------------------------------------------------------------------
# Online retraining
# ---------------------------------------------------------------------------

def retrain_level_model(
    params: dict,
    opt_state: optax.OptState,
    dataset: dict,
    rng: jax.Array,
    cfg: dict,
    n_steps: int,
) -> Tuple[dict, optax.OptState, dict]:
    """Run n_steps gradient steps on the dataset to update the level model.

    Args:
        params: current model params (plain dict or FrozenDict)
        opt_state: current optimizer state
        dataset: dict with keys grids, static_targets, p_ema, success_obs_count
        rng: PRNGKey
        cfg: coevo config dict
        n_steps: number of gradient steps

    Returns:
        (updated_params, updated_opt_state, final_metrics)
    """
    loss_cfg = _build_loss_cfg(cfg)
    batch_size = cfg["level_model_batch_size"]
    n = len(dataset["grids"])

    optimizer = optax.adam(cfg["level_model_lr"])

    # JIT the loss + grad computation for speed
    @jax.jit
    def _step(params, opt_state, batch, rng):
        (loss, aux), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            params, batch, rng, loss_cfg
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, aux

    last_aux: dict = {}
    for step in range(n_steps):
        rng, rng_batch, rng_loss = jax.random.split(rng, 3)

        # Sample a minibatch (with replacement for small buffers)
        replace = n < batch_size
        idx = np.random.choice(n, size=min(batch_size, n), replace=replace)
        batch = {k: jnp.array(v[idx], dtype=jnp.float32) for k, v in dataset.items()}

        params, opt_state, loss, aux = _step(params, opt_state, batch, rng_loss)
        last_aux = {k: float(v) for k, v in aux.items()}

    return params, opt_state, last_aux


# ---------------------------------------------------------------------------
# Buffer extraction
# ---------------------------------------------------------------------------

def extract_buffer_grids(
    sampler: dict,
    height: int,
    width: int,
) -> np.ndarray:
    """Extract valid buffer levels as (N, H, W, 3) float32 grids.

    Args:
        sampler: LevelSampler state dict
        height: maze height
        width: maze width

    Returns:
        grids: (N, H, W, 3) float32 numpy array
    """
    size = int(sampler["size"])
    levels = sampler["levels"]

    wall_maps = np.array(levels.wall_map[:size], dtype=np.float32)   # (N, H, W)
    goal_pos = np.array(levels.goal_pos[:size], dtype=np.int32)       # (N, 2) [x, y]
    agent_pos = np.array(levels.agent_pos[:size], dtype=np.int32)     # (N, 2) [x, y]

    grids = np.zeros((size, height, width, 3), dtype=np.float32)
    grids[:, :, :, 0] = wall_maps

    n = np.arange(size)
    # goal_pos[:, 0] = x (col), goal_pos[:, 1] = y (row)
    grids[n, goal_pos[:, 1], goal_pos[:, 0], 1] = 1.0
    grids[n, agent_pos[:, 1], agent_pos[:, 0], 2] = 1.0

    return grids


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_batch_np(
    params: dict,
    grids: np.ndarray,
    latent_dim: int,
    height: int,
    width: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode a batch of grids to latent means using the encoder.

    Args:
        params: full MazeTaskAwareVAE params (plain dict or FrozenDict)
        grids: (N, H, W, 3) float32 numpy array
        latent_dim: latent dimension
        height: maze height
        width: maze width
        batch_size: encoding batch size

    Returns:
        latents: (N, latent_dim) float32 numpy array
    """
    enc_params = params["MazeEncoder_0"]
    encoder = MazeEncoder(latent_dim=latent_dim, variational=True)

    @jax.jit
    def _encode(batch):
        mean, _ = encoder.apply({"params": enc_params}, batch)
        return mean

    n = len(grids)
    out_chunks: list[np.ndarray] = []
    for i in range(0, n, batch_size):
        chunk = jnp.array(grids[i : i + batch_size], dtype=jnp.float32)
        out_chunks.append(np.array(_encode(chunk)))

    return np.concatenate(out_chunks, axis=0) if out_chunks else np.zeros((0, latent_dim), dtype=np.float32)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def compute_pca_from_latents(
    latents: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PCA of latent vectors.

    Args:
        latents: (N, D) float32 numpy array

    Returns:
        eigvecs: (D, D) where ROWS are principal components (sorted descending by variance)
        eigvals: (D,) eigenvalues (variances), non-negative, sorted descending
    """
    if latents.shape[0] < 2:
        d = latents.shape[1]
        return np.eye(d, dtype=np.float32), np.ones(d, dtype=np.float32)

    centered = latents - latents.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)  # (D, D)

    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order, columns = eigvecs

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals = np.maximum(eigvals, 0.0).astype(np.float32)
    eigvecs = eigvecs.T.astype(np.float32)  # rows = principal components

    return eigvecs, eigvals


# ---------------------------------------------------------------------------
# Reservoir dataset
# ---------------------------------------------------------------------------

def build_reservoir_dataset(
    sample_random_level_fn,
    rng: jax.Array,
    n_levels: int,
    height: int,
    width: int,
) -> dict:
    """Build a reservoir of random maze levels as a training dataset.

    This is used to prevent latent collapse during online retraining by
    mixing in structurally diverse random levels.

    Args:
        sample_random_level_fn: callable (rng) -> Level
        rng: PRNGKey
        n_levels: number of random levels to generate
        height: maze height
        width: maze width

    Returns:
        dataset dict with keys: grids, static_targets, p_ema, success_obs_count
    """
    rngs = jax.random.split(rng, n_levels)
    levels = jax.jit(jax.vmap(sample_random_level_fn))(rngs)

    _grid_fn = jax.jit(
        jax.vmap(lambda wm, gp, ap: maze_level_to_grid(wm, gp, ap, height, width))
    )
    grids = np.array(
        _grid_fn(levels.wall_map, levels.goal_pos, levels.agent_pos),
        dtype=np.float32,
    )

    static_targets = np.array(
        jax.jit(compute_structural_targets_from_grids)(jnp.array(grids)),
        dtype=np.float32,
    )

    return {
        "grids": grids,
        "static_targets": static_targets,
        "p_ema": np.zeros(n_levels, dtype=np.float32),
        "success_obs_count": np.zeros(n_levels, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Dataset mixing
# ---------------------------------------------------------------------------

def build_dataset_with_reservoir(
    buffer_grids: np.ndarray,
    buffer_static: np.ndarray,
    buffer_p_ema: np.ndarray,
    buffer_obs_count: np.ndarray,
    reservoir: dict,
    reservoir_fraction: float = 0.3,
) -> dict:
    """Mix buffer data with reservoir to prevent latent collapse.

    Args:
        buffer_grids: (N, H, W, 3)
        buffer_static: (N, 7)
        buffer_p_ema: (N,)
        buffer_obs_count: (N,)
        reservoir: dict from build_reservoir_dataset
        reservoir_fraction: fraction of each batch from reservoir

    Returns:
        combined dataset dict
    """
    n_buf = len(buffer_grids)
    n_res = len(reservoir["grids"])
    n_reservoir_keep = min(n_res, max(1, int(n_buf * reservoir_fraction / (1 - reservoir_fraction + 1e-8))))

    if n_reservoir_keep > 0 and n_res > 0:
        res_idx = np.random.choice(n_res, size=min(n_reservoir_keep, n_res), replace=False)
        grids = np.concatenate([buffer_grids, reservoir["grids"][res_idx]], axis=0)
        static_targets = np.concatenate([buffer_static, reservoir["static_targets"][res_idx]], axis=0)
        p_ema = np.concatenate([buffer_p_ema, reservoir["p_ema"][res_idx]], axis=0)
        obs_count = np.concatenate([buffer_obs_count, reservoir["success_obs_count"][res_idx]], axis=0)
    else:
        grids = buffer_grids
        static_targets = buffer_static
        p_ema = buffer_p_ema
        obs_count = buffer_obs_count

    return {
        "grids": grids,
        "static_targets": static_targets,
        "p_ema": p_ema,
        "success_obs_count": obs_count,
    }
