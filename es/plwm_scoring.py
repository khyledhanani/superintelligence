"""PLWM candidate scoring functions.

Includes the legacy structural surrogate and the task-aware objective used for
beta-VAE-guided candidate selection.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax

from es.env_bridge import bfs_path_length


def structural_difficulty_surrogate(
    wall_map,
    goal_pos,
    agent_pos,
    *,
    weight_bfs: float,
    weight_slack: float,
    weight_dead_ends: float,
    weight_walls: float,
    weight_branches: float,
    require_solvable: bool,
):
    """Legacy structural proxy used for candidate ranking."""
    h = wall_map.shape[1]
    w = wall_map.shape[2]
    inf = h * w

    bfs = jax.vmap(bfs_path_length)(wall_map, agent_pos, goal_pos).astype(jnp.float32)
    manhattan = (
        jnp.abs(goal_pos[:, 0].astype(jnp.int32) - agent_pos[:, 0].astype(jnp.int32))
        + jnp.abs(goal_pos[:, 1].astype(jnp.int32) - agent_pos[:, 1].astype(jnp.int32))
    ).astype(jnp.float32)
    slack = bfs - manhattan

    walls = wall_map.astype(jnp.bool_)
    free = ~walls

    up = jnp.roll(free, -1, axis=1).at[:, -1, :].set(False)
    down = jnp.roll(free, 1, axis=1).at[:, 0, :].set(False)
    left = jnp.roll(free, -1, axis=2).at[:, :, -1].set(False)
    right = jnp.roll(free, 1, axis=2).at[:, :, 0].set(False)
    degree = up.astype(jnp.int32) + down.astype(jnp.int32) + left.astype(jnp.int32) + right.astype(jnp.int32)

    dead_ends = jnp.sum(free & (degree == 1), axis=(1, 2)).astype(jnp.float32)
    branch_points = jnp.sum(free & (degree >= 3), axis=(1, 2)).astype(jnp.float32)
    wall_count = jnp.sum(walls, axis=(1, 2)).astype(jnp.float32)

    score = (
        weight_bfs * bfs
        + weight_slack * slack
        + weight_dead_ends * dead_ends
        + weight_walls * wall_count
        + weight_branches * branch_points
    )
    if require_solvable:
        score = jnp.where(bfs < inf, score, -1e9)
    return score


def band_penalty(p_pred: jnp.ndarray, low: float, high: float) -> jnp.ndarray:
    below = jnp.maximum(low - p_pred, 0.0)
    above = jnp.maximum(p_pred - high, 0.0)
    return below + above


def task_aware_objective(
    *,
    p_pred: jnp.ndarray,
    learnability_pred: jnp.ndarray,
    invalid_prob: jnp.ndarray,
    bfs_norm_pred: jnp.ndarray,
    wall_density_pred: jnp.ndarray,
    parent_bfs_norm: jnp.ndarray,
    parent_wall_density: jnp.ndarray,
    a: float = 1.0,
    b: float = 2.0,
    c: float = 0.5,
    d: float = 0.25,
    e: float = 0.1,
    low: float = 0.3,
    high: float = 0.7,
    delta_bfs_norm: float = 2.0 / 169.0,
) -> jnp.ndarray:
    """Task-aware PLWM objective.

    Higher is better.
    """
    target_bfs = jnp.clip(parent_bfs_norm + delta_bfs_norm, 0.0, 1.0)
    p_term = band_penalty(p_pred, low=low, high=high)
    bfs_term = jnp.abs(bfs_norm_pred - target_bfs)
    wall_term = jnp.abs(wall_density_pred - parent_wall_density)

    return (
        a * learnability_pred
        - b * invalid_prob
        - c * p_term
        - d * bfs_term
        - e * wall_term
    )
