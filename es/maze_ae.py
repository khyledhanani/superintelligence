"""
Grid-based Maze beta-VAE with task-aware auxiliary heads for PLWM.

This module keeps the existing encode/decode helper surface while upgrading the
training model to include:
- beta-VAE bottleneck (mean/logvar + reparameterization)
- static structural head
- curriculum head (success + learnability)
- validity head

Static targets (per level):
  s1 solvable, s2 wall_density, s3 bfs_norm, s4 manhattan_norm,
  s5 slack_norm, s6 branch_ratio, s7 dead_end_ratio.
"""

from __future__ import annotations

import pickle
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from es.env_bridge import bfs_path_length
from jaxued.environments.maze.level import Level


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class MazeEncoder(nn.Module):
    latent_dim: int = 64
    variational: bool = True

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, 3)
        h = nn.Conv(32, (3, 3), padding="SAME")(x)
        h = nn.leaky_relu(h, negative_slope=0.2)

        h = nn.Conv(64, (3, 3), strides=(2, 2), padding="SAME")(h)  # (B, 7, 7, 64)
        h = nn.leaky_relu(h, negative_slope=0.2)

        h = nn.Conv(128, (3, 3), strides=(2, 2), padding="SAME")(h)  # (B, 4, 4, 128)
        h = nn.leaky_relu(h, negative_slope=0.2)

        h = h.reshape(h.shape[0], -1)  # (B, 2048)
        h = nn.Dense(512)(h)
        h = nn.leaky_relu(h, negative_slope=0.2)

        if self.variational:
            mean = nn.Dense(self.latent_dim, name="mean_layer")(h)
            logvar = nn.Dense(self.latent_dim, name="logvar_layer")(h)
            return mean, logvar

        # Backward-compatible deterministic path (legacy checkpoints).
        z = nn.Dense(self.latent_dim)(h)
        return z


class MazeDecoder(nn.Module):
    height: int = 13
    width: int = 13

    @nn.compact
    def __call__(self, z):
        # z: (B, latent_dim)
        h = nn.Dense(512)(z)
        h = nn.leaky_relu(h, negative_slope=0.2)

        h = nn.Dense(4 * 4 * 128)(h)
        h = nn.leaky_relu(h, negative_slope=0.2)
        h = h.reshape(h.shape[0], 4, 4, 128)

        h = nn.ConvTranspose(64, (3, 3), strides=(2, 2), padding="SAME")(h)  # (B, 8, 8, 64)
        h = nn.leaky_relu(h, negative_slope=0.2)

        h = nn.ConvTranspose(32, (3, 3), strides=(2, 2), padding="SAME")(h)  # (B, 16, 16, 32)
        h = nn.leaky_relu(h, negative_slope=0.2)

        h = h[:, : self.height, : self.width, :]  # (B, 13, 13, 32)

        wh = nn.Conv(16, (3, 3), padding="SAME")(h)
        wh = nn.leaky_relu(wh, negative_slope=0.2)
        wall_logits = nn.Conv(1, (1, 1))(wh)[:, :, :, 0]  # (B, H, W)

        gap = h.mean(axis=(1, 2))
        ph = nn.Dense(256)(gap)
        ph = nn.leaky_relu(ph, negative_slope=0.2)
        goal_logits = nn.Dense(self.height * self.width)(ph)
        agent_logits = nn.Dense(self.height * self.width)(ph)
        return wall_logits, goal_logits, agent_logits


class StaticHead(nn.Module):
    """Predicts static structural targets: solvable + 6 regression dims."""

    @nn.compact
    def __call__(self, z):
        h = nn.Dense(128)(z)
        h = nn.relu(h)
        solvable_logit = nn.Dense(1)(h)[:, 0]
        static_reg = nn.Dense(6)(h)
        return solvable_logit, static_reg


class CurriculumHead(nn.Module):
    """Predicts dynamic curriculum targets: success probability + learnability."""

    @nn.compact
    def __call__(self, z):
        h = nn.Dense(128)(z)
        h = nn.relu(h)
        p_logit = nn.Dense(1)(h)[:, 0]
        # Bounded to [0,1] for a stable learnability regression target.
        learnability = nn.sigmoid(nn.Dense(1)(h)[:, 0])
        return p_logit, learnability


class ValidHead(nn.Module):
    @nn.compact
    def __call__(self, z):
        h = nn.Dense(64)(z)
        h = nn.relu(h)
        return nn.Dense(1)(h)[:, 0]


class MazeTaskAwareVAE(nn.Module):
    latent_dim: int = 64
    height: int = 13
    width: int = 13

    @nn.compact
    def __call__(self, x, z_rng: jax.Array | None = None, deterministic: bool = False):
        mean, logvar = MazeEncoder(self.latent_dim, variational=True)(x)
        if deterministic or z_rng is None:
            z = mean
        else:
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(z_rng, mean.shape)
            z = mean + eps * std

        wall_logits, goal_logits, agent_logits = MazeDecoder(self.height, self.width)(z)
        solvable_logit, static_reg = StaticHead()(z)
        p_logit, learnability = CurriculumHead()(z)
        valid_logit = ValidHead()(z)
        return {
            "z": z,
            "mean": mean,
            "logvar": logvar,
            "wall_logits": wall_logits,
            "goal_logits": goal_logits,
            "agent_logits": agent_logits,
            "solvable_logit": solvable_logit,
            "static_reg": static_reg,
            "p_logit": p_logit,
            "learnability": learnability,
            "valid_logit": valid_logit,
        }


# Backward import compatibility.
class MazeAE(MazeTaskAwareVAE):
    pass


# ---------------------------------------------------------------------------
# Structural targets
# ---------------------------------------------------------------------------


def _grid_positions_from_onehot(grids: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (goal_pos, agent_pos) as [x, y] from one-hot grid channels."""
    b, h, w, _ = grids.shape
    goal_flat = grids[:, :, :, 1].reshape(b, -1).argmax(axis=-1)
    agent_flat = grids[:, :, :, 2].reshape(b, -1).argmax(axis=-1)

    goal_pos = jnp.stack([goal_flat % w, goal_flat // w], axis=-1).astype(jnp.uint32)
    agent_pos = jnp.stack([agent_flat % w, agent_flat // w], axis=-1).astype(jnp.uint32)
    return goal_pos, agent_pos


def compute_structural_targets(
    wall_map: jnp.ndarray,
    goal_pos: jnp.ndarray,
    agent_pos: jnp.ndarray,
) -> jnp.ndarray:
    """Compute s1..s7 for a batch of mazes.

    Args:
        wall_map: (B, H, W) bool
        goal_pos: (B, 2) uint32 [x,y]
        agent_pos: (B, 2) uint32 [x,y]

    Returns:
        (B, 7) float32 in the order:
          s1 solvable, s2 wall_density, s3 bfs_norm, s4 manhattan_norm,
          s5 slack_norm, s6 branch_ratio, s7 dead_end_ratio.
    """
    wall_map = wall_map.astype(jnp.bool_)
    b, h, w = wall_map.shape
    inf = float(h * w)

    bfs = jax.vmap(bfs_path_length)(wall_map, agent_pos, goal_pos).astype(jnp.float32)
    solvable = (bfs < inf).astype(jnp.float32)

    manhattan = (
        jnp.abs(goal_pos[:, 0].astype(jnp.int32) - agent_pos[:, 0].astype(jnp.int32))
        + jnp.abs(goal_pos[:, 1].astype(jnp.int32) - agent_pos[:, 1].astype(jnp.int32))
    ).astype(jnp.float32)

    walls = wall_map
    free = ~walls

    up = jnp.roll(free, -1, axis=1).at[:, -1, :].set(False)
    down = jnp.roll(free, 1, axis=1).at[:, 0, :].set(False)
    left = jnp.roll(free, -1, axis=2).at[:, :, -1].set(False)
    right = jnp.roll(free, 1, axis=2).at[:, :, 0].set(False)
    degree = (
        up.astype(jnp.int32)
        + down.astype(jnp.int32)
        + left.astype(jnp.int32)
        + right.astype(jnp.int32)
    )

    wall_count = walls.reshape(b, -1).sum(axis=-1).astype(jnp.float32)
    free_count = jnp.maximum((h * w) - wall_count, 1.0)
    branch_points = (free & (degree >= 3)).reshape(b, -1).sum(axis=-1).astype(jnp.float32)
    dead_ends = (free & (degree == 1)).reshape(b, -1).sum(axis=-1).astype(jnp.float32)

    wall_density = wall_count / float(h * w)
    bfs_norm = jnp.minimum(bfs, inf) / inf
    manhattan_norm = manhattan / float(2 * (w - 1))
    slack_norm = jnp.clip((bfs - manhattan) / inf, 0.0, 1.0)
    branch_ratio = branch_points / free_count
    dead_end_ratio = dead_ends / free_count

    return jnp.stack(
        [
            solvable,
            wall_density,
            bfs_norm,
            manhattan_norm,
            slack_norm,
            branch_ratio,
            dead_end_ratio,
        ],
        axis=-1,
    )


def compute_structural_targets_from_grids(grids: jnp.ndarray) -> jnp.ndarray:
    wall_map = grids[:, :, :, 0] > 0.5
    goal_pos, agent_pos = _grid_positions_from_onehot(grids)
    return compute_structural_targets(wall_map, goal_pos, agent_pos)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def load_maze_ae_params(checkpoint_path: str) -> dict:
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    return data["params"]


def extract_maze_encoder_params(full_params: dict) -> dict:
    return full_params["MazeEncoder_0"]


def extract_maze_decoder_params(full_params: dict) -> dict:
    return full_params["MazeDecoder_0"]


def extract_maze_static_head_params(full_params: dict) -> dict:
    return full_params["StaticHead_0"]


def extract_maze_curriculum_head_params(full_params: dict) -> dict:
    return full_params["CurriculumHead_0"]


def extract_maze_valid_head_params(full_params: dict) -> dict:
    return full_params["ValidHead_0"]


# ---------------------------------------------------------------------------
# PLWM helpers
# ---------------------------------------------------------------------------


def maze_level_to_grid(
    wall_map: jnp.ndarray,
    goal_pos: jnp.ndarray,
    agent_pos: jnp.ndarray,
    height: int = 13,
    width: int = 13,
) -> jnp.ndarray:
    """Convert Level components to (H, W, 3) float32 grid."""
    grid = jnp.zeros((height, width, 3), dtype=jnp.float32)
    grid = grid.at[:, :, 0].set(wall_map.astype(jnp.float32))
    grid = grid.at[goal_pos[1], goal_pos[0], 1].set(1.0)
    grid = grid.at[agent_pos[1], agent_pos[0], 2].set(1.0)
    return grid


def _is_variational_encoder_params(encoder_params: dict) -> bool:
    return "mean_layer" in encoder_params and "logvar_layer" in encoder_params


def encode_maze_levels(encoder_params: dict, grids: jnp.ndarray) -> jnp.ndarray:
    """Encode a batch of grids to latent vectors.

    Uses the encoder mean for variational checkpoints and preserves legacy
    behavior for deterministic legacy checkpoints.
    """
    if _is_variational_encoder_params(encoder_params):
        mean, _ = MazeEncoder(variational=True).apply({"params": encoder_params}, grids)
        return mean
    return MazeEncoder(variational=False).apply({"params": encoder_params}, grids)


def predict_task_targets(full_params: dict, grids: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Predict task-aware targets used by PLWM candidate scoring."""
    out = MazeTaskAwareVAE().apply({"params": full_params}, grids, deterministic=True)

    # static_reg dims correspond to [s2, s3, s4, s5, s6, s7]
    static_reg = out["static_reg"]
    wall_density_pred = jnp.clip(static_reg[:, 0], 0.0, 1.0)
    bfs_norm_pred = jnp.clip(static_reg[:, 1], 0.0, 1.0)

    valid_prob = jax.nn.sigmoid(out["valid_logit"])
    invalid_prob = 1.0 - valid_prob
    p_pred = jax.nn.sigmoid(out["p_logit"])
    l_pred = jnp.clip(out["learnability"], 0.0, 1.0)

    return {
        "p_pred": p_pred,
        "l_pred": l_pred,
        "invalid_prob": invalid_prob,
        "bfs_norm_pred": bfs_norm_pred,
        "wall_density_pred": wall_density_pred,
    }


def _decode_single_level(
    wall_logits: jnp.ndarray,
    goal_logits: jnp.ndarray,
    agent_logits: jnp.ndarray,
    rng: jax.Array,
    wall_threshold: float,
    temperature: float,
    height: int,
    width: int,
) -> Level:
    """Decode single-sample decoder outputs to a jaxued Level."""
    rng_goal, rng_agent, rng_dir = jax.random.split(rng, 3)

    wall_map = jax.nn.sigmoid(wall_logits) > wall_threshold

    gumbel_g = jax.random.gumbel(rng_goal, (height * width,))
    goal_flat = jnp.argmax(goal_logits / temperature + gumbel_g)
    goal_col = (goal_flat % width).astype(jnp.uint32)
    goal_row = (goal_flat // width).astype(jnp.uint32)

    agent_mask = jnp.zeros(height * width).at[goal_flat].set(-jnp.inf)
    gumbel_a = jax.random.gumbel(rng_agent, (height * width,))
    agent_flat = jnp.argmax((agent_logits + agent_mask) / temperature + gumbel_a)
    agent_col = (agent_flat % width).astype(jnp.uint32)
    agent_row = (agent_flat // width).astype(jnp.uint32)

    wall_map = wall_map.at[goal_row, goal_col].set(False)
    wall_map = wall_map.at[agent_row, agent_col].set(False)

    agent_dir = jax.random.randint(rng_dir, (), 0, 4).astype(jnp.uint8)

    return Level(
        wall_map=wall_map,
        goal_pos=jnp.array([goal_col, goal_row], dtype=jnp.uint32),
        agent_pos=jnp.array([agent_col, agent_row], dtype=jnp.uint32),
        agent_dir=agent_dir,
        width=width,
        height=height,
    )


def decode_maze_latents(
    decoder_params: dict,
    z: jnp.ndarray,
    rng_keys: jnp.ndarray,
    wall_threshold: float = 0.5,
    temperature: float = 0.25,
    height: int = 13,
    width: int = 13,
) -> Level:
    """Decode a batch of latent vectors to jaxued Level objects."""
    wall_logits, goal_logits, agent_logits = MazeDecoder(height, width).apply(
        {"params": decoder_params}, z
    )

    return jax.vmap(_decode_single_level, in_axes=(0, 0, 0, 0, None, None, None, None))(
        wall_logits,
        goal_logits,
        agent_logits,
        rng_keys,
        wall_threshold,
        temperature,
        height,
        width,
    )
