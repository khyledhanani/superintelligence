"""
JAX-friendly MAP-Elites mutation service for ACCEL integration.

This module keeps a fixed-size MAP-Elites archive (quality-diversity grid)
in JAX arrays so it can be carried inside the training state. It exposes:

- parent sampling with mixed uniform + fitness-biased selection
  (with optional staleness decay to deprioritise stale archive entries)
- latent Gaussian mutation
- VAE decode + CLUTTR repair + Maze level conversion
- archive insertion by behavior cell with elitist replacement
  (tracks insertion step per cell for staleness computation)

Descriptor modes
----------------
behavior : (obs_count, manhattan_distance)     — 8 × 6 = 48 cells
latent   : (latent_proj_1, latent_proj_2)      — configurable
hybrid   : (manhattan_distance, latent_proj_1) — 6 × n_latent cells
bfs      : (bfs_path_length, obs_count)        — 8 × 8 = 64 cells  [recommended]

The "bfs" mode is preferred because BFS path length captures the actual
navigational difficulty (must route around walls), not just Euclidean
separation. Combined with obstacle count it produces a grid where every
row/column corresponds to a meaningfully different curriculum challenge.
"""

from flax import struct
import chex
import jax
import jax.numpy as jnp

from es.vae_decoder import decode_latent_to_env, repair_cluttr_sequence
from es.env_bridge import cluttr_sequence_to_level, bfs_path_length


# ---------------------------------------------------------------------------
# Behavior descriptor bins
# ---------------------------------------------------------------------------

# Obstacle-count bins (original "behavior" mode axis)
OBS_BINS = jnp.array([5, 10, 15, 20, 25, 30, 35, 40, 50], dtype=jnp.int32)

# Manhattan-distance bins
DIST_BINS = jnp.array([3, 6, 9, 12, 15, 18, 24], dtype=jnp.int32)

# BFS path-length bins — logarithmically spaced to give finer resolution
# where short-path mazes (hardest to discover, easiest to learn) cluster.
BFS_PATH_BINS = jnp.array([3, 5, 8, 12, 18, 28, 45, 80, 169], dtype=jnp.int32)

# Denser obstacle-count bins for the BFS mode's second axis; more resolution
# in the 5-22 range where the target difficulty band (20-80% solve rate) lives.
DENSE_OBS_BINS = jnp.array([5, 8, 12, 17, 22, 28, 35, 42, 50], dtype=jnp.int32)

# Default latent projection bins (used by "latent" and "hybrid" modes)
DEFAULT_LATENT_BINS = jnp.linspace(-4.0, 4.0, 13, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Archive dataclass
# ---------------------------------------------------------------------------

@struct.dataclass
class MapElitesArchive:
    latents:     chex.Array   # (num_cells, latent_dim)
    sequences:   chex.Array   # (num_cells, seq_len)
    fitness:     chex.Array   # (num_cells,)
    occupied:    chex.Array   # (num_cells,)  bool
    last_update: chex.Array   # (num_cells,)  int32 — mutation step of last insertion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def num_cells(axis1_bins=OBS_BINS, axis2_bins=DIST_BINS):
    return int((axis1_bins.shape[0] - 1) * (axis2_bins.shape[0] - 1))


def init_map_elites_archive(
    latent_dim: int, seq_len: int = 52, cells: int | None = None
) -> MapElitesArchive:
    """Initialize an empty fixed-size MAP-Elites archive."""
    cells = int(cells) if cells is not None else num_cells()
    return MapElitesArchive(
        latents=jnp.zeros((cells, latent_dim), dtype=jnp.float32),
        sequences=jnp.zeros((cells, seq_len), dtype=jnp.int32),
        fitness=jnp.full((cells,), -jnp.inf, dtype=jnp.float32),
        occupied=jnp.zeros((cells,), dtype=jnp.bool_),
        last_update=jnp.zeros((cells,), dtype=jnp.int32),
    )


def compute_behavior_descriptors(sequences: chex.Array, inner_dim: int = 13):
    """Compute (obs_count, manhattan_distance) for a batch of sequences."""
    obs_count = jnp.sum(sequences[:, :50] > 0, axis=1)
    goal_idx  = sequences[:, 50]
    agent_idx = sequences[:, 51]
    goal_row  = (goal_idx  - 1) // inner_dim
    goal_col  = (goal_idx  - 1) % inner_dim
    agent_row = (agent_idx - 1) // inner_dim
    agent_col = (agent_idx - 1) % inner_dim
    manhattan = jnp.abs(goal_row - agent_row) + jnp.abs(goal_col - agent_col)
    return obs_count, manhattan


def compute_bfs_descriptors(sequences: chex.Array, inner_dim: int = 13):
    """Compute (bfs_path_length, obs_count) for a batch of sequences.

    BFS path length is the shortest navigable distance from agent to goal,
    routing around walls. Returns inner_dim**2 when unreachable.
    This is more informative than Manhattan distance because it captures
    wall density effects directly.

    Args:
        sequences: (batch, 52) integer arrays.
        inner_dim: Grid width/height (13 for CLUTTR mazes).

    Returns:
        bfs_lengths: (batch,) int32
        obs_count:   (batch,) int32
    """
    obs_count = jnp.sum(sequences[:, :50] > 0, axis=1)

    def _single_bfs(seq):
        obstacles = seq[:50]
        goal_idx  = seq[50]
        agent_idx = seq[51]

        # Build wall map from obstacle indices
        obs_rows = (obstacles - 1) // inner_dim
        obs_cols = (obstacles - 1) % inner_dim
        obs_valid = obstacles > 0
        flat_idx  = obs_rows * inner_dim + obs_cols
        flat_wall = jnp.zeros(inner_dim * inner_dim, dtype=jnp.bool_)
        flat_wall = flat_wall.at[flat_idx].max(obs_valid)
        wall_map  = flat_wall.reshape(inner_dim, inner_dim)

        # Clear agent/goal cells (repair should already have done this)
        goal_col  = (goal_idx  - 1) % inner_dim
        goal_row  = (goal_idx  - 1) // inner_dim
        agent_col = (agent_idx - 1) % inner_dim
        agent_row = (agent_idx - 1) // inner_dim
        wall_map  = wall_map.at[goal_row,  goal_col ].set(False)
        wall_map  = wall_map.at[agent_row, agent_col].set(False)

        agent_pos = jnp.array([agent_col, agent_row], dtype=jnp.uint32)
        goal_pos  = jnp.array([goal_col,  goal_row],  dtype=jnp.uint32)

        return bfs_path_length(wall_map, agent_pos, goal_pos,
                               H=inner_dim, W=inner_dim)

    bfs_lengths = jax.vmap(_single_bfs)(sequences)
    return bfs_lengths, obs_count


def descriptor_to_cell(axis1_values, axis2_values, axis1_bins=OBS_BINS, axis2_bins=DIST_BINS):
    """Map descriptor values to flattened MAP-Elites cell ids.

    Returns (cell_id, in_bounds). cell_id is -1 for out-of-bounds entries.
    """
    n_axis2_bins = axis2_bins.shape[0] - 1
    row = jnp.searchsorted(axis1_bins, axis1_values, side="right") - 1
    col = jnp.searchsorted(axis2_bins, axis2_values, side="right") - 1
    in_bounds = (
        (row >= 0)
        & (row < (axis1_bins.shape[0] - 1))
        & (col >= 0)
        & (col < n_axis2_bins)
    )
    cell = row * n_axis2_bins + col
    return jnp.where(in_bounds, cell, -1), in_bounds


def make_latent_projections(latent_dim: int, seed: int = 0):
    """Build two deterministic orthonormal projection vectors for latent descriptors."""
    key  = jax.random.PRNGKey(seed)
    vecs = jax.random.normal(key, (2, latent_dim), dtype=jnp.float32)
    p1   = vecs[0] / jnp.maximum(jnp.linalg.norm(vecs[0]), 1e-8)
    v2   = vecs[1] - jnp.dot(vecs[1], p1) * p1
    p2   = v2 / jnp.maximum(jnp.linalg.norm(v2), 1e-8)
    return jnp.stack([p1, p2], axis=0)


# ---------------------------------------------------------------------------
# Parent sampling
# ---------------------------------------------------------------------------

def _sample_parent_latents(
    rng: chex.PRNGKey,
    archive: MapElitesArchive,
    batch_size: int,
    uniform_fraction: float = 0.5,
    softmax_temperature: float = 1.0,
    current_step: chex.Array | None = None,
    staleness_decay_rate: float = 0.0,
):
    """Sample parent latent vectors from occupied archive cells.

    Mixing strategy:
      - uniform_fraction of parents drawn uniformly from occupied cells
        (exploration — visits underrepresented parts of the archive)
      - (1 - uniform_fraction) drawn fitness-weighted via softmax
        (exploitation — targets currently hard environments)

    Staleness decay:
      When staleness_decay_rate > 0 and current_step is provided, the
      effective fitness used for softmax weighting is discounted by how
      long ago the cell was last updated:
          effective_fitness = fitness * exp(-decay_rate * staleness)
      This prevents the sampler collapsing onto environments that were
      hard early in training but have since been mastered.

    Falls back to random N(0,1) latents if the archive is empty.
    """
    occupied  = archive.occupied
    occ_count = occupied.sum()

    def _random_fallback():
        return jax.random.normal(rng, (batch_size, archive.latents.shape[-1]),
                                 dtype=jnp.float32)

    def _sample_from_archive():
        rng_uniform, rng_fit, rng_mix = jax.random.split(rng, 3)
        occupied_f    = occupied.astype(jnp.float32)
        uniform_probs = occupied_f / jnp.maximum(occupied_f.sum(), 1.0)

        # Optionally apply staleness decay to fitness before softmax
        if staleness_decay_rate > 0.0 and current_step is not None:
            staleness        = jnp.maximum(0, current_step - archive.last_update).astype(jnp.float32)
            decay            = jnp.exp(-staleness_decay_rate * staleness)
            effective_fitness = archive.fitness * decay
        else:
            effective_fitness = archive.fitness

        masked_fitness = jnp.where(occupied, effective_fitness, -jnp.inf)
        temp   = jnp.maximum(jnp.asarray(softmax_temperature, dtype=jnp.float32), 1e-4)
        max_f  = jnp.max(jnp.where(occupied, masked_fitness, -jnp.inf))
        logits = jnp.where(occupied, (masked_fitness - max_f) / temp, -1e9)
        fit_probs = jax.nn.softmax(logits)
        fit_probs = fit_probs / jnp.maximum(fit_probs.sum(), 1.0)

        idx_uniform = jax.random.choice(
            rng_uniform, archive.fitness.shape[0], shape=(batch_size,), p=uniform_probs
        )
        idx_fit = jax.random.choice(
            rng_fit, archive.fitness.shape[0], shape=(batch_size,), p=fit_probs
        )
        choose_uniform = jax.random.bernoulli(
            rng_mix, p=jnp.clip(uniform_fraction, 0.0, 1.0), shape=(batch_size,)
        )
        idx = jnp.where(choose_uniform, idx_uniform, idx_fit)
        return archive.latents[idx]

    return jax.lax.cond(occ_count > 0, _sample_from_archive, _random_fallback)


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def map_elites_mutate_levels(
    rng: chex.PRNGKey,
    archive: MapElitesArchive,
    decoder_params,
    batch_size: int,
    latent_sigma: float = 0.5,
    decode_temperature: float = 0.25,
    uniform_fraction: float = 0.5,
    softmax_temperature: float = 1.0,
    current_step: chex.Array | None = None,
    staleness_decay_rate: float = 0.0,
):
    """Sample parents from the archive, mutate in latent space, decode to Maze levels.

    Args:
        rng:                  PRNG key.
        archive:              Current MAP-Elites archive.
        decoder_params:       VAE decoder parameters.
        batch_size:           Number of children to generate.
        latent_sigma:         Gaussian noise std for latent mutation.
                              Should be matched to bin width: sigma ≈ 1.5 × bin_width.
                              For 8 latent bins over [-4,4] (bin_width=1.0), use ~0.3–0.5.
        decode_temperature:   VAE decoder sampling temperature.
        uniform_fraction:     Fraction of parents sampled uniformly (vs fitness-biased).
        softmax_temperature:  Temperature for fitness softmax (higher = more uniform).
        current_step:         Current mutation update count (for staleness decay).
        staleness_decay_rate: Staleness decay coefficient (0 = disabled).

    Returns:
        child_levels:    Batched Maze Level pytree, shape (batch_size, ...).
        child_latents:   (batch_size, latent_dim) mutated latent vectors.
        child_sequences: (batch_size, 52) decoded + repaired CLUTTR sequences.
    """
    rng_parent, rng_noise, rng_decode, rng_dir = jax.random.split(rng, 4)
    parents = _sample_parent_latents(
        rng_parent, archive, batch_size,
        uniform_fraction=uniform_fraction,
        softmax_temperature=softmax_temperature,
        current_step=current_step,
        staleness_decay_rate=staleness_decay_rate,
    )
    noise         = jax.random.normal(rng_noise, parents.shape, dtype=parents.dtype)
    child_latents = (parents + latent_sigma * noise).astype(jnp.float32)

    if decode_temperature > 0.0:
        sequences = decode_latent_to_env(
            decoder_params, child_latents, rng_key=rng_decode, temperature=decode_temperature
        )
    else:
        sequences = decode_latent_to_env(
            decoder_params, child_latents, rng_key=None, temperature=0.0
        )

    sequences    = jax.vmap(repair_cluttr_sequence)(sequences).astype(jnp.int32)
    dir_keys     = jax.random.split(rng_dir, batch_size)
    child_levels = jax.vmap(cluttr_sequence_to_level)(sequences, dir_keys)
    return child_levels, child_latents, sequences


# ---------------------------------------------------------------------------
# Archive insertion
# ---------------------------------------------------------------------------

def map_elites_insert_batch(
    archive: MapElitesArchive,
    latents: chex.Array,
    sequences: chex.Array,
    fitness: chex.Array,
    descriptor_mode: str = "bfs",
    axis1_bins: chex.Array = BFS_PATH_BINS,
    axis2_bins: chex.Array = DENSE_OBS_BINS,
    latent_projections: chex.Array | None = None,
    min_obstacles: int = 5,
    min_distance: int = 3,
    current_step: chex.Array | None = None,
):
    """Insert a batch into the archive using elitist replacement per cell.

    Descriptor modes:
      "bfs"      — axis1 = BFS path length, axis2 = obstacle count  [recommended]
      "behavior" — axis1 = obstacle count,  axis2 = manhattan distance
      "latent"   — both axes are random latent projections
      "hybrid"   — axis1 = manhattan distance, axis2 = latent projection

    Complexity gate (min_obstacles + min_distance) filters trivially easy
    environments before insertion. In "bfs" mode, min_distance applies to
    the BFS path length (strictly more informative than Manhattan).

    Args:
        archive:            Current archive.
        latents:            (batch, latent_dim) latent vectors.
        sequences:          (batch, 52) CLUTTR sequences.
        fitness:            (batch,) fitness scores (higher = better).
        descriptor_mode:    One of "bfs", "behavior", "latent", "hybrid".
        axis1_bins:         Bin edges for the first descriptor axis.
        axis2_bins:         Bin edges for the second descriptor axis.
        latent_projections: (2, latent_dim) projection vectors (latent/hybrid modes).
        min_obstacles:      Minimum obstacle count to pass complexity gate.
        min_distance:       Minimum path distance to pass complexity gate.
        current_step:       Current mutation step (stored in last_update on insertion).

    Returns:
        (updated_archive, num_insertions)
    """
    obs_count, manhattan = compute_behavior_descriptors(sequences)
    step = jnp.asarray(0 if current_step is None else current_step, dtype=jnp.int32)

    if descriptor_mode == "behavior":
        axis1_values   = obs_count.astype(jnp.float32)
        axis2_values   = manhattan.astype(jnp.float32)
        complex_enough = (obs_count >= min_obstacles) & (manhattan >= min_distance)

    elif descriptor_mode == "latent":
        if latent_projections is None:
            raise ValueError("latent_projections must be provided for descriptor_mode='latent'.")
        axis1_values   = jnp.dot(latents, latent_projections[0])
        axis2_values   = jnp.dot(latents, latent_projections[1])
        complex_enough = (obs_count >= min_obstacles) & (manhattan >= min_distance)

    elif descriptor_mode == "hybrid":
        if latent_projections is None:
            raise ValueError("latent_projections must be provided for descriptor_mode='hybrid'.")
        axis1_values   = manhattan.astype(jnp.float32)
        axis2_values   = jnp.dot(latents, latent_projections[0])
        complex_enough = (obs_count >= min_obstacles) & (manhattan >= min_distance)

    elif descriptor_mode == "bfs":
        bfs_lengths, _ = compute_bfs_descriptors(sequences)
        axis1_values   = bfs_lengths.astype(jnp.float32)
        axis2_values   = obs_count.astype(jnp.float32)
        # Use BFS length for the distance gate — strictly more informative than Manhattan
        complex_enough = (bfs_lengths >= min_distance) & (obs_count >= min_obstacles)

    else:
        raise ValueError(f"Unknown descriptor_mode: {descriptor_mode!r}")

    cells, in_bounds = descriptor_to_cell(
        axis1_values, axis2_values, axis1_bins=axis1_bins, axis2_bins=axis2_bins
    )
    valid = in_bounds & complex_enough

    def _body(i, carry):
        state, insertions = carry
        cell  = jnp.maximum(cells[i], 0)
        fit_i = fitness[i]

        def _try_insert(c):
            st, ins = c
            replace = (~st.occupied[cell]) | (fit_i > st.fitness[cell])

            def _replace(c2):
                st2, ins2 = c2
                st2 = st2.replace(
                    latents=st2.latents.at[cell].set(latents[i]),
                    sequences=st2.sequences.at[cell].set(sequences[i]),
                    fitness=st2.fitness.at[cell].set(fit_i),
                    occupied=st2.occupied.at[cell].set(True),
                    last_update=st2.last_update.at[cell].set(step),
                )
                return st2, ins2 + 1

            return jax.lax.cond(replace, _replace, lambda x: x, c)

        return jax.lax.cond(valid[i], _try_insert, lambda x: x, (state, insertions))

    return jax.lax.fori_loop(
        0, latents.shape[0], _body, (archive, jnp.array(0, dtype=jnp.int32))
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def map_elites_stats(archive: MapElitesArchive, current_step: chex.Array | None = None):
    """Compute archive coverage, fitness, and optional staleness statistics."""
    occupied_count = archive.occupied.astype(jnp.float32).sum()
    total          = jnp.asarray(archive.occupied.shape[0], dtype=jnp.float32)
    coverage       = occupied_count / jnp.maximum(total, 1.0)

    masked_fitness = jnp.where(archive.occupied, archive.fitness, 0.0)
    mean_fitness   = masked_fitness.sum() / jnp.maximum(occupied_count, 1.0)
    best_fitness   = jnp.where(
        occupied_count > 0,
        jnp.max(jnp.where(archive.occupied, archive.fitness, -jnp.inf)),
        0.0,
    )

    stats = {
        "occupied_cells": occupied_count,
        "coverage":       coverage,
        "best_fitness":   best_fitness,
        "mean_fitness":   mean_fitness,
    }

    if current_step is not None:
        staleness      = jnp.where(archive.occupied,
                                   current_step - archive.last_update, 0).astype(jnp.float32)
        mean_staleness = staleness.sum() / jnp.maximum(occupied_count, 1.0)
        stats["mean_staleness"] = mean_staleness

    return stats
