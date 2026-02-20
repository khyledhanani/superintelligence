"""
JAX-friendly MAP-Elites mutation service for ACCEL integration.

This module keeps a fixed-size MAP-Elites archive (quality-diversity grid)
in JAX arrays so it can be carried inside the training state. It exposes:

- parent sampling with mixed uniform + fitness-biased selection
- latent Gaussian mutation
- VAE decode + CLUTTR repair + Maze level conversion
- archive insertion by behavior cell with elitist replacement
"""

from flax import struct
import chex
import jax
import jax.numpy as jnp

from es.vae_decoder import decode_latent_to_env, repair_cluttr_sequence
from es.env_bridge import cluttr_sequence_to_level


# Behavior descriptor bins (match es/map_elites.py defaults)
OBS_BINS = jnp.array([5, 10, 15, 20, 25, 30, 35, 40, 50], dtype=jnp.int32)
DIST_BINS = jnp.array([3, 6, 9, 12, 15, 18, 24], dtype=jnp.int32)
DEFAULT_LATENT_BINS = jnp.linspace(-3.0, 3.0, 13, dtype=jnp.float32)


@struct.dataclass
class MapElitesArchive:
    latents: chex.Array      # (num_cells, latent_dim)
    sequences: chex.Array    # (num_cells, seq_len)
    fitness: chex.Array      # (num_cells,)
    occupied: chex.Array     # (num_cells,)


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
    )


def compute_behavior_descriptors(sequences: chex.Array, inner_dim: int = 13):
    """Compute obstacle-count and Manhattan distance descriptors."""
    obs_count = jnp.sum(sequences[:, :50] > 0, axis=1)
    goal_idx = sequences[:, 50]
    agent_idx = sequences[:, 51]
    goal_row = (goal_idx - 1) // inner_dim
    goal_col = (goal_idx - 1) % inner_dim
    agent_row = (agent_idx - 1) // inner_dim
    agent_col = (agent_idx - 1) % inner_dim
    manhattan = jnp.abs(goal_row - agent_row) + jnp.abs(goal_col - agent_col)
    return obs_count, manhattan


def descriptor_to_cell(axis1_values, axis2_values, axis1_bins=OBS_BINS, axis2_bins=DIST_BINS):
    """Map descriptors to flattened MAP-Elites cell ids."""
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
    """Build two deterministic unit projection vectors for latent descriptors."""
    key = jax.random.PRNGKey(seed)
    vecs = jax.random.normal(key, (2, latent_dim), dtype=jnp.float32)
    p1 = vecs[0] / jnp.maximum(jnp.linalg.norm(vecs[0]), 1e-8)
    v2 = vecs[1] - jnp.dot(vecs[1], p1) * p1
    p2 = v2 / jnp.maximum(jnp.linalg.norm(v2), 1e-8)
    return jnp.stack([p1, p2], axis=0)


def _sample_parent_latents(
    rng: chex.PRNGKey,
    archive: MapElitesArchive,
    batch_size: int,
    uniform_fraction: float = 0.5,
    softmax_temperature: float = 0.5,
):
    """Sample parent latents from occupied archive cells (or random fallback)."""
    occupied = archive.occupied
    occ_count = occupied.sum()

    def _random_fallback():
        return jax.random.normal(rng, (batch_size, archive.latents.shape[-1]), dtype=jnp.float32)

    def _sample_from_archive():
        rng_uniform, rng_fit, rng_mix = jax.random.split(rng, 3)
        occupied_f = occupied.astype(jnp.float32)
        uniform_probs = occupied_f / jnp.maximum(occupied_f.sum(), 1.0)

        masked_fitness = jnp.where(occupied, archive.fitness, -jnp.inf)
        temp = jnp.maximum(jnp.asarray(softmax_temperature, dtype=jnp.float32), 1e-4)
        max_fit = jnp.max(jnp.where(occupied, masked_fitness, -jnp.inf))
        logits = jnp.where(occupied, (masked_fitness - max_fit) / temp, -1e9)
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


def map_elites_mutate_levels(
    rng: chex.PRNGKey,
    archive: MapElitesArchive,
    decoder_params,
    batch_size: int,
    latent_sigma: float = 0.5,
    decode_temperature: float = 0.25,
    uniform_fraction: float = 0.5,
    softmax_temperature: float = 0.5,
):
    """Sample parents, mutate in latent space, decode, and convert to Maze levels."""
    rng_parent, rng_noise, rng_decode, rng_dir = jax.random.split(rng, 4)
    parents = _sample_parent_latents(
        rng_parent,
        archive,
        batch_size=batch_size,
        uniform_fraction=uniform_fraction,
        softmax_temperature=softmax_temperature,
    )
    noise = jax.random.normal(rng_noise, parents.shape, dtype=parents.dtype)
    child_latents = (parents + latent_sigma * noise).astype(jnp.float32)

    if decode_temperature > 0.0:
        sequences = decode_latent_to_env(
            decoder_params, child_latents, rng_key=rng_decode, temperature=decode_temperature
        )
    else:
        sequences = decode_latent_to_env(
            decoder_params, child_latents, rng_key=None, temperature=0.0
        )

    sequences = jax.vmap(repair_cluttr_sequence)(sequences).astype(jnp.int32)
    dir_keys = jax.random.split(rng_dir, batch_size)
    child_levels = jax.vmap(cluttr_sequence_to_level)(sequences, dir_keys)
    return child_levels, child_latents, sequences


def map_elites_insert_batch(
    archive: MapElitesArchive,
    latents: chex.Array,
    sequences: chex.Array,
    fitness: chex.Array,
    descriptor_mode: str = "behavior",
    axis1_bins: chex.Array = OBS_BINS,
    axis2_bins: chex.Array = DIST_BINS,
    latent_projections: chex.Array | None = None,
    min_obstacles: int = 5,
    min_distance: int = 3,
):
    """Insert a batch into the archive using elitist replacement per descriptor cell."""
    obs_count, manhattan = compute_behavior_descriptors(sequences)

    if descriptor_mode == "behavior":
        axis1_values = obs_count.astype(jnp.float32)
        axis2_values = manhattan.astype(jnp.float32)
    elif descriptor_mode == "latent":
        if latent_projections is None:
            raise ValueError("latent_projections must be provided for descriptor_mode='latent'.")
        axis1_values = jnp.dot(latents, latent_projections[0])
        axis2_values = jnp.dot(latents, latent_projections[1])
    elif descriptor_mode == "hybrid":
        if latent_projections is None:
            raise ValueError("latent_projections must be provided for descriptor_mode='hybrid'.")
        # Hybrid: keep task-relevant structure (distance) while diversifying along latent geometry.
        axis1_values = manhattan.astype(jnp.float32)
        axis2_values = jnp.dot(latents, latent_projections[0])
    else:
        raise ValueError(f"Unknown descriptor_mode: {descriptor_mode}")

    cells, in_bounds = descriptor_to_cell(
        axis1_values, axis2_values, axis1_bins=axis1_bins, axis2_bins=axis2_bins
    )
    complex_enough = (obs_count >= min_obstacles) & (manhattan >= min_distance)
    valid = in_bounds & complex_enough

    def _body(i, carry):
        state, insertions = carry
        cell = jnp.maximum(cells[i], 0)
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
                )
                return st2, ins2 + 1

            return jax.lax.cond(replace, _replace, lambda x: x, c)

        return jax.lax.cond(valid[i], _try_insert, lambda x: x, (state, insertions))

    return jax.lax.fori_loop(
        0, latents.shape[0], _body, (archive, jnp.array(0, dtype=jnp.int32))
    )


def map_elites_stats(archive: MapElitesArchive):
    """Compute archive coverage and score stats for logging."""
    occupied_count = archive.occupied.astype(jnp.float32).sum()
    total = jnp.asarray(archive.occupied.shape[0], dtype=jnp.float32)
    coverage = occupied_count / jnp.maximum(total, 1.0)

    masked_fitness = jnp.where(archive.occupied, archive.fitness, 0.0)
    mean_fitness = masked_fitness.sum() / jnp.maximum(occupied_count, 1.0)
    best_fitness = jnp.where(
        occupied_count > 0,
        jnp.max(jnp.where(archive.occupied, archive.fitness, -jnp.inf)),
        0.0,
    )

    return {
        "occupied_cells": occupied_count,
        "coverage": coverage,
        "best_fitness": best_fitness,
        "mean_fitness": mean_fitness,
    }
