"""
UED Interface: VAE + MAP-Elites online level generation.

This replaces DCD's random level generator (make_level_generator) and
wall-flip mutator (make_level_mutator_minimax) with:
  - VAE latent-space sampling / mutation for generating CLUTTR sequences
  - MAP-Elites archive tracking diversity over (num_obstacles, distance) cells

The Archive class is reused from es/map_elites.py. Level decoding and
conversion utilities are imported directly from the es/ pipeline.

Public interface:
    load_vae(checkpoint_path)      -> decoder_params (JAX pytree)
    generate_candidates(...)       -> latents (numpy array, n x latent_dim)
    mutate_latents(latents, ...)   -> child_latents (numpy array, n x latent_dim)
    update_archive(archive, ...)   -> (modifies archive in-place)
    build_eval_fn(...)             -> JIT-compiled (rng, params, latents) -> results
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

# Add project root to sys.path so we can import from es/ and src/
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'es'))

from vae_decoder import load_vae_params, extract_decoder_params, decode_latent_to_env, repair_cluttr_sequence
from env_bridge import cluttr_sequence_to_level, flood_fill_solvable
from regret_fitness import compute_complexity_mask
from agent_loader import ActorCritic

from jaxued.utils import compute_max_returns, max_mc
from jaxued.wrappers import AutoReplayWrapper
from jaxued.environments.maze import Maze

from archive import Archive, compute_behavior_descriptors


# ---------------------------------------------------------------------------
# VAE loading
# ---------------------------------------------------------------------------

def load_vae(checkpoint_path):
    """Load and extract decoder parameters from a VAE checkpoint.

    Args:
        checkpoint_path: Path to the pkl checkpoint (relative to project root
                         or absolute).

    Returns:
        decoder_params: JAX-compatible dict of decoder parameters.
    """
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(_ROOT, checkpoint_path)
    full_params = load_vae_params(checkpoint_path)
    return extract_decoder_params(full_params)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(archive, rng_np, n, latent_dim=64,
                        mutation_sigma=0.5, random_fraction=0.3):
    """Generate n candidate latent vectors.

    Uses archive mutation when archive is sufficiently populated, falls back
    to random sampling otherwise.

    Args:
        archive:         MAP-Elites Archive (from map_elites.py).
        rng_np:          numpy.random.Generator for numpy-side operations.
        n:               Number of candidates to generate.
        latent_dim:      Dimensionality of VAE latent space (64).
        mutation_sigma:  Std dev of Gaussian perturbation for archive parents.
        random_fraction: Fraction of candidates sampled randomly (vs. mutation).

    Returns:
        latents: numpy float32 array of shape (n, latent_dim).
    """
    if archive.num_filled < 2:
        # Archive too empty — full random
        return rng_np.standard_normal((n, latent_dim)).astype(np.float32)

    n_random = max(1, int(round(n * random_fraction)))
    n_mutate = n - n_random

    random_part = rng_np.standard_normal((n_random, latent_dim)).astype(np.float32)

    parents = archive.sample_parents(rng_np, n_mutate)  # (n_mutate, latent_dim)
    noise = rng_np.standard_normal((n_mutate, latent_dim)).astype(np.float32)
    mutated_part = parents + mutation_sigma * noise

    return np.concatenate([random_part, mutated_part], axis=0)


def mutate_latents(latents, rng_np, mutation_sigma=0.5):
    """Mutate a batch of latent vectors with Gaussian noise.

    Used for the ACCEL mutation step: takes the latents of the last replay
    batch and perturbs them to find harder variants.

    Args:
        latents:        numpy array (n, latent_dim).
        rng_np:         numpy.random.Generator.
        mutation_sigma: Std dev of perturbation.

    Returns:
        child_latents: numpy array (n, latent_dim).
    """
    noise = rng_np.standard_normal(latents.shape).astype(np.float32)
    return latents + mutation_sigma * noise


# ---------------------------------------------------------------------------
# Archive update
# ---------------------------------------------------------------------------

def update_archive(archive, latents_np, sequences_np, regrets_np, valid_np):
    """Insert valid candidates into the MAP-Elites archive.

    Args:
        archive:       Archive object (mutated in-place).
        latents_np:    (n, latent_dim) float32 numpy array.
        sequences_np:  (n, 52) int numpy array.
        regrets_np:    (n,) float numpy array.
        valid_np:      (n,) bool numpy array (solvable & complex enough).
    """
    obs_counts, dists = compute_behavior_descriptors(sequences_np)
    for i in range(len(latents_np)):
        if valid_np[i]:
            archive.try_insert(
                latents_np[i], sequences_np[i], float(regrets_np[i]),
                obs_counts[i], dists[i],
            )


# ---------------------------------------------------------------------------
# JIT-compiled evaluation function builder
# ---------------------------------------------------------------------------

def build_eval_fn(decoder_params, network, env_params, latent_dim,
                  decode_temperature=0.25, num_steps=128,
                  min_obstacles=5, min_distance=3):
    """Build and return a JIT-compiled function that evaluates candidate latents.

    The returned function has signature:
        eval_fn(rng, agent_params, latents) ->
            (sequences, levels, regrets, max_returns, valid)

    All shapes are (n, ...) where n = latents.shape[0].

    Args:
        decoder_params:     Extracted VAE decoder parameters (static).
        network:            ActorCritic instance (static).
        env_params:         Maze env_params (static).
        latent_dim:         64.
        decode_temperature: VAE decode temperature.
        num_steps:          Rollout length for regret estimation.
        min_obstacles:      Complexity filter.
        min_distance:       Complexity filter.

    Returns:
        eval_fn: JIT-compiled callable.
    """
    # Build env inside the closure so it's static to JIT
    maze_env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    wrapped_env = AutoReplayWrapper(maze_env)

    @jax.jit
    def eval_fn(rng, variable_dict, latents):
        """Decode latents, run agent, compute MaxMC regret.

        Args:
            rng:           JAX PRNGKey.
            variable_dict: Full Flax variable dict (train_state.params).
            latents:       (n, latent_dim) float32 array.

        Returns:
            sequences:   (n, 52) int array
            levels:      batched Level pytree, shape (n, ...)
            regrets:     (n,) float — 0 for invalid envs
            max_returns: (n,) float
            valid:       (n,) bool — solvable & complex enough
        """
        n = latents.shape[0]
        rng_decode, rng_dirs, rng_rollout = jax.random.split(rng, 3)

        # 1. Decode latents -> CLUTTR sequences
        sequences = decode_latent_to_env(
            decoder_params, latents, rng_key=rng_decode, temperature=decode_temperature
        )
        sequences = jax.vmap(repair_cluttr_sequence)(sequences)

        # 2. Convert to Maze Levels
        dir_keys = jax.random.split(rng_dirs, n)
        levels = jax.vmap(cluttr_sequence_to_level)(sequences, dir_keys)

        # 3. Validity checks
        solvable = jax.vmap(flood_fill_solvable)(
            levels.wall_map, levels.agent_pos, levels.goal_pos
        )
        complex_enough = compute_complexity_mask(
            sequences, min_obstacles=min_obstacles, min_distance=min_distance,
        )
        valid = solvable & complex_enough

        # 4. Agent rollout to compute MaxMC regret (deterministic policy)
        init_obs, init_state = jax.vmap(
            wrapped_env.reset_to_level, in_axes=(0, 0, None)
        )(jax.random.split(rng_rollout, n), levels, env_params)

        init_hstate = ActorCritic.initialize_carry((n,))

        def _step(carry, _):
            rng_s, hstate, obs, state, done = carry
            rng_s, rng_st = jax.random.split(rng_s)
            x = jax.tree_util.tree_map(lambda a: a[None, ...], (obs, done))
            hstate, pi, value = network.apply(variable_dict, x, hstate)
            action = jnp.argmax(pi.logits, axis=-1).squeeze(0)
            value = value.squeeze(0)
            next_obs, next_state, reward, next_done, _ = jax.vmap(
                wrapped_env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(rng_st, n), state, action, env_params)
            return (rng_s, hstate, next_obs, next_state, next_done), (reward, value, next_done)

        _, (rewards, values, dones) = jax.lax.scan(
            _step,
            (rng_rollout, init_hstate, init_obs, init_state, jnp.zeros(n, dtype=jnp.bool_)),
            None,
            length=num_steps,
        )

        max_returns = compute_max_returns(dones, rewards)
        regrets = max_mc(dones, values, max_returns, incomplete_value=0.0)
        regrets = jnp.where(valid, regrets, 0.0)

        return sequences, levels, regrets, max_returns, valid

    return eval_fn, wrapped_env
