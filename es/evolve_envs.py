"""
Evolve CLUTTR gridworld environments using CMA-ES in the VAE latent space.

Approach:
    1. Load trained VAE decoder to map 64-dim latent vectors -> 52-element CLUTTR sequences.
    2. Use CMA-ES (evosax) to search the continuous latent space.
    3. Evaluate decoded environments with a fitness function (placeholder: structural complexity).
    4. Save evolved environments as .npy files.

Usage:
    cd /path/to/superintelligence/vae
    python evolve_envs.py
    python evolve_envs.py --num_generations 50 --pop_size 16 --no_warm_start
"""

import jax
import jax.numpy as jnp
import numpy as np
import yaml
import os
import argparse
import sys
from functools import partial

# Add parent directory to path to import from vae folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evosax.algorithms import CMA_ES

from vae_decoder import (
    load_vae_params,
    extract_decoder_params,
    decode_latent_to_env,
    repair_cluttr_sequence,
    CluttrDecoder,
)
from vae.sample_envs import generate_cluttr_batch_jax


# ---------------------------------------------------------------------------
# Fitness function (placeholder)
# ---------------------------------------------------------------------------

def placeholder_fitness(sequences, inner_dim=13, w_obstacles=0.4, w_distance=0.4, w_validity=0.2):
    """Evaluate environment complexity as a proxy fitness.

    Combines obstacle density, agent-goal distance, and structural validity.
    Returns NEGATED scores because evosax MINIMIZES.

    This function is designed to be swapped for RL-based evaluation later.
    Future signature: (key, sequences, train_state, env, env_params) -> scores.

    Args:
        sequences: Decoded integer sequences, shape (pop_size, 52).
        inner_dim: Grid inner dimension (default 13 for 13x13).
        w_obstacles: Weight for obstacle count component.
        w_distance: Weight for agent-goal Manhattan distance.
        w_validity: Weight for structural validity bonus.

    Returns:
        Fitness scores, shape (pop_size,). Lower is better (evosax minimizes).
    """
    obstacles = sequences[:, :50]
    goal_idx = sequences[:, 50]
    agent_idx = sequences[:, 51]

    # Obstacle density (0 to 1)
    obs_count = jnp.sum(obstacles > 0, axis=1).astype(jnp.float32)
    obs_score = obs_count / 50.0

    # Manhattan distance between agent and goal (normalized)
    agent_row = (agent_idx - 1) // inner_dim
    agent_col = (agent_idx - 1) % inner_dim
    goal_row = (goal_idx - 1) // inner_dim
    goal_col = (goal_idx - 1) % inner_dim
    manhattan = (jnp.abs(agent_row - goal_row) + jnp.abs(agent_col - goal_col)).astype(jnp.float32)
    dist_score = manhattan / (2.0 * (inner_dim - 1))  # max Manhattan = 2*(dim-1)

    # Validity: goal and agent in [1, 169] and distinct
    valid = (
        (goal_idx >= 1) & (goal_idx <= inner_dim**2) &
        (agent_idx >= 1) & (agent_idx <= inner_dim**2) &
        (goal_idx != agent_idx)
    ).astype(jnp.float32)

    fitness = w_obstacles * obs_score + w_distance * dist_score + w_validity * valid
    return -fitness  # negate: evosax minimizes


# ---------------------------------------------------------------------------
# Warm-start: encode random environments to seed CMA-ES mean
# ---------------------------------------------------------------------------

def compute_warm_start_mean(key, full_vae_params, pop_size, max_obs=50, inner_dim=13):
    """Encode random valid environments through the VAE encoder to get an initial CMA-ES mean.

    Args:
        key: PRNG key.
        full_vae_params: Full VAE parameters (encoder + decoder).
        pop_size: Number of environments to encode.
        max_obs: Maximum obstacles per environment.
        inner_dim: Grid inner dimension.

    Returns:
        Mean latent vector, shape (latent_dim,).
    """
    from vae.train_vae import CluttrVAE

    key, envs = generate_cluttr_batch_jax(key, pop_size, max_obs, inner_dim)
    z_rng = jax.random.PRNGKey(0)
    _, means, _ = CluttrVAE().apply({'params': full_vae_params}, envs, z_rng)
    return means.mean(axis=0)  # centroid of encoded environments


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def run_evolution(config):
    """Run CMA-ES evolution in the VAE latent space.

    Args:
        config: Dictionary with evolution and VAE configuration.
    """
    key = jax.random.PRNGKey(config["seed"])

    # 1. Load VAE
    print("Loading VAE checkpoint...")
    full_vae_params = load_vae_params(config["checkpoint_path"])
    decoder_params = extract_decoder_params(full_vae_params)

    # 2. Initialize CMA-ES
    latent_dim = config["latent_dim"]
    dummy_solution = jnp.zeros(latent_dim)
    es = CMA_ES(population_size=config["pop_size"], solution=dummy_solution)
    es_params = es.default_params

    # Warm-start: use VAE-encoded random environments as initial mean
    if config.get("warm_start", True):
        print("Computing warm-start mean from random environments...")
        key, ws_key = jax.random.split(key)
        init_mean = compute_warm_start_mean(ws_key, full_vae_params, config["pop_size"])
    else:
        init_mean = dummy_solution

    key, init_key = jax.random.split(key)
    es_state = es.init(init_key, init_mean, es_params)

    # 3. JIT-compile decode + fitness
    @jax.jit
    def evaluate_population(population):
        sequences = decode_latent_to_env(decoder_params, population)
        sequences = jax.vmap(repair_cluttr_sequence)(sequences)
        fitness = placeholder_fitness(
            sequences,
            inner_dim=config.get("inner_dim", 13),
            w_obstacles=config.get("w_obstacles", 0.4),
            w_distance=config.get("w_distance", 0.4),
            w_validity=config.get("w_validity", 0.2),
        )
        return fitness, sequences

    # 4. Evolution loop
    best_fitness_history = []
    mean_fitness_history = []
    num_gens = config["num_generations"]

    print(f"Starting CMA-ES evolution: {num_gens} generations, pop_size={config['pop_size']}, latent_dim={latent_dim}")
    print("-" * 70)

    for gen in range(num_gens):
        key, ask_key, tell_key = jax.random.split(key, 3)

        # Ask
        population, es_state = es.ask(ask_key, es_state, es_params)

        # Evaluate
        fitness, sequences = evaluate_population(population)

        # Tell
        es_state, metrics = es.tell(tell_key, population, fitness, es_state, es_params)

        # Track (negate back since we negated for minimization)
        best_idx = jnp.argmin(fitness)
        best_fit = float(-fitness[best_idx])
        mean_fit = float(-fitness.mean())
        best_fitness_history.append(best_fit)
        mean_fitness_history.append(mean_fit)

        if gen % config.get("log_freq", 10) == 0:
            std_val = float(es_state.std) if es_state.std.ndim == 0 else float(es_state.std.mean())
            print(
                f"Gen {gen:4d} | "
                f"Best: {best_fit:.4f} | "
                f"Mean: {mean_fit:.4f} | "
                f"Std: {std_val:.4f}"
            )

    print("-" * 70)
    print("Evolution complete.")

    # 5. Extract and save results
    # Decode the final population
    key, final_key = jax.random.split(key)
    final_pop, _ = es.ask(final_key, es_state, es_params)
    _, final_sequences = evaluate_population(final_pop)

    # Decode the best solution
    best_z = es_state.best_solution[None, :]
    best_seq = decode_latent_to_env(decoder_params, best_z)
    best_seq = jax.vmap(repair_cluttr_sequence)(best_seq)

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "evolved_envs.npy"), np.array(final_sequences))
    np.save(os.path.join(output_dir, "best_env.npy"), np.array(best_seq))
    np.save(os.path.join(output_dir, "fitness_history.npy"), np.array(best_fitness_history))
    np.save(os.path.join(output_dir, "best_latent.npy"), np.array(es_state.best_solution))

    print(f"Saved {config['pop_size']} evolved environments to {output_dir}/")
    print(f"Best fitness: {best_fitness_history[-1]:.4f}")

    return es_state, final_sequences


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evolve CLUTTR environments via CMA-ES in VAE latent space"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_generations", type=int, default=200)
    parser.add_argument("--pop_size", type=int, default=32)
    parser.add_argument("--sigma_init", type=float, default=1.0)
    parser.add_argument("--no_warm_start", action="store_true",
                        help="Disable warm-start (use zero mean instead)")
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--w_obstacles", type=float, default=0.4)
    parser.add_argument("--w_distance", type=float, default=0.4)
    parser.add_argument("--w_validity", type=float, default=0.2)
    args = parser.parse_args()

    # Load VAE config for model dimensions and paths
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vae', 'vae_train_config.yml')
    with open(config_path, 'r') as f:
        vae_config = yaml.safe_load(f)

    # Load evolution config defaults (overridden by CLI args)
    evolve_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evolve_config.yml')
    evolve_defaults = {}
    if os.path.exists(evolve_config_path):
        with open(evolve_config_path, 'r') as f:
            evolve_defaults = yaml.safe_load(f)

    # Checkpoint path: VAE model in vae/model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "..", "vae", "model", "checkpoint_330000.pkl")

    # Output directory
    output_subdir = evolve_defaults.get("output_subdir", "evolved")
    output_dir = os.path.join(script_dir, output_subdir)

    config = {
        **evolve_defaults,
        **vars(args),
        "warm_start": not args.no_warm_start,
        "latent_dim": vae_config["latent_dim"],
        "inner_dim": 13,
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir,
    }

    run_evolution(config)
