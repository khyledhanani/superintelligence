"""
Evolve CLUTTR gridworld environments using CMA-ES in the VAE latent space.

Supports two fitness modes:
    - "placeholder": structural complexity (obstacle density + distance + validity)
    - "regret": ACCEL-inspired MaxMC regret from a frozen RL agent

Usage:
    # Placeholder fitness (no agent needed)
    python evolve_envs.py --fitness_mode placeholder

    # Regret fitness (requires agent checkpoint)
    python evolve_envs.py --fitness_mode regret --agent_checkpoint_dir agent_folder/119
"""

import jax
import jax.numpy as jnp
import numpy as np
import yaml
import os
import argparse
import sys
import time

# Add parent directory to path to import from vae folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evosax.algorithms import CMA_ES

from vae_decoder import (
    load_vae_params,
    extract_decoder_params,
    decode_latent_to_env,
    repair_cluttr_sequence,
)
from vae.sample_envs import generate_cluttr_batch_jax


# ---------------------------------------------------------------------------
# Fitness function (placeholder)
# ---------------------------------------------------------------------------

def placeholder_fitness(sequences, inner_dim=13, w_obstacles=0.4, w_distance=0.4, w_validity=0.2):
    """Evaluate environment complexity as a proxy fitness.

    Combines obstacle density, agent-goal distance, and structural validity.
    Returns NEGATED scores because evosax MINIMIZES.

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

    obs_count = jnp.sum(obstacles > 0, axis=1).astype(jnp.float32)
    obs_score = obs_count / 50.0

    agent_row = (agent_idx - 1) // inner_dim
    agent_col = (agent_idx - 1) % inner_dim
    goal_row = (goal_idx - 1) // inner_dim
    goal_col = (goal_idx - 1) % inner_dim
    manhattan = (jnp.abs(agent_row - goal_row) + jnp.abs(agent_col - goal_col)).astype(jnp.float32)
    dist_score = manhattan / (2.0 * (inner_dim - 1))

    valid = (
        (goal_idx >= 1) & (goal_idx <= inner_dim**2) &
        (agent_idx >= 1) & (agent_idx <= inner_dim**2) &
        (goal_idx != agent_idx)
    ).astype(jnp.float32)

    fitness = w_obstacles * obs_score + w_distance * dist_score + w_validity * valid
    return -fitness


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
    return means.mean(axis=0)


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def run_evolution(config):
    """Run CMA-ES evolution in the VAE latent space.

    Args:
        config: Dictionary with evolution and VAE configuration.
    """
    key = jax.random.PRNGKey(config["seed"])
    fitness_mode = config.get("fitness_mode", "placeholder")

    # 1. Load VAE
    print("Loading VAE checkpoint...")
    full_vae_params = load_vae_params(config["checkpoint_path"])
    decoder_params = extract_decoder_params(full_vae_params)

    # 2. Load agent + env if using regret fitness
    agent_params = None
    network = None
    wrapped_env = None
    env_params = None

    if fitness_mode == "regret":
        from agent_loader import load_agent, verify_agent_contract
        from regret_fitness import regret_fitness
        from jaxued.environments.maze import Maze
        from jaxued.wrappers import AutoReplayWrapper

        agent_checkpoint_dir = config.get("agent_checkpoint_dir")
        if not agent_checkpoint_dir:
            raise ValueError("--agent_checkpoint_dir required for regret fitness mode")

        print(f"Loading agent from {agent_checkpoint_dir}...")
        agent_params, network = load_agent(agent_checkpoint_dir, action_dim=7)
        verify_agent_contract(agent_params, network)

        maze_env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
        wrapped_env = AutoReplayWrapper(maze_env)
        env_params = wrapped_env.default_params
        print("Agent and Maze environment initialized.")

    # 3. Initialize CMA-ES
    latent_dim = config["latent_dim"]
    dummy_solution = jnp.zeros(latent_dim)
    es = CMA_ES(population_size=config["pop_size"], solution=dummy_solution)
    es_params = es.default_params

    # Warm-start (disabled by default in regret mode to avoid cwd config loading issues)
    if config.get("warm_start", False):
        print("Computing warm-start mean from random environments...")
        key, ws_key = jax.random.split(key)
        init_mean = compute_warm_start_mean(ws_key, full_vae_params, config["pop_size"])
    else:
        init_mean = dummy_solution

    key, init_key = jax.random.split(key)
    es_state = es.init(init_key, init_mean, es_params)

    # 4. JIT-compile decode + fitness
    deterministic = config.get("eval_policy_mode", "deterministic") == "deterministic"
    rollout_steps = config.get("rollout_steps", 256)
    decode_temperature = float(config.get("decode_temperature", 0.0))
    min_obstacles = int(config.get("min_obstacles", 0))
    min_distance = int(config.get("min_distance", 0))
    inner_dim = int(config.get("inner_dim", 13))

    if fitness_mode == "regret":
        @jax.jit
        def evaluate_population(eval_key, population):
            decode_key, regret_key = jax.random.split(eval_key)
            if decode_temperature > 0:
                sequences = decode_latent_to_env(
                    decoder_params, population, rng_key=decode_key, temperature=decode_temperature
                )
            else:
                sequences = decode_latent_to_env(decoder_params, population)
            sequences = jax.vmap(repair_cluttr_sequence)(sequences)
            fitness, info = regret_fitness(
                regret_key, sequences, agent_params, network, wrapped_env, env_params,
                num_steps=rollout_steps, deterministic=deterministic,
                min_obstacles=min_obstacles, min_distance=min_distance, inner_dim=inner_dim,
            )
            return fitness, sequences, info
    else:
        @jax.jit
        def evaluate_population(eval_key, population):
            if decode_temperature > 0:
                sequences = decode_latent_to_env(
                    decoder_params, population, rng_key=eval_key, temperature=decode_temperature
                )
            else:
                sequences = decode_latent_to_env(decoder_params, population)
            sequences = jax.vmap(repair_cluttr_sequence)(sequences)
            fitness = placeholder_fitness(
                sequences,
                inner_dim=config.get("inner_dim", 13),
                w_obstacles=config.get("w_obstacles", 0.4),
                w_distance=config.get("w_distance", 0.4),
                w_validity=config.get("w_validity", 0.2),
            )
            return fitness, sequences, {}

    # 5. Evolution loop
    num_gens = config["num_generations"]
    log_freq = config.get("log_freq", 10)

    # Metrics tracking
    if fitness_mode == "regret":
        from metrics import EvolutionMetrics, compute_generation_metrics
        evo_metrics = EvolutionMetrics()
    best_fitness_history = []
    mean_fitness_history = []

    print(f"Starting CMA-ES evolution: {num_gens} generations, pop_size={config['pop_size']}, "
          f"latent_dim={latent_dim}, fitness_mode={fitness_mode}")
    print("-" * 80)

    for gen in range(num_gens):
        t_start = time.time()
        key, ask_key, tell_key, eval_key = jax.random.split(key, 4)

        # Ask
        population, es_state = es.ask(ask_key, es_state, es_params)

        # Evaluate
        fitness, sequences, info = evaluate_population(eval_key, population)

        # Tell
        es_state, es_metrics = es.tell(tell_key, population, fitness, es_state, es_params)

        t_end = time.time()
        gen_time = t_end - t_start

        # Track fitness (negate back since we negated for minimization)
        best_idx = jnp.argmin(fitness)
        best_fit = float(-fitness[best_idx])
        mean_fit = float(-fitness.mean())
        best_fitness_history.append(best_fit)
        mean_fitness_history.append(mean_fit)

        # Track regret-specific metrics
        if fitness_mode == "regret" and info:
            gen_data = compute_generation_metrics(info, population, sequences, es_state, gen_time)
            gen_data['best_fitness'] = best_fit
            gen_data['mean_fitness'] = mean_fit
            evo_metrics.record(gen_data)

            if gen % log_freq == 0:
                print(
                    f"Gen {gen:4d} | "
                    f"Regret: best={gen_data['best_regret']:.4f} mean={gen_data['mean_regret']:.4f} | "
                    f"Solvable: {gen_data['solvability_rate']:.0%} | "
                    f"Complex: {gen_data['complexity_pass_rate']:.0%} | "
                    f"Div(L2): {gen_data['latent_diversity']:.3f} | "
                    f"sigma: {gen_data['cma_sigma']:.4f} | "
                    f"Time: {gen_time:.2f}s"
                )
        else:
            if gen % log_freq == 0:
                std_val = float(es_state.std) if es_state.std.ndim == 0 else float(es_state.std.mean())
                print(
                    f"Gen {gen:4d} | "
                    f"Best: {best_fit:.4f} | "
                    f"Mean: {mean_fit:.4f} | "
                    f"Std: {std_val:.4f}"
                )

    print("-" * 80)
    print("Evolution complete.")

    # 6. Extract and save results
    key, final_key, final_eval_key = jax.random.split(key, 3)
    final_pop, _ = es.ask(final_key, es_state, es_params)
    _, final_sequences, _ = evaluate_population(final_eval_key, final_pop)

    best_z = es_state.best_solution[None, :]
    # Keep final "best env" decode deterministic for interpretability.
    best_seq = decode_latent_to_env(decoder_params, best_z, rng_key=None)
    best_seq = jax.vmap(repair_cluttr_sequence)(best_seq)

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "evolved_envs.npy"), np.array(final_sequences))
    np.save(os.path.join(output_dir, "best_env.npy"), np.array(best_seq))
    np.save(os.path.join(output_dir, "fitness_history.npy"),
            np.column_stack([best_fitness_history, mean_fitness_history]))
    np.save(os.path.join(output_dir, "best_latent.npy"), np.array(es_state.best_solution))

    if fitness_mode == "regret":
        evo_metrics.save(output_dir)

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

    # Placeholder fitness weights
    parser.add_argument("--w_obstacles", type=float, default=0.4)
    parser.add_argument("--w_distance", type=float, default=0.4)
    parser.add_argument("--w_validity", type=float, default=0.2)

    # Fitness mode
    parser.add_argument("--fitness_mode", type=str, default="placeholder",
                        choices=["placeholder", "regret"],
                        help="Fitness function: 'placeholder' (structural) or 'regret' (agent-based)")
    parser.add_argument("--agent_checkpoint_dir", type=str, default=None,
                        help="Path to orbax agent checkpoint directory (required for regret mode)")
    parser.add_argument("--rollout_steps", type=int, default=256,
                        help="Steps per rollout for regret evaluation")
    parser.add_argument("--eval_policy_mode", type=str, default="deterministic",
                        choices=["deterministic", "stochastic"],
                        help="Agent policy during eval: argmax (deterministic) or sample (stochastic)")
    parser.add_argument("--decode_temperature", type=float, default=None,
                        help="Decoder sampling temperature. <=0 disables sampling (argmax decode).")
    parser.add_argument("--min_obstacles", type=int, default=None,
                        help="Complexity gate: minimum non-zero obstacle tokens (regret mode).")
    parser.add_argument("--min_distance", type=int, default=None,
                        help="Complexity gate: minimum Manhattan distance agent-goal (regret mode).")

    # Output
    parser.add_argument("--output_subdir", type=str, default=None,
                        help="Output subdirectory (default: 'evolved' or 'evolved_regret')")

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
            evolve_defaults = yaml.safe_load(f) or {}

    # Checkpoint path: VAE model in vae/model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "..", "vae", "model", "checkpoint_420000.pkl")

    # Output directory
    default_subdir = "evolved_regret" if args.fitness_mode == "regret" else "evolved"
    output_subdir = args.output_subdir or evolve_defaults.get("output_subdir", default_subdir)
    output_dir = os.path.join(script_dir, output_subdir)

    # Warm-start: disabled by default in regret mode (cwd config loading trap)
    if args.fitness_mode == "regret" and not args.no_warm_start:
        warm_start = evolve_defaults.get("warm_start", False)
    else:
        warm_start = not args.no_warm_start

    # Resolve defaults after mode is known.
    if args.decode_temperature is None:
        if args.fitness_mode == "regret":
            decode_temperature = evolve_defaults.get("decode_temperature", 1.0)
        else:
            decode_temperature = 0.0
    else:
        decode_temperature = args.decode_temperature
    if args.min_obstacles is None:
        min_obstacles = evolve_defaults.get("min_obstacles", 5)
    else:
        min_obstacles = args.min_obstacles
    if args.min_distance is None:
        min_distance = evolve_defaults.get("min_distance", 3)
    else:
        min_distance = args.min_distance

    config = {
        **evolve_defaults,
        **vars(args),
        "warm_start": warm_start,
        "decode_temperature": decode_temperature,
        "min_obstacles": min_obstacles,
        "min_distance": min_distance,
        "latent_dim": vae_config["latent_dim"],
        "inner_dim": 13,
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir,
    }

    run_evolution(config)
