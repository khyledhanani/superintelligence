#!/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python
"""
Evaluate agent checkpoints on a fixed test set of mazes.

Runs each checkpoint's agent on:
  1. All 11 jaxued prefab mazes (TrivialMaze..StandardMaze3)
  2. 50 random VAE-decoded mazes (frozen seed, solvable + complex)
  = 61 total test levels

For each (checkpoint, level), records:
  - solved: bool (agent reached goal at least once in 256 steps)
  - total_return: sum of rewards over 256 steps
  - regret (MaxMC): how much worse than optimal

Usage:
    python scripts/evaluate_checkpoints.py --run_dir runs/phase5-cma-es
    python scripts/evaluate_checkpoints.py --run_dir runs/phase5-ns-es
    python scripts/evaluate_checkpoints.py --run_dir runs/phase5-sv-cma-es
    python scripts/evaluate_checkpoints.py --all   # evaluate all three
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np

# Project imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'accel_training'))
sys.path.insert(0, os.path.join(ROOT, 'es'))

import jax
import jax.numpy as jnp

from jaxued.environments.maze import Maze
from jaxued.environments import MazeRenderer
from jaxued.environments.maze.level import Level, prefabs
from jaxued.wrappers import AutoReplayWrapper
from jaxued.utils import compute_max_returns, max_mc

from agent_loader import ActorCritic, load_agent_params
from vae_decoder import load_vae_params, extract_decoder_params, decode_latent_to_env, repair_cluttr_sequence
from env_bridge import cluttr_sequence_to_level, flood_fill_solvable
from regret_fitness import compute_complexity_mask


# ──────────────────────────────────────────────────────────────────────────────
# Test set construction
# ──────────────────────────────────────────────────────────────────────────────

PREFAB_NAMES = list(prefabs.keys())  # 11 mazes

def build_test_set(vae_checkpoint, n_random=50, seed=999):
    """Build a fixed set of test levels: prefabs + VAE-decoded random mazes.

    Returns:
        levels: stacked Level pytree, shape (N, ...)
        names: list of str, one per level
        n_prefab: int, number of prefab levels
    """
    # 1. Prefab levels
    prefab_levels = Level.load_prefabs(PREFAB_NAMES)
    n_prefab = len(PREFAB_NAMES)

    # 2. VAE-decoded levels (deterministic via fixed seed)
    full_params = load_vae_params(vae_checkpoint)
    decoder_params = extract_decoder_params(full_params)

    rng = jax.random.PRNGKey(seed)
    # Over-sample to get enough valid ones
    n_sample = n_random * 5
    rng, rng_z, rng_dec, rng_dirs = jax.random.split(rng, 4)
    z = jax.random.normal(rng_z, (n_sample, 64))

    sequences = decode_latent_to_env(decoder_params, z, rng_key=rng_dec, temperature=0.25)
    sequences = jax.vmap(repair_cluttr_sequence)(sequences)

    dir_keys = jax.random.split(rng_dirs, n_sample)
    levels_all = jax.vmap(cluttr_sequence_to_level)(sequences, dir_keys)

    solvable = jax.vmap(flood_fill_solvable)(
        levels_all.wall_map, levels_all.agent_pos, levels_all.goal_pos
    )
    complex_enough = compute_complexity_mask(sequences, min_obstacles=5, min_distance=3)
    valid = np.asarray(solvable & complex_enough)

    valid_idx = np.where(valid)[0][:n_random]
    if len(valid_idx) < n_random:
        print(f"  Warning: only {len(valid_idx)} valid random levels (wanted {n_random})")

    random_levels = jax.tree_util.tree_map(lambda x: x[valid_idx], levels_all)
    random_names = [f"random_{i:03d}" for i in range(len(valid_idx))]

    # 3. Stack all
    all_levels = jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate([a, b], axis=0),
        prefab_levels, random_levels
    )
    all_names = PREFAB_NAMES + random_names

    return all_levels, all_names, n_prefab


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_agent(agent_params, network, env, env_params, levels, num_steps=256):
    """Run agent on all levels and compute metrics.

    Returns dict with:
        solved: (N,) bool
        total_return: (N,) float
        regret: (N,) float (MaxMC)
    """
    n_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]

    rng = jax.random.PRNGKey(0)
    rng, rng_reset = jax.random.split(rng)

    init_obs, init_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(rng_reset, n_levels), levels, env_params
    )
    init_hstate = ActorCritic.initialize_carry((n_levels,))

    @jax.jit
    def _rollout(rng, init_hstate, init_obs, init_state):
        def step_fn(carry, _):
            rng_c, hstate, obs, state, done = carry
            rng_c, rng_act, rng_step = jax.random.split(rng_c, 3)

            x = jax.tree_util.tree_map(lambda a: a[None, ...], (obs, done))
            hstate, pi, value = network.apply(agent_params, x, hstate)

            action = jnp.argmax(pi.logits, axis=-1).squeeze(0)
            value = value.squeeze(0)

            next_obs, next_state, reward, next_done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(rng_step, n_levels), state, action, env_params)

            return (rng_c, hstate, next_obs, next_state, next_done), (reward, value, next_done)

        _, (rewards, values, dones) = jax.lax.scan(
            step_fn,
            (rng, init_hstate, init_obs, init_state, jnp.zeros(n_levels, dtype=jnp.bool_)),
            None,
            length=num_steps,
        )
        return rewards, values, dones

    rewards, values, dones = _rollout(rng, init_hstate, init_obs, init_state)

    # Metrics
    total_return = rewards.sum(axis=0)  # (N,)
    solved = dones.any(axis=0)           # (N,) — agent reached goal at least once

    max_returns = compute_max_returns(dones, rewards)
    regret = max_mc(dones, values, max_returns, incomplete_value=0.0)

    return {
        'solved': np.asarray(solved),
        'total_return': np.asarray(total_return),
        'regret': np.asarray(regret),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_run(run_dir, test_levels, level_names, n_prefab, network, env, env_params):
    """Evaluate all checkpoints in a run directory."""
    run_name = os.path.basename(run_dir)
    ckpt_dirs = sorted([
        d for d in os.listdir(run_dir)
        if d.startswith('checkpoint_') and os.path.isdir(os.path.join(run_dir, d))
    ])

    if not ckpt_dirs:
        print(f"  No checkpoints found in {run_dir}")
        return None

    results = {'run': run_name, 'checkpoints': []}

    for ckpt_name in ckpt_dirs:
        ckpt_path = os.path.join(run_dir, ckpt_name)
        step = int(ckpt_name.split('_')[1])
        print(f"  {run_name} / {ckpt_name} (step {step})...", end=' ', flush=True)

        agent_params = load_agent_params(ckpt_path)

        metrics = evaluate_agent(agent_params, network, env, env_params, test_levels)

        # Summary stats
        n_total = len(level_names)
        n_solved = int(metrics['solved'].sum())
        solve_rate = n_solved / n_total
        mean_return = float(metrics['total_return'].mean())
        mean_regret = float(metrics['regret'].mean())

        # Breakdown: prefab vs random
        prefab_solved = int(metrics['solved'][:n_prefab].sum())
        random_solved = int(metrics['solved'][n_prefab:].sum())
        n_random = n_total - n_prefab

        print(f"solved {n_solved}/{n_total} ({solve_rate:.1%}) | "
              f"prefab {prefab_solved}/{n_prefab} | random {random_solved}/{n_random} | "
              f"return={mean_return:.2f} | regret={mean_regret:.4f}")

        ckpt_result = {
            'step': step,
            'solve_rate': solve_rate,
            'n_solved': n_solved,
            'n_total': n_total,
            'prefab_solved': prefab_solved,
            'random_solved': random_solved,
            'mean_return': mean_return,
            'mean_regret': mean_regret,
            'per_level': {
                name: {
                    'solved': bool(metrics['solved'][i]),
                    'total_return': float(metrics['total_return'][i]),
                    'regret': float(metrics['regret'][i]),
                }
                for i, name in enumerate(level_names)
            },
        }
        results['checkpoints'].append(ckpt_result)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate agent checkpoints on fixed test set')
    parser.add_argument('--run_dir', type=str, help='Path to a single run directory')
    parser.add_argument('--all', action='store_true', help='Evaluate all phase5 runs')
    parser.add_argument('--vae', type=str, default='vae/model/checkpoint_final.pkl',
                        help='VAE checkpoint path')
    parser.add_argument('--n_random', type=int, default=50,
                        help='Number of random test mazes')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Directory for output JSON files')
    args = parser.parse_args()

    os.chdir(ROOT)

    # Determine which runs to evaluate
    if args.all:
        run_dirs = sorted([
            os.path.join('runs', d) for d in os.listdir('runs')
            if d.startswith('phase5-') and os.path.isdir(os.path.join('runs', d))
        ])
    elif args.run_dir:
        run_dirs = [args.run_dir]
    else:
        parser.error('Specify --run_dir or --all')

    print(f"Runs to evaluate: {[os.path.basename(d) for d in run_dirs]}")

    # Build environment
    maze_env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    env = AutoReplayWrapper(maze_env)
    env_params = env.default_params
    network = ActorCritic(action_dim=7)

    # Build test set
    print("Building test set...")
    test_levels, level_names, n_prefab = build_test_set(args.vae, n_random=args.n_random)
    print(f"  {len(level_names)} test levels: {n_prefab} prefab + {len(level_names)-n_prefab} random")

    # Evaluate
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for run_dir in run_dirs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {run_dir}")
        print(f"{'='*60}")
        result = evaluate_run(run_dir, test_levels, level_names, n_prefab,
                              network, env, env_params)
        if result:
            all_results.append(result)

            # Save per-run JSON
            out_path = os.path.join(args.output_dir, f"{result['run']}_eval.json")
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {out_path}")

    # Save combined summary
    if all_results:
        summary_path = os.path.join(args.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved: {summary_path}")

        # Print comparison table
        print(f"\n{'='*80}")
        print(f"{'Strategy':<20} {'Step':>6} {'Solve%':>7} {'Prefab':>8} {'Random':>8} {'Return':>8} {'Regret':>8}")
        print(f"{'-'*80}")
        for r in all_results:
            for c in r['checkpoints']:
                print(f"{r['run']:<20} {c['step']:>6} {c['solve_rate']:>6.1%} "
                      f"{c['prefab_solved']:>3}/{n_prefab:<4} "
                      f"{c['random_solved']:>3}/{len(level_names)-n_prefab:<4} "
                      f"{c['mean_return']:>8.2f} {c['mean_regret']:>8.4f}")


if __name__ == '__main__':
    main()
