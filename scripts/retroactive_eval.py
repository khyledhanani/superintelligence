#!/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python
"""
Retroactive evaluation of all saved checkpoints using ACCEL-identical protocol.

Matches maze_plr.py eval exactly:
  - 10 stochastic rollouts per level (pi.sample, not argmax)
  - solve_rate = fraction of attempts with cumulative reward > 0
  - 8 DCD paper benchmark levels (SixteenRooms..StandardMaze3)

Evaluates every checkpoint for: CMA-ES, NS-ES, SV-CMA-ES, SV-CMA-ES-v2, ACCEL.
Runs on CPU (no GPU needed — agent is small).

Usage:
    python scripts/retroactive_eval.py
    python scripts/retroactive_eval.py --strategies cma-es ns-es
    python scripts/retroactive_eval.py --output_dir eval_retroactive
"""

import os
import sys
import argparse
import pickle
import json
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'accel_training'))

# Force CPU (avoids GPU contention and works on head nodes)
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np

from jaxued.environments.maze import Maze
from jaxued.environments.maze.level import Level
from jaxued.wrappers import AutoReplayWrapper

from agent_loader import ActorCritic, load_agent_params

# ---------------------------------------------------------------------------
# Constants — exact match to maze_plr.py defaults
# ---------------------------------------------------------------------------

BENCHMARK_NAMES = [
    "SixteenRooms", "SixteenRooms2",
    "Labyrinth", "LabyrinthFlipped", "Labyrinth2",
    "StandardMaze", "StandardMaze2", "StandardMaze3",
]

N_EVAL_ATTEMPTS = 10  # maze_plr.py line 847: --eval_num_attempts default=10

# ES checkpoints: runs/<strategy>/checkpoint_NNNNNN/agent_params.pkl
ES_STRATEGIES = {
    'cma-es':       'runs/phase5-cma-es',
    'ns-es':        'runs/phase5-ns-es',
    'sv-cma-es':    'runs/phase5-sv-cma-es',
    'sv-cma-es-v2': 'runs/phase5-sv-cma-es-v2',
}

# ACCEL checkpoint: extracted pickle (from orbax via extract_accel_params.py)
ACCEL_CHECKPOINT_PKL = 'checkpoints/phase5-accel/42/agent_params.pkl'


# ---------------------------------------------------------------------------
# Eval function — mirrors maze_plr.py evaluate_rnn + eval
# ---------------------------------------------------------------------------

def eval_single_attempt(rng, env, env_params, agent_params, network, levels, num_steps):
    """Single stochastic rollout on all levels. Returns cumulative rewards (n_levels,)."""
    n_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]

    rng, rng_reset = jax.random.split(rng)
    init_obs, init_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(rng_reset, n_levels), levels, env_params
    )
    init_hstate = ActorCritic.initialize_carry((n_levels,))

    def step_fn(carry, _):
        rng_c, hstate, obs, state, done, mask, ep_len = carry
        rng_c, rng_act, rng_step = jax.random.split(rng_c, 3)

        x = jax.tree_util.tree_map(lambda a: a[None, ...], (obs, done))
        hstate, pi, _ = network.apply(agent_params, x, hstate)

        # Stochastic sampling — same as maze_plr.py line 180
        action = pi.sample(seed=rng_act).squeeze(0)

        next_obs, next_state, reward, next_done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, n_levels), state, action, env_params)

        next_mask = mask & ~next_done
        ep_len = ep_len + mask

        return (rng_c, hstate, next_obs, next_state, next_done, next_mask, ep_len), reward

    (_, _, _, _, _, _, ep_lengths), rewards = jax.lax.scan(
        step_fn,
        (rng, init_hstate, init_obs, init_state,
         jnp.zeros(n_levels, dtype=jnp.bool_),
         jnp.ones(n_levels, dtype=jnp.bool_),
         jnp.zeros(n_levels, dtype=jnp.int32)),
        None,
        length=num_steps,
    )

    # mask rewards to episode length (same as maze_plr.py line 740)
    mask = jnp.arange(num_steps)[:, None] < ep_lengths[None, :]
    cum_rewards = (rewards * mask).sum(axis=0)  # (n_levels,)
    return cum_rewards


def eval_multi_attempt(rng, env, env_params, agent_params, network, levels, num_steps):
    """N_EVAL_ATTEMPTS stochastic rollouts, averaged. Returns solve_rates, mean_returns."""
    rngs = jax.random.split(rng, N_EVAL_ATTEMPTS)

    # vmap over attempts
    all_cum = jax.vmap(
        lambda r: eval_single_attempt(r, env, env_params, agent_params, network, levels, num_steps)
    )(rngs)  # (N_EVAL_ATTEMPTS, n_levels)

    solve_rates = jnp.where(all_cum > 0, 1.0, 0.0).mean(axis=0)  # (n_levels,)
    mean_returns = all_cum.mean(axis=0)  # (n_levels,)
    return solve_rates, mean_returns


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_es_checkpoints(run_dir):
    """Find all checkpoint directories in an ES run, sorted by step."""
    if not os.path.isdir(run_dir):
        return []
    ckpts = []
    for d in sorted(os.listdir(run_dir)):
        if d.startswith('checkpoint_') and os.path.isdir(os.path.join(run_dir, d)):
            step = int(d.split('_')[1])
            ckpts.append((step, os.path.join(run_dir, d)))
    return ckpts


def find_accel_checkpoints(models_dir):
    """Find ACCEL orbax checkpoint steps."""
    if not os.path.isdir(models_dir):
        return []
    ckpts = []
    for d in sorted(os.listdir(models_dir)):
        full = os.path.join(models_dir, d)
        if os.path.isdir(full) and d.isdigit():
            step = int(d)
            ckpts.append((step, full))
    return ckpts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Retroactive ACCEL-protocol evaluation')
    parser.add_argument('--strategies', nargs='+', default=None,
                        help='Which strategies to evaluate (default: all)')
    parser.add_argument('--output_dir', type=str, default='eval_retroactive',
                        help='Output directory for JSON results')
    parser.add_argument('--final_only', action='store_true',
                        help='Only evaluate the final (latest) checkpoint per strategy')
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Eval protocol: {N_EVAL_ATTEMPTS} stochastic attempts, {len(BENCHMARK_NAMES)} DCD benchmark levels")
    print()

    # Environment setup
    maze_env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    env = AutoReplayWrapper(maze_env)
    env_params = env.default_params
    network = ActorCritic(action_dim=7)
    num_steps = env_params.max_steps_in_episode  # 250

    # Load benchmark levels
    benchmark_levels = Level.load_prefabs(BENCHMARK_NAMES)
    print(f"Benchmark levels loaded: {BENCHMARK_NAMES}")
    print(f"Rollout length: {num_steps} steps")
    print()

    # JIT compile eval function once
    print("JIT compiling eval function (first call will be slow)...")
    _eval_jit = jax.jit(lambda rng, params: eval_multi_attempt(
        rng, env, env_params, params, network, benchmark_levels, num_steps
    ))

    # Determine which strategies to evaluate
    strategies_to_eval = {}

    if args.strategies:
        for s in args.strategies:
            s_lower = s.lower()
            if s_lower == 'accel':
                strategies_to_eval['accel'] = ('accel_pkl', ACCEL_CHECKPOINT_PKL)
            elif s_lower in ES_STRATEGIES:
                strategies_to_eval[s_lower] = ('es', ES_STRATEGIES[s_lower])
            else:
                print(f"Warning: unknown strategy '{s}', skipping")
    else:
        # All strategies
        for name, path in ES_STRATEGIES.items():
            if os.path.isdir(path):
                strategies_to_eval[name] = ('es', path)
        if os.path.isfile(ACCEL_CHECKPOINT_PKL):
            strategies_to_eval['accel'] = ('accel_pkl', ACCEL_CHECKPOINT_PKL)

    if not strategies_to_eval:
        print("No strategies found to evaluate!")
        return

    print(f"Strategies: {list(strategies_to_eval.keys())}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for strat_name, (strat_type, strat_path) in strategies_to_eval.items():
        print(f"{'='*70}")
        print(f"  {strat_name.upper()}")
        print(f"{'='*70}")

        # Find checkpoints
        if strat_type == 'es':
            checkpoints = find_es_checkpoints(strat_path)
        elif strat_type == 'accel_pkl':
            # Single pickle checkpoint — use step 0 (ACCEL saves final only)
            checkpoints = [(0, strat_path)] if os.path.isfile(strat_path) else []
        else:
            checkpoints = find_accel_checkpoints(strat_path)

        if not checkpoints:
            print(f"  No checkpoints found in {strat_path}")
            continue

        if args.final_only:
            checkpoints = [checkpoints[-1]]

        strat_results = []

        for step, ckpt_path in checkpoints:
            print(f"  Step {step:>6d} ... ", end='', flush=True)
            t0 = time.time()

            try:
                if strat_type == 'accel_pkl':
                    # Direct pickle: contains raw network params (no 'params' wrapper)
                    with open(ckpt_path, 'rb') as f:
                        raw_params = pickle.load(f)
                    agent_params = {'params': jax.tree_util.tree_map(jnp.asarray, raw_params)}
                else:
                    agent_params = load_agent_params(ckpt_path)
                    # load_agent_params returns raw params; wrap for network.apply
                    if isinstance(agent_params, dict) and 'params' not in agent_params:
                        agent_params = {'params': agent_params}

                rng = jax.random.PRNGKey(42)
                solve_rates, mean_returns = _eval_jit(rng, agent_params)

                solve_rates_np = np.asarray(solve_rates)
                mean_returns_np = np.asarray(mean_returns)

                dt = time.time() - t0

                per_level = {
                    name: {'solve_rate': float(solve_rates_np[i]),
                           'mean_return': float(mean_returns_np[i])}
                    for i, name in enumerate(BENCHMARK_NAMES)
                }

                result = {
                    'step': step,
                    'mean_solve_rate': float(solve_rates_np.mean()),
                    'mean_return': float(mean_returns_np.mean()),
                    'per_level': per_level,
                }
                strat_results.append(result)

                # Print summary
                level_str = ' '.join(f"{s:.0%}" for s in solve_rates_np)
                print(f"mean_solve={solve_rates_np.mean():.1%}  [{level_str}]  ({dt:.1f}s)")

            except Exception as e:
                print(f"ERROR: {e}")
                continue

        all_results[strat_name] = strat_results

        # Save per-strategy JSON
        out_path = os.path.join(args.output_dir, f'{strat_name}_eval.json')
        with open(out_path, 'w') as f:
            json.dump({strat_name: strat_results}, f, indent=2)

    # ---------------------------------------------------------------------------
    # Print comparison table
    # ---------------------------------------------------------------------------
    print()
    print(f"{'='*100}")
    print(f"  COMPARISON TABLE (ACCEL-protocol: {N_EVAL_ATTEMPTS} stochastic attempts)")
    print(f"{'='*100}")

    # Header
    header = f"{'Strategy':<18} {'Step':>6} {'Mean':>6}"
    for name in BENCHMARK_NAMES:
        short = name[:10]
        header += f" {short:>10}"
    print(header)
    print('-' * len(header))

    for strat_name, results in all_results.items():
        if not results:
            continue
        # Print final checkpoint only in the table
        r = results[-1]
        row = f"{strat_name:<18} {r['step']:>6} {r['mean_solve_rate']:>5.1%}"
        for name in BENCHMARK_NAMES:
            sr = r['per_level'][name]['solve_rate']
            row += f" {sr:>10.0%}"
        print(row)

    # Save combined results
    combined_path = os.path.join(args.output_dir, 'combined_results.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.output_dir}/")

    # ---------------------------------------------------------------------------
    # Learning curve summary (all checkpoints)
    # ---------------------------------------------------------------------------
    print()
    print(f"{'='*70}")
    print(f"  LEARNING CURVES (mean solve rate over training)")
    print(f"{'='*70}")
    print(f"{'Strategy':<18} ", end='')
    # Get all unique steps
    all_steps = sorted(set(r['step'] for results in all_results.values() for r in results))
    for s in all_steps:
        print(f"{s:>7}", end='')
    print()
    print('-' * (18 + 7 * len(all_steps)))

    for strat_name, results in all_results.items():
        step_to_sr = {r['step']: r['mean_solve_rate'] for r in results}
        print(f"{strat_name:<18} ", end='')
        for s in all_steps:
            if s in step_to_sr:
                print(f"{step_to_sr[s]:>6.1%} ", end='')
            else:
                print(f"{'--':>7}", end='')
        print()


if __name__ == '__main__':
    main()
