"""
Cross-evaluate agent checkpoints on buffer levels from different training runs.

Loads an agent from one run's checkpoint and evaluates it on buffer levels
from another run's buffer dump (.npz with 'tokens' key).

Usage:
    # Single cross-eval: ACCEL agent on CMA-ES buffer
    python examples/cross_evaluate.py \
        --agent_checkpoint_dir checkpoints/accel_only/0 \
        --buffer_npz /tmp/buffer_dumps/cmaes_accel/0/buffer_dump_30k.npz \
        --num_attempts 10

    # Batch mode: all combinations of agents x buffers
    python examples/cross_evaluate.py --batch \
        --agent_dirs checkpoints/accel_only/0 checkpoints/cmaes_accel/0 \
        --buffer_npzs /tmp/buffer_dumps/accel_only/0/buffer_dump_30k.npz \
                      /tmp/buffer_dumps/cmaes_accel/0/buffer_dump_30k.npz \
        --num_attempts 10 \
        --output_dir results/cross_eval/

    # Download buffers from GCS first if needed:
    gcloud storage cp -r gs://bucket/accel/buffer_dumps/run_name/seed/ /tmp/buffer_dumps/run_name/seed/
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from functools import partial

sys.path.insert(0, os.path.dirname(__file__))

from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator
from jaxued.level_sampler import LevelSampler
from jaxued.wrappers import AutoReplayWrapper

from maze_plr import ActorCritic, TrainState, evaluate_rnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vae'))
from vae_level_utils import tokens_to_level


def load_agent(checkpoint_dir, checkpoint_step=-1):
    """Load agent params and config from an orbax checkpoint.

    Returns (train_state, config, env, env_params).
    """
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    base_env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    env = AutoReplayWrapper(base_env)
    env_params = env.default_params
    sample_random_level = make_level_generator(base_env.max_height, base_env.max_width, config.get("n_walls", 25))

    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config.get("topk_k", 4)},
        duplicate_check=config.get("buffer_duplicate_check", True),
    )

    rng = jax.random.PRNGKey(0)
    obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
    obs_batch = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
        obs,
    )
    init_x = (obs_batch, jnp.zeros((256, config["num_train_envs"])))
    network = ActorCritic(env.action_space(env_params).n)
    network_params = network.init(rng, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))

    import optax
    tx = optax.chain(
        optax.clip_by_global_norm(config.get("max_grad_norm", 0.5)),
        optax.adam(learning_rate=config.get("lr", 1e-4), eps=1e-5),
    )

    pholder_level = sample_random_level(jax.random.PRNGKey(0))
    sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
    pholder_level_batch = jax.tree_util.tree_map(
        lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), pholder_level
    )

    template_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
        sampler=sampler,
        update_state=0,
        es_state=None,
        num_dr_updates=0,
        num_replay_updates=0,
        num_mutation_updates=0,
        dr_last_level_batch=pholder_level_batch,
        replay_last_level_batch=pholder_level_batch,
        mutation_last_level_batch=pholder_level_batch,
    )

    models_dir = os.path.join(checkpoint_dir, "models")
    checkpoint_manager = ocp.CheckpointManager(
        models_dir, item_handlers=ocp.StandardCheckpointHandler()
    )
    available = sorted(checkpoint_manager.all_steps())

    if checkpoint_step == -1:
        step = checkpoint_manager.latest_step()
    elif checkpoint_step not in available:
        # Find closest available step
        below = [s for s in available if s <= checkpoint_step]
        above = [s for s in available if s >= checkpoint_step]
        step = max(below) if below else min(above)
        print(f"[Agent] Step {checkpoint_step} not available, using closest: {step}")
    else:
        step = checkpoint_step

    eval_freq = config.get("eval_freq", 250)
    updates_at_step = (step + 1) * eval_freq

    # If a specific step was requested, check that the match is close enough
    if checkpoint_step >= 0:
        target_updates = (checkpoint_step + 1) * eval_freq
        update_gap = abs(updates_at_step - target_updates)
        max_gap = 2500  # max acceptable gap in updates
        if update_gap > max_gap:
            print(f"[Agent] SKIP: closest checkpoint is {updates_at_step} updates, "
                  f"target was {target_updates} (gap={update_gap} > {max_gap})")
            print(f"[Agent] Available steps: {available[0]}..{available[-1]} ({len(available)} total)")
            return None, config, None, None

    print(f"[Agent] Loading step {step} (~{updates_at_step} updates) from {models_dir}")
    print(f"[Agent] Available steps: {available[0]}..{available[-1]} ({len(available)} total)")

    # Try standard restore first; fall back to tensorstore for TPU->CPU
    try:
        restored = checkpoint_manager.restore(step)
    except ValueError as e:
        if "was not found in jax.local_devices()" in str(e) or "Topology mismatch" in str(e):
            print("[Agent] TPU checkpoint detected, restoring params via tensorstore...")
            restored = {"params": _restore_params_from_ocdbt(
                os.path.join(models_dir, str(step), "default"),
                network_params,
            )}
        else:
            raise

    train_state = template_state.replace(params=restored["params"])
    return train_state, config, env, env_params


def _restore_params_from_ocdbt(ckpt_default_dir, template_params):
    """Restore network params from OCDBT checkpoint, bypassing TPU sharding.

    Reads the _METADATA JSON for tree structure, then uses tensorstore
    with the zarr driver to read each array from the ocdbt.process_0 store.
    """
    import ast
    import tensorstore as ts

    ckpt_dir = os.path.abspath(ckpt_default_dir)
    ocdbt_base = "file://" + os.path.join(ckpt_dir, "ocdbt.process_0")

    # Read tree metadata
    with open(os.path.join(ckpt_dir, "_METADATA"), "r") as f:
        meta = json.load(f)

    # Extract param entries: keys like "('params', 'params', 'Conv_0', 'bias')"
    param_entries = {
        k: v for k, v in meta["tree_metadata"].items()
        if k.startswith("('params',")
    }

    params = {}
    for key_str in sorted(param_entries.keys()):
        key_tuple = ast.literal_eval(key_str)
        # key_tuple = ('params', 'params', 'Conv_0', 'bias')
        # tensorstore path uses dot-separated full key
        ts_path = ".".join(key_tuple)

        spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "ocdbt",
                "base": ocdbt_base,
                "path": ts_path,
            },
        }
        store = ts.open(spec, read=True, open=True).result()
        arr = jnp.array(store.read().result())

        # Build nested dict: skip first 'params' (TrainState.params), keep rest
        path = key_tuple[1:]  # ('params', 'Conv_0', 'bias')
        d = params
        for p in path[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[path[-1]] = arr

    return params


def tokens_to_levels_batch(tokens):
    """Convert (N, 52) token array to batched Level pytree."""
    return jax.vmap(tokens_to_level)(jnp.array(tokens))


def evaluate_agent_on_levels(train_state, env, env_params, levels, num_attempts, batch_size=4096):
    """Run agent on levels and return per-level solve rates.

    Args:
        train_state: Agent's TrainState (only params used).
        env: AutoReplayWrapper(Maze).
        env_params: Environment params.
        levels: Batched Level pytree (N levels).
        num_attempts: Number of rollouts per level (averaged for solve rate).
        batch_size: Max levels per evaluation chunk (to avoid OOM).

    Returns:
        solve_rates: (N,) float array, fraction of attempts that solved each level.
    """
    eval_env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]
    max_steps = env_params.max_steps_in_episode

    all_attempt_solves = []

    for attempt_idx in range(num_attempts):
        rng = jax.random.PRNGKey(attempt_idx + 1000)
        chunk_solves = []

        for start in range(0, num_levels, batch_size):
            end = min(start + batch_size, num_levels)
            chunk_levels = jax.tree_util.tree_map(lambda x: x[start:end], levels)
            n_chunk = end - start

            rng, rng_eval, rng_reset = jax.random.split(rng, 3)
            init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
                jax.random.split(rng_reset, n_chunk), chunk_levels, env_params
            )

            states, rewards, episode_lengths = evaluate_rnn(
                rng_eval, eval_env, env_params, train_state,
                ActorCritic.initialize_carry((n_chunk,)),
                init_obs, init_env_state, max_steps,
            )

            mask = np.arange(max_steps)[:, None] < np.asarray(episode_lengths)[None, :]
            cum_rewards = (np.asarray(rewards) * mask).sum(axis=0)
            solved = (cum_rewards > 0).astype(float)
            chunk_solves.append(solved)

        attempt_solve = np.concatenate(chunk_solves)
        all_attempt_solves.append(attempt_solve)

    solve_rates = np.stack(all_attempt_solves).mean(axis=0)
    return solve_rates


def run_single_cross_eval(agent_checkpoint_dir, agent_step, buffer_npz_path,
                          num_attempts, output_dir=None):
    """Run a single cross-evaluation and return results dict."""
    t0 = time.time()

    # Load agent
    train_state, config, env, env_params = load_agent(agent_checkpoint_dir, agent_step)
    agent_name = config.get("run_name", os.path.basename(agent_checkpoint_dir))
    agent_seed = config.get("seed", "?")

    if train_state is None:
        print(f"[Cross-eval] SKIP: no matching checkpoint for {agent_name}/seed{agent_seed}")
        return None

    # Load buffer tokens
    buffer_data = np.load(buffer_npz_path, allow_pickle=True)
    tokens = buffer_data["tokens"]
    size = int(buffer_data.get("size", len(tokens)))
    tokens = tokens[:size]
    update_num = int(buffer_data.get("update_num", 0))

    # Extract buffer identity from path
    # Expected: .../run_name/seed/buffer_dump_{N}k.npz
    buffer_path_parts = buffer_npz_path.replace("\\", "/").split("/")
    buffer_timestep = os.path.basename(buffer_npz_path).replace("buffer_dump", "").replace(".npz", "")
    # Try to extract run_name and seed from path: .../run_name/seed/buffer_dump_...
    try:
        buf_seed = buffer_path_parts[-2]
        buffer_run_name = buffer_path_parts[-3]
    except IndexError:
        buf_seed = "?"
        buffer_run_name = "unknown"
    buffer_tag = f"{buffer_run_name}_s{buf_seed}{buffer_timestep}"

    print(f"[Cross-eval] Agent: {agent_name}/seed{agent_seed} (step {agent_step})")
    print(f"[Cross-eval] Buffer: {buffer_run_name}/seed{buf_seed} @ {buffer_timestep} ({size} levels)")

    # Convert tokens to levels
    levels = tokens_to_levels_batch(tokens)

    # Evaluate
    solve_rates = evaluate_agent_on_levels(
        train_state, env, env_params, levels, num_attempts
    )

    results = {
        "agent_name": agent_name,
        "agent_seed": agent_seed,
        "agent_step": agent_step,
        "buffer_npz": buffer_npz_path,
        "buffer_run_name": buffer_run_name,
        "buffer_seed": buf_seed,
        "buffer_timestep": buffer_timestep,
        "buffer_tag": buffer_tag,
        "buffer_update_num": update_num,
        "num_levels": size,
        "num_attempts": num_attempts,
        "mean_solve_rate": float(solve_rates.mean()),
        "median_solve_rate": float(np.median(solve_rates)),
        "unsolved_count": int((solve_rates == 0).sum()),
        "fully_solved_count": int((solve_rates == 1.0).sum()),
        "solve_rates": solve_rates,
        "elapsed_s": time.time() - t0,
    }

    print(f"  Mean solve rate: {results['mean_solve_rate']:.1%}")
    print(f"  Unsolved: {results['unsolved_count']}/{size} | Fully solved: {results['fully_solved_count']}/{size}")
    print(f"  Elapsed: {results['elapsed_s']:.1f}s")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = f"cross_eval_{agent_name}_s{agent_seed}_on_{buffer_tag}.npz"
        out_path = os.path.join(output_dir, fname)
        np.savez_compressed(out_path, solve_rates=solve_rates, **{
            k: v for k, v in results.items() if k != "solve_rates"
        })
        print(f"  Saved: {out_path}")

    return results


def run_batch_cross_eval(agent_dirs, buffer_npzs, num_attempts, output_dir,
                         agent_step=-1):
    """Run all agent x buffer combinations and produce summary."""
    import csv

    all_results = []
    for agent_dir in agent_dirs:
        for buffer_npz in buffer_npzs:
            print(f"\n{'='*60}")
            results = run_single_cross_eval(
                agent_dir, agent_step, buffer_npz, num_attempts, output_dir
            )
            if results is not None:
                all_results.append(results)

    # Summary table
    print(f"\n{'='*60}")
    print("CROSS-EVALUATION SUMMARY")
    print(f"{'='*60}")
    header = f"{'Agent':>30s} {'Buffer':>20s} {'Solve%':>8s} {'Unsolved':>8s} {'N':>6s}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        label = f"{r['agent_name']}/s{r['agent_seed']}"
        print(f"{label:>30s} {r['buffer_tag']:>20s} {r['mean_solve_rate']:>7.1%} "
              f"{r['unsolved_count']:>8d} {r['num_levels']:>6d}")

    # Save CSV summary
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "cross_eval_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "agent_name", "agent_seed", "agent_step",
                "buffer_npz", "buffer_tag", "buffer_update_num",
                "num_levels", "mean_solve_rate", "median_solve_rate",
                "unsolved_count", "fully_solved_count", "elapsed_s",
            ])
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r[k] for k in writer.fieldnames})
        print(f"\n[Summary] Saved: {csv_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-evaluate agents on buffer levels from different runs"
    )
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: evaluate all agent x buffer combinations")

    # Single mode
    parser.add_argument("--agent_checkpoint_dir", type=str, default=None,
                        help="Path to agent checkpoint dir (has config.json + models/)")
    parser.add_argument("--agent_step", type=int, default=-1,
                        help="Orbax checkpoint step to load (-1 for latest)")
    parser.add_argument("--agent_updates", type=int, default=None,
                        help="Target update count (auto-converts to checkpoint step via eval_freq)")
    parser.add_argument("--buffer_npz", type=str, default=None,
                        help="Path to buffer_dump_{N}k.npz from another run")

    # Batch mode
    parser.add_argument("--agent_dirs", nargs="+", default=[],
                        help="List of agent checkpoint directories")
    parser.add_argument("--buffer_npzs", nargs="+", default=[],
                        help="List of buffer .npz paths to evaluate on")

    # Common
    parser.add_argument("--num_attempts", type=int, default=10,
                        help="Evaluation attempts per level (averaged for solve rate)")
    parser.add_argument("--output_dir", type=str, default="results/cross_eval/",
                        help="Directory to save results")

    args = parser.parse_args()

    # Convert --agent_updates to --agent_step if provided
    agent_step = args.agent_step
    if args.agent_updates is not None:
        # Load eval_freq from the first checkpoint dir's config
        ckpt_dir = args.agent_checkpoint_dir or (args.agent_dirs[0] if args.agent_dirs else None)
        if ckpt_dir:
            with open(os.path.join(ckpt_dir, "config.json")) as f:
                cfg = json.load(f)
            eval_freq = cfg.get("eval_freq", 250)
            # eval_step N means (N+1)*eval_freq updates completed
            agent_step = (args.agent_updates // eval_freq) - 1
            print(f"[Config] --agent_updates={args.agent_updates} -> target eval_step={agent_step} (eval_freq={eval_freq})")

    if args.batch:
        if not args.agent_dirs or not args.buffer_npzs:
            parser.error("--batch requires --agent_dirs and --buffer_npzs")
        run_batch_cross_eval(
            args.agent_dirs, args.buffer_npzs,
            args.num_attempts, args.output_dir, agent_step,
        )
    else:
        if not args.agent_checkpoint_dir or not args.buffer_npz:
            parser.error("Single mode requires --agent_checkpoint_dir and --buffer_npz")
        run_single_cross_eval(
            args.agent_checkpoint_dir, agent_step,
            args.buffer_npz, args.num_attempts, args.output_dir,
        )


if __name__ == "__main__":
    main()
