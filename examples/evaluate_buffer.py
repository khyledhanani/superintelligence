"""
Evaluate a trained agent on PLR buffer levels and extract trajectories.

Loads an orbax checkpoint, extracts the PLR buffer levels, runs the agent
on each level, and saves per-level solve rates, paths, and difficulty metrics.
Optionally renders the hardest levels with agent paths overlaid.

Usage:
    python examples/evaluate_buffer.py \
        --checkpoint_dir checkpoints/cmaes_accel/0 \
        --checkpoint_step -1 \
        --num_attempts 10 \
        --output_dir results/buffer_eval/
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

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator
from jaxued.level_sampler import LevelSampler
from jaxued.wrappers import AutoReplayWrapper

# Import shared components from maze_plr
from maze_plr import ActorCritic, TrainState, evaluate_rnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vae'))
from vae_level_utils import level_to_tokens


def load_checkpoint(checkpoint_dir, checkpoint_step, config):
    """Load agent from orbax checkpoint."""
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

    # Create a template train state to get the right pytree structure
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

    # Load checkpoint
    models_dir = os.path.join(checkpoint_dir, "models")
    checkpoint_manager = ocp.CheckpointManager(models_dir)
    step = checkpoint_manager.latest_step() if checkpoint_step == -1 else checkpoint_step
    print(f"[Checkpoint] Loading step {step} from {models_dir}")

    restored = checkpoint_manager.restore(step)

    # Orbax restores Level as a plain dict — convert back to Level namedtuple
    sampler = restored["sampler"]
    if isinstance(sampler["levels"], dict):
        sampler = dict(sampler)
        sampler["levels"] = Level(
            wall_map=sampler["levels"]["wall_map"],
            goal_pos=sampler["levels"]["goal_pos"],
            agent_pos=sampler["levels"]["agent_pos"],
            agent_dir=sampler["levels"]["agent_dir"],
            width=sampler["levels"]["width"],
            height=sampler["levels"]["height"],
        )

    train_state = template_state.replace(
        params=restored["params"],
        sampler=sampler,
    )

    return train_state, env, env_params, network


def evaluate_on_levels(train_state, env, env_params, levels, num_attempts, batch_size=32):
    """Run agent on levels and collect trajectories.

    Returns dict with per-level results.
    """
    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]
    max_steps = env_params.max_steps_in_episode

    all_solve_rates = []
    all_mean_lengths = []
    all_paths = []  # list of (num_levels, max_steps, 2) arrays per attempt

    eval_env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    n_chunks = (num_levels + batch_size - 1) // batch_size

    for attempt_batch_start in range(0, num_attempts, 1):
        t_attempt = time.time()
        rng = jax.random.PRNGKey(attempt_batch_start + 1000)

        # Evaluate in chunks to avoid OOM
        chunk_solve = []
        chunk_lengths = []
        chunk_paths = []

        for chunk_idx, start in enumerate(range(0, num_levels, batch_size)):
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

            # Extract paths (agent positions at each timestep)
            agent_positions = np.asarray(states.agent_pos)  # (max_steps, n_chunk, 2)

            mask = np.arange(max_steps)[:, None] < np.asarray(episode_lengths)[None, :]
            cum_rewards = (np.asarray(rewards) * mask).sum(axis=0)
            solved = (cum_rewards > 0).astype(float)

            chunk_solve.append(solved)
            chunk_lengths.append(np.asarray(episode_lengths))
            chunk_paths.append(agent_positions)

            if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
                print(f"    Attempt {attempt_batch_start+1}/{num_attempts} | chunk {chunk_idx+1}/{n_chunks} | {end}/{num_levels} levels")

        attempt_solve = np.concatenate(chunk_solve)
        attempt_lengths = np.concatenate(chunk_lengths)
        attempt_paths = np.concatenate(chunk_paths, axis=1)  # (max_steps, num_levels, 2)

        all_solve_rates.append(attempt_solve)
        all_mean_lengths.append(attempt_lengths)
        all_paths.append(attempt_paths)
        print(f"  [Attempt {attempt_batch_start+1}/{num_attempts}] solve={attempt_solve.mean():.1%} | {time.time()-t_attempt:.1f}s")

    # Aggregate across attempts
    solve_rates = np.stack(all_solve_rates).mean(axis=0)  # (num_levels,)
    mean_lengths = np.stack(all_mean_lengths).mean(axis=0)  # (num_levels,)

    return {
        "solve_rates": solve_rates,
        "mean_lengths": mean_lengths,
        "paths": all_paths[0],  # paths from first attempt: (max_steps, num_levels, 2)
    }


def render_levels_with_paths(env, env_params, levels, paths, episode_lengths,
                             solve_rates, scores, n_show=16, output_path="buffer_overview.png"):
    """Render the hardest levels with agent paths overlaid."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    renderer = MazeRenderer(env, tile_size=16)

    # Sort by solve rate (hardest first)
    order = np.argsort(solve_rates)
    n_show = min(n_show, len(order))

    ncols = min(4, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    for idx in range(n_show):
        level_idx = order[idx]
        ax = axes[idx // ncols, idx % ncols]

        level = jax.tree_util.tree_map(lambda x: x[level_idx], levels)
        img = np.asarray(renderer.render_level(level, env_params))
        ax.imshow(img)

        # Draw agent path
        path = paths[:, level_idx, :]  # (max_steps, 2) — (x, y) positions
        ep_len = int(episode_lengths[level_idx]) if episode_lengths is not None else paths.shape[0]
        path = path[:ep_len]

        tile_size = 16
        # Convert grid positions to pixel centers (accounting for border)
        px = (path[:, 0].astype(float) + 0.5) * tile_size
        py = (path[:, 1].astype(float) + 0.5) * tile_size

        ax.plot(px, py, 'r-', linewidth=1.5, alpha=0.7)
        if len(px) > 0:
            ax.plot(px[0], py[0], 'go', markersize=6)  # start
            ax.plot(px[-1], py[-1], 'rs', markersize=6)  # end

        score_str = f"{scores[level_idx]:.2f}" if scores is not None else "?"
        ax.set_title(f"Solve: {solve_rates[level_idx]:.0%} | Score: {score_str}", fontsize=9)
        ax.axis("off")

    # Hide empty axes
    for idx in range(n_show, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    plt.suptitle(f"Hardest {n_show} Buffer Levels (sorted by solve rate)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on PLR buffer levels")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint directory (contains config.json and models/)")
    parser.add_argument("--checkpoint_step", type=int, default=-1,
                        help="Checkpoint step to load (-1 for latest)")
    parser.add_argument("--num_attempts", type=int, default=10,
                        help="Number of evaluation attempts per level")
    parser.add_argument("--output_dir", type=str, default="results/buffer_eval/")
    parser.add_argument("--n_render", type=int, default=16,
                        help="Number of hardest levels to render with paths")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    t_total = time.time()

    # Load config
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    print(f"[Config] Loaded from {config_path}")
    print(f"  run_name={config.get('run_name')}, seed={config.get('seed')}, "
          f"use_cmaes={config.get('use_cmaes')}, use_accel={config.get('use_accel')}")

    # Load checkpoint
    t0 = time.time()
    train_state, env, env_params, network = load_checkpoint(
        args.checkpoint_dir, args.checkpoint_step, config
    )
    print(f"[Checkpoint] Loaded in {time.time()-t0:.1f}s")

    # Extract buffer levels
    sampler = train_state.sampler
    size = int(sampler["size"])
    print(f"[Buffer] {size} levels in PLR buffer")

    buffer_levels = jax.tree_util.tree_map(lambda x: x[:size], sampler["levels"])
    buffer_scores = np.asarray(sampler["scores"][:size])

    # Evaluate agent on buffer levels
    t0 = time.time()
    print(f"[Eval] Running {args.num_attempts} attempts on {size} levels (batch_size=32)...")
    results = evaluate_on_levels(train_state, env, env_params, buffer_levels, args.num_attempts)
    print(f"[Eval] Completed in {time.time()-t0:.1f}s")

    # Convert buffer to tokens for downstream use
    t0 = time.time()
    tokens = np.asarray(jax.vmap(level_to_tokens)(buffer_levels))
    print(f"[Tokens] Converted {size} levels to tokens in {time.time()-t0:.1f}s")

    # Save results
    t0 = time.time()
    np.savez_compressed(
        os.path.join(args.output_dir, "buffer_eval.npz"),
        solve_rates=results["solve_rates"],
        mean_lengths=results["mean_lengths"],
        buffer_scores=buffer_scores,
        tokens=tokens,
        paths=results["paths"],
    )
    print(f"[Saved] {os.path.join(args.output_dir, 'buffer_eval.npz')} ({time.time()-t0:.1f}s)")

    # Print summary
    print(f"\n--- Buffer Evaluation Summary ---")
    print(f"  Levels evaluated: {size}")
    print(f"  Mean solve rate: {results['solve_rates'].mean():.1%}")
    print(f"  Median solve rate: {np.median(results['solve_rates']):.1%}")
    print(f"  Unsolved levels (0% solve): {(results['solve_rates'] == 0).sum()}")
    print(f"  Fully solved levels (100%): {(results['solve_rates'] == 1.0).sum()}")
    print(f"  Mean episode length: {results['mean_lengths'].mean():.1f}")

    # Render hardest levels with paths
    if args.n_render > 0:
        t0 = time.time()
        render_levels_with_paths(
            Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True),
            env_params, buffer_levels,
            results["paths"],
            results["mean_lengths"],
            results["solve_rates"],
            buffer_scores,
            n_show=args.n_render,
            output_path=os.path.join(args.output_dir, "hardest_levels.png"),
        )
        print(f"[Render] Completed in {time.time()-t0:.1f}s")

    print(f"\n[Done] Total time: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
