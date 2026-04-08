"""Compute 257D behavioral embeddings with solved-rollout filtering.

For each level, runs num_rollouts rollouts and returns:
  - mean embedding of SOLVED rollouts only (if any solved)
  - mean embedding of ALL rollouts (fallback if none solved)
  - per-level solved flag (True if at least 1 rollout solved)
  - per-level solve rate

This differs from plot_tsne_training_evolution.compute_embeddings() which
averages ALL rollouts regardless of solve status.

Usage:
    # Compute embeddings for a buffer dump
    python analysis/tsne_perplexity_investigation/compute_embeddings.py \
        --buffer buffer_dumps/llm_injection_fresh/seed1/buffer_dumps/buffer_dump_2500.npz \
        --checkpoint_dir buffer_dumps/llm_injection_fresh/seed1/checkpoints \
        --checkpoint_step 9 \
        --output /tmp/emb_solved_s1_t2500.npz

    # Compute embeddings for eval levels only
    python analysis/tsne_perplexity_investigation/compute_embeddings.py \
        --eval_only \
        --checkpoint_dir buffer_dumps/llm_injection_fresh/seed1/checkpoints \
        --checkpoint_step 9 \
        --output /tmp/eval_solved_s1_t2500.npz
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


EVAL_LEVEL_NAMES = [
    "SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped",
    "Labyrinth2", "StandardMaze", "StandardMaze2", "StandardMaze3",
]


def compute_embeddings_solved(tokens, checkpoint_dir, ckpt_step,
                               batch_size=256, num_rollouts=10):
    """Compute 257D embeddings, averaging only solved rollouts.

    Returns:
        embeddings: (N, 257) mean embedding (solved-only if any solved, else all)
        solved: (N,) bool — True if at least 1 rollout solved
        solve_rates: (N,) float — fraction of rollouts that solved
        embeddings_all: (N, 257) mean embedding of ALL rollouts (for comparison)
    """
    from cross_evaluate import load_agent, tokens_to_levels_batch
    from maze_plr import ActorCritic, sample_trajectories_rnn, compute_insertion_embeddings
    from jaxued.environments import Maze
    from jaxued.wrappers import AutoReplayWrapper
    import jax
    import jax.numpy as jnp

    train_state, config, env, env_params = load_agent(
        checkpoint_dir, checkpoint_step=ckpt_step)
    if train_state is None:
        return None, None, None, None

    eval_env = Maze(max_height=13, max_width=13,
                    agent_view_size=config["agent_view_size"], normalize_obs=True)
    wrapped_env = AutoReplayWrapper(eval_env)
    max_steps = env_params.max_steps_in_episode

    levels = tokens_to_levels_batch(tokens)
    n_levels = len(tokens)

    # Accumulate per-rollout embeddings and solve status
    all_rollout_embs = np.zeros((num_rollouts, n_levels, 257), dtype=np.float32)
    all_rollout_solved = np.zeros((num_rollouts, n_levels), dtype=bool)

    for rollout_idx in range(num_rollouts):
        rollout_embs = []
        rollout_solved = []

        for start in range(0, n_levels, batch_size):
            end = min(start + batch_size, n_levels)
            chunk_levels = jax.tree_util.tree_map(lambda x: x[start:end], levels)
            n_chunk = end - start

            rng = jax.random.PRNGKey(rollout_idx * 1000 + start)
            rng, rng_reset, rng_eval = jax.random.split(rng, 3)

            init_obs, init_state = jax.vmap(
                wrapped_env.reset_to_level, (0, 0, None)
            )(jax.random.split(rng_reset, n_chunk), chunk_levels, env_params)

            init_hstate = ActorCritic.initialize_carry((n_chunk,))

            (_, _, _, _, _, _), traj = sample_trajectories_rnn(
                rng_eval, wrapped_env, env_params, train_state,
                init_hstate, init_obs, init_state,
                n_chunk, max_steps,
            )
            _, actions, rewards, dones, _, _, _, _, hstates = traj

            embeddings = compute_insertion_embeddings(hstates, actions, dones)
            rollout_embs.append(np.array(embeddings))

            # Check if solved: any done with positive reward
            episode_returns = np.array(jnp.sum(rewards * dones, axis=0))
            solved_chunk = episode_returns > 0
            rollout_solved.append(solved_chunk)

        all_rollout_embs[rollout_idx] = np.concatenate(rollout_embs, axis=0)
        all_rollout_solved[rollout_idx] = np.concatenate(rollout_solved, axis=0)

        n_solved = all_rollout_solved[rollout_idx].sum()
        print(f"  Rollout {rollout_idx+1}/{num_rollouts}: "
              f"{n_solved}/{n_levels} solved", flush=True)

    # Compute mean of ALL rollouts
    embeddings_all = all_rollout_embs.mean(axis=0)  # (N, 257)

    # Compute mean of SOLVED rollouts only
    solve_counts = all_rollout_solved.sum(axis=0)  # (N,)
    solved_mask = solve_counts > 0  # (N,) — at least 1 rollout solved
    solve_rates = solve_counts / num_rollouts  # (N,)

    embeddings_solved = np.zeros_like(embeddings_all)
    for i in range(n_levels):
        if solved_mask[i]:
            # Average only the rollouts that solved
            rollout_mask = all_rollout_solved[:, i]
            embeddings_solved[i] = all_rollout_embs[rollout_mask, i].mean(axis=0)
        else:
            # No rollout solved — fall back to all-rollout mean
            embeddings_solved[i] = embeddings_all[i]

    print(f"\nSummary: {solved_mask.sum()}/{n_levels} levels solved by >= 1 rollout")
    print(f"  Mean solve rate: {solve_rates.mean():.3f}")

    return embeddings_solved, solved_mask, solve_rates, embeddings_all


def get_eval_level_tokens():
    """Get token arrays for the 8 eval benchmark levels."""
    from jaxued.environments.maze.level import prefabs, Level
    from vae.vae_level_utils import level_to_tokens
    tokens_list = []
    for name in EVAL_LEVEL_NAMES:
        level = Level.from_str(prefabs[name])
        tok = np.asarray(level_to_tokens(level))
        tokens_list.append(tok)
    return np.stack(tokens_list)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--buffer", type=str, default=None,
                        help="Buffer dump .npz path")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Agent checkpoint directory (must contain config.json + models/)")
    parser.add_argument("--checkpoint_step", type=int, default=-1,
                        help="Checkpoint step (-1 for latest)")
    parser.add_argument("--num_rollouts", type=int, default=10,
                        help="Number of rollouts per level")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output", type=str, required=True,
                        help="Output .npz path")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only compute embeddings for 8 eval benchmark levels")
    args = parser.parse_args()

    if args.eval_only:
        print("Computing eval level embeddings...")
        tokens = get_eval_level_tokens()
    elif args.buffer:
        print(f"Loading buffer from {args.buffer}...")
        buf = np.load(args.buffer, allow_pickle=True)
        size = int(buf["size"])
        tokens = buf["tokens"][:size]
        print(f"  {size} levels")
    else:
        parser.error("Must provide --buffer or --eval_only")

    ckpt_dir = os.path.abspath(args.checkpoint_dir)
    print(f"Checkpoint: {ckpt_dir}, step={args.checkpoint_step}")
    print(f"Rollouts: {args.num_rollouts}")

    emb_solved, solved, solve_rates, emb_all = compute_embeddings_solved(
        tokens, ckpt_dir, args.checkpoint_step,
        batch_size=args.batch_size, num_rollouts=args.num_rollouts)

    if emb_solved is None:
        print("Failed to load agent.")
        return

    save_data = {
        "embeddings_solved": emb_solved,
        "embeddings_all": emb_all,
        "solved": solved,
        "solve_rates": solve_rates,
    }

    # Pass through buffer metadata if available
    if args.buffer and not args.eval_only:
        buf = np.load(args.buffer, allow_pickle=True)
        size = int(buf["size"])
        for key in ["tokens", "scores", "origins", "ancestor_ids"]:
            if key in buf:
                save_data[key] = buf[key][:size]

    if args.eval_only:
        save_data["level_names"] = np.array(EVAL_LEVEL_NAMES)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **save_data)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
