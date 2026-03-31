"""t-SNE/MDS visualization of buffer embeddings with LLM-generated mazes.

Loads buffer with pre-computed embeddings, rolls out agent on seed/eligible
mazes to compute their embeddings, then plots everything in t-SNE and MDS.

Usage:
  # Buffer-only (no agent rollout needed)
  python vae/tsne_buffer_embeddings.py \
      --buffer /tmp/buffer_dump_emb_final.npz

  # With seeds + eligible mazes (requires agent checkpoint for rollouts)
  python vae/tsne_buffer_embeddings.py \
      --buffer /tmp/buffer_dump_emb_final.npz \
      --seeds /path/to/seeds.npz \
      --eligible /path/to/eligible_pool.npz \
      --agent_checkpoint /path/to/checkpoint_dir
"""
import argparse
import json
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import pairwise_distances

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from vae_level_utils import tokens_to_level


def load_buffer(path):
    d = np.load(path)
    size = int(d["size"])
    embeddings = d["mean_embeddings"][:size]
    scores = d["scores"][:size]
    tokens = d["tokens"][:size]
    print(f"Buffer: {size} levels, embeddings: {embeddings.shape}")
    return embeddings, scores, tokens


def select_kmedoids(dist_matrix, n_medoids, max_iter=100):
    """PAM k-medoids: BUILD (greedy init) + SWAP."""
    n = dist_matrix.shape[0]
    medoids = []
    medoids.append(int(np.argmin(dist_matrix.sum(axis=1))))

    for _ in range(1, n_medoids):
        d_nearest = dist_matrix[:, medoids].min(axis=1)
        best, best_gain = -1, -1
        for c in range(n):
            if c in medoids:
                continue
            gain = np.sum(np.maximum(d_nearest - dist_matrix[:, c], 0))
            if gain > best_gain:
                best_gain = gain
                best = c
        medoids.append(best)

    for _ in range(max_iter):
        total_cost = dist_matrix[:, medoids].min(axis=1).sum()
        improved = False
        for i, m in enumerate(medoids):
            for c in range(n):
                if c in medoids:
                    continue
                new_medoids = medoids.copy()
                new_medoids[i] = c
                new_cost = dist_matrix[:, new_medoids].min(axis=1).sum()
                if new_cost < total_cost - 1e-8:
                    medoids[i] = c
                    total_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return np.array(medoids)


def compute_embeddings_for_tokens(tokens, agent_checkpoint_dir, batch_size=256):
    """Roll out agent on mazes defined by tokens to get (N, 257) embeddings.

    Args:
        tokens: (N, 52) int32 array of maze token encodings
        agent_checkpoint_dir: Path to orbax checkpoint directory
        batch_size: Max levels per rollout batch

    Returns:
        (N, 257) mean LSTM embeddings
    """
    from cross_evaluate import load_agent, tokens_to_levels_batch
    from maze_plr import ActorCritic, sample_trajectories_rnn, compute_insertion_embeddings
    from jaxued.environments import Maze
    from jaxued.wrappers import AutoReplayWrapper

    train_state, config, env, env_params = load_agent(agent_checkpoint_dir)
    eval_env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"],
                    normalize_obs=True)
    max_steps = env_params.max_steps_in_episode

    levels = tokens_to_levels_batch(tokens)
    n_levels = len(tokens)
    all_embeddings = []

    for start in range(0, n_levels, batch_size):
        end = min(start + batch_size, n_levels)
        chunk_levels = jax.tree_util.tree_map(lambda x: x[start:end], levels)
        n_chunk = end - start
        print(f"  Rolling out levels {start}-{end} ({n_chunk})...")

        rng = jax.random.PRNGKey(42)
        rng, rng_reset, rng_eval = jax.random.split(rng, 3)

        init_obs, init_env_state = jax.vmap(
            eval_env.reset_to_level, (0, 0, None)
        )(jax.random.split(rng_reset, n_chunk), chunk_levels, env_params)

        # Wrap in AutoReplayWrapper state format
        wrapped_env = AutoReplayWrapper(eval_env)
        init_obs_w, init_env_state_w = jax.vmap(
            wrapped_env.reset_to_level, (0, 0, None)
        )(jax.random.split(rng_reset, n_chunk), chunk_levels, env_params)

        init_hstate = ActorCritic.initialize_carry((n_chunk,))

        (_, _, _, _, _, _), traj = sample_trajectories_rnn(
            rng_eval, wrapped_env, env_params, train_state,
            init_hstate, init_obs_w, init_env_state_w,
            n_chunk, max_steps,
        )
        # traj = (obs, action, reward, done, log_prob, value, info, agent_pos, hstate_h)
        _, actions, _, dones, _, _, _, _, hstates = traj

        embeddings = compute_insertion_embeddings(hstates, actions, dones)
        all_embeddings.append(np.array(embeddings))

    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, required=True,
                        help="Buffer .npz with mean_embeddings")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Seeds .npz (reference mazes given to LLM)")
    parser.add_argument("--eligible", type=str, default=None,
                        help="Eligible pool .npz (accepted LLM mazes)")
    parser.add_argument("--agent_checkpoint", type=str, default=None,
                        help="Agent checkpoint dir (needed for --seeds/--eligible)")
    parser.add_argument("--n_refs", type=int, default=5,
                        help="Number of k-medoid references (only used if no --seeds)")
    parser.add_argument("--n_buffer", type=int, default=500,
                        help="Number of buffer samples for background (-1 for all)")
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--output", type=str, default="vae/tsne_injection.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Need agent checkpoint if computing new embeddings
    if (args.seeds or args.eligible) and not args.agent_checkpoint:
        parser.error("--agent_checkpoint required when using --seeds or --eligible")

    # Load buffer
    buf_embeddings, buf_scores, buf_tokens = load_buffer(args.buffer)
    n_buf_total = len(buf_embeddings)

    # Filter zero-norm
    norms = np.linalg.norm(buf_embeddings, axis=1)
    valid = norms > 1e-6
    buf_embeddings = buf_embeddings[valid]
    buf_scores = buf_scores[valid]
    buf_tokens = buf_tokens[valid]
    print(f"After filtering zero-norm: {len(buf_embeddings)} buffer levels")

    # Subsample buffer for background
    np.random.seed(args.seed)
    if args.n_buffer == -1:
        buf_sample_idx = np.arange(len(buf_embeddings))
    else:
        n_sample = min(args.n_buffer, len(buf_embeddings))
        buf_sample_idx = np.sort(np.random.choice(len(buf_embeddings), n_sample, replace=False))
    buf_emb_sampled = buf_embeddings[buf_sample_idx]
    buf_scores_sampled = buf_scores[buf_sample_idx]
    n_buf = len(buf_sample_idx)
    print(f"Buffer background: {n_buf} points")

    # Compute embeddings for seeds (references)
    seed_embeddings = None
    n_seeds = 0
    if args.seeds:
        print("\n=== Computing seed (reference) embeddings ===")
        seed_data = np.load(args.seeds)
        seed_tokens = seed_data["tokens"]
        n_seeds = len(seed_tokens)
        print(f"Seed mazes: {n_seeds}")
        seed_embeddings = compute_embeddings_for_tokens(
            seed_tokens, args.agent_checkpoint)
        print(f"Seed embeddings: {seed_embeddings.shape}")
    else:
        # Use k-medoids on buffer as references
        print(f"\nSelecting {args.n_refs} k-medoid references from buffer...")
        full_dist = pairwise_distances(buf_embeddings, metric="euclidean").astype(np.float32)
        medoid_idx = select_kmedoids(full_dist, args.n_refs)
        seed_embeddings = buf_embeddings[medoid_idx]
        n_seeds = len(medoid_idx)
        print(f"Medoid indices: {medoid_idx}, scores: {buf_scores[medoid_idx]}")

    # Compute embeddings for eligible (accepted) mazes
    eligible_embeddings = None
    eligible_scores = None
    n_eligible = 0
    if args.eligible:
        print("\n=== Computing eligible (accepted) maze embeddings ===")
        elig_data = np.load(args.eligible)
        elig_tokens = elig_data["tokens"]
        eligible_scores = elig_data.get("scores", None)
        if eligible_scores is not None:
            eligible_scores = np.array(eligible_scores)
        n_eligible = len(elig_tokens)
        print(f"Eligible mazes: {n_eligible}")
        eligible_embeddings = compute_embeddings_for_tokens(
            elig_tokens, args.agent_checkpoint)
        print(f"Eligible embeddings: {eligible_embeddings.shape}")

    # Build combined matrix: [seeds, eligible, buffer_sample]
    parts = [seed_embeddings]
    if eligible_embeddings is not None:
        parts.append(eligible_embeddings)
    parts.append(buf_emb_sampled)
    all_emb = np.concatenate(parts, axis=0)

    # Pairwise L2
    print(f"\nComputing pairwise distances ({len(all_emb)} points)...")
    dist_matrix = pairwise_distances(all_emb, metric="euclidean").astype(np.float32)

    # t-SNE
    perp = min(args.perplexity, len(all_emb) - 1)
    print(f"Running t-SNE (perplexity={perp})...")
    tsne = TSNE(n_components=2, metric="precomputed", perplexity=perp,
                random_state=args.seed, init="random")
    coords_tsne = tsne.fit_transform(dist_matrix)

    # MDS
    print("Running MDS...")
    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=args.seed, normalized_stress="auto")
    coords_mds = mds.fit_transform(dist_matrix)

    # Score-based coloring for buffer
    buf_start = n_seeds + n_eligible
    buf_sc = buf_scores_sampled
    sc_max = buf_sc.max() if buf_sc.max() > 0 else 1.0
    buf_colors = buf_sc / sc_max

    for name, coords, out_path in [
        ("t-SNE", coords_tsne, args.output),
        ("MDS", coords_mds, args.output.replace(".png", "_mds.png")),
    ]:
        fig, ax = plt.subplots(figsize=(12, 9))

        # Buffer background
        scatter = ax.scatter(
            coords[buf_start:, 0], coords[buf_start:, 1],
            c=buf_colors, cmap="YlOrRd", s=12, alpha=0.4,
            zorder=1, edgecolors="none",
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("SFL score (normalized)", fontsize=9)

        # Eligible (accepted) mazes
        if n_eligible > 0:
            elig_start = n_seeds
            elig_end = n_seeds + n_eligible
            ax.scatter(
                coords[elig_start:elig_end, 0], coords[elig_start:elig_end, 1],
                c="green", s=120, marker="*", edgecolors="black",
                linewidths=0.8, zorder=3, label=f"Accepted mazes ({n_eligible})",
                alpha=0.7,
            )

        # Seeds / references
        ax.scatter(
            coords[:n_seeds, 0], coords[:n_seeds, 1],
            c="blue", s=180, marker="o", edgecolors="black",
            linewidths=1.5, zorder=4, label=f"References ({n_seeds})",
        )
        for i in range(n_seeds):
            ax.annotate(
                f"R{i}", (coords[i, 0], coords[i, 1]),
                fontsize=8, fontweight="bold", ha="center", va="bottom",
                xytext=(0, 8), textcoords="offset points",
            )

        # Dashed lines from eligible to nearest reference
        if n_eligible > 0 and n_seeds > 0:
            for j in range(n_eligible):
                elig_idx = n_seeds + j
                # Find nearest reference
                ref_dists = dist_matrix[elig_idx, :n_seeds]
                nearest_ref = int(np.argmin(ref_dists))
                d_val = ref_dists[nearest_ref]
                # Draw dashed line
                ax.plot(
                    [coords[elig_idx, 0], coords[nearest_ref, 0]],
                    [coords[elig_idx, 1], coords[nearest_ref, 1]],
                    'k--', alpha=0.15, linewidth=0.5, zorder=2,
                )

        ax.set_xlabel(f"{name} dim 1", fontsize=11)
        ax.set_ylabel(f"{name} dim 2", fontsize=11)

        title_parts = [f"Embedding Diversity ({name})"]
        title_parts.append(f"{n_buf} buffer | {n_seeds} refs")
        if n_eligible > 0:
            title_parts[-1] += f" | {n_eligible} accepted"
        ax.set_title("\n".join(title_parts), fontsize=12)
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
