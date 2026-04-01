"""Visualize initial buffer embeddings: LLM-lineage vs ACCEL-organic levels.

Computes 257D embeddings (LSTM hidden + mean action) for all merged buffer
levels using the initial agent checkpoint, then plots a grid:
  - Rows: seeds (0, 1, 2)
  - Columns: injection percentages (5%, 10%, 15%, 20%, 25%)

Colors: grey = organic (ACCEL), green = LLM mutation, blue = LLM original.
Produces both a PCA-2D grid and a t-SNE-2D grid.

Usage:
    python vae/plot_embedding_evolution.py \
        --agent_checkpoint gcs_artifacts/agent/39
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gcs_artifacts", "plot_data",
)

SEEDS = [0, 1, 2]
PCTS = ["5pct", "10pct", "15pct", "20pct", "25pct"]
PCT_LABELS = ["5%", "10%", "15%", "20%", "25%"]


def load_merged_buffer(seed, pct, data_root=None):
    """Load pre-training merged buffer (tokens + origins, no embeddings)."""
    root = data_root or DEFAULT_DATA_ROOT
    path = os.path.join(root, f"llm_inject_seed{seed}", f"merged_buffer_{pct}.npz")
    d = np.load(path)
    size = int(d["size"]) if "size" in d else len(d["tokens"])
    return {
        "tokens": d["tokens"][:size],
        "origins": d["origins"][:size],
        "scores": d["scores"][:size],
        "size": size,
    }


def compute_embeddings_initial_agent(tokens, agent_checkpoint_dir, batch_size=256):
    """Compute 257D embeddings using the initial agent checkpoint."""
    from cross_evaluate import load_agent, tokens_to_levels_batch
    from maze_plr import ActorCritic, sample_trajectories_rnn, compute_insertion_embeddings
    from jaxued.environments import Maze
    from jaxued.wrappers import AutoReplayWrapper
    import jax
    import jax.numpy as jnp

    agent_checkpoint_dir = os.path.abspath(agent_checkpoint_dir)
    train_state, config, env, env_params = load_agent(agent_checkpoint_dir)
    eval_env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"],
                    normalize_obs=True)
    wrapped_env = AutoReplayWrapper(eval_env)
    max_steps = env_params.max_steps_in_episode

    levels = tokens_to_levels_batch(tokens)
    n_levels = len(tokens)
    all_embeddings = []

    for start in range(0, n_levels, batch_size):
        end = min(start + batch_size, n_levels)
        chunk_levels = jax.tree_util.tree_map(lambda x: x[start:end], levels)
        n_chunk = end - start
        print(f"    Embedding levels {start}-{end} / {n_levels}...")

        rng = jax.random.PRNGKey(42)
        rng, rng_reset, rng_eval = jax.random.split(rng, 3)

        init_obs_w, init_env_state_w = jax.vmap(
            wrapped_env.reset_to_level, (0, 0, None)
        )(jax.random.split(rng_reset, n_chunk), chunk_levels, env_params)

        init_hstate = ActorCritic.initialize_carry((n_chunk,))

        (_, _, _, _, _, _), traj = sample_trajectories_rnn(
            rng_eval, wrapped_env, env_params, train_state,
            init_hstate, init_obs_w, init_env_state_w,
            n_chunk, max_steps,
        )
        _, actions, _, dones, _, _, _, _, hstates = traj
        embeddings = compute_insertion_embeddings(hstates, actions, dones)
        all_embeddings.append(np.array(embeddings))

    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_checkpoint", type=str,
                        default="/cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued/gcs_artifacts/agent/39")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--pcts", type=str, default="5pct,10pct,15pct,20pct,25pct")
    parser.add_argument("--tsne_perplexity", type=float, default=40)
    parser.add_argument("--output_dir", type=str, default="vae/plots/embedding_evolution")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root dir with llm_inject_seed{s}/ layout (default: gcs_artifacts/plot_data)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache computed embeddings to avoid recomputation")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    pcts = args.pcts.split(",")
    pct_labels = [p.replace("pct", "%") for p in pcts]
    data_root = args.data_root

    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    n_rows = len(seeds)
    n_cols = len(pcts)

    # --- Step 1: Load all buffers and compute embeddings ---
    print("=== Step 1: Loading buffers and computing embeddings ===")
    data = {}  # (seed, pct) -> {embeddings, origins, scores}

    for seed in seeds:
        for pct in pcts:
            key = (seed, pct)
            print(f"\n  Seed {seed}, {pct}:")

            # Check cache first
            cache_path = None
            if args.cache_dir:
                cache_path = os.path.join(args.cache_dir, f"emb_s{seed}_{pct}.npz")
                if os.path.exists(cache_path):
                    print(f"    Loading from cache: {cache_path}")
                    cached = np.load(cache_path)
                    data[key] = {
                        "embeddings": cached["embeddings"],
                        "origins": cached["origins"],
                        "scores": cached["scores"],
                    }
                    continue

            buf = load_merged_buffer(seed, pct, data_root=data_root)
            n_org = (buf["origins"] == 0).sum()
            n_orig = (buf["origins"] == 1).sum()
            n_mut = (buf["origins"] == 2).sum()
            print(f"    {buf['size']} levels: organic={n_org}, LLM orig={n_orig}, LLM mut={n_mut}")

            embeddings = compute_embeddings_initial_agent(
                buf["tokens"], args.agent_checkpoint)

            data[key] = {
                "embeddings": embeddings,
                "origins": buf["origins"],
                "scores": buf["scores"],
            }

            # Save cache
            if cache_path:
                np.savez_compressed(cache_path,
                                    embeddings=embeddings,
                                    origins=buf["origins"],
                                    scores=buf["scores"])
                print(f"    Cached to {cache_path}")

    # --- Step 2: Fit a single PCA on all data combined ---
    print("\n=== Step 2: Fitting global PCA ===")
    all_emb = np.concatenate([data[k]["embeddings"] for k in data], axis=0)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_emb)
    ev = pca.explained_variance_ratio_
    print(f"  PCA variance explained: PC1={ev[0]:.1%}, PC2={ev[1]:.1%}, total={ev[:2].sum():.1%}")

    # --- Step 3: PCA grid plot ---
    print("\n=== Step 3: PCA grid ===")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                              squeeze=False)

    for i, seed in enumerate(seeds):
        for j, pct in enumerate(pcts):
            ax = axes[i][j]
            d = data[(seed, pct)]
            coords = pca.transform(d["embeddings"])
            origins = d["origins"]

            is_organic = origins == 0
            is_original = origins == 1
            is_mutation = origins == 2

            # Organic background
            ax.scatter(coords[is_organic, 0], coords[is_organic, 1],
                       c='lightgrey', s=4, alpha=0.25, edgecolors='none',
                       rasterized=True)
            # LLM mutations
            if is_mutation.sum() > 0:
                ax.scatter(coords[is_mutation, 0], coords[is_mutation, 1],
                           c='green', s=10, alpha=0.5, edgecolors='none',
                           rasterized=True)
            # LLM originals
            if is_original.sum() > 0:
                ax.scatter(coords[is_original, 0], coords[is_original, 1],
                           c='blue', s=40, marker='*', alpha=0.9,
                           edgecolors='black', linewidths=0.3)

            n_mut = is_mutation.sum()
            n_orig = is_original.sum()
            ax.set_title(f"Seed {seed}, {pct_labels[j]}\n"
                         f"({n_orig} orig, {n_mut} mut)", fontsize=10)

            if i == n_rows - 1:
                ax.set_xlabel(f'PC1 ({ev[0]:.1%})', fontsize=9)
            if j == 0:
                ax.set_ylabel(f'PC2 ({ev[1]:.1%})', fontsize=9)
            ax.tick_params(labelsize=7)

    # Add shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey',
               markersize=6, label='Organic (ACCEL)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=8, label='LLM mutation'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='blue',
               markersize=12, markeredgecolor='black', label='LLM original'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Initial Buffer Embeddings (PCA 2D) — Initial Agent Checkpoint',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    pca_path = os.path.join(args.output_dir, "grid_pca2d_initial.png")
    fig.savefig(pca_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {pca_path}")
    plt.close(fig)

    # --- Step 4: t-SNE grid plot ---
    print("\n=== Step 4: t-SNE grid ===")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                              squeeze=False)

    for i, seed in enumerate(seeds):
        for j, pct in enumerate(pcts):
            ax = axes[i][j]
            d = data[(seed, pct)]
            origins = d["origins"]

            is_organic = origins == 0
            is_original = origins == 1
            is_mutation = origins == 2

            print(f"  t-SNE: seed {seed}, {pct} ({len(d['embeddings'])} pts)...")
            perp = min(args.tsne_perplexity, len(d["embeddings"]) - 1)
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                        init='pca', learning_rate='auto')
            coords = tsne.fit_transform(d["embeddings"])

            ax.scatter(coords[is_organic, 0], coords[is_organic, 1],
                       c='lightgrey', s=4, alpha=0.25, edgecolors='none',
                       rasterized=True)
            if is_mutation.sum() > 0:
                ax.scatter(coords[is_mutation, 0], coords[is_mutation, 1],
                           c='green', s=10, alpha=0.5, edgecolors='none',
                           rasterized=True)
            if is_original.sum() > 0:
                ax.scatter(coords[is_original, 0], coords[is_original, 1],
                           c='blue', s=40, marker='*', alpha=0.9,
                           edgecolors='black', linewidths=0.3)

            n_mut = is_mutation.sum()
            n_orig = is_original.sum()
            ax.set_title(f"Seed {seed}, {pct_labels[j]}\n"
                         f"({n_orig} orig, {n_mut} mut)", fontsize=10)
            ax.tick_params(labelsize=7)

    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Initial Buffer Embeddings (t-SNE 2D) — Initial Agent Checkpoint',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    tsne_path = os.path.join(args.output_dir, "grid_tsne2d_initial.png")
    fig.savefig(tsne_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {tsne_path}")
    plt.close(fig)

    print(f"\nDone. Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
