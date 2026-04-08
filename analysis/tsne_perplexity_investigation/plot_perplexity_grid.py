"""t-SNE perplexity comparison grid.

Plots buffer embeddings at multiple timesteps and perplexity values.
Uses solved-rollout-only embeddings from compute_embeddings.py:
  - Buffer/LLM levels: mean embedding of solved rollouts (fallback to all if unsolved)
  - Eval levels: cyan diamond if >=1 rollout solved, black diamond if unsolved

Grid: rows = perplexity values, columns = timesteps.

Usage:
    python analysis/tsne_perplexity_investigation/plot_perplexity_grid.py
    python analysis/tsne_perplexity_investigation/plot_perplexity_grid.py --seed 2
    python analysis/tsne_perplexity_investigation/plot_perplexity_grid.py \
        --timesteps 9750,10000,10250 --seed 1 \
        --output analysis/tsne_perplexity_investigation/tsne_perplexity_grid_seed1_10k.png
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

EVAL_LEVEL_NAMES = [
    "SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped",
    "Labyrinth2", "StandardMaze", "StandardMaze2", "StandardMaze3",
]
EVAL_LEVEL_SHORT = {
    "SixteenRooms": "16R", "SixteenRooms2": "16R2",
    "Labyrinth": "Lab", "LabyrinthFlipped": "LabF",
    "Labyrinth2": "Lab2", "StandardMaze": "SM",
    "StandardMaze2": "SM2", "StandardMaze3": "SM3",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", type=str,
                        default="analysis/tsne_perplexity_investigation/cache_solved",
                        help="Directory with solved-rollout embedding .npz files")
    parser.add_argument("--fallback_cache_dir", type=str,
                        default="/tmp/tsne_cache_fresh_behav",
                        help="Fallback to old-style cache if solved cache missing")
    parser.add_argument("--inject_pct", type=str, default="fresh")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=str, default="2250,2500,2750")
    parser.add_argument("--perplexities", type=str, default="5,20,40,100,400")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    timesteps = [int(t) for t in args.timesteps.split(",")]
    perplexities = [int(p) for p in args.perplexities.split(",")]

    if args.output is None:
        args.output = f"analysis/tsne_perplexity_investigation/tsne_perplexity_grid_seed{args.seed}.png"

    n_rows = len(perplexities)
    n_cols = len(timesteps)

    # Load buffer data (prefer solved-rollout cache, fallback to old cache)
    data = {}
    for ts in timesteps:
        # Try solved cache first
        solved_path = os.path.join(args.cache_dir, f"emb_solved_s{args.seed}_t{ts}.npz")
        old_path = os.path.join(args.fallback_cache_dir,
                                f"emb_{args.inject_pct}_s{args.seed}_t{ts}.npz")

        if os.path.exists(solved_path):
            cached = np.load(solved_path)
            data[ts] = {
                "embeddings": cached["embeddings_solved"],
                "origins": cached["origins"] if "origins" in cached else np.zeros(len(cached["embeddings_solved"]), dtype=np.int32),
                "scores": cached["scores"] if "scores" in cached else np.zeros(len(cached["embeddings_solved"]), dtype=np.float32),
                "ancestor_ids": cached["ancestor_ids"] if "ancestor_ids" in cached else np.full(len(cached["embeddings_solved"]), -1, dtype=np.int32),
                "solved": cached["solved"],
                "solve_rates": cached["solve_rates"],
            }
            print(f"Loaded t={ts}: {len(data[ts]['embeddings'])} levels (solved-rollout embeddings)")
        elif os.path.exists(old_path):
            cached = np.load(old_path)
            n = len(cached["embeddings"])
            data[ts] = {
                "embeddings": cached["embeddings"],
                "origins": cached["origins"],
                "scores": cached["scores"] if "scores" in cached else np.zeros(n, dtype=np.float32),
                "ancestor_ids": cached["ancestor_ids"] if "ancestor_ids" in cached else np.full(n, -1, dtype=np.int32),
                "solved": np.ones(n, dtype=bool),  # assume all solved for old cache
                "solve_rates": np.ones(n, dtype=np.float32),
            }
            print(f"Loaded t={ts}: {n} levels (FALLBACK: old all-rollout embeddings)")
        else:
            print(f"Missing cache for t={ts}")

    if not data:
        print("No data loaded.")
        return

    # Load eval level data per timestep
    eval_data = {}
    for ts in timesteps:
        eval_path = os.path.join(args.cache_dir, f"eval_solved_s{args.seed}_t{ts}.npz")
        if os.path.exists(eval_path):
            ed = np.load(eval_path, allow_pickle=True)
            eval_data[ts] = {
                "embeddings": ed["embeddings_solved"],
                "solved": ed["solved"],
                "solve_rates": ed["solve_rates"],
            }
            n_solved = ed["solved"].sum()
            print(f"  Eval levels t={ts}: {n_solved}/{len(ed['solved'])} solved")

    # Build lineage color map
    all_anc = set()
    for d in data.values():
        llm_mask = d["origins"] > 0
        if llm_mask.any():
            all_anc.update(d["ancestor_ids"][llm_mask].tolist())
    all_anc.discard(-1)
    sorted_anc = sorted(all_anc)
    cmap_obj = plt.get_cmap('tab10')
    lineage_colors = {aid: cmap_obj(i % 10) for i, aid in enumerate(sorted_anc)}
    lineage_colors[-1] = (0.5, 0.5, 0.5, 1.0)

    # SFL colormap
    sfl_cmap = mcolors.LinearSegmentedColormap.from_list("sfl", ["#e0e0e0", "gold", "red"])
    sfl_norm = mcolors.Normalize(vmin=0, vmax=0.25)

    # Plot grid
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 4 * n_rows),
                              squeeze=False)

    for i, perp in enumerate(perplexities):
        for j, ts in enumerate(timesteps):
            ax = axes[i][j]

            if ts not in data:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='red')
                ax.set_title(f"t={ts}, perp={perp}", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            emb = data[ts]["embeddings"]
            origins = data[ts]["origins"]
            ancestor_ids = data[ts]["ancestor_ids"]
            scores = data[ts]["scores"]

            # Include eval levels in t-SNE
            eval_d = eval_data.get(ts)
            n_eval = len(eval_d["embeddings"]) if eval_d is not None else 0
            if n_eval > 0:
                combined = np.concatenate([emb, eval_d["embeddings"]], axis=0)
            else:
                combined = emb

            effective_perp = min(perp, len(combined) - 1)
            print(f"  t-SNE: t={ts}, perplexity={effective_perp}...")

            tsne = TSNE(n_components=2, perplexity=effective_perp,
                        random_state=42, max_iter=1000,
                        learning_rate='auto', init='pca')
            all_coords = tsne.fit_transform(combined)
            coords = all_coords[:len(emb)]
            eval_coords = all_coords[len(emb):] if n_eval > 0 else None

            is_organic = origins == 0
            is_mutation = origins == 2
            is_original = origins == 1

            # Organic background colored by SFL
            if is_organic.sum() > 0:
                ax.scatter(coords[is_organic, 0], coords[is_organic, 1],
                           c=scores[is_organic], cmap=sfl_cmap, norm=sfl_norm,
                           s=4, alpha=0.4, edgecolors='none',
                           rasterized=True)

            # LLM levels by lineage
            llm_mask = origins > 0
            if llm_mask.sum() > 0:
                for aid in sorted(set(ancestor_ids[llm_mask].tolist())):
                    color = lineage_colors.get(aid, (0.5, 0.5, 0.5, 1.0))
                    mut_mask = (ancestor_ids == aid) & is_mutation
                    if mut_mask.sum() > 0:
                        ax.scatter(coords[mut_mask, 0], coords[mut_mask, 1],
                                   c=[color], s=8, alpha=0.6, edgecolors='none',
                                   rasterized=True)
                    orig_mask = (ancestor_ids == aid) & is_original
                    if orig_mask.sum() > 0:
                        ax.scatter(coords[orig_mask, 0], coords[orig_mask, 1],
                                   c=[color], s=50, marker='*', alpha=0.95,
                                   edgecolors='black', linewidths=0.4, zorder=8)

            # Eval benchmark levels: cyan if solved, black if unsolved
            if eval_coords is not None and eval_d is not None:
                eval_solved = eval_d["solved"]
                for ei, name in enumerate(EVAL_LEVEL_NAMES[:n_eval]):
                    color = 'cyan' if eval_solved[ei] else 'black'
                    edge = 'black' if eval_solved[ei] else 'white'
                    ax.scatter(eval_coords[ei, 0], eval_coords[ei, 1],
                               c=color, s=30, marker='D', alpha=0.9,
                               edgecolors=edge, linewidths=0.5, zorder=10)
                    ax.annotate(EVAL_LEVEL_SHORT.get(name, name),
                                (eval_coords[ei, 0], eval_coords[ei, 1]),
                                fontsize=5, ha='center', va='bottom',
                                xytext=(0, 4), textcoords='offset points',
                                fontweight='bold')

            n_llm = (origins > 0).sum()
            ax.set_title(f"t={ts}, perp={perp}\n({n_llm} LLM / {len(origins)} total)",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    # SFL colorbar on last panel
    sm = plt.cm.ScalarMappable(cmap=sfl_cmap, norm=sfl_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[-1, -1], fraction=0.046, pad=0.04)
    cb.set_label("SFL", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e0e0e0',
               markersize=4, alpha=0.5, label='Organic (SFL=0)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=4, alpha=0.5, label='Organic (SFL=0.25)'),
    ]
    for aid in sorted(lineage_colors.keys()):
        c = lineage_colors[aid]
        lbl = 'Unknown anc.' if aid == -1 else f'Seed {aid}'
        legend_els.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                   markersize=5, label=lbl))
    legend_els.append(
        Line2D([0], [0], marker='*', color='w', markerfacecolor='grey',
               markersize=8, markeredgecolor='black', markeredgewidth=0.4,
               label='LLM original'))
    legend_els.append(
        Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan',
               markersize=6, markeredgecolor='black', markeredgewidth=0.5,
               label='Eval (solved)'))
    legend_els.append(
        Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
               markersize=6, markeredgecolor='white', markeredgewidth=0.5,
               label='Eval (unsolved)'))
    fig.legend(handles=legend_els, fontsize=7, loc='lower center',
               framealpha=0.7, ncol=len(legend_els),
               bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(f"t-SNE Perplexity Comparison — seed{args.seed}, llm_injection_fresh\n"
                 f"(257D solved-rollout behavioral embeddings)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
