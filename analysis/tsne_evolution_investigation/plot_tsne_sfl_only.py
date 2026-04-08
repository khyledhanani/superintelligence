"""t-SNE evolution — SFL-colored view (no lineage coloring).

Like plot_tsne_evolution.py but all buffer levels (organic + LLM mutations)
are colored by SFL score.  LLM originals still shown as stars, eval levels
still shown as diamonds (cyan=solved, black=unsolved).

Usage:
    python analysis/tsne_evolution_investigation/plot_tsne_sfl_only.py --seed 1 --start 2250

    # Custom grid size
    python analysis/tsne_evolution_investigation/plot_tsne_sfl_only.py \
        --seed 1 --start 2250 --n_checkpoints 10 --cols 5
"""
import argparse
import math
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

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
                        default="analysis/tsne_evolution_investigation/llm_injection_fresh/cache_solved",
                        help="Directory with solved-rollout embedding .npz files")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start", type=int, required=True,
                        help="First timestep (e.g. 2250)")
    parser.add_argument("--n_checkpoints", type=int, default=20,
                        help="Number of consecutive checkpoints to plot")
    parser.add_argument("--interval", type=int, default=250,
                        help="Step interval between checkpoints")
    parser.add_argument("--perplexity", type=int, default=40)
    parser.add_argument("--cols", type=int, default=5,
                        help="Number of columns in the grid")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    timesteps = [args.start + i * args.interval for i in range(args.n_checkpoints)]
    n_panels = len(timesteps)
    n_cols = min(args.cols, n_panels)
    n_rows = math.ceil(n_panels / n_cols)

    if args.output is None:
        args.output = (f"analysis/tsne_evolution_investigation/llm_injection_fresh/"
                       f"tsne_sfl_seed{args.seed}_t{args.start}.png")

    # Load buffer data
    data = {}
    for ts in timesteps:
        path = os.path.join(args.cache_dir, f"emb_solved_s{args.seed}_t{ts}.npz")
        if os.path.exists(path):
            cached = np.load(path)
            n = len(cached["embeddings_solved"])
            data[ts] = {
                "embeddings": cached["embeddings_solved"],
                "origins": cached["origins"] if "origins" in cached else np.zeros(n, dtype=np.int32),
                "scores": cached["scores"] if "scores" in cached else np.zeros(n, dtype=np.float32),
                "solved": cached["solved"],
                "solve_rates": cached["solve_rates"],
            }
            print(f"Loaded t={ts}: {n} levels")
        else:
            print(f"Missing cache for t={ts}: {path}")

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

    # SFL colormap
    sfl_cmap = mcolors.LinearSegmentedColormap.from_list("sfl", ["#e0e0e0", "gold", "red"])
    sfl_norm = mcolors.Normalize(vmin=0, vmax=0.25)

    # Plot grid
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 4 * n_rows),
                              squeeze=False)

    for panel_idx, ts in enumerate(timesteps):
        row = panel_idx // n_cols
        col = panel_idx % n_cols
        ax = axes[row][col]

        if ts not in data:
            ax.text(0.5, 0.5, f"t={ts}\nN/A", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='red')
            ax.set_xticks([]); ax.set_yticks([])
            continue

        emb = data[ts]["embeddings"]
        origins = data[ts]["origins"]
        scores = data[ts]["scores"]

        # Include eval levels in t-SNE
        eval_d = eval_data.get(ts)
        n_eval = len(eval_d["embeddings"]) if eval_d is not None else 0
        if n_eval > 0:
            combined = np.concatenate([emb, eval_d["embeddings"]], axis=0)
        else:
            combined = emb

        effective_perp = min(args.perplexity, len(combined) - 1)
        print(f"  t-SNE: t={ts}, n={len(combined)}, perplexity={effective_perp}...")

        tsne = TSNE(n_components=2, perplexity=effective_perp,
                    random_state=42, max_iter=1000,
                    learning_rate='auto', init='pca')
        all_coords = tsne.fit_transform(combined)
        coords = all_coords[:len(emb)]
        eval_coords = all_coords[len(emb):] if n_eval > 0 else None

        is_original = origins == 1
        is_not_original = ~is_original

        # All non-original buffer levels (organic + mutations) colored by SFL
        if is_not_original.sum() > 0:
            ax.scatter(coords[is_not_original, 0], coords[is_not_original, 1],
                       c=scores[is_not_original], cmap=sfl_cmap, norm=sfl_norm,
                       s=4, alpha=0.4, edgecolors='none',
                       rasterized=True)

        # LLM originals as stars, colored by SFL
        if is_original.sum() > 0:
            ax.scatter(coords[is_original, 0], coords[is_original, 1],
                       c=scores[is_original], cmap=sfl_cmap, norm=sfl_norm,
                       s=50, marker='*', alpha=0.95,
                       edgecolors='black', linewidths=0.4, zorder=8)

        # Eval benchmark levels: cyan diamond if solved, black diamond if unsolved
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
        n_solved_eval = eval_d["solved"].sum() if eval_d is not None else 0
        ax.set_title(f"t={ts}\n({n_llm} LLM / {len(origins)} buf, "
                     f"{n_solved_eval}/8 eval solved)",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # Hide unused panels
    for panel_idx in range(n_panels, n_rows * n_cols):
        row = panel_idx // n_cols
        col = panel_idx % n_cols
        axes[row][col].set_visible(False)

    # SFL colorbar on last visible panel
    last_row = (n_panels - 1) // n_cols
    last_col = (n_panels - 1) % n_cols
    sm = plt.cm.ScalarMappable(cmap=sfl_cmap, norm=sfl_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[last_row, last_col], fraction=0.046, pad=0.04)
    cb.set_label("SFL", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Legend
    legend_els = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e0e0e0',
               markersize=4, alpha=0.5, label='Buffer level (SFL=0)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=4, alpha=0.5, label='Buffer level (SFL=0.25)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
               markersize=8, markeredgecolor='black', markeredgewidth=0.4,
               label='LLM original'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan',
               markersize=6, markeredgecolor='black', markeredgewidth=0.5,
               label='Eval (solved)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
               markersize=6, markeredgecolor='white', markeredgewidth=0.5,
               label='Eval (unsolved)'),
    ]
    fig.legend(handles=legend_els, fontsize=7, loc='lower center',
               framealpha=0.7, ncol=len(legend_els),
               bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(f"t-SNE Evolution — SFL colored (perp={args.perplexity}) — seed {args.seed}\n"
                 f"(257D solved-rollout behavioral embeddings)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
