"""t-SNE evolution colored by SFL (staleness-weighted regret score).

All buffer levels colored by SFL on a continuous colormap.
LLM originals shown as stars, eval levels as diamonds.

Usage:
    python analysis/final_results_plotting/plot_tsne_sfl.py \
        --seed 3 --timesteps 2250 2500 3000 4000 5500 8000
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
from sklearn.manifold import TSNE
from adjustText import adjust_text

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
                        default="analysis/final_results_plotting/cache_solved")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--timesteps", type=int, nargs='+', required=True)
    parser.add_argument("--perplexity", type=int, default=40)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--sfl_max", type=float, default=None,
                        help="Max SFL for colorbar (auto if not set)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    timesteps = args.timesteps
    n_panels = len(timesteps)
    n_cols = min(args.cols, n_panels)
    n_rows = math.ceil(n_panels / n_cols)

    if args.output is None:
        args.output = (f"analysis/final_results_plotting/tsne_sfl/"
                       f"tsne_sfl_seed{args.seed}.png")

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
            print(f"Loaded t={ts}: {n} levels, "
                  f"SFL range=[{data[ts]['scores'].min():.3f}, {data[ts]['scores'].max():.3f}]")
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

    # Determine SFL colorbar range
    if args.sfl_max is not None:
        sfl_vmax = args.sfl_max
    else:
        all_scores = np.concatenate([d["scores"] for d in data.values()])
        sfl_vmax = np.percentile(all_scores, 95)
    sfl_cmap = mcolors.LinearSegmentedColormap.from_list("sfl", ["#e0e0e0", "gold", "red"])
    sfl_norm = mcolors.Normalize(vmin=0, vmax=sfl_vmax)

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
        print(f"  t-SNE: t={ts}, perplexity={effective_perp}...")

        tsne = TSNE(n_components=2, perplexity=effective_perp,
                    random_state=42, max_iter=1000,
                    learning_rate='auto', init='pca')
        all_coords = tsne.fit_transform(combined)
        coords = all_coords[:len(emb)]
        eval_coords = all_coords[len(emb):] if n_eval > 0 else None

        is_original = origins == 1
        is_not_original = ~is_original

        # All non-original buffer levels colored by SFL
        if is_not_original.sum() > 0:
            ax.scatter(coords[is_not_original, 0], coords[is_not_original, 1],
                       c=scores[is_not_original], cmap=sfl_cmap, norm=sfl_norm,
                       s=6, alpha=0.6, edgecolors='none', rasterized=True)

        # LLM originals as stars, also colored by SFL
        if is_original.sum() > 0:
            ax.scatter(coords[is_original, 0], coords[is_original, 1],
                       c=scores[is_original], cmap=sfl_cmap, norm=sfl_norm,
                       s=60, marker='*', alpha=0.95,
                       edgecolors='black', linewidths=0.4, zorder=8)

        # Eval benchmark levels
        eval_texts = []
        if eval_coords is not None and eval_d is not None:
            eval_solved = eval_d["solved"]
            for ei, name in enumerate(EVAL_LEVEL_NAMES[:n_eval]):
                color = 'cyan' if eval_solved[ei] else 'black'
                edge = 'black' if eval_solved[ei] else 'white'
                ax.scatter(eval_coords[ei, 0], eval_coords[ei, 1],
                           c=color, s=30, marker='D', alpha=0.9,
                           edgecolors=edge, linewidths=0.5, zorder=10)
                eval_texts.append(ax.text(
                    eval_coords[ei, 0], eval_coords[ei, 1],
                    EVAL_LEVEL_SHORT.get(name, name),
                    fontsize=5, fontweight='bold'))
            if eval_texts:
                adjust_text(eval_texts, ax=ax,
                            arrowprops=dict(arrowstyle='-', color='grey',
                                            lw=0.5, alpha=0.5))

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

    # SFL colorbar
    sm = plt.cm.ScalarMappable(cmap=sfl_cmap, norm=sfl_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("SFL (regret score)", fontsize=10)

    # Legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='grey',
               markersize=10, markeredgecolor='black', markeredgewidth=0.4,
               label='LLM original'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan',
               markersize=7, markeredgecolor='black', markeredgewidth=0.5,
               label='Eval (solved)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
               markersize=7, markeredgecolor='white', markeredgewidth=0.5,
               label='Eval (unsolved)'),
    ]
    fig.legend(handles=legend_els, fontsize=9, loc='lower center',
               framealpha=0.8, ncol=3,
               bbox_to_anchor=(0.45, -0.02), columnspacing=1.5,
               handletextpad=0.5)

    plt.suptitle(f"t-SNE Evolution (SFL) — seed {args.seed}",
                 fontsize=14, y=1.01)
    fig.subplots_adjust(right=0.90)
    plt.tight_layout(rect=[0, 0, 0.90, 1])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
