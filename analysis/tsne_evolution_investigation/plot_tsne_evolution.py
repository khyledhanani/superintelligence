"""t-SNE evolution across training timesteps (perplexity=40).

Plots buffer embeddings at 20 consecutive checkpoints (250-step intervals)
starting from --start, using solved-rollout-only embeddings:
  - Buffer/LLM levels: mean embedding of solved rollouts (fallback to all if unsolved)
  - Eval levels: cyan diamond if >=1 rollout solved, black diamond if unsolved

Grid: 4 rows x 5 cols = 20 panels.

Usage:
    python analysis/tsne_evolution_investigation/plot_tsne_evolution.py --seed 1 --start 2500
    python analysis/tsne_evolution_investigation/plot_tsne_evolution.py --seed 2 --start 10000
    python analysis/tsne_evolution_investigation/plot_tsne_evolution.py \
        --seed 1 --start 2500 --n_checkpoints 10 --cols 5
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start", type=int, required=True,
                        help="First timestep (e.g. 2500)")
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
        args.output = (f"analysis/tsne_evolution_investigation/"
                       f"tsne_evolution_seed{args.seed}_t{args.start}.png")

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
                "ancestor_ids": cached["ancestor_ids"] if "ancestor_ids" in cached else np.full(n, -1, dtype=np.int32),
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

    # Determine which ancestor_ids are NEW at start+interval (the injection point)
    inject_ts = args.start + args.interval
    pre_ancs = set()
    if args.start in data:
        pre_mask = data[args.start]["origins"] > 0
        if pre_mask.any():
            pre_ancs = set(data[args.start]["ancestor_ids"][pre_mask].tolist())
    pre_ancs.discard(-1)

    post_ancs = set()
    if inject_ts in data:
        post_mask = data[inject_ts]["origins"] > 0
        if post_mask.any():
            post_ancs = set(data[inject_ts]["ancestor_ids"][post_mask].tolist())
    post_ancs.discard(-1)

    new_ancs = post_ancs - pre_ancs
    print(f"Injection at t={inject_ts}: {len(new_ancs)} new lineages: {sorted(new_ancs)}")

    # Build lineage color map ONLY for new lineages
    sorted_anc = sorted(new_ancs)
    cmap_obj = plt.get_cmap('tab10')
    lineage_colors = {aid: cmap_obj(i % 10) for i, aid in enumerate(sorted_anc)}

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
        ancestor_ids = data[ts]["ancestor_ids"]
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

        is_mutation = origins == 2
        is_original = origins == 1

        # Identify levels belonging to NEW lineages from the injection event
        is_new_lineage = np.zeros(len(origins), dtype=bool)
        for aid in new_ancs:
            is_new_lineage |= (ancestor_ids == aid)
        is_new_llm = is_new_lineage & (origins > 0)

        # Grey background: everything NOT in the new lineages
        is_grey = ~is_new_llm
        if is_grey.sum() > 0:
            ax.scatter(coords[is_grey, 0], coords[is_grey, 1],
                       c='#d0d0d0', s=4, alpha=0.3, edgecolors='none',
                       rasterized=True)

        # New LLM levels by lineage (colored)
        if is_new_llm.sum() > 0:
            for aid in sorted(new_ancs):
                color = lineage_colors[aid]
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

        n_new_llm = is_new_llm.sum()
        n_solved_eval = eval_d["solved"].sum() if eval_d is not None else 0
        ax.set_title(f"t={ts}\n({n_new_llm} tracked / {len(origins)} buf, "
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
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d0d0d0',
               markersize=4, alpha=0.5, label='Buffer (grey)'),
    ]
    for aid in sorted(lineage_colors.keys()):
        c = lineage_colors[aid]
        legend_els.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                   markersize=5, label=f'Lineage {aid}'))
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
               framealpha=0.7, ncol=min(len(legend_els), 8),
               bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(f"t-SNE Evolution (perp={args.perplexity}) — seed{args.seed}, "
                 f"llm_injection_fresh\n(257D solved-rollout behavioral embeddings)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
