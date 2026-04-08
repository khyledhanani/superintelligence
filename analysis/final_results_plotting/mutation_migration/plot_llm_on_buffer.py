"""Interactive t-SNE: LLM mutations on pre-injection buffer, slider for flip count.

Fits t-SNE on buffer + eval + LLM originals only. Mutations are projected
into the t-SNE space via k-NN weighted interpolation of their high-D
neighbours' t-SNE coordinates.

Usage:
    python analysis/final_results_plotting/mutation_migration/plot_llm_on_buffer.py --seed 3
    python analysis/final_results_plotting/mutation_migration/plot_llm_on_buffer.py --seed 3 --static
"""
import argparse
import math
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from adjustText import adjust_text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

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


def project_via_knn(ref_emb_hd, ref_coords_2d, query_emb_hd, k=10):
    """Project query points into t-SNE space via k-NN weighted interpolation."""
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(ref_emb_hd)
    dists, inds = nn.kneighbors(query_emb_hd)
    # Inverse-distance weighting (add small epsilon to avoid div by zero)
    weights = 1.0 / (dists + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    # Weighted average of reference t-SNE coords
    projected = np.zeros((len(query_emb_hd), 2))
    for i in range(len(query_emb_hd)):
        projected[i] = (weights[i, :, None] * ref_coords_2d[inds[i]]).sum(axis=0)
    return projected


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--buffer_cache_dir", type=str,
                        default="analysis/final_results_plotting/cache_solved")
    parser.add_argument("--buffer_timestep", type=int, default=2250,
                        help="Buffer timestep to use (default: 2250)")
    parser.add_argument("--mutations_cache", type=str,
                        default="analysis/final_results_plotting/mutation_migration/test_cache/mutations_embeddings.npz")
    parser.add_argument("--llm_orig_cache", type=str,
                        default="analysis/final_results_plotting/mutation_migration/test_cache/emb_llm_originals_t2250.npz")
    parser.add_argument("--perplexity", type=int, default=40)
    parser.add_argument("--knn_k", type=int, default=10,
                        help="k for k-NN projection of mutations")
    parser.add_argument("--tsne_cache", type=str,
                        default="analysis/final_results_plotting/mutation_migration/test_cache/tsne_coords_knn.npz")
    parser.add_argument("--static", action="store_true",
                        help="Save a static PNG instead of interactive plot")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"analysis/final_results_plotting/mutation_migration/llm_on_buffer_seed{args.seed}.png"

    # Load buffer
    bt = args.buffer_timestep
    buf_path = os.path.join(args.buffer_cache_dir, f"emb_solved_s{args.seed}_t{bt}.npz")
    buf = np.load(buf_path, allow_pickle=True)
    buf_emb = buf["embeddings_solved"]
    n_buf = len(buf_emb)
    print(f"Buffer (t={bt}): {n_buf} levels")

    # Load eval levels
    eval_path = os.path.join(args.buffer_cache_dir, f"eval_solved_s{args.seed}_t{bt}.npz")
    eval_d = np.load(eval_path, allow_pickle=True)
    eval_emb = eval_d["embeddings_solved"]
    eval_solved = eval_d["solved"]
    n_eval = len(eval_emb)
    print(f"Eval: {n_eval} levels, {eval_solved.sum()}/8 solved")

    # Load LLM originals
    llm_orig = np.load(args.llm_orig_cache, allow_pickle=True)
    llm_emb = llm_orig["embeddings_solved"]
    n_llm = len(llm_emb)
    print(f"LLM originals: {n_llm}")

    # Load mutations
    mut = np.load(args.mutations_cache, allow_pickle=True)
    mut_emb = mut["embeddings_solved"]
    mut_flips = mut["flip_counts"]
    mut_aids = mut["ancestor_ids"]
    n_mut = len(mut_emb)
    print(f"Mutations: {n_mut} (flips 0..{mut_flips.max()})")

    # Subsample mutations (50 per flip count for plotting)
    n_per_flip = 50
    sub_idx = []
    rng = np.random.RandomState(42)
    for f in range(mut_flips.max() + 1):
        f_idx = np.where(mut_flips == f)[0]
        if len(f_idx) > n_per_flip:
            f_idx = rng.choice(f_idx, n_per_flip, replace=False)
        sub_idx.extend(f_idx.tolist())
    sub_idx = np.array(sub_idx)
    mut_emb_sub = mut_emb[sub_idx]
    mut_flips_sub = mut_flips[sub_idx]
    mut_aids_sub = mut_aids[sub_idx]
    n_mut_sub = len(sub_idx)
    print(f"Subsampled mutations: {n_mut_sub}")

    if os.path.exists(args.tsne_cache):
        print(f"Loading cached t-SNE + projections: {args.tsne_cache}")
        tc = np.load(args.tsne_cache)
        buf_coords = tc["buf_coords"]
        eval_coords = tc["eval_coords"]
        llm_coords = tc["llm_coords"]
        mut_coords = tc["mut_coords"]
        mut_flips_sub = tc["mut_flips"]
        mut_aids_sub = tc["mut_aids"]
    else:
        # t-SNE on reference points only (buffer + eval + LLM originals)
        ref_emb = np.concatenate([buf_emb, eval_emb, llm_emb], axis=0)
        effective_perp = min(args.perplexity, len(ref_emb) - 1)
        print(f"t-SNE on reference: {len(ref_emb)} points, perplexity={effective_perp}...")

        tsne = TSNE(n_components=2, perplexity=effective_perp,
                    random_state=42, max_iter=1000,
                    learning_rate='auto', init='pca')
        ref_coords = tsne.fit_transform(ref_emb)

        buf_coords = ref_coords[:n_buf]
        eval_coords = ref_coords[n_buf:n_buf + n_eval]
        llm_coords = ref_coords[n_buf + n_eval:]

        # Project mutations via k-NN interpolation
        print(f"Projecting {n_mut_sub} mutations via k={args.knn_k} NN...")
        mut_coords = project_via_knn(ref_emb, ref_coords, mut_emb_sub, k=args.knn_k)

        np.savez_compressed(args.tsne_cache,
            buf_coords=buf_coords, eval_coords=eval_coords,
            llm_coords=llm_coords, mut_coords=mut_coords,
            mut_flips=mut_flips_sub, mut_aids=mut_aids_sub)
        print(f"Cached: {args.tsne_cache}")

    if args.static:
        matplotlib.use('Agg')
        _plot_static(buf_coords, eval_coords, eval_solved, llm_coords,
                     mut_coords, mut_flips_sub, args)
    else:
        _plot_interactive(buf_coords, eval_coords, eval_solved, llm_coords,
                          mut_coords, mut_flips_sub, args)


def _plot_interactive(buf_coords, eval_coords, eval_solved, llm_coords,
                      mut_coords, mut_flips, args):
    """Interactive plot with slider for flip count."""
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(bottom=0.12)

    ax.scatter(buf_coords[:, 0], buf_coords[:, 1],
               c='#707070', s=4, alpha=0.5, edgecolors='none',
               rasterized=True, label='PLR buffer')

    for ei, name in enumerate(EVAL_LEVEL_NAMES[:len(eval_coords)]):
        color = 'cyan' if eval_solved[ei] else 'black'
        edge = 'black' if eval_solved[ei] else 'white'
        ax.scatter(eval_coords[ei, 0], eval_coords[ei, 1],
                   c=color, s=40, marker='D', alpha=0.9,
                   edgecolors=edge, linewidths=0.5, zorder=10)

    ax.scatter(llm_coords[:, 0], llm_coords[:, 1],
               c='tab:red', s=120, marker='*', alpha=0.95,
               edgecolors='black', linewidths=0.5, zorder=8,
               label='LLM originals')

    init_flip = 0
    mask = mut_flips == init_flip
    mut_scatter = ax.scatter(mut_coords[mask, 0], mut_coords[mask, 1],
                              c='tab:orange', s=15, alpha=0.6,
                              edgecolors='none', zorder=5,
                              label='Mutations')

    ax.set_xticks([]); ax.set_yticks([])
    title = ax.set_title(f'Flip count: {init_flip}  (n={mask.sum()})', fontsize=13)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.8)

    ax_slider = plt.axes([0.15, 0.02, 0.7, 0.03])
    slider = Slider(ax_slider, 'Flips', 0, int(mut_flips.max()),
                    valinit=init_flip, valstep=1)

    def update(val):
        f = int(slider.val)
        mask = mut_flips == f
        if mask.sum() > 0:
            mut_scatter.set_offsets(mut_coords[mask])
        else:
            mut_scatter.set_offsets(np.empty((0, 2)))
        title.set_text(f'Flip count: {f}  (n={mask.sum()})')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def _plot_static(buf_coords, eval_coords, eval_solved, llm_coords,
                 mut_coords, mut_flips, args):
    """Save a static PNG with selected flip counts."""
    flip_values = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80]
    flip_values = [f for f in flip_values if f <= mut_flips.max()]
    n_panels = len(flip_values)
    n_cols = 4
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                              squeeze=False)

    for idx, f in enumerate(flip_values):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]

        ax.scatter(buf_coords[:, 0], buf_coords[:, 1],
                   c='#707070', s=2, alpha=0.4, edgecolors='none', rasterized=True)

        mask = mut_flips == f
        if mask.sum() > 0:
            ax.scatter(mut_coords[mask, 0], mut_coords[mask, 1],
                       c='tab:orange', s=10, alpha=0.6, edgecolors='none', zorder=5)

        ax.scatter(llm_coords[:, 0], llm_coords[:, 1],
                   c='tab:red', s=80, marker='*', alpha=0.95,
                   edgecolors='black', linewidths=0.4, zorder=8)

        for ei, name in enumerate(EVAL_LEVEL_NAMES[:len(eval_coords)]):
            color = 'cyan' if eval_solved[ei] else 'black'
            edge = 'black' if eval_solved[ei] else 'white'
            ax.scatter(eval_coords[ei, 0], eval_coords[ei, 1],
                       c=color, s=25, marker='D', alpha=0.9,
                       edgecolors=edge, linewidths=0.5, zorder=10)

        ax.set_title(f'{f} flips (n={mask.sum()})', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#707070',
               markersize=6, alpha=0.5, label='PLR buffer'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:orange',
               markersize=6, label='Mutations'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='tab:red',
               markersize=10, markeredgecolor='black', markeredgewidth=0.4,
               label='LLM originals'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan',
               markersize=7, markeredgecolor='black', markeredgewidth=0.5,
               label='Eval (solved)'),
    ]
    fig.legend(handles=legend_els, fontsize=9, loc='lower center',
               framealpha=0.8, ncol=4, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f'LLM mutations on pre-injection buffer — seed {args.seed}',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
