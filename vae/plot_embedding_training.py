"""Visualize buffer embeddings during training: LLM-lineage vs ACCEL-organic.

Reads buffer dumps at specified training updates and plots a grid of PCA-2D
projections:
  - Rows: seeds (0, 1, 2)
  - Columns: injection percentages (5%, 10%, 15%, 20%, 25%)

One grid per update step. Uses stored embeddings from buffer dumps (computed
by the evolving agent at that training step).

Colors: light blue (transparent) = organic (ACCEL, origin=0),
        green = LLM mutation descendants (origin=2),
        red = LLM originals (origin=1).

Usage:
    python vae/plot_embedding_training.py
    python vae/plot_embedding_training.py --updates 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gcs_artifacts", "plot_data",
)


def load_buffer_dump(seed, pct, update, data_root=None):
    """Load training buffer dump with embeddings and origins."""
    root = data_root or DEFAULT_DATA_ROOT
    path = os.path.join(
        root, f"llm_inject_seed{seed}",
        f"training_{pct}", "buffer_dumps", f"buffer_dump_{update}.npz")
    d = np.load(path)
    size = int(d["size"])
    return {
        "embeddings": d["embeddings"][:size],
        "origins": d["origins"][:size],
        "scores": d["scores"][:size],
        "size": size,
    }


def _plot_cell(ax, coords, origins, pct_label, seed, show_xlabel=False, show_ylabel=False,
               xlabel='', ylabel=''):
    """Plot a single scatter cell."""
    is_organic = origins == 0
    is_original = origins == 1
    is_mutation = origins == 2
    n_org = is_organic.sum()
    n_orig = is_original.sum()
    n_mut = is_mutation.sum()

    # Organic — light blue, transparent
    ax.scatter(coords[is_organic, 0], coords[is_organic, 1],
               c='#6BAED6', s=4, alpha=0.15, edgecolors='none',
               rasterized=True)
    # LLM mutations — green
    if n_mut > 0:
        ax.scatter(coords[is_mutation, 0], coords[is_mutation, 1],
                   c='#2CA02C', s=12, alpha=0.5, edgecolors='none',
                   rasterized=True)
    # LLM originals — red stars
    if n_orig > 0:
        ax.scatter(coords[is_original, 0], coords[is_original, 1],
                   c='red', s=50, marker='*', alpha=0.9,
                   edgecolors='black', linewidths=0.3)

    ax.set_title(f"Seed {seed}, {pct_label}\n"
                 f"({n_orig} orig, {n_mut} mut, {n_org} organic)",
                 fontsize=10)
    ax.tick_params(labelsize=7)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def make_grid_pca(seeds, pcts, pct_labels, update, pca_model, output_dir, data_root=None):
    """Create a PCA grid plot for one training update step."""
    n_rows = len(seeds)
    n_cols = len(pcts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                              squeeze=False)
    ev = pca_model.explained_variance_ratio_

    for i, seed in enumerate(seeds):
        for j, pct in enumerate(pcts):
            buf = load_buffer_dump(seed, pct, update, data_root=data_root)
            coords = pca_model.transform(buf["embeddings"])
            _plot_cell(axes[i][j], coords, buf["origins"], pct_labels[j], seed,
                       show_xlabel=(i == n_rows - 1), show_ylabel=(j == 0),
                       xlabel=f'PC1 ({ev[0]:.1%})', ylabel=f'PC2 ({ev[1]:.1%})')

    _add_legend_and_save(fig, f'Buffer Embeddings at Update {update} (PCA 2D)',
                         os.path.join(output_dir, f"grid_pca_u{update}.png"))


def make_grid_tsne(seeds, pcts, pct_labels, update, output_dir, perplexity=40, data_root=None):
    """Create a t-SNE grid plot for one training update step."""
    n_rows = len(seeds)
    n_cols = len(pcts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                              squeeze=False)

    for i, seed in enumerate(seeds):
        for j, pct in enumerate(pcts):
            buf = load_buffer_dump(seed, pct, update, data_root=data_root)
            print(f"    t-SNE: seed {seed}, {pct}...")
            perp = min(perplexity, buf["size"] - 1)
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                        init='pca', learning_rate='auto')
            coords = tsne.fit_transform(buf["embeddings"])
            _plot_cell(axes[i][j], coords, buf["origins"], pct_labels[j], seed,
                       show_xlabel=(i == n_rows - 1), show_ylabel=(j == 0),
                       xlabel='t-SNE dim 1', ylabel='t-SNE dim 2')

    _add_legend_and_save(fig, f'Buffer Embeddings at Update {update} (t-SNE 2D)',
                         os.path.join(output_dir, f"grid_tsne_u{update}.png"))


def _add_legend_and_save(fig, title, out_path):
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#6BAED6',
               markersize=6, alpha=0.5, label='Organic (ACCEL)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2CA02C',
               markersize=8, label='LLM mutation descendant'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markersize=12, markeredgecolor='black', label='LLM original'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--pcts", type=str, default="5pct,10pct,15pct,20pct,25pct")
    parser.add_argument("--updates", type=str,
                        default="1000,2000,3000,4000,5000,6000,7000,8000,9000,10000",
                        help="Comma-separated update steps to plot")
    parser.add_argument("--method", type=str, default="both",
                        choices=["pca", "tsne", "both"])
    parser.add_argument("--tsne_perplexity", type=float, default=40)
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root dir with llm_inject_seed{s}/ layout (default: gcs_artifacts/plot_data)")
    parser.add_argument("--output_dir", type=str,
                        default="vae/plots/embedding_training")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    pcts = args.pcts.split(",")
    pct_labels = [p.replace("pct", "%") for p in pcts]
    updates = [int(u) for u in args.updates.split(",")]
    data_root = args.data_root

    os.makedirs(args.output_dir, exist_ok=True)

    # Fit global PCA on all data across all updates for consistent axes
    print(f"=== Fitting global PCA across {len(updates)} updates ===")
    all_emb = []
    for update in updates:
        for seed in seeds:
            for pct in pcts:
                try:
                    buf = load_buffer_dump(seed, pct, update, data_root=data_root)
                    all_emb.append(buf["embeddings"])
                except FileNotFoundError:
                    print(f"  WARNING: missing s{seed}/{pct}/u{update}, skipping")
    all_emb = np.concatenate(all_emb, axis=0)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_emb)
    ev = pca.explained_variance_ratio_
    print(f"  PC1={ev[0]:.1%}, PC2={ev[1]:.1%}, total={ev[:2].sum():.1%}")
    print(f"  Fit on {len(all_emb)} total points\n")
    del all_emb

    for update in updates:
        print(f"\n=== Update {update} ===")
        try:
            if args.method in ("pca", "both"):
                make_grid_pca(seeds, pcts, pct_labels, update, pca, args.output_dir,
                              data_root=data_root)
            if args.method in ("tsne", "both"):
                make_grid_tsne(seeds, pcts, pct_labels, update, args.output_dir,
                               perplexity=args.tsne_perplexity, data_root=data_root)
        except FileNotFoundError as e:
            print(f"  Skipping update {update}: {e}")

    print(f"\nDone. {len(updates)} plots in {args.output_dir}/")


if __name__ == "__main__":
    main()
