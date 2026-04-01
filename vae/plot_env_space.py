"""Visualize buffer levels in environment (structural) space.

Instead of agent-behavior embeddings, projects levels based on their
structural features: flattened wall map (169D) + agent/goal positions (4D).

Grid layout: rows = seeds, columns = injection percentages.
One grid per update step. Supports both initial (merged) buffers and
training buffer dumps.

Usage:
    # Initial buffers (pre-training)
    python vae/plot_env_space.py --source initial

    # Training snapshots
    python vae/plot_env_space.py --source training --updates 1000,5000,10000

    # Both
    python vae/plot_env_space.py --source both --updates 1000,5000,10000
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gcs_artifacts", "plot_data",
)
GRID_SIZE = 13


def tokens_to_structural_features(tokens_batch):
    """Convert (N, 52) token array to (N, 173) structural feature vectors.

    Features: flattened wall map (169D bool) + agent_pos (x,y) + goal_pos (x,y).
    """
    N = len(tokens_batch)
    features = np.zeros((N, 169 + 4), dtype=np.float32)

    for i in range(N):
        tokens = tokens_batch[i]
        wall_tokens = tokens[:-2]
        goal_idx = tokens[-2]
        agent_idx = tokens[-1]

        # Wall map: 1-based indices to flat 169 binary
        wall_flat = np.zeros(169, dtype=np.float32)
        for w in wall_tokens:
            if w > 0:
                wall_flat[int(w) - 1] = 1.0
        features[i, :169] = wall_flat

        # Agent position (normalized to [0,1])
        if agent_idx > 0:
            a0 = int(agent_idx) - 1
            features[i, 169] = (a0 % GRID_SIZE) / (GRID_SIZE - 1)
            features[i, 170] = (a0 // GRID_SIZE) / (GRID_SIZE - 1)

        # Goal position (normalized to [0,1])
        if goal_idx > 0:
            g0 = int(goal_idx) - 1
            features[i, 171] = (g0 % GRID_SIZE) / (GRID_SIZE - 1)
            features[i, 172] = (g0 // GRID_SIZE) / (GRID_SIZE - 1)

    return features


def load_merged_buffer(seed, pct, data_root=None):
    root = data_root or DEFAULT_DATA_ROOT
    path = os.path.join(root, f"llm_inject_seed{seed}", f"merged_buffer_{pct}.npz")
    d = np.load(path)
    size = int(d["size"]) if "size" in d else len(d["tokens"])
    return {"tokens": d["tokens"][:size], "origins": d["origins"][:size], "size": size}


def load_buffer_dump(seed, pct, update, data_root=None):
    root = data_root or DEFAULT_DATA_ROOT
    path = os.path.join(root, f"llm_inject_seed{seed}",
                        f"training_{pct}", "buffer_dumps", f"buffer_dump_{update}.npz")
    d = np.load(path)
    size = int(d["size"])
    return {"tokens": d["tokens"][:size], "origins": d["origins"][:size], "size": size}


def _plot_cell(ax, coords, origins, pct_label, seed,
               show_xlabel=False, show_ylabel=False, xlabel='', ylabel=''):
    is_organic = origins == 0
    is_original = origins == 1
    is_mutation = origins == 2

    ax.scatter(coords[is_organic, 0], coords[is_organic, 1],
               c='#6BAED6', s=4, alpha=0.15, edgecolors='none', rasterized=True)
    if is_mutation.sum() > 0:
        ax.scatter(coords[is_mutation, 0], coords[is_mutation, 1],
                   c='#2CA02C', s=12, alpha=0.5, edgecolors='none', rasterized=True)
    if is_original.sum() > 0:
        ax.scatter(coords[is_original, 0], coords[is_original, 1],
                   c='red', s=50, marker='*', alpha=0.9,
                   edgecolors='black', linewidths=0.3)

    n_orig = is_original.sum()
    n_mut = is_mutation.sum()
    n_org = is_organic.sum()
    ax.set_title(f"Seed {seed}, {pct_label}\n({n_orig} orig, {n_mut} mut, {n_org} organic)",
                 fontsize=10)
    ax.tick_params(labelsize=7)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


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


def make_grid(seeds, pcts, pct_labels, features_dict, origins_dict,
              method, output_dir, tag, pca_model=None, perplexity=40):
    """Create one grid plot.

    features_dict/origins_dict: keyed by (seed, pct).
    """
    n_rows = len(seeds)
    n_cols = len(pcts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                              squeeze=False)

    for i, seed in enumerate(seeds):
        for j, pct in enumerate(pcts):
            key = (seed, pct)
            feats = features_dict[key]
            origins = origins_dict[key]

            if method == "pca":
                coords = pca_model.transform(feats)
                ev = pca_model.explained_variance_ratio_
                xl = f'PC1 ({ev[0]:.1%})'
                yl = f'PC2 ({ev[1]:.1%})'
            else:
                print(f"    t-SNE: seed {seed}, {pct}...")
                perp_val = min(perplexity, len(feats) - 1)
                tsne = TSNE(n_components=2, perplexity=perp_val, random_state=42,
                            init='pca', learning_rate='auto')
                coords = tsne.fit_transform(feats)
                xl = 't-SNE dim 1'
                yl = 't-SNE dim 2'

            _plot_cell(axes[i][j], coords, origins, pct_labels[j], seed,
                       show_xlabel=(i == n_rows - 1), show_ylabel=(j == 0),
                       xlabel=xl, ylabel=yl)

    method_label = "PCA" if method == "pca" else "t-SNE"
    _add_legend_and_save(fig, f'Environment Space ({method_label} 2D) — {tag}',
                         os.path.join(output_dir, f"grid_env_{method}_{tag}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--pcts", type=str, default="5pct,10pct,15pct,20pct,25pct")
    parser.add_argument("--source", type=str, default="both",
                        choices=["initial", "training", "both"])
    parser.add_argument("--updates", type=str,
                        default="1000,2000,3000,4000,5000,6000,7000,8000,9000,10000")
    parser.add_argument("--method", type=str, default="both",
                        choices=["pca", "tsne", "both"])
    parser.add_argument("--tsne_perplexity", type=float, default=40)
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root dir with llm_inject_seed{s}/ layout (default: gcs_artifacts/plot_data)")
    parser.add_argument("--output_dir", type=str, default="vae/plots/env_space")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    pcts = args.pcts.split(",")
    pct_labels = [p.replace("pct", "%") for p in pcts]
    updates = [int(u) for u in args.updates.split(",")]
    methods = ["pca", "tsne"] if args.method == "both" else [args.method]
    data_root = args.data_root

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all features for global PCA fit
    print("=== Computing structural features ===")
    all_features_for_pca = []
    snapshots = []  # list of (tag, features_dict, origins_dict)

    if args.source in ("initial", "both"):
        features_dict = {}
        origins_dict = {}
        for seed in seeds:
            for pct in pcts:
                try:
                    buf = load_merged_buffer(seed, pct, data_root=data_root)
                    feats = tokens_to_structural_features(buf["tokens"])
                    features_dict[(seed, pct)] = feats
                    origins_dict[(seed, pct)] = buf["origins"]
                    all_features_for_pca.append(feats)
                    print(f"  initial s{seed}/{pct}: {buf['size']} levels")
                except FileNotFoundError:
                    print(f"  WARNING: missing initial s{seed}/{pct}")
        if features_dict:
            snapshots.append(("initial", features_dict, origins_dict))

    if args.source in ("training", "both"):
        for update in updates:
            features_dict = {}
            origins_dict = {}
            for seed in seeds:
                for pct in pcts:
                    try:
                        buf = load_buffer_dump(seed, pct, update, data_root=data_root)
                        feats = tokens_to_structural_features(buf["tokens"])
                        features_dict[(seed, pct)] = feats
                        origins_dict[(seed, pct)] = buf["origins"]
                        all_features_for_pca.append(feats)
                    except FileNotFoundError:
                        print(f"  WARNING: missing s{seed}/{pct}/u{update}")
            if features_dict:
                snapshots.append((f"u{update}", features_dict, origins_dict))
                print(f"  training u{update}: {len(features_dict)} cells loaded")

    # Fit global PCA
    print("\n=== Fitting global PCA ===")
    all_feats = np.concatenate(all_features_for_pca, axis=0)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_feats)
    ev = pca.explained_variance_ratio_
    print(f"  PC1={ev[0]:.1%}, PC2={ev[1]:.1%}, total={ev[:2].sum():.1%}")
    print(f"  Fit on {len(all_feats)} total points")
    del all_feats, all_features_for_pca

    # Generate plots
    for tag, features_dict, origins_dict in snapshots:
        for method in methods:
            print(f"\n=== {tag} ({method.upper()}) ===")
            make_grid(seeds, pcts, pct_labels, features_dict, origins_dict,
                      method, args.output_dir, tag, pca_model=pca,
                      perplexity=args.tsne_perplexity)

    print(f"\nDone. Plots in {args.output_dir}/")


if __name__ == "__main__":
    main()
