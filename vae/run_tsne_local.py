"""Run t-SNE evolution plot on local buffer dumps with pre-computed embeddings.

Adapted from plot_tsne_training_evolution.py to work directly with local data
from buffer_dumps/llm_injection_fresh/seed0/.

Usage:
    python vae/run_tsne_local.py --data_dir buffer_dumps/llm_injection_fresh/seed0
    python vae/run_tsne_local.py --data_dir buffer_dumps/llm_injection_fresh/seed0 --mode structural
    python vae/run_tsne_local.py --data_dir buffer_dumps/llm_injection_fresh/seed0 --show_difficulty
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from plot_tsne_training_evolution import tokens_to_structural_features


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to seed directory (e.g. buffer_dumps/llm_injection_fresh/seed0)")
    parser.add_argument("--timesteps", type=str, default=None,
                        help="Comma-separated update steps. Auto-detected if not specified.")
    parser.add_argument("--mode", type=str, default="behavioral", choices=["behavioral", "structural"],
                        help="behavioral: use pre-computed 257D embeddings. structural: compute 173D env features.")
    parser.add_argument("--tsne_perplexity", type=float, default=40)
    parser.add_argument("--output_dir", type=str, default="vae/plots/tsne_training_evolution")
    parser.add_argument("--grid_size", type=int, default=13)
    parser.add_argument("--show_difficulty", action="store_true",
                        help="Color by SFL learnability instead of origin type")
    args = parser.parse_args()

    buf_dir = os.path.join(args.data_dir, "buffer_dumps")
    if not os.path.isdir(buf_dir):
        print(f"Error: {buf_dir} not found")
        sys.exit(1)

    # Auto-detect timesteps
    if args.timesteps:
        timesteps = [int(t) for t in args.timesteps.split(",")]
    else:
        timesteps = []
        for f in sorted(os.listdir(buf_dir)):
            if f.startswith("buffer_dump_") and f.endswith(".npz") and "final" not in f:
                ts = int(f.replace("buffer_dump_", "").replace(".npz", ""))
                timesteps.append(ts)
        timesteps.sort()
        print(f"Auto-detected timesteps: {timesteps}")

    # Load data
    data = {}
    for ts in timesteps:
        path = os.path.join(buf_dir, f"buffer_dump_{ts}.npz")
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found")
            continue
        d = np.load(path, allow_pickle=True)
        size = int(d["size"])
        tokens = d["tokens"][:size]
        origins = d["origins"][:size] if "origins" in d else np.zeros(size, dtype=np.int32)
        scores = d["scores"][:size]

        if args.mode == "structural":
            embeddings = tokens_to_structural_features(tokens, grid_size=args.grid_size)
        else:
            if "embeddings" not in d:
                print(f"  SKIP ts={ts}: no pre-computed embeddings in dump")
                continue
            embeddings = d["embeddings"][:size]

        n0 = (origins == 0).sum()
        n1 = (origins == 1).sum()
        n2 = (origins == 2).sum()
        print(f"  ts={ts}: {size} levels (organic={n0}, llm_orig={n1}, llm_mut={n2})")
        data[ts] = {"embeddings": embeddings, "origins": origins, "scores": scores}

    if not data:
        print("No data loaded.")
        sys.exit(1)

    # Plot grid: single row, one column per timestep
    valid_ts = [ts for ts in timesteps if ts in data]
    n_cols = len(valid_ts)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5), squeeze=False)

    for j, ts in enumerate(valid_ts):
        ax = axes[0][j]
        emb = data[ts]["embeddings"]
        origins = data[ts]["origins"]
        scores = data[ts]["scores"]

        print(f"  t-SNE for {ts} upd ({len(emb)} pts)...")
        tsne = TSNE(n_components=2, perplexity=min(args.tsne_perplexity, len(emb) - 1),
                    random_state=42, max_iter=1000, learning_rate='auto', init='pca')
        coords = tsne.fit_transform(emb)

        is_organic = origins == 0
        is_original = origins == 1
        is_mutation = origins == 2

        if args.show_difficulty:
            cmap = mcolors.LinearSegmentedColormap.from_list("sfl", ["yellow", "red"])
            norm = mcolors.Normalize(vmin=0, vmax=0.25)

            if is_organic.sum() > 0:
                ax.scatter(coords[is_organic, 0], coords[is_organic, 1],
                           c=scores[is_organic], cmap=cmap, norm=norm,
                           s=3, alpha=0.25, edgecolors='none', rasterized=True)
            if is_mutation.sum() > 0:
                ax.scatter(coords[is_mutation, 0], coords[is_mutation, 1],
                           c=scores[is_mutation], cmap=cmap, norm=norm,
                           s=15, alpha=0.8, edgecolors='green', linewidths=0.5,
                           rasterized=True, zorder=5)
            if is_original.sum() > 0:
                ax.scatter(coords[is_original, 0], coords[is_original, 1],
                           c=scores[is_original], cmap=cmap, norm=norm,
                           s=60, marker='*', alpha=0.95,
                           edgecolors='blue', linewidths=0.6, zorder=8)

            if j == 0:
                from matplotlib.lines import Line2D
                legend_els = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
                           markersize=4, alpha=0.5, label='Organic'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
                           markersize=6, markeredgecolor='green', markeredgewidth=0.5,
                           label='LLM mutation'),
                    Line2D([0], [0], marker='*', color='w', markerfacecolor='grey',
                           markersize=10, markeredgecolor='blue', markeredgewidth=0.6,
                           label='LLM original'),
                ]
                ax.legend(handles=legend_els, fontsize=5, loc='upper left', framealpha=0.7)
            if j == n_cols - 1:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label("SFL", fontsize=7)
                cb.ax.tick_params(labelsize=6)
        else:
            if is_organic.sum() > 0:
                ax.scatter(coords[is_organic, 0], coords[is_organic, 1],
                           c='lightgrey', s=3, alpha=0.25, edgecolors='none',
                           rasterized=True, label=f'Organic ({is_organic.sum()})')
            if is_mutation.sum() > 0:
                ax.scatter(coords[is_mutation, 0], coords[is_mutation, 1],
                           c='green', s=8, alpha=0.5, edgecolors='none',
                           rasterized=True, label=f'LLM mut ({is_mutation.sum()})')
            if is_original.sum() > 0:
                ax.scatter(coords[is_original, 0], coords[is_original, 1],
                           c='blue', s=35, marker='*', alpha=0.9,
                           edgecolors='black', linewidths=0.3,
                           label=f'LLM orig ({is_original.sum()})')
            if j == 0:
                ax.legend(fontsize=6, loc='lower left', framealpha=0.7)

        n_llm = is_original.sum() + is_mutation.sum()
        ax.set_title(f"{ts} upd\n({n_llm} LLM / {len(origins)} total)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    seed_name = os.path.basename(args.data_dir)
    mode_label = "structural" if args.mode == "structural" else "behavioral"
    plt.suptitle(f"Buffer t-SNE Evolution — {seed_name} ({mode_label})", fontsize=13, y=1.02)
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    mode_tag = "env" if args.mode == "structural" else "behav"
    diff_tag = "_difficulty" if args.show_difficulty else ""
    out_path = os.path.join(args.output_dir, f"tsne_{mode_tag}_{seed_name}{diff_tag}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
