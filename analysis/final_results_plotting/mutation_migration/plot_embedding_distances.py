"""Plot buffer neighbourhood density vs tile flips from LLM ancestor.

For each LLM descendant, counts how many organic buffer levels lie within
radius R in the 257D embedding space, binned by tile flips from ancestor.
Plots a grid of subplots for multiple radii.

Usage:
    python analysis/final_results_plotting/mutation_migration/plot_embedding_distances.py \
        --seed 3 --timesteps 2250 2500 3000 4000 5500 8000 15000 30000
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str,
                        default="analysis/final_results_plotting/cache_solved")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--timesteps", type=int, nargs='+', default=None,
                        help="Timesteps to use (default: auto-discover from cache_dir)")
    parser.add_argument("--radii", type=float, nargs='+',
                        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"analysis/final_results_plotting/mutation_migration/embedding_distances_seed{args.seed}.png"

    radii = sorted(args.radii)
    max_r = max(radii)

    # Auto-discover timesteps if not specified
    if args.timesteps is None:
        import glob, re
        pattern = os.path.join(args.cache_dir, f"emb_solved_s{args.seed}_t*.npz")
        args.timesteps = sorted(
            int(re.search(r'_t(\d+)\.npz', f).group(1))
            for f in glob.glob(pattern)
        )
        print(f"Auto-discovered {len(args.timesteps)} timesteps")

    # Collect per-level: tile_diffs and embedding distances to all organic levels
    # We store (tile_diffs, distances_to_organic) per descendant per timestep,
    # then count within each radius.
    # More efficient: use BallTree.query_radius at max radius, then threshold.

    # Per-radius records: radius -> list of (tile_diffs, count)
    records = {r: [] for r in radii}

    for ts in args.timesteps:
        path = os.path.join(args.cache_dir, f"emb_solved_s{args.seed}_t{ts}.npz")
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        d = np.load(path, allow_pickle=True)
        emb = d['embeddings_solved']
        origins = d['origins']
        tile_diffs = d['tile_diffs'] if 'tile_diffs' in d else None

        if tile_diffs is None:
            print(f"t={ts}: no tile_diffs, skipping")
            continue

        organic_mask = origins == 0
        organic_emb = emb[organic_mask]
        if len(organic_emb) == 0:
            print(f"t={ts}: no organic levels, skipping")
            continue

        tree = BallTree(organic_emb, metric='euclidean')

        desc_mask = (origins > 0) & (tile_diffs >= 0)
        desc_indices = np.where(desc_mask)[0]
        if len(desc_indices) == 0:
            print(f"t={ts}: no trackable descendants")
            continue

        desc_embs = emb[desc_indices]

        # Query at max radius, get actual distances
        _, dists_list = tree.query_radius(desc_embs, r=max_r,
                                          return_distance=True)

        for j, i in enumerate(desc_indices):
            td = tile_diffs[i]
            d_arr = dists_list[j]
            for r in radii:
                records[r].append((td, int((d_arr <= r).sum())))

        print(f"t={ts}: {len(desc_indices)} LLM levels, {len(organic_emb)} organic")

    if not records[radii[0]]:
        print("No data collected.")
        return

    # --- Plot grid ---
    n_radii = len(radii)
    n_cols = 5
    n_rows = (n_radii + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                              squeeze=False, sharex=True)

    for idx, r in enumerate(radii):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        arr = np.array(records[r])
        flips = arr[:, 0].astype(int)
        neighbours = arr[:, 1].astype(float)

        bin_width = 5
        bin_edges = np.arange(0, flips.max() + bin_width + 1, bin_width)
        bin_flips, bin_mean, bin_std, bin_counts = [], [], [], []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (flips >= lo) & (flips < hi)
            n = mask.sum()
            if n < 3:
                continue
            bin_flips.append((lo + hi) / 2)
            bin_counts.append(n)
            bin_mean.append(neighbours[mask].mean())
            bin_std.append(neighbours[mask].std() / np.sqrt(n))

        bin_flips = np.array(bin_flips)
        bin_mean = np.array(bin_mean)
        bin_std = np.array(bin_std)

        ax.plot(bin_flips, bin_mean, 'o-', color='tab:blue', linewidth=1.5,
                markersize=3)
        ax.fill_between(bin_flips, bin_mean - bin_std,
                        bin_mean + bin_std, color='tab:blue', alpha=0.15)

        ax.set_title(f'r = {r:.2f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Organic neighbours', fontsize=9)
        if row == n_rows - 1:
            ax.set_xlabel('Tile flips', fontsize=9)

    # Hide unused panels
    for idx in range(n_radii, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.suptitle(f'Buffer neighbourhood density vs structural distance — seed {args.seed}',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
