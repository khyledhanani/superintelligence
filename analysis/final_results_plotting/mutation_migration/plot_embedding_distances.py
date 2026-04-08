"""Plot buffer neighbourhood density vs tile flips from LLM ancestor.

Two modes:
  --normalized: adaptive R per timestep, y-axis = ratio to organic mean (baseline=1.0)
  --radii 0.4: fixed radius, y-axis = raw neighbour count

Usage:
    python analysis/final_results_plotting/mutation_migration/plot_embedding_distances.py \
        --seed 3 --normalized --target 100 --percentile 50 80
    python analysis/final_results_plotting/mutation_migration/plot_embedding_distances.py \
        --seed 3 --radii 0.4 --percentile 50 80
"""
import argparse
import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, BallTree


def find_radius_for_target(organic_emb, target_neighbours):
    """Find radius R such that mean organic-to-organic neighbour count ~ target."""
    k = min(target_neighbours + 1, len(organic_emb))
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(organic_emb)
    dists, _ = nn.kneighbors(organic_emb)
    if k <= target_neighbours:
        return dists[:, -1].mean()
    return dists[:, target_neighbours].mean()


def bin_data(flips, values, percentiles, sigma=0):
    """Bin by tile flips and compute mean + percentile bands."""
    from scipy.ndimage import gaussian_filter1d

    bin_edges = np.arange(0, flips.max() + 2, 1)
    bin_flips, bin_mean, bin_counts = [], [], []
    bin_bands = {p: ([], []) for p in percentiles}

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (flips >= lo) & (flips < hi)
        n = mask.sum()
        if n < 3:
            continue
        bin_flips.append((lo + hi) / 2)
        bin_counts.append(n)
        bin_mean.append(values[mask].mean())
        for p in percentiles:
            lo_pct = (100 - p) / 2
            hi_pct = 100 - lo_pct
            bin_bands[p][0].append(np.percentile(values[mask], lo_pct))
            bin_bands[p][1].append(np.percentile(values[mask], hi_pct))

    bin_flips = np.array(bin_flips)
    bin_mean = np.array(bin_mean)

    if len(bin_mean) > 3 and sigma > 0:
        bin_mean = gaussian_filter1d(bin_mean, sigma=sigma)

    for p in percentiles:
        lo_arr = np.array(bin_bands[p][0])
        hi_arr = np.array(bin_bands[p][1])
        if len(lo_arr) > 3 and sigma > 0:
            lo_arr = gaussian_filter1d(lo_arr, sigma=sigma)
            hi_arr = gaussian_filter1d(hi_arr, sigma=sigma)
        bin_bands[p] = (lo_arr, hi_arr)

    return bin_flips, bin_mean, bin_counts, bin_bands


def plot_on_ax(ax, bin_flips, bin_mean, bin_bands, percentiles, baseline=None,
               baseline_label=None):
    """Plot mean line + percentile bands on an axis."""
    pcts_sorted = sorted(percentiles, reverse=True)
    n_bands = len(pcts_sorted)
    for bi, p in enumerate(pcts_sorted):
        lo_arr, hi_arr = bin_bands[p]
        alpha = 0.1 + 0.15 * (bi / max(n_bands - 1, 1))
        ax.fill_between(bin_flips, lo_arr, hi_arr,
                        color='tab:blue', alpha=alpha,
                        label=f'{int(p)}% band')

    ax.plot(bin_flips, bin_mean, '-', color='tab:blue', linewidth=2, label='Mean')

    if baseline is not None:
        ax.axhline(baseline, color='tab:red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=baseline_label or f'{baseline}')

    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str,
                        default="analysis/final_results_plotting/cache_solved")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--timesteps", type=int, nargs='+', default=None)
    parser.add_argument("--normalized", action="store_true",
                        help="Adaptive R per timestep, y-axis = ratio to organic mean")
    parser.add_argument("--target", type=int, default=100,
                        help="Target mean organic neighbours for adaptive R")
    parser.add_argument("--radii", type=float, nargs='+', default=None,
                        help="Fixed radii (one plot per radius)")
    parser.add_argument("--percentile", type=float, nargs='+', default=[90])
    parser.add_argument("--sigma", type=float, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"analysis/final_results_plotting/mutation_migration/embedding_distances_seed{args.seed}.png"

    percentiles = sorted(args.percentile, reverse=True)

    # Auto-discover timesteps
    if args.timesteps is None:
        import glob, re
        pattern = os.path.join(args.cache_dir, f"emb_solved_s{args.seed}_t*.npz")
        args.timesteps = sorted(
            int(re.search(r'_t(\d+)\.npz', f).group(1))
            for f in glob.glob(pattern)
        )
        print(f"Auto-discovered {len(args.timesteps)} timesteps")

    if args.normalized:
        _run_normalized(args, percentiles)
    else:
        if args.radii is None:
            args.radii = [0.4]
        _run_fixed(args, percentiles)


def _run_normalized(args, percentiles):
    """Adaptive R per timestep, y-axis = count / organic_mean."""
    target = args.target
    records = []  # (tile_diffs, ratio)

    for ts in args.timesteps:
        path = os.path.join(args.cache_dir, f"emb_solved_s{args.seed}_t{ts}.npz")
        if not os.path.exists(path):
            continue

        d = np.load(path, allow_pickle=True)
        emb = d['embeddings_solved']
        origins = d['origins']
        tile_diffs = d['tile_diffs'] if 'tile_diffs' in d else None
        if tile_diffs is None:
            continue

        organic_emb = emb[origins == 0]
        if len(organic_emb) < target + 1:
            print(f"t={ts}: only {len(organic_emb)} organic, skipping")
            continue

        R = find_radius_for_target(organic_emb, target)
        tree = BallTree(organic_emb, metric='euclidean')

        # Compute actual organic mean at this R (may differ slightly from target)
        org_counts = tree.query_radius(organic_emb, r=R, count_only=True) - 1  # subtract self
        org_mean = org_counts.mean()

        desc_mask = (origins > 0) & (tile_diffs >= 0)
        desc_indices = np.where(desc_mask)[0]
        if len(desc_indices) == 0:
            continue

        desc_counts = tree.query_radius(emb[desc_indices], r=R, count_only=True)

        for j, i in enumerate(desc_indices):
            ratio = desc_counts[j] / org_mean if org_mean > 0 else 0
            records.append((tile_diffs[i], ratio))

        print(f"t={ts}: R={R:.4f}, org_mean={org_mean:.1f}, {len(desc_indices)} LLM")

    if not records:
        print("No data.")
        return

    arr = np.array(records)
    flips = arr[:, 0].astype(int)
    ratios = arr[:, 1]

    bin_flips, bin_mean, bin_counts, bin_bands = bin_data(
        flips, ratios, percentiles, sigma=args.sigma)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_on_ax(ax, bin_flips, bin_mean, bin_bands, percentiles,
               baseline=1.0, baseline_label='Organic mean (1.0)')

    ax.set_xlabel('Grid hamming distance from LLM ancestor', fontsize=12)
    ax.set_ylabel('Neighbour density ratio (vs organic mean)', fontsize=12)
    ax.set_title(f'Normalized buffer neighbourhood density — seed {args.seed}', fontsize=13)
    ax.legend(fontsize=9, framealpha=0.8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.output}")


def _run_fixed(args, percentiles):
    """Fixed radii, y-axis = raw neighbour count."""
    radii = sorted(args.radii)
    max_r = max(radii)

    records = {r: [] for r in radii}
    organic_baseline = {r: [] for r in radii}

    for ts in args.timesteps:
        path = os.path.join(args.cache_dir, f"emb_solved_s{args.seed}_t{ts}.npz")
        if not os.path.exists(path):
            continue

        d = np.load(path, allow_pickle=True)
        emb = d['embeddings_solved']
        origins = d['origins']
        tile_diffs = d['tile_diffs'] if 'tile_diffs' in d else None
        if tile_diffs is None:
            continue

        organic_emb = emb[origins == 0]
        if len(organic_emb) == 0:
            continue

        tree = BallTree(organic_emb, metric='euclidean')

        # Organic baseline
        _, org_dists_list = tree.query_radius(organic_emb, r=max_r,
                                              return_distance=True)
        for r in radii:
            counts = np.array([((d_arr <= r).sum() - 1) for d_arr in org_dists_list])
            organic_baseline[r].append(counts.mean())

        desc_mask = (origins > 0) & (tile_diffs >= 0)
        desc_indices = np.where(desc_mask)[0]
        if len(desc_indices) == 0:
            continue

        _, dists_list = tree.query_radius(emb[desc_indices], r=max_r,
                                          return_distance=True)
        for j, i in enumerate(desc_indices):
            td = tile_diffs[i]
            d_arr = dists_list[j]
            for r in radii:
                records[r].append((td, int((d_arr <= r).sum())))

        print(f"t={ts}: {len(desc_indices)} LLM, {len(organic_emb)} organic")

    if not records[radii[0]]:
        print("No data.")
        return

    n_radii = len(radii)
    n_cols = min(5, n_radii)
    n_rows = math.ceil(n_radii / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3.5 * n_rows) if n_radii > 1
                              else (10, 6),
                              squeeze=False, sharex=True)

    for idx, r in enumerate(radii):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]

        arr = np.array(records[r])
        flips = arr[:, 0].astype(int)
        neighbours = arr[:, 1].astype(float)

        bin_flips, bin_mean, bin_counts, bin_bands = bin_data(
            flips, neighbours, percentiles, sigma=args.sigma)

        org_mean = np.mean(organic_baseline[r]) if organic_baseline[r] else None
        plot_on_ax(ax, bin_flips, bin_mean, bin_bands, percentiles,
                   baseline=org_mean,
                   baseline_label='Organic mean' if org_mean else None)

        if n_radii > 1:
            ax.set_title(f'r = {r:.2f}', fontsize=10)
        else:
            ax.set_title(f'Buffer neighbourhood density (r={r}) — seed {args.seed}',
                         fontsize=13)
        if col == 0:
            ax.set_ylabel('Organic neighbours', fontsize=10)
        if row == n_rows - 1:
            ax.set_xlabel('Grid hamming distance from LLM ancestor', fontsize=10)
        if idx == 0:
            ax.legend(fontsize=9, framealpha=0.8)

    for idx in range(n_radii, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    if n_radii > 1:
        plt.suptitle(f'Buffer neighbourhood density vs structural distance — seed {args.seed}',
                     fontsize=13, y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
