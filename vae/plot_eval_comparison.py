"""Plot comparative bar charts from eval_test_levels.py results.

Loads all .npz eval results, aggregates across seeds per method,
and produces bar charts with error bars for key metrics.

Usage:
    python vae/plot_eval_comparison.py \
        --results_dir /cs/student/project_msc/2025/csml/rhautier/eval_results_13x13 \
        --output_dir vae/plots/eval_comparison_13x13
"""
import argparse
import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all_results(results_dir):
    """Load all .npz eval results, grouped by method."""
    methods = {}
    for method_dir in sorted(glob.glob(os.path.join(results_dir, "*"))):
        if not os.path.isdir(method_dir):
            continue
        method = os.path.basename(method_dir)
        npz_files = sorted(glob.glob(os.path.join(method_dir, "*.npz")))
        if not npz_files:
            continue

        seeds = []
        for f in npz_files:
            d = dict(np.load(f, allow_pickle=True))
            seeds.append(d)
        methods[method] = seeds
        print(f"  {method}: {len(seeds)} seeds, levels={list(seeds[0]['level_names'])}")

    return methods


def aggregate_metric(methods, key):
    """Get mean and std across seeds for a scalar metric."""
    names = []
    means = []
    stds = []
    for method, seeds in methods.items():
        vals = [float(s[key]) for s in seeds]
        names.append(method)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return names, np.array(means), np.array(stds)


def aggregate_per_level(methods, key):
    """Get mean and std across seeds for a per-level metric."""
    result = {}
    for method, seeds in methods.items():
        # Stack: (n_seeds, n_levels)
        stacked = np.stack([s[key] for s in seeds], axis=0)
        result[method] = {
            "mean": stacked.mean(axis=0),
            "std": stacked.std(axis=0),
            "level_names": seeds[0]["level_names"],
        }
    return result


def plot_grouped_bars(ax, methods_data, title, ylabel, level_names):
    """Plot grouped bars: one group per level, one bar per method."""
    n_methods = len(methods_data)
    n_levels = len(level_names)
    x = np.arange(n_levels)
    width = 0.8 / n_methods

    for i, (method, data) in enumerate(methods_data.items()):
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, data["mean"], width, yerr=data["std"],
               label=method, capsize=2, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(level_names, rotation=45, ha="right", fontsize=7)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(axis="y", alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading results...")
    methods = load_all_results(args.results_dir)
    if not methods:
        print("No results found.")
        return

    level_names = list(list(methods.values())[0][0]["level_names"])

    # ===== Plot 1: Aggregate metrics bar chart =====
    agg_metrics = [
        ("mean_solve_rate", "Mean Solve Rate"),
        ("iqm_solve_rate", "IQM Solve Rate"),
        ("mean_return_agg", "Mean Return"),
        ("mean_optimality_gap", "Mean Optimality Gap"),
    ]

    fig, axes = plt.subplots(1, len(agg_metrics), figsize=(4.5 * len(agg_metrics), 5))
    for ax, (key, title) in zip(axes, agg_metrics):
        names, means, stds = aggregate_metric(methods, key)
        colors = [f"C{i}" for i in range(len(names))]
        bars = ax.bar(names, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(title)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        # Add value labels
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("Aggregate Metrics (mean ± std across seeds)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.output_dir, "aggregate_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ===== Plot 2: Per-level solve rates =====
    per_level_sr = aggregate_per_level(methods, "solve_rates")
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_grouped_bars(ax, per_level_sr, "Solve Rate per Level (100 rollouts)", "Solve Rate", level_names)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="grey", linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(args.output_dir, "per_level_solve_rate.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ===== Plot 3: Per-level optimality gap =====
    per_level_og = aggregate_per_level(methods, "optimality_gaps")
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_grouped_bars(ax, per_level_og, "Optimality Gap per Level (V* - V^π)", "Gap", level_names)
    plt.tight_layout()
    path = os.path.join(args.output_dir, "per_level_optimality_gap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ===== Plot 4: Per-level path ratio =====
    per_level_pr = aggregate_per_level(methods, "path_ratio_means")
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_grouped_bars(ax, per_level_pr, "Path Ratio per Level (agent_steps / BFS_shortest)", "Ratio", level_names)
    ax.axhline(y=1.0, color="grey", linestyle="--", alpha=0.3, label="Optimal")
    plt.tight_layout()
    path = os.path.join(args.output_dir, "per_level_path_ratio.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ===== Plot 5: Per-level episode length =====
    per_level_el = aggregate_per_level(methods, "ep_length_means")
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_grouped_bars(ax, per_level_el, "Episode Length per Level", "Steps", level_names)
    # Add BFS shortest path reference
    shortest = list(methods.values())[0][0]["shortest_paths"]
    x = np.arange(len(level_names))
    ax.scatter(x, shortest, color="black", marker="_", s=200, zorder=5, label="BFS shortest")
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    path = os.path.join(args.output_dir, "per_level_episode_length.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ===== Plot 6: Per-level learnability =====
    per_level_learn = aggregate_per_level(methods, "learnability")
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_grouped_bars(ax, per_level_learn, "Learnability per Level (SFL = p(1-p))", "SFL", level_names)
    ax.axhline(y=0.25, color="grey", linestyle="--", alpha=0.3, label="Max learnable")
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    path = os.path.join(args.output_dir, "per_level_learnability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # ===== Summary table printed =====
    print(f"\n{'='*70}")
    print(f"{'Method':<25} {'MeanSR':>8} {'IQM':>8} {'Return':>8} {'OptGap':>8}")
    print(f"{'='*70}")
    for method, seeds in methods.items():
        sr = np.mean([float(s["mean_solve_rate"]) for s in seeds])
        iqm = np.mean([float(s["iqm_solve_rate"]) for s in seeds])
        ret = np.mean([float(s["mean_return_agg"]) for s in seeds])
        og = np.mean([float(s["mean_optimality_gap"]) for s in seeds])
        print(f"{method:<25} {sr:>8.3f} {iqm:>8.3f} {ret:>8.3f} {og:>8.3f}")

    print(f"\nDone. 6 plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
