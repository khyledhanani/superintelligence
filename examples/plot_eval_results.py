"""
Visualise post-hoc evaluation results from eval_test_levels.py.

Reads eval_summary.csv (or NPZ files) and produces ACCEL-paper-style bar plots:
  1. Solve rate per level (grouped by run, mean±std across seeds)
  2. Mean return per level
  3. Optimality gap per level
  4. Aggregate comparison (IQM, mean solve rate, mean opt gap)

Usage:
    python examples/plot_eval_results.py \
        --results_dir /path/to/eval_results/21x21_final/ \
        --output_dir /path/to/eval_results/21x21_final/plots/
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# ── Consistent run display names and colors ──
RUN_DISPLAY = {
    # 21x21 runs
    "accel_sfl_baseline": "ACCEL (SFL)",
    "latent_mut_sfl_21x21": "Latent Mut (SFL)",
    "cycle_2k3k5k_decay999_sfl": "CMA-ES Cycle (SFL)",
    # 13x13 runs
    "accel_sfl_13x13": "ACCEL",
    "plr_sfl_13x13": "PLR",
    "cmaes_accel_sfl_13x13": "CMA-ES + ACCEL",
    "latent_mut_sfl_13x13": "Latent Mut",
    "cmaes_latent_mut_sfl_13x13": "CMA-ES + Latent Mut",
}

RUN_COLORS = {
    # 21x21 runs
    "accel_sfl_baseline": "#1f77b4",
    "latent_mut_sfl_21x21": "#ff7f0e",
    "cycle_2k3k5k_decay999_sfl": "#2ca02c",
    # 13x13 runs
    "accel_sfl_13x13": "#1f77b4",
    "plr_sfl_13x13": "#9467bd",
    "cmaes_accel_sfl_13x13": "#2ca02c",
    "latent_mut_sfl_13x13": "#ff7f0e",
    "cmaes_latent_mut_sfl_13x13": "#d62728",
}


def load_csv(csv_path):
    """Load eval_summary.csv into list of dicts."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_level_names_from_csv(rows):
    """Extract level names from CSV column headers."""
    levels = []
    for key in rows[0].keys():
        if key.startswith("solve_rate/"):
            levels.append(key.split("/", 1)[1])
    return levels


def aggregate_by_run(rows, level_names, metric_prefix="solve_rate"):
    """Group rows by run_name and compute mean±std across seeds for a given metric."""
    by_run = defaultdict(lambda: defaultdict(list))
    for row in rows:
        run = row["run_name"]
        for name in level_names:
            key = f"{metric_prefix}/{name}"
            val = row.get(key, "")
            if val:
                by_run[run][name].append(float(val))

    agg = {}
    for run in by_run:
        means = []
        stds = []
        for name in level_names:
            vals = by_run[run].get(name, [])
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0.0)
        agg[run] = {"means": np.array(means), "stds": np.array(stds)}
    return agg


def plot_grouped_bars(agg, level_names, ylabel, title, output_path,
                      ylim=None, percent=False):
    """ACCEL-style grouped bar chart."""
    runs = sorted(agg.keys())
    n_levels = len(level_names)
    n_runs = len(runs)
    bar_width = 0.8 / n_runs
    x = np.arange(n_levels)

    fig, ax = plt.subplots(figsize=(max(14, n_levels * 1.2), 5))

    for i, run in enumerate(runs):
        means = agg[run]["means"]
        stds = agg[run]["stds"]
        if percent:
            means = means * 100
            stds = stds * 100
        offset = (i - n_runs / 2 + 0.5) * bar_width
        label = RUN_DISPLAY.get(run, run)
        color = RUN_COLORS.get(run, None)
        ax.bar(x + offset, means, bar_width * 0.9, yerr=stds,
               label=label, color=color, capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(level_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_aggregate_comparison(rows, output_path):
    """Bar chart comparing aggregate metrics (mean solve rate, IQM, mean opt gap) per run."""
    by_run = defaultdict(lambda: {"solve": [], "iqm": [], "opt_gap": []})
    for row in rows:
        run = row["run_name"]
        by_run[run]["solve"].append(float(row["mean_solve_rate"]))
        by_run[run]["iqm"].append(float(row["iqm_solve_rate"]))
        by_run[run]["opt_gap"].append(float(row["mean_optimality_gap"]))

    runs = sorted(by_run.keys())
    metrics = ["solve", "iqm", "opt_gap"]
    metric_labels = ["Mean Solve Rate", "IQM Solve Rate", "Mean Opt. Gap"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    bar_width = 0.6
    x = np.arange(len(runs))
    labels = [RUN_DISPLAY.get(r, r) for r in runs]
    colors = [RUN_COLORS.get(r, None) for r in runs]

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        means = [np.mean(by_run[r][metric]) for r in runs]
        stds = [np.std(by_run[r][metric]) for r in runs]
        n_seeds = [len(by_run[r][metric]) for r in runs]
        ax.bar(x, means, bar_width, yerr=stds, color=colors, capsize=5,
               edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        # Annotate n_seeds
        for xi, n in zip(x, n_seeds):
            ax.text(xi, ax.get_ylim()[1] * 0.02, f"n={n}", ha="center", fontsize=7, color="gray")

    fig.suptitle("Aggregate Evaluation Metrics (mean ± std across seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot eval_test_levels results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing eval_summary.csv")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save plots (default: results_dir/plots/)")
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, "eval_summary.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        return

    output_dir = args.output_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    rows = load_csv(csv_path)
    level_names = get_level_names_from_csv(rows)
    print(f"Loaded {len(rows)} eval rows, {len(level_names)} levels")

    # 1. Solve rate bar chart
    agg_sr = aggregate_by_run(rows, level_names, "solve_rate")
    plot_grouped_bars(agg_sr, level_names,
                      ylabel="Solve Rate (%)", title="Solve Rate per Test Level",
                      output_path=os.path.join(output_dir, "solve_rates.png"),
                      ylim=(0, 105), percent=True)

    # 2. Mean return bar chart
    agg_ret = aggregate_by_run(rows, level_names, "return")
    plot_grouped_bars(agg_ret, level_names,
                      ylabel="Mean Return", title="Mean Return per Test Level",
                      output_path=os.path.join(output_dir, "mean_returns.png"))

    # 3. Optimality gap bar chart
    agg_og = aggregate_by_run(rows, level_names, "opt_gap")
    plot_grouped_bars(agg_og, level_names,
                      ylabel="Optimality Gap", title="Optimality Gap per Test Level (lower = better)",
                      output_path=os.path.join(output_dir, "optimality_gaps.png"))

    # 4. Learnability bar chart
    agg_lr = aggregate_by_run(rows, level_names, "learnability")
    plot_grouped_bars(agg_lr, level_names,
                      ylabel="Learnability p(1-p)", title="Learnability per Test Level",
                      output_path=os.path.join(output_dir, "learnability.png"),
                      ylim=(0, 0.28))

    # 5. Path ratio bar chart
    agg_spr = aggregate_by_run(rows, level_names, "path_ratio")
    plot_grouped_bars(agg_spr, level_names,
                      ylabel="Path Ratio (agent/BFS)", title="Shortest Path Ratio per Test Level (1.0 = optimal)",
                      output_path=os.path.join(output_dir, "path_ratios.png"))

    # 6. Aggregate comparison
    plot_aggregate_comparison(rows, os.path.join(output_dir, "aggregate_comparison.png"))

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
