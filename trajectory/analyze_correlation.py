"""
Analyze whether agent trajectory behavior correlates with regret.

Extracts behavioral features directly from the raw position sequence
(no VAE needed) and tests their correlation with regret scores.

Features capture the spatial structure of the agent's path:
  - Path efficiency (how directly agent reaches goal)
  - Backtracking / revisit patterns
  - Spatial coverage and entropy
  - Wall-bumping (stalling on same cell)
  - Directional changes (turning frequency)
  - Goal-progress curve shape

Usage:
    python trajectory/analyze_correlation.py \
        --data_path trajectory/data/trajectories.npz \
        --output_dir trajectory/analysis
"""
import argparse
import os
import sys

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID_SIZE = 13


def pos_to_xy(pos):
    """Convert 1-based position index to (x, y). 0 → (-1, -1) for padding."""
    x = (pos - 1) % GRID_SIZE
    y = (pos - 1) // GRID_SIZE
    # mask padding
    valid = pos > 0
    x = np.where(valid, x, -1)
    y = np.where(valid, y, -1)
    return x, y


def extract_features(trajectories, reached_goal):
    """Extract behavioral features from raw position trajectories.

    Args:
        trajectories: (N, max_steps) int32 position tokens (1-based, 0=pad)
        reached_goal: (N,) bool

    Returns:
        features: dict of {name: (N,) array}
    """
    N, T = trajectories.shape
    features = {}

    # Basic: trajectory length, unique positions
    mask = trajectories > 0  # (N, T)
    traj_len = mask.sum(axis=1).astype(float)
    features["traj_length"] = traj_len

    unique_pos = np.array([len(np.unique(t[t > 0])) for t in trajectories], dtype=float)
    features["unique_positions"] = unique_pos

    # Revisit ratio: steps / unique cells (>1 means backtracking)
    features["revisit_ratio"] = traj_len / np.maximum(unique_pos, 1)

    # Coverage: fraction of 13x13 grid visited
    features["grid_coverage"] = unique_pos / (GRID_SIZE * GRID_SIZE)

    # Convert to xy for spatial features
    # Work per-trajectory for variable-length sequences
    stall_counts = np.zeros(N)
    backtrack_counts = np.zeros(N)
    turn_counts = np.zeros(N)
    total_displacement = np.zeros(N)
    max_displacement = np.zeros(N)
    spatial_entropy = np.zeros(N)
    mean_step_size = np.zeros(N)
    goal_pos_x = np.zeros(N)
    goal_pos_y = np.zeros(N)
    start_goal_dist = np.zeros(N)
    final_goal_dist = np.zeros(N)
    path_straightness = np.zeros(N)

    for i in range(N):
        t = trajectories[i]
        valid = t > 0
        n_steps = valid.sum()
        if n_steps < 2:
            continue

        pos = t[valid]
        x = (pos - 1) % GRID_SIZE
        y = (pos - 1) // GRID_SIZE

        # Stalling: consecutive identical positions (agent bumped a wall)
        stall_counts[i] = np.sum(pos[1:] == pos[:-1])

        # Step displacements
        dx = np.diff(x).astype(float)
        dy = np.diff(y).astype(float)
        step_sizes = np.abs(dx) + np.abs(dy)  # Manhattan per step
        mean_step_size[i] = step_sizes.mean() if len(step_sizes) > 0 else 0

        # Total Manhattan displacement from start
        total_displacement[i] = abs(x[-1] - x[0]) + abs(y[-1] - y[0])

        # Max displacement from start during trajectory
        disp_from_start = np.abs(x - x[0]) + np.abs(y - y[0])
        max_displacement[i] = disp_from_start.max()

        # Backtracking: how many times agent revisits a cell it already visited
        seen = set()
        bt = 0
        for p in pos:
            if p in seen:
                bt += 1
            seen.add(p)
        backtrack_counts[i] = bt

        # Turning: direction changes (count transitions between different move directions)
        if len(dx) >= 2:
            # Direction as (sign(dx), sign(dy))
            dirs = np.stack([np.sign(dx), np.sign(dy)], axis=1)
            # Count direction changes (excluding stalls where dir=(0,0))
            moving = (dirs != 0).any(axis=1)
            moving_dirs = dirs[moving]
            if len(moving_dirs) >= 2:
                turns = np.sum((moving_dirs[1:] != moving_dirs[:-1]).any(axis=1))
                turn_counts[i] = turns

        # Spatial entropy of visited positions
        _, counts = np.unique(pos, return_counts=True)
        probs = counts / counts.sum()
        spatial_entropy[i] = -np.sum(probs * np.log2(probs + 1e-10))

        # Path straightness: displacement / path_length
        if n_steps > 1:
            path_straightness[i] = total_displacement[i] / max(n_steps - 1, 1)

        # Goal position (last unique position before padding — for solved levels this is the goal)
        if reached_goal[i]:
            goal_pos_x[i] = x[-1]
            goal_pos_y[i] = y[-1]
            start_goal_dist[i] = abs(x[-1] - x[0]) + abs(y[-1] - y[0])
            # Final distance to goal (should be 0 for solved)
            final_goal_dist[i] = 0
        else:
            goal_pos_x[i] = -1
            goal_pos_y[i] = -1

    features["stall_count"] = stall_counts
    features["stall_fraction"] = stall_counts / np.maximum(traj_len - 1, 1)
    features["backtrack_count"] = backtrack_counts
    features["backtrack_fraction"] = backtrack_counts / np.maximum(traj_len, 1)
    features["turn_count"] = turn_counts
    features["turn_rate"] = turn_counts / np.maximum(traj_len - 2, 1)
    features["total_displacement"] = total_displacement
    features["max_displacement"] = max_displacement
    features["spatial_entropy"] = spatial_entropy
    features["mean_step_size"] = mean_step_size
    features["path_straightness"] = path_straightness

    # Solved-specific: path efficiency (start-goal distance / steps taken)
    # Only meaningful for solved levels
    solved = reached_goal.astype(bool)
    path_efficiency = np.zeros(N)
    path_efficiency[solved] = start_goal_dist[solved] / np.maximum(traj_len[solved], 1)
    features["path_efficiency_solved"] = path_efficiency

    return features


def plot_feature_correlations(features, regret, output_dir, reached_goal=None, tag="all"):
    """Scatter plots of each feature vs regret with Spearman rho."""
    feat_names = [k for k in features.keys() if not k.endswith("_solved")]
    n_feats = len(feat_names)
    n_cols = 4
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for j, name in enumerate(feat_names):
        ax = axes[j]
        feat = features[name]
        ax.scatter(feat, regret, alpha=0.1, s=3, rasterized=True)
        rho, pval = stats.spearmanr(feat, regret)
        ax.set_xlabel(name.replace("_", " "))
        ax.set_ylabel("Regret")
        ax.set_title(f"ρ={rho:+.3f} (p={pval:.1e})", fontsize=9)

    # Hide unused axes
    for j in range(n_feats, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Trajectory Features vs Regret — {tag} (N={len(regret)})", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"features_vs_regret_{tag}.png"), dpi=150)
    plt.close()


def run_regression_analysis(features, regret, output_dir, tag="all"):
    """Ridge regression: all features → regret, with feature importances."""
    feat_names = [k for k in features.keys() if not k.endswith("_solved")]
    X = np.column_stack([features[k] for k in feat_names])

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5-fold CV R²
    ridge = Ridge(alpha=1.0)
    r2_scores = cross_val_score(ridge, X_scaled, regret, cv=5, scoring="r2")

    # Fit on all data for feature importances
    ridge.fit(X_scaled, regret)
    coefs = ridge.coef_

    # Per-feature Spearman
    spearman = []
    for j, name in enumerate(feat_names):
        rho, pval = stats.spearmanr(X[:, j], regret)
        spearman.append((name, rho, pval, coefs[j]))
    spearman.sort(key=lambda x: -abs(x[1]))

    print(f"\n  --- {tag} (N={len(regret)}) ---")
    print(f"  Combined Ridge R² (5-fold CV): {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"  {'Feature':>25s} | {'Spearman ρ':>12s} | {'p-value':>10s} | {'Ridge coef':>10s}")
    print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")
    for name, rho, pval, coef in spearman:
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {name:>25s} | {rho:+12.4f} | {pval:10.2e} | {coef:+10.4f} {sig}")

    # Feature importance bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    names_sorted = [x[0] for x in spearman]
    rhos_sorted = [x[1] for x in spearman]
    colors = ["#2196F3" if r > 0 else "#F44336" for r in rhos_sorted]
    ax.barh(range(len(names_sorted)), rhos_sorted, color=colors)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels([n.replace("_", " ") for n in names_sorted], fontsize=9)
    ax.set_xlabel("Spearman ρ with regret")
    ax.set_title(f"Feature-Regret Correlations — {tag}\nCombined R²={r2_scores.mean():.4f}")
    ax.axvline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"feature_importance_{tag}.png"), dpi=150)
    plt.close()

    return {
        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std(),
        "spearman": spearman,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory behavior vs regret")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to trajectories.npz from collect_trajectories.py")
    parser.add_argument("--output_dir", type=str, default="trajectory/analysis")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of samples (for speed)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    print("[1/3] Loading data...")
    data = np.load(args.data_path, allow_pickle=True)
    trajectories = data["trajectories"]
    regret = data["regret"]
    reached_goal = data["reached_goal"] if "reached_goal" in data else np.zeros(len(trajectories), dtype=bool)

    if args.max_samples and len(trajectories) > args.max_samples:
        idx = np.random.RandomState(42).choice(len(trajectories), args.max_samples, replace=False)
        trajectories = trajectories[idx]
        regret = regret[idx]
        reached_goal = reached_goal[idx]

    print(f"  {len(trajectories)} samples, regret range [{regret.min():.4f}, {regret.max():.4f}]")
    print(f"  Reached goal: {reached_goal.sum()}/{len(reached_goal)} ({reached_goal.mean()*100:.1f}%)")

    # --- Extract features ---
    print("\n[2/3] Extracting trajectory features...")
    features = extract_features(trajectories, reached_goal)
    print(f"  Extracted {len(features)} features")

    # --- Analysis: all levels ---
    print("\n[3/3] Correlation analysis...")
    plot_feature_correlations(features, regret, args.output_dir, reached_goal, tag="all")
    results_all = run_regression_analysis(features, regret, args.output_dir, tag="all")

    # --- Analysis: solved only ---
    solved = reached_goal.astype(bool)
    n_solved = solved.sum()
    if n_solved > 30:
        features_solved = {k: v[solved] for k, v in features.items()}
        regret_solved = regret[solved]
        plot_feature_correlations(features_solved, regret_solved, args.output_dir, tag="solved")
        results_solved = run_regression_analysis(features_solved, regret_solved, args.output_dir, tag="solved")
    else:
        print(f"\n  Too few solved levels ({n_solved}) for solved-only analysis")
        results_solved = None

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  All levels:    R² = {results_all['r2_mean']:.4f} ± {results_all['r2_std']:.4f}")
    if results_solved:
        print(f"  Solved only:   R² = {results_solved['r2_mean']:.4f} ± {results_solved['r2_std']:.4f}")
    print(f"\n  Plots saved to {args.output_dir}/")

    # Save results
    import pickle
    results_path = os.path.join(args.output_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump({"all": results_all, "solved": results_solved}, f)


if __name__ == "__main__":
    main()
