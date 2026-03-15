"""
Analyze whether the trajectory VAE latent space correlates with regret.

Compares:
  1. Trajectory VAE latent z_traj → regret correlation
  2. Level VAE latent z_level → regret correlation (baseline)

Outputs:
  - Per-dimension Spearman correlations
  - Linear regression R² (z → regret)
  - PCA visualization colored by regret
  - Summary statistics

Usage:
    python trajectory/analyze_correlation.py \
        --data_path trajectory/data/trajectories.npz \
        --traj_vae_path trajectory/checkpoints/run1/best_model.pkl \
        --level_vae_path /path/to/vae_checkpoint.pkl \
        --level_vae_config /path/to/vae_config.yaml \
        --output_dir trajectory/analysis
"""
import argparse
import os
import sys
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vae"))

from trajectory_vae import TrajectoryVAE


def encode_trajectories(model, params, trajectories, batch_size=512):
    """Encode trajectories to latent vectors."""
    @jax.jit
    def _encode(batch):
        return model.apply({"params": params}, batch, train=False, method=model.encode)

    all_means = []
    for i in range(0, len(trajectories), batch_size):
        batch = jnp.array(trajectories[i:i+batch_size])
        means, _ = _encode(batch)
        all_means.append(np.array(means))
    return np.concatenate(all_means, axis=0)


def encode_levels(level_vae, level_params, tokens, batch_size=512):
    """Encode level tokens to latent vectors."""
    @jax.jit
    def _encode(batch):
        return level_vae.apply({"params": level_params}, batch, train=False, method=level_vae.encode)

    all_means = []
    for i in range(0, len(tokens), batch_size):
        batch = jnp.array(tokens[i:i+batch_size])
        means, _ = _encode(batch)
        all_means.append(np.array(means))
    return np.concatenate(all_means, axis=0)


def compute_correlations(z, regret, name=""):
    """Compute per-dimension and aggregate correlations."""
    n_dims = z.shape[1]

    # Per-dimension Spearman correlation
    spearman_per_dim = []
    for d in range(n_dims):
        rho, pval = stats.spearmanr(z[:, d], regret)
        spearman_per_dim.append((d, rho, pval))
    spearman_per_dim.sort(key=lambda x: -abs(x[1]))

    # Linear regression R² (5-fold CV)
    ridge = Ridge(alpha=1.0)
    r2_scores = cross_val_score(ridge, z, regret, cv=5, scoring="r2")

    print(f"\n{'='*60}")
    print(f"  {name} ({n_dims} dims)")
    print(f"{'='*60}")
    print(f"  Linear R² (5-fold CV): {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"  Top 10 Spearman correlations (|rho|):")
    for d, rho, pval in spearman_per_dim[:10]:
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"    dim {d:3d}: rho={rho:+.4f} (p={pval:.2e}) {sig}")

    # Mean absolute Spearman
    mean_abs_rho = np.mean([abs(x[1]) for x in spearman_per_dim])
    max_abs_rho = max(abs(x[1]) for x in spearman_per_dim)
    print(f"  Mean |rho|: {mean_abs_rho:.4f}")
    print(f"  Max  |rho|: {max_abs_rho:.4f}")

    return {
        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std(),
        "r2_scores": r2_scores,
        "mean_abs_rho": mean_abs_rho,
        "max_abs_rho": max_abs_rho,
        "spearman_per_dim": spearman_per_dim,
    }


def plot_pca_comparison(z_traj, z_level, regret, output_dir):
    """PCA visualization of both latent spaces colored by regret."""
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, z, title in [
        (axes[0], z_traj, "Trajectory VAE"),
        (axes[1], z_level, "Level VAE"),
    ]:
        if z is None:
            ax.set_title(f"{title}\n(not available)")
            continue
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z)
        sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=regret, cmap="viridis",
                        alpha=0.3, s=5, rasterized=True)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"{title}\nPCA colored by regret")
        plt.colorbar(sc, ax=ax, label="regret")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved PCA comparison to {output_dir}/pca_comparison.png")


def plot_correlation_bars(results_traj, results_level, output_dir):
    """Bar chart comparing key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # R² comparison
    ax = axes[0]
    names = []
    r2s = []
    r2_errs = []
    if results_traj:
        names.append("Trajectory VAE")
        r2s.append(results_traj["r2_mean"])
        r2_errs.append(results_traj["r2_std"])
    if results_level:
        names.append("Level VAE")
        r2s.append(results_level["r2_mean"])
        r2_errs.append(results_level["r2_std"])
    ax.bar(names, r2s, yerr=r2_errs, capsize=5, color=["#2196F3", "#FF9800"][:len(names)])
    ax.set_ylabel("R² (5-fold CV)")
    ax.set_title("Linear Regression: z → regret")
    ax.set_ylim(bottom=0)

    # Mean |rho|
    ax = axes[1]
    vals = []
    if results_traj:
        vals.append(results_traj["mean_abs_rho"])
    if results_level:
        vals.append(results_level["mean_abs_rho"])
    ax.bar(names, vals, color=["#2196F3", "#FF9800"][:len(names)])
    ax.set_ylabel("Mean |Spearman ρ|")
    ax.set_title("Average per-dim correlation")
    ax.set_ylim(bottom=0)

    # Max |rho|
    ax = axes[2]
    vals = []
    if results_traj:
        vals.append(results_traj["max_abs_rho"])
    if results_level:
        vals.append(results_level["max_abs_rho"])
    ax.bar(names, vals, color=["#2196F3", "#FF9800"][:len(names)])
    ax.set_ylabel("Max |Spearman ρ|")
    ax.set_title("Best single-dim correlation")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved correlation comparison to {output_dir}/correlation_comparison.png")


def plot_trajectory_features(trajectories, regret, output_dir):
    """Plot simple trajectory features vs regret (no VAE needed)."""
    n = len(trajectories)

    # Feature 1: trajectory length (non-zero positions)
    traj_lengths = (trajectories > 0).sum(axis=1)

    # Feature 2: number of unique positions visited
    unique_positions = np.array([len(np.unique(t[t > 0])) for t in trajectories])

    # Feature 3: revisit ratio = total_steps / unique_positions
    revisit_ratio = traj_lengths / np.maximum(unique_positions, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, feat, name in [
        (axes[0], traj_lengths, "Trajectory length"),
        (axes[1], unique_positions, "Unique positions"),
        (axes[2], revisit_ratio, "Revisit ratio"),
    ]:
        ax.scatter(feat, regret, alpha=0.1, s=3, rasterized=True)
        rho, pval = stats.spearmanr(feat, regret)
        ax.set_xlabel(name)
        ax.set_ylabel("Regret")
        ax.set_title(f"{name} vs Regret\nSpearman ρ={rho:.4f} (p={pval:.2e})")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_features_vs_regret.png"), dpi=150)
    plt.close()
    print(f"  Saved trajectory features plot to {output_dir}/trajectory_features_vs_regret.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory vs level VAE latent-regret correlation")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to trajectories.npz from collect_trajectories.py")
    parser.add_argument("--traj_vae_path", type=str, default=None,
                        help="Path to trained trajectory VAE checkpoint")
    parser.add_argument("--level_vae_path", type=str, default=None,
                        help="Path to level VAE checkpoint (for baseline comparison)")
    parser.add_argument("--level_vae_config", type=str, default=None,
                        help="Path to level VAE config.yaml")
    parser.add_argument("--output_dir", type=str, default="trajectory/analysis")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of samples (for speed)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    print("[1/4] Loading data...")
    data = np.load(args.data_path, allow_pickle=True)
    trajectories = data["trajectories"]
    regret = data["regret"]
    tokens = data["tokens"] if "tokens" in data else None

    if args.max_samples and len(trajectories) > args.max_samples:
        idx = np.random.RandomState(42).choice(len(trajectories), args.max_samples, replace=False)
        trajectories = trajectories[idx]
        regret = regret[idx]
        if tokens is not None:
            tokens = tokens[idx]

    print(f"  {len(trajectories)} samples, regret range [{regret.min():.4f}, {regret.max():.4f}]")

    # --- Raw trajectory features (no VAE needed) ---
    print("\n[2/4] Analyzing raw trajectory features...")
    plot_trajectory_features(trajectories, regret, args.output_dir)

    # Raw feature correlations
    traj_lengths = (trajectories > 0).sum(axis=1)
    unique_pos = np.array([len(np.unique(t[t > 0])) for t in trajectories])
    revisit_ratio = traj_lengths / np.maximum(unique_pos, 1)

    for feat, name in [(traj_lengths, "traj_length"), (unique_pos, "unique_positions"), (revisit_ratio, "revisit_ratio")]:
        rho, pval = stats.spearmanr(feat, regret)
        print(f"  {name:20s}: Spearman ρ = {rho:+.4f} (p={pval:.2e})")

    # --- Trajectory VAE encoding ---
    results_traj = None
    z_traj = None
    if args.traj_vae_path:
        print("\n[3/4] Encoding with trajectory VAE...")
        with open(args.traj_vae_path, "rb") as f:
            traj_ckpt = pickle.load(f)
        traj_cfg = traj_ckpt["config"]
        traj_model = TrajectoryVAE(
            vocab_size=170,
            embed_dim=traj_cfg["embed_dim"],
            latent_dim=traj_cfg["latent_dim"],
            max_steps=int(data["max_steps"]),
        )
        z_traj = encode_trajectories(traj_model, traj_ckpt["params"], trajectories)
        results_traj = compute_correlations(z_traj, regret, "Trajectory VAE")
    else:
        print("\n[3/4] Skipping trajectory VAE (no --traj_vae_path)")

    # --- Level VAE encoding (baseline) ---
    results_level = None
    z_level = None
    if args.level_vae_path and args.level_vae_config and tokens is not None:
        print("\n[4/4] Encoding with level VAE (baseline)...")
        import yaml
        from vae_model import CluttrVAE
        with open(args.level_vae_config) as f:
            lvl_cfg = yaml.safe_load(f)
        level_vae = CluttrVAE(
            vocab_size=lvl_cfg["vocab_size"], embed_dim=lvl_cfg["embed_dim"],
            latent_dim=lvl_cfg["latent_dim"], seq_len=lvl_cfg["seq_len"],
        )
        with open(args.level_vae_path, "rb") as f:
            lvl_ckpt = pickle.load(f)
        lvl_params = lvl_ckpt["params"] if isinstance(lvl_ckpt, dict) and "params" in lvl_ckpt else lvl_ckpt
        z_level = encode_levels(level_vae, lvl_params, tokens)
        results_level = compute_correlations(z_level, regret, "Level VAE (baseline)")
    else:
        print("\n[4/4] Skipping level VAE baseline")

    # --- Comparison plots ---
    if z_traj is not None or z_level is not None:
        plot_pca_comparison(z_traj, z_level, regret, args.output_dir)
    if results_traj or results_level:
        plot_correlation_bars(results_traj, results_level, args.output_dir)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if results_traj:
        print(f"  Trajectory VAE:  R² = {results_traj['r2_mean']:.4f} ± {results_traj['r2_std']:.4f}, "
              f"mean |ρ| = {results_traj['mean_abs_rho']:.4f}")
    if results_level:
        print(f"  Level VAE:       R² = {results_level['r2_mean']:.4f} ± {results_level['r2_std']:.4f}, "
              f"mean |ρ| = {results_level['mean_abs_rho']:.4f}")
    if results_traj and results_level:
        r2_diff = results_traj["r2_mean"] - results_level["r2_mean"]
        print(f"\n  R² difference: {r2_diff:+.4f} ({'trajectory better' if r2_diff > 0 else 'level better'})")

    # Save results
    results_path = os.path.join(args.output_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump({
            "trajectory_vae": results_traj,
            "level_vae": results_level,
        }, f)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
