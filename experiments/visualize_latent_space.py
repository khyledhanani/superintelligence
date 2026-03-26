"""Visualize LLM-generated seeds and buffer levels in VAE latent space (PCA projection).

Encodes both buffer levels and LLM seeds through the VAE encoder, projects to
PCA space, and creates scatter plots to answer:
  - Do LLM seeds cluster separately from buffer levels?
  - Are LLM seeds in dense or sparse regions of the latent space?
  - Would Gaussian sampling around LLM seeds be useful?

Also plots Gaussian contours around LLM seed clusters to show what
vae_noise mutations would cover.

Usage:
    python experiments/visualize_latent_space.py \
        --buffer_npz /tmp/buffer_dumps/run/seed/buffer_dump_30k.npz \
        --seeds_dir experiments/seeds/run1 \
        --vae_checkpoint_path vae/runs/best/checkpoint.pkl \
        --vae_config_path vae/runs/best/config.yaml \
        --output_dir experiments/plots

Outputs:
    latent_pca_2d.png       — 2D PCA scatter: buffer (grey) vs LLM seeds (red)
    latent_pca_3d.png       — 3D PCA scatter (optional)
    latent_pca_with_gaussian.png — Same + Gaussian contours at sigma=0.3,0.5,1.0
    latent_stats.json       — PCA explained variance, distances, cluster stats
"""
import argparse
import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [
    os.path.join(_project_root, "src"),
    os.path.join(_project_root, "vae"),
    _project_root,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vae_level_utils import level_to_tokens, tokens_to_level


def load_vae(ckpt_path, cfg_path):
    import yaml
    from vae_model import CluttrVAE

    with open(cfg_path) as f:
        vae_cfg = yaml.safe_load(f)

    vae = CluttrVAE(
        vocab_size=vae_cfg["vocab_size"],
        embed_dim=vae_cfg["embed_dim"],
        latent_dim=vae_cfg["latent_dim"],
        seq_len=vae_cfg["seq_len"],
    )

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    vae_params = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt

    return vae, vae_params, vae_cfg["latent_dim"]


def encode_tokens_batch(vae, vae_params, tokens_array):
    """Encode a batch of token sequences to latent means.

    Args:
        tokens_array: (N, 52) int32 array

    Returns:
        (N, latent_dim) array of latent means
    """
    tokens_jax = jnp.array(tokens_array, dtype=jnp.int32)
    # Batch encode
    mean, logvar = vae.apply(
        {"params": vae_params}, tokens_jax,
        train=False, method=vae.encode,
    )
    return np.asarray(mean), np.asarray(logvar)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load VAE ---
    print("[1/4] Loading VAE...")
    vae, vae_params, latent_dim = load_vae(args.vae_checkpoint_path, args.vae_config_path)

    # --- Load buffer tokens ---
    print("[2/4] Loading buffer...")
    buf_data = np.load(args.buffer_npz, allow_pickle=True)
    buf_tokens = buf_data["tokens"]
    buf_scores = buf_data["scores"]
    buf_size = int(buf_data["size"])
    buf_tokens = buf_tokens[:buf_size]
    buf_scores = buf_scores[:buf_size]
    print(f"  Buffer: {buf_size} levels")

    # --- Load seed tokens ---
    print("[3/4] Loading LLM seeds...")
    seeds_npz = os.path.join(args.seeds_dir, "seeds.npz")
    seeds_data = np.load(seeds_npz, allow_pickle=True)
    seed_tokens = seeds_data["tokens"]
    n_seeds = len(seed_tokens)
    print(f"  Seeds: {n_seeds} levels")

    # Also load mutants if available
    mutant_tokens = None
    inject_path = os.path.join(args.seeds_dir, "..", "results")
    # Check for injected_levels.npz in output_dir or nearby
    for candidate in [
        os.path.join(args.output_dir, "injected_levels.npz"),
        os.path.join(args.seeds_dir, "injected_levels.npz"),
    ]:
        if os.path.exists(candidate):
            mut_data = np.load(candidate, allow_pickle=True)
            mutant_tokens = mut_data["tokens"]
            print(f"  Mutants: {len(mutant_tokens)} levels (from {candidate})")
            break

    # --- Encode all ---
    print("[4/4] Encoding to latent space...")
    buf_z, buf_logvar = encode_tokens_batch(vae, vae_params, buf_tokens)
    seed_z, seed_logvar = encode_tokens_batch(vae, vae_params, seed_tokens)

    mut_z = None
    if mutant_tokens is not None:
        mut_z, _ = encode_tokens_batch(vae, vae_params, mutant_tokens)

    # --- PCA ---
    pca = PCA(n_components=min(10, latent_dim))
    all_z = np.concatenate([buf_z, seed_z])
    pca.fit(all_z)

    buf_pca = pca.transform(buf_z)
    seed_pca = pca.transform(seed_z)
    mut_pca = pca.transform(mut_z) if mut_z is not None else None

    explained = pca.explained_variance_ratio_
    print(f"  PCA explained variance (first 3): {explained[:3]}")

    # --- Compute statistics ---
    # Distance of each LLM seed to nearest buffer level (in full latent space)
    from scipy.spatial.distance import cdist
    dists = cdist(seed_z, buf_z, metric="euclidean")
    nearest_dists = dists.min(axis=1)

    # Buffer pairwise stats (subsample for speed)
    n_sub = min(500, buf_size)
    buf_sub = buf_z[np.random.choice(buf_size, n_sub, replace=False)]
    buf_pairwise = cdist(buf_sub, buf_sub, metric="euclidean")
    np.fill_diagonal(buf_pairwise, np.inf)
    buf_nn_dists = buf_pairwise.min(axis=1)

    stats = {
        "pca_explained_variance": explained.tolist(),
        "n_buffer": buf_size,
        "n_seeds": n_seeds,
        "latent_dim": latent_dim,
        "seed_nearest_buffer_dist": {
            "mean": float(nearest_dists.mean()),
            "std": float(nearest_dists.std()),
            "min": float(nearest_dists.min()),
            "max": float(nearest_dists.max()),
            "values": nearest_dists.tolist(),
        },
        "buffer_nn_dist": {
            "mean": float(buf_nn_dists.mean()),
            "std": float(buf_nn_dists.std()),
        },
        "seed_centroid_dist_from_buffer_centroid": float(
            np.linalg.norm(seed_z.mean(axis=0) - buf_z.mean(axis=0))
        ),
    }
    with open(os.path.join(args.output_dir, "latent_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # --- Plot 1: 2D PCA scatter ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Buffer levels colored by score
    sc = ax.scatter(
        buf_pca[:, 0], buf_pca[:, 1],
        c=buf_scores, cmap="viridis", s=8, alpha=0.4, label=f"Buffer ({buf_size})",
    )
    plt.colorbar(sc, ax=ax, label="Buffer Score (regret/SFL)")

    # LLM seeds
    ax.scatter(
        seed_pca[:, 0], seed_pca[:, 1],
        c="red", s=100, marker="*", edgecolors="black", linewidths=0.5,
        label=f"LLM Seeds ({n_seeds})", zorder=5,
    )

    # Mutants
    if mut_pca is not None:
        ax.scatter(
            mut_pca[:, 0], mut_pca[:, 1],
            c="orange", s=15, alpha=0.5, marker="^",
            label=f"Mutants ({len(mut_pca)})", zorder=4,
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
    ax.set_title("VAE Latent Space: Buffer vs LLM Seeds (PCA)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "latent_pca_2d.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved latent_pca_2d.png")

    # --- Plot 2: With Gaussian contours ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.scatter(
        buf_pca[:, 0], buf_pca[:, 1],
        c="lightgrey", s=5, alpha=0.3, label=f"Buffer ({buf_size})",
    )

    # Draw Gaussian ellipses around each LLM seed
    sigmas = [0.3, 0.5, 1.0]
    colors_sigma = ["#ff000030", "#ff000020", "#ff000010"]
    for sigma, color in zip(sigmas, colors_sigma):
        for j in range(n_seeds):
            # Approximate: sigma in latent space → scale in PCA space
            # The PCA components have unit variance after fitting on buffer
            # So sigma in latent ≈ sigma in PCA (rough approximation)
            ellipse = Ellipse(
                xy=(seed_pca[j, 0], seed_pca[j, 1]),
                width=2 * sigma * np.sqrt(explained[0]) * latent_dim**0.5,
                height=2 * sigma * np.sqrt(explained[1]) * latent_dim**0.5,
                facecolor=color, edgecolor="red", linewidth=0.5,
                linestyle="--" if sigma > 0.5 else "-",
            )
            ax.add_patch(ellipse)

    ax.scatter(
        seed_pca[:, 0], seed_pca[:, 1],
        c="red", s=100, marker="*", edgecolors="black", linewidths=0.5,
        label=f"LLM Seeds ({n_seeds})", zorder=5,
    )

    ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
    ax.set_title("VAE Latent Space: Gaussian Sampling Regions Around LLM Seeds")
    ax.legend(loc="upper right")

    # Add text annotation for sigma values
    ax.text(0.02, 0.02,
            f"Ellipses show σ={sigmas} noise regions\n"
            f"Seed→buffer centroid dist: {stats['seed_centroid_dist_from_buffer_centroid']:.2f}\n"
            f"Mean seed→nearest buffer: {stats['seed_nearest_buffer_dist']['mean']:.2f}",
            transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "latent_pca_with_gaussian.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved latent_pca_with_gaussian.png")

    # --- Plot 3: PC1 vs PC3 for additional perspective ---
    if explained.shape[0] >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, (pc_x, pc_y, ax) in enumerate([
            (0, 1, axes[0]),
            (0, 2, axes[1]),
        ]):
            sc = ax.scatter(
                buf_pca[:, pc_x], buf_pca[:, pc_y],
                c=buf_scores, cmap="viridis", s=5, alpha=0.3,
            )
            ax.scatter(
                seed_pca[:, pc_x], seed_pca[:, pc_y],
                c="red", s=100, marker="*", edgecolors="black", linewidths=0.5,
                zorder=5,
            )
            if mut_pca is not None:
                ax.scatter(
                    mut_pca[:, pc_x], mut_pca[:, pc_y],
                    c="orange", s=10, alpha=0.4, marker="^", zorder=4,
                )
            ax.set_xlabel(f"PC{pc_x+1} ({explained[pc_x]:.1%})")
            ax.set_ylabel(f"PC{pc_y+1} ({explained[pc_y]:.1%})")

        plt.colorbar(sc, ax=axes[1], label="Score")
        fig.suptitle("VAE Latent Space: Multiple PCA Views")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "latent_pca_multi.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved latent_pca_multi.png")

    print(f"\n[Done] Plots and stats saved to {args.output_dir}/")
    print(f"  Key finding: LLM seeds are {stats['seed_nearest_buffer_dist']['mean']:.2f} "
          f"from nearest buffer level (buffer NN avg: {stats['buffer_nn_dist']['mean']:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LLM seeds in VAE latent space")
    parser.add_argument("--buffer_npz", type=str, required=True)
    parser.add_argument("--seeds_dir", type=str, required=True)
    parser.add_argument("--vae_checkpoint_path", type=str, required=True)
    parser.add_argument("--vae_config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/plots")
    args = parser.parse_args()
    main(args)
