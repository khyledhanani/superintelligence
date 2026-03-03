"""
Compare PLR buffer levels from different training runs in VAE latent space.

Encodes buffer dump tokens through the VAE encoder, fits PCA on the combined
latent representations, and plots levels on PC1 vs PC2 colored by source and score.

Usage:
    python vae/buffer_latent_analysis.py \
        --buffer_dumps /tmp/buffer_dumps/vanilla_accel/0/buffer_dump.npz \
                       /tmp/buffer_dumps/cmaes_accel/0/buffer_dump.npz \
        --labels "Vanilla ACCEL" "CMA-ES ACCEL" \
        --vae_checkpoint_path vae/runs/.../checkpoints/checkpoint_80000.pkl \
        --vae_config_path vae/runs/.../config.yaml \
        --output_dir vae/plots/buffer_comparison/
"""
import argparse
import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, os.path.dirname(__file__))
from vae_model import CluttrVAE


def load_vae(checkpoint_path, config_path):
    """Load VAE model and params from checkpoint."""
    with open(config_path) as f:
        vae_cfg = yaml.safe_load(f)

    vae = CluttrVAE(
        vocab_size=vae_cfg["vocab_size"],
        embed_dim=vae_cfg["embed_dim"],
        latent_dim=vae_cfg["latent_dim"],
        seq_len=vae_cfg["seq_len"],
    )

    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt

    return vae, params, vae_cfg


def encode_tokens(vae, params, tokens, batch_size=512):
    """Encode token sequences to latent means. Batched to avoid OOM."""
    all_means = []
    for i in range(0, len(tokens), batch_size):
        batch = jnp.array(tokens[i:i + batch_size])
        mean, _ = vae.apply({"params": params}, batch, train=False, method=vae.encode)
        all_means.append(np.asarray(mean))
    return np.concatenate(all_means, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Compare buffer levels in VAE latent space")
    parser.add_argument("--buffer_dumps", nargs="+", required=True,
                        help="Paths to buffer_dump.npz files")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels for each buffer dump (same order)")
    parser.add_argument("--vae_checkpoint_path", type=str, required=True)
    parser.add_argument("--vae_config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="vae/plots/buffer_comparison/")
    parser.add_argument("--n_components", type=int, default=2)
    args = parser.parse_args()

    assert len(args.buffer_dumps) == len(args.labels), "Must have same number of dumps and labels"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load VAE
    print("[VAE] Loading model...")
    vae, params, vae_cfg = load_vae(args.vae_checkpoint_path, args.vae_config_path)
    print(f"[VAE] latent_dim={vae_cfg['latent_dim']}, seq_len={vae_cfg['seq_len']}")

    # Encode each buffer dump
    all_latents = []
    all_scores = []
    all_labels = []
    all_sizes = []

    for path, label in zip(args.buffer_dumps, args.labels):
        print(f"[Buffer] Loading {path} ({label})...")
        data = np.load(path)
        tokens = data["tokens"]
        scores = data["scores"]
        size = int(data["size"]) if "size" in data else len(tokens)
        tokens = tokens[:size]
        scores = scores[:size]

        print(f"  {size} levels, score range: [{scores.min():.3f}, {scores.max():.3f}]")

        latents = encode_tokens(vae, params, tokens)
        all_latents.append(latents)
        all_scores.append(scores)
        all_labels.extend([label] * size)
        all_sizes.append(size)

    combined_latents = np.concatenate(all_latents, axis=0)
    combined_scores = np.concatenate(all_scores, axis=0)

    # PCA
    print(f"[PCA] Fitting on {combined_latents.shape[0]} latent vectors...")
    pca = PCA(n_components=args.n_components)
    projected = pca.fit_transform(combined_latents)
    print(f"[PCA] Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}")

    # --- Plot 1: Colored by source ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    colors = plt.cm.tab10.colors
    offset = 0
    for i, (label, size) in enumerate(zip(args.labels, all_sizes)):
        axes[0].scatter(
            projected[offset:offset + size, 0],
            projected[offset:offset + size, 1],
            c=[colors[i % len(colors)]], alpha=0.4, s=8, label=label
        )
        offset += size

    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].set_title("Buffer Levels by Source")
    axes[0].legend(markerscale=3)

    # --- Plot 2: Colored by score ---
    valid_mask = np.isfinite(combined_scores) & (combined_scores > -1e6)
    sc = axes[1].scatter(
        projected[valid_mask, 0],
        projected[valid_mask, 1],
        c=combined_scores[valid_mask],
        cmap="viridis", alpha=0.4, s=8
    )
    plt.colorbar(sc, ax=axes[1], label="Score (regret)")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[1].set_title("Buffer Levels by Score")

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "buffer_pca_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")

    # --- Plot 3: Per-source score distributions ---
    fig, ax = plt.subplots(figsize=(10, 5))
    offset = 0
    for i, (label, size) in enumerate(zip(args.labels, all_sizes)):
        scores_i = combined_scores[offset:offset + size]
        valid = scores_i[np.isfinite(scores_i) & (scores_i > -1e6)]
        ax.hist(valid, bins=50, alpha=0.5, label=f"{label} (n={len(valid)})", color=colors[i % len(colors)])
        offset += size
    ax.set_xlabel("Score (regret)")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution by Source")
    ax.legend()
    plt.tight_layout()
    out_path2 = os.path.join(args.output_dir, "buffer_score_distributions.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path2}")

    # Save PCA results for further analysis
    np.savez_compressed(
        os.path.join(args.output_dir, "pca_results.npz"),
        projected=projected,
        scores=combined_scores,
        labels=np.array(all_labels),
        explained_variance=pca.explained_variance_ratio_,
        components=pca.components_,
        mean=pca.mean_,
    )
    print(f"[Saved] {os.path.join(args.output_dir, 'pca_results.npz')}")


if __name__ == "__main__":
    main()
