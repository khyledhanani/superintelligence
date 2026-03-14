"""
Diagnostics for the adapter: visualisations, perturbation tests, convergence analysis.

1. UMAP visualisation: z vs z' = z + adapter(z), coloured by regret
2. Perturbation diagnostic: does solve agreement improve in z'-space?
3. Reconstruction quality: how well does the decoder reconstruct after adapter correction?

Usage:
    python adapter/diagnostics.py \
        --data_path /tmp/adapter_data/train_data.npz \
        --adapter_path /tmp/adapter_data/adapter.pkl \
        --vae_checkpoint_path /tmp/vae_beta10/checkpoint_500000.pkl \
        --vae_config_path /tmp/vae_beta10/config.yaml \
        --output_dir /tmp/adapter_data/diagnostics/
"""
import argparse
import os
import sys
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vae"))

from adapter_model import AdapterMLP
from vae_model import CluttrVAE


def load_adapter(adapter_path):
    with open(adapter_path, "rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["config"]
    model = AdapterMLP(
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
    )
    return model, ckpt["params"], cfg


def load_vae_decoder(vae_checkpoint_path, vae_config_path):
    with open(vae_config_path) as f:
        vae_cfg = yaml.safe_load(f)
    vae = CluttrVAE(
        vocab_size=vae_cfg["vocab_size"],
        embed_dim=vae_cfg["embed_dim"],
        latent_dim=vae_cfg["latent_dim"],
        seq_len=vae_cfg["seq_len"],
    )
    with open(vae_checkpoint_path, "rb") as f:
        vae_ckpt = pickle.load(f)
    vae_params = vae_ckpt["params"] if isinstance(vae_ckpt, dict) and "params" in vae_ckpt else vae_ckpt
    return vae, vae_params, vae_cfg


def plot_pca_comparison(z, z_prime, regret, output_dir):
    """Plot PCA projection of z and z' side by side, coloured by regret."""
    from sklearn.decomposition import PCA

    n = min(len(z), 5000)
    idx = np.random.choice(len(z), n, replace=False)
    z_sub = np.array(z[idx])
    z_prime_sub = np.array(z_prime[idx])
    regret_sub = np.array(regret[idx])

    # Fit PCA on original z, project both z and z' using same PCA
    print("  Fitting PCA on z...")
    pca = PCA(n_components=2, random_state=42)
    z_2d = pca.fit_transform(z_sub)
    z_prime_2d = pca.transform(z_prime_sub)  # same projection for comparability
    explained = pca.explained_variance_ratio_

    # Also fit a separate PCA on z' for its own structure
    pca2 = PCA(n_components=2, random_state=42)
    z_prime_2d_own = pca2.fit_transform(z_prime_sub)
    explained2 = pca2.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Panel 1: z in its own PCA
    sc1 = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=regret_sub, cmap="RdYlGn_r", s=6, alpha=0.5)
    axes[0].set_title(f"Original z (PCA, {explained[0]:.1%}+{explained[1]:.1%} var)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    plt.colorbar(sc1, ax=axes[0], label="regret")

    # Panel 2: z' projected into z's PCA (shows how adapter shifts points)
    sc2 = axes[1].scatter(z_prime_2d[:, 0], z_prime_2d[:, 1], c=regret_sub, cmap="RdYlGn_r", s=6, alpha=0.5)
    axes[1].set_title(f"Adapted z' (in z's PCA space)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    plt.colorbar(sc2, ax=axes[1], label="regret")

    # Panel 3: z' in its own PCA
    sc3 = axes[2].scatter(z_prime_2d_own[:, 0], z_prime_2d_own[:, 1], c=regret_sub, cmap="RdYlGn_r", s=6, alpha=0.5)
    axes[2].set_title(f"Adapted z' (own PCA, {explained2[0]:.1%}+{explained2[1]:.1%} var)")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    plt.colorbar(sc3, ax=axes[2], label="regret")

    plt.tight_layout()
    path = os.path.join(output_dir, "pca_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved PCA plot to {path}")

    # Also log to wandb if active
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"diagnostics/pca_comparison": wandb.Image(path)})
    except Exception:
        pass

    return pca, explained


def plot_umap_comparison(z, z_prime, regret, output_dir):
    """Plot UMAP of z and z' side by side, coloured by regret."""
    try:
        from umap import UMAP
    except ImportError:
        print("  UMAP not installed, skipping UMAP plot. Install with: pip install umap-learn")
        return

    n = min(len(z), 5000)  # subsample for speed
    idx = np.random.choice(len(z), n, replace=False)
    z_sub = np.array(z[idx])
    z_prime_sub = np.array(z_prime[idx])
    regret_sub = np.array(regret[idx])

    print("  Fitting UMAP on z...")
    reducer = UMAP(n_components=2, random_state=42)
    z_2d = reducer.fit_transform(z_sub)

    print("  Fitting UMAP on z'...")
    reducer2 = UMAP(n_components=2, random_state=42)
    z_prime_2d = reducer2.fit_transform(z_prime_sub)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sc1 = ax1.scatter(z_2d[:, 0], z_2d[:, 1], c=regret_sub, cmap="viridis", s=4, alpha=0.5)
    ax1.set_title("Original z (coloured by regret)")
    plt.colorbar(sc1, ax=ax1, label="regret")

    sc2 = ax2.scatter(z_prime_2d[:, 0], z_prime_2d[:, 1], c=regret_sub, cmap="viridis", s=4, alpha=0.5)
    ax2.set_title("Adapted z' = z + adapter(z)")
    plt.colorbar(sc2, ax=ax2, label="regret")

    plt.tight_layout()
    path = os.path.join(output_dir, "umap_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved UMAP plot to {path}")


def plot_delta_z_stats(z, delta_z, regret, output_dir):
    """Analyse adapter corrections: magnitude vs regret, per-dim distribution."""
    delta_norm = np.sqrt(np.sum(delta_z ** 2, axis=-1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. ||delta_z|| distribution
    axes[0].hist(delta_norm, bins=50, alpha=0.7)
    axes[0].axvline(np.mean(delta_norm), color="red", linestyle="--", label=f"mean={np.mean(delta_norm):.3f}")
    axes[0].set_xlabel("||delta_z||")
    axes[0].set_ylabel("count")
    axes[0].set_title("Correction magnitude distribution")
    axes[0].legend()

    # 2. ||delta_z|| vs regret
    axes[1].scatter(regret, delta_norm, s=2, alpha=0.3)
    axes[1].set_xlabel("regret")
    axes[1].set_ylabel("||delta_z||")
    axes[1].set_title("Correction magnitude vs regret")

    # 3. Per-dim mean correction
    mean_delta_per_dim = np.mean(delta_z, axis=0)
    std_delta_per_dim = np.std(delta_z, axis=0)
    dims = np.arange(len(mean_delta_per_dim))
    axes[2].bar(dims, mean_delta_per_dim, yerr=std_delta_per_dim, alpha=0.7, capsize=1)
    axes[2].set_xlabel("latent dimension")
    axes[2].set_ylabel("mean delta_z")
    axes[2].set_title("Per-dimension correction")
    axes[2].axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "delta_z_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved delta_z analysis to {path}")


def evaluate_reconstruction(z, z_prime, tokens, vae, vae_params, output_dir, n_samples=1000):
    """Compare reconstruction accuracy: decode(z) vs decode(z') vs original tokens."""
    idx = np.random.choice(len(z), min(n_samples, len(z)), replace=False)
    z_sub = jnp.array(z[idx])
    z_prime_sub = jnp.array(z_prime[idx])
    tokens_sub = jnp.array(tokens[idx])

    @jax.jit
    def decode_batch(z_batch):
        return jax.vmap(lambda z: vae.apply({"params": vae_params}, z, method=vae.decode))(z_batch)

    # Decode original and adapted
    logits_orig = decode_batch(z_sub)
    logits_adapted = decode_batch(z_prime_sub)

    pred_orig = jnp.argmax(logits_orig, axis=-1)
    pred_adapted = jnp.argmax(logits_adapted, axis=-1)

    # Token-level accuracy
    acc_orig = float(jnp.mean(pred_orig == tokens_sub))
    acc_adapted = float(jnp.mean(pred_adapted == tokens_sub))

    # Level-level accuracy (all 52 tokens match)
    level_acc_orig = float(jnp.mean(jnp.all(pred_orig == tokens_sub, axis=-1)))
    level_acc_adapted = float(jnp.mean(jnp.all(pred_adapted == tokens_sub, axis=-1)))

    # Agreement between original and adapted decodings
    agreement = float(jnp.mean(pred_orig == pred_adapted))

    print(f"\n  Reconstruction analysis ({len(idx)} samples):")
    print(f"    Token accuracy (z → tokens):  {acc_orig:.4f}")
    print(f"    Token accuracy (z' → tokens): {acc_adapted:.4f}")
    print(f"    Level accuracy (z → tokens):  {level_acc_orig:.4f}")
    print(f"    Level accuracy (z' → tokens): {level_acc_adapted:.4f}")
    print(f"    decode(z) ↔ decode(z') agreement: {agreement:.4f}")

    # Save to file
    with open(os.path.join(output_dir, "reconstruction_metrics.txt"), "w") as f:
        f.write(f"token_acc_original: {acc_orig:.6f}\n")
        f.write(f"token_acc_adapted: {acc_adapted:.6f}\n")
        f.write(f"level_acc_original: {level_acc_orig:.6f}\n")
        f.write(f"level_acc_adapted: {level_acc_adapted:.6f}\n")
        f.write(f"decode_agreement: {agreement:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Adapter diagnostics")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Training data .npz (with 'source' field: 1=buffer, 0=prior)")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Separate test .npz (auto-detects *_test.npz if not given)")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--vae_checkpoint_path", type=str, required=True)
    parser.add_argument("--vae_config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load adapter
    print("Loading adapter...")
    adapter_model, adapter_params, adapter_cfg = load_adapter(args.adapter_path)

    @jax.jit
    def compute_z_prime(z_batch):
        delta_z = adapter_model.apply({"params": adapter_params}, z_batch)
        return z_batch + delta_z, delta_z

    def apply_adapter(z_arr):
        """Apply adapter to array, return (z_prime, delta_z)."""
        batch_size = 1024
        z_prime_parts, delta_z_parts = [], []
        for i in range(0, len(z_arr), batch_size):
            end = min(i + batch_size, len(z_arr))
            zp, dz = compute_z_prime(jnp.array(z_arr[i:end]))
            z_prime_parts.append(np.array(zp))
            delta_z_parts.append(np.array(dz))
        return np.concatenate(z_prime_parts), np.concatenate(delta_z_parts)

    # --- Load training data ---
    print("Loading training data...")
    data = np.load(args.data_path, allow_pickle=True)
    z_all = np.array(data["z"])
    tokens_all = np.array(data["tokens"])
    regret_all = np.array(data["regret"])
    source_all = np.array(data["source"]) if "source" in data else None
    print(f"  {len(z_all)} training samples")

    # --- Load test data ---
    test_path = args.test_data_path
    if test_path is None:
        auto_test = args.data_path.replace(".npz", "_test.npz")
        if os.path.exists(auto_test):
            test_path = auto_test
    test_z, test_tokens, test_regret, test_source = None, None, None, None
    if test_path and os.path.exists(test_path):
        print(f"Loading test data from {test_path}...")
        test_data = np.load(test_path, allow_pickle=True)
        test_z = np.array(test_data["z"])
        test_tokens = np.array(test_data["tokens"])
        test_regret = np.array(test_data["regret"])
        test_source = np.array(test_data["source"]) if "source" in test_data else None
        print(f"  {len(test_z)} test samples")

    # --- Extract buffer-only subset ---
    if source_all is not None:
        buf_mask = source_all == 1
        z_buf = z_all[buf_mask]
        tokens_buf = tokens_all[buf_mask]
        regret_buf = regret_all[buf_mask]
        print(f"  Buffer subset: {len(z_buf)} levels")
    else:
        z_buf = z_all
        tokens_buf = tokens_all
        regret_buf = regret_all

    # --- Compute adapted vectors ---
    print("\nComputing adapted latent vectors...")
    z_prime_all, delta_z_all = apply_adapter(z_all)
    z_prime_buf, delta_z_buf = apply_adapter(z_buf)

    delta_norms = np.sqrt(np.sum(delta_z_all ** 2, axis=-1))
    print(f"  ||delta_z|| (all): mean={delta_norms.mean():.4f}, std={delta_norms.std():.4f}, max={delta_norms.max():.4f}")
    delta_norms_buf = np.sqrt(np.sum(delta_z_buf ** 2, axis=-1))
    print(f"  ||delta_z|| (buf): mean={delta_norms_buf.mean():.4f}, std={delta_norms_buf.std():.4f}")

    # ===== DIAGNOSTICS =====

    # --- PCA on buffer-only ---
    print("\n--- PCA comparison (BUFFER only) ---")
    plot_pca_comparison(z_buf, z_prime_buf, regret_buf,
                        args.output_dir)
    # Rename to distinguish
    buf_pca_path = os.path.join(args.output_dir, "pca_comparison.png")
    buf_pca_renamed = os.path.join(args.output_dir, "pca_comparison_buffer.png")
    if os.path.exists(buf_pca_path):
        os.rename(buf_pca_path, buf_pca_renamed)
        print(f"  -> {buf_pca_renamed}")

    # --- PCA on test set ---
    if test_z is not None:
        print("\n--- PCA comparison (TEST set) ---")
        test_z_prime, _ = apply_adapter(test_z)
        plot_pca_comparison(test_z, test_z_prime, test_regret,
                            args.output_dir)
        test_pca_path = os.path.join(args.output_dir, "pca_comparison.png")
        test_pca_renamed = os.path.join(args.output_dir, "pca_comparison_test.png")
        if os.path.exists(test_pca_path):
            os.rename(test_pca_path, test_pca_renamed)
            print(f"  -> {test_pca_renamed}")

    # --- Delta-z analysis ---
    print("\n--- Delta-z analysis ---")
    plot_delta_z_stats(z_all, delta_z_all, regret_all, args.output_dir)

    # --- Reconstruction analysis ---
    print("\n--- Reconstruction analysis ---")
    vae, vae_params, vae_cfg = load_vae_decoder(args.vae_checkpoint_path, args.vae_config_path)
    evaluate_reconstruction(z_buf, z_prime_buf, tokens_buf, vae, vae_params, args.output_dir)

    print(f"\nAll diagnostics saved to {args.output_dir}")
    print(f"  pca_comparison_buffer.png  — PCA on buffer levels only")
    if test_z is not None:
        print(f"  pca_comparison_test.png    — PCA on held-out test set")


if __name__ == "__main__":
    main()
