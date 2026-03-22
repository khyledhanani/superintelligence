"""Compare pretrained vs fine-tuned VAE: latent drift, param drift, reconstruction."""
import argparse, pickle, yaml
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from flax import linen as nn
from vae_model import CluttrVAE


def load_vae(checkpoint_path, config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    vae = CluttrVAE(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        latent_dim=cfg["latent_dim"],
        seq_len=cfg["seq_len"],
    )
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    # Handle both raw params and {"params": ..., "config": ..., "step": ...} format
    if isinstance(ckpt, dict) and "params" in ckpt:
        params = ckpt["params"]
    else:
        params = ckpt
    # Ensure params are wrapped for flax: {"params": ...}
    if isinstance(params, dict) and "params" not in params:
        params = {"params": params}
    return vae, params, cfg


def param_drift(params_pre, params_ft):
    """Print per-layer parameter drift between pretrained and fine-tuned."""
    print("\n=== Parameter Drift ===")
    flat_pre = jax.tree_util.tree_leaves_with_path(params_pre)
    flat_ft = jax.tree_util.tree_leaves_with_path(params_ft)

    total_l2 = 0.0
    total_cos_num = 0.0
    total_cos_den_a = 0.0
    total_cos_den_b = 0.0

    for (path_pre, v_pre), (path_ft, v_ft) in zip(flat_pre, flat_ft):
        if not isinstance(v_pre, (jnp.ndarray, np.ndarray)):
            continue
        name = "/".join(str(k) for k in path_pre)
        v_pre_f = v_pre.flatten().astype(float)
        v_ft_f = v_ft.flatten().astype(float)
        diff = v_ft_f - v_pre_f
        l2 = float(np.linalg.norm(diff))
        norm_pre = float(np.linalg.norm(v_pre_f))
        cos = float(np.dot(v_pre_f, v_ft_f) / (norm_pre * np.linalg.norm(v_ft_f) + 1e-12))
        rel = l2 / (norm_pre + 1e-12)
        print(f"  {name:60s}  L2={l2:.6f}  cos={cos:.6f}  rel={rel:.6f}")
        total_l2 += l2 ** 2
        total_cos_num += float(np.dot(v_pre_f, v_ft_f))
        total_cos_den_a += float(np.dot(v_pre_f, v_pre_f))
        total_cos_den_b += float(np.dot(v_ft_f, v_ft_f))

    print(f"\n  Total L2 = {np.sqrt(total_l2):.6f}")
    print(f"  Overall cosine similarity = {total_cos_num / (np.sqrt(total_cos_den_a) * np.sqrt(total_cos_den_b) + 1e-12):.6f}")


def latent_drift(vae, params_pre, params_ft, tokens, n=2000):
    """Compare latent means from pretrained vs fine-tuned encoder."""
    print(f"\n=== Latent Drift (n={n}) ===")
    tokens = tokens[:n]

    mean_pre, _ = vae.apply(params_pre, tokens, train=False, method=vae.encode)
    mean_ft, _ = vae.apply(params_ft, tokens, train=False, method=vae.encode)

    z_pre = np.array(mean_pre)
    z_ft = np.array(mean_ft)

    diff = z_ft - z_pre
    per_sample_l2 = np.linalg.norm(diff, axis=-1)
    cos_sim = np.sum(z_pre * z_ft, axis=-1) / (
        np.linalg.norm(z_pre, axis=-1) * np.linalg.norm(z_ft, axis=-1) + 1e-12)

    print(f"  Per-sample L2:  mean={per_sample_l2.mean():.4f}, std={per_sample_l2.std():.4f}, "
          f"max={per_sample_l2.max():.4f}")
    print(f"  Per-sample cos: mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f}, "
          f"min={cos_sim.min():.4f}")

    # Per-dimension drift
    per_dim_shift = np.abs(diff).mean(axis=0)
    top_dims = np.argsort(per_dim_shift)[::-1][:10]
    print(f"\n  Top 10 shifted dimensions:")
    for d in top_dims:
        print(f"    dim {d:2d}: mean_shift={per_dim_shift[d]:.4f}, "
              f"pre_mean={z_pre[:, d].mean():.3f}, ft_mean={z_ft[:, d].mean():.3f}")

    # Variance ratio
    var_pre = z_pre.var(axis=0)
    var_ft = z_ft.var(axis=0)
    var_ratio = var_ft / (var_pre + 1e-12)
    print(f"\n  Variance ratio (ft/pre): mean={var_ratio.mean():.4f}, "
          f"min={var_ratio.min():.4f}, max={var_ratio.max():.4f}")


def reconstruction_comparison(vae, params_pre, params_ft, tokens, n=500):
    """Compare reconstruction accuracy: pretrained, fine-tuned, and cross."""
    print(f"\n=== Reconstruction Accuracy ===")
    tokens = tokens[:n]
    rng = jax.random.PRNGKey(0)

    # Pretrained enc + pretrained dec
    logits_pre, _, _ = vae.apply(params_pre, tokens, rng, train=False)
    pred_pre = jnp.argmax(logits_pre, axis=-1)
    acc_pre = float((pred_pre == tokens).mean())

    # Fine-tuned enc + fine-tuned dec
    logits_ft, _, _ = vae.apply(params_ft, tokens, rng, train=False)
    pred_ft = jnp.argmax(logits_ft, axis=-1)
    acc_ft = float((pred_ft == tokens).mean())

    # Cross: pretrained enc -> fine-tuned dec
    mean_pre, logvar_pre = vae.apply(params_pre, tokens, train=False, method=vae.encode)
    logits_cross = vae.apply(params_ft, mean_pre, method=vae.decode)
    pred_cross = jnp.argmax(logits_cross, axis=-1)
    acc_cross = float((pred_cross == tokens).mean())

    print(f"  Pretrained (enc+dec):   {acc_pre * 100:.4f}%")
    print(f"  Fine-tuned (enc+dec):   {acc_ft * 100:.4f}%")
    print(f"  Cross (pre_enc+ft_dec): {acc_cross * 100:.4f}%")


def _score_correlated_axes(z, scores):
    """Find 2 orthogonal axes in latent space most correlated with score.

    Uses OLS regression to find direction most correlated with score,
    then finds second direction orthogonal to first via residuals.

    Returns:
        projections: (N, 2) array of projections onto the two axes
        correlations: (r1, r2) Pearson correlations with score
        per_dim_corr: (latent_dim,) per-dimension correlations
    """
    # Per-dimension correlation
    per_dim_corr = np.array([np.corrcoef(z[:, d], scores)[0, 1] for d in range(z.shape[1])])

    # First axis: OLS regression z -> score
    z_centered = z - z.mean(axis=0)
    s_centered = scores - scores.mean()
    # beta = (Z^T Z)^{-1} Z^T s
    beta = np.linalg.lstsq(z_centered, s_centered, rcond=None)[0]
    axis1 = beta / (np.linalg.norm(beta) + 1e-12)
    proj1 = z_centered @ axis1
    r1 = float(np.corrcoef(proj1, scores)[0, 1])

    # Second axis: regress residuals
    residuals = s_centered - proj1 * (s_centered @ proj1) / (proj1 @ proj1 + 1e-12)
    beta2 = np.linalg.lstsq(z_centered, residuals, rcond=None)[0]
    # Orthogonalize against axis1
    beta2 = beta2 - np.dot(beta2, axis1) * axis1
    axis2 = beta2 / (np.linalg.norm(beta2) + 1e-12)
    proj2 = z_centered @ axis2
    r2 = float(np.corrcoef(proj2, scores)[0, 1])

    projections = np.stack([proj1, proj2], axis=-1)
    return projections, (r1, r2), per_dim_corr


def score_correlated_plot(vae, params_pre, params_ft, buffer_path,
                          output_path="vae_score_axes.png", top_k=200):
    """Plot buffer levels on score-correlated axes for both VAEs."""
    buf = np.load(buffer_path, allow_pickle=True)
    tokens = jnp.array(buf["tokens"])
    scores = np.array(buf["scores"])
    if "size" in buf:
        buf_size = int(buf["size"])
        tokens = tokens[:buf_size]
        scores = scores[:buf_size]

    # Encode with both
    mean_pre, _ = vae.apply(params_pre, tokens, train=False, method=vae.encode)
    mean_ft, _ = vae.apply(params_ft, tokens, train=False, method=vae.encode)
    z_pre = np.array(mean_pre)
    z_ft = np.array(mean_ft)

    # Find score-correlated axes for each
    proj_pre, (r1_pre, r2_pre), corrs_pre = _score_correlated_axes(z_pre, scores)
    proj_ft, (r1_ft, r2_ft), corrs_ft = _score_correlated_axes(z_ft, scores)

    # Project
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, proj, title, r1, r2 in [
        (axes[0], proj_pre, "Pretrained encoder", r1_pre, r2_pre),
        (axes[1], proj_ft, "Fine-tuned encoder", r1_ft, r2_ft),
    ]:
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=scores, cmap="viridis",
                        s=3, alpha=0.5, vmin=0, vmax=scores.max())
        # Highlight top-k
        top_idx = np.argsort(scores)[-top_k:]
        ax.scatter(proj[top_idx, 0], proj[top_idx, 1], c=scores[top_idx],
                   cmap="viridis", s=15, alpha=0.8, edgecolors="red", linewidths=0.5,
                   vmin=0, vmax=scores.max())
        ax.set_xlabel(f"Score axis 1 (r={r1:.3f})")
        ax.set_ylabel(f"Score axis 2 (r={r2:.3f})")
        ax.set_title(title)

    # Per-dim correlation comparison
    ax = axes[2]
    dims = np.arange(len(corrs_pre))
    w = 0.35
    ax.bar(dims - w / 2, np.abs(corrs_pre), w, label="Pretrained", alpha=0.7)
    ax.bar(dims + w / 2, np.abs(corrs_ft), w, label="Fine-tuned", alpha=0.7)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("|Correlation with SFL score|")
    ax.set_title("Per-dim score correlation")
    ax.legend()

    fig.colorbar(sc, ax=axes[:2], label="SFL Score", shrink=0.8)
    fig.suptitle(f"Score-correlated latent axes (buffer: {len(tokens)} levels, top {top_k} highlighted)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved: {output_path}")
    print(f"  Pretrained: axis1 r={r1_pre:.4f}, axis2 r={r2_pre:.4f}")
    print(f"  Fine-tuned: axis1 r={r1_ft:.4f}, axis2 r={r2_ft:.4f}")


def pca_buffer_plot(vae, params_pre, params_ft, buffer_path,
                    output_path="vae_pca_comparison.png", bg_tokens=None, top_k=200):
    """2x2 PCA projection of buffer levels using both encoders and both PCA bases."""
    buf = np.load(buffer_path, allow_pickle=True)
    tokens = jnp.array(buf["tokens"])
    scores = np.array(buf["scores"])
    if "size" in buf:
        buf_size = int(buf["size"])
        tokens = tokens[:buf_size]
        scores = scores[:buf_size]

    # Encode with both
    mean_pre, _ = vae.apply(params_pre, tokens, train=False, method=vae.encode)
    mean_ft, _ = vae.apply(params_ft, tokens, train=False, method=vae.encode)
    z_pre = np.array(mean_pre)
    z_ft = np.array(mean_ft)

    # Fit PCA on each
    pca_pre = PCA(n_components=2).fit(z_pre)
    pca_ft = PCA(n_components=2).fit(z_ft)

    # Background tokens
    if bg_tokens is not None:
        bg_mean_pre, _ = vae.apply(params_pre, jnp.array(bg_tokens), train=False, method=vae.encode)
        bg_mean_ft, _ = vae.apply(params_ft, jnp.array(bg_tokens), train=False, method=vae.encode)
        bg_z_pre = np.array(bg_mean_pre)
        bg_z_ft = np.array(bg_mean_ft)

    top_idx = np.argsort(scores)[-top_k:]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    configs = [
        (0, 0, z_pre, pca_pre, "Pretrained enc + Pretrained PCA"),
        (0, 1, z_pre, pca_ft, "Pretrained enc + Fine-tuned PCA"),
        (1, 0, z_ft, pca_pre, "Fine-tuned enc + Pretrained PCA"),
        (1, 1, z_ft, pca_ft, "Fine-tuned enc + Fine-tuned PCA"),
    ]

    for r, c, z, pca, title in configs:
        ax = axes[r][c]
        proj = pca.transform(z)

        # Background
        if bg_tokens is not None:
            bg_z = bg_z_pre if "Pretrained enc" in title else bg_z_ft
            bg_proj = pca.transform(bg_z)
            ax.scatter(bg_proj[:, 0], bg_proj[:, 1], c="lightgray", s=1, alpha=0.3)

        # All buffer levels
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=scores, cmap="viridis",
                        s=5, alpha=0.5, vmin=0, vmax=scores.max())
        # Top-k highlighted
        ax.scatter(proj[top_idx, 0], proj[top_idx, 1], c=scores[top_idx],
                   cmap="viridis", s=20, alpha=0.9, edgecolors="red", linewidths=0.5,
                   vmin=0, vmax=scores.max())
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    fig.colorbar(sc, ax=axes, label="SFL Score", shrink=0.6)
    fig.suptitle(f"PCA projections: pretrained vs fine-tuned VAE\n"
                 f"({len(tokens)} levels, score range [{scores.min():.3f}, {scores.max():.3f}])",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt", required=True)
    parser.add_argument("--finetuned_ckpt", required=True)
    parser.add_argument("--config", required=True, help="VAE config.yaml")
    parser.add_argument("--data", required=True, help="Token .npy file for latent comparison")
    parser.add_argument("--buffer", default=None, help="Buffer .npz for PCA plot")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--top_k", type=int, default=200, help="Number of top-scoring levels to highlight")
    parser.add_argument("--output", default="vae_pca_comparison.png")
    args = parser.parse_args()

    vae, params_pre, cfg = load_vae(args.pretrained_ckpt, args.config)
    _, params_ft, _ = load_vae(args.finetuned_ckpt, args.config)
    tokens = np.load(args.data)[:args.n_samples]
    tokens = jnp.array(tokens)

    param_drift(params_pre, params_ft)
    latent_drift(vae, params_pre, params_ft, tokens, n=args.n_samples)
    reconstruction_comparison(vae, params_pre, params_ft, tokens, n=min(500, args.n_samples))

    if args.buffer:
        score_correlated_plot(vae, params_pre, params_ft, args.buffer,
                              output_path=args.output.replace(".png", "_score_axes.png"),
                              top_k=args.top_k)
        pca_buffer_plot(vae, params_pre, params_ft, args.buffer,
                        output_path=args.output, bg_tokens=np.array(tokens), top_k=args.top_k)
