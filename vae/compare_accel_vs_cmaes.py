#!/usr/bin/env python3
"""
Plain ACCEL vs CMA-ES+VAE Training Comparison
Converts compare_accel_vs_cmaes.ipynb to a standalone script.
Saves all plots to vae/plots/comparison/ and uploads to GCS.
"""
import os
import sys
import subprocess
import glob as glob_mod
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless TPU
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae_model import CluttrVAE
from buffer_latent_analysis import load_vae, encode_tokens

# ── Config ──────────────────────────────────────────────────────────────────
GCS_BUCKET = "ucl-ued-project-bucket"
GCS_PREFIX = "accel"
WANDB_PROJECT = "JAXUED_VAE_COMPARISON"
WANDB_ENTITY = None

CONDITIONS = {
    "plain_accel":        {"label": "Plain ACCEL",        "color": "#1f77b4", "marker": "o"},
    "cmaes_vae_beta1.0":  {"label": "CMA-ES (beta=1.0)", "color": "#ff7f0e", "marker": "s"},
    "cmaes_vae_beta1.5":  {"label": "CMA-ES (beta=1.5)", "color": "#2ca02c", "marker": "^"},
    "cmaes_vae_beta2.0":  {"label": "CMA-ES (beta=2.0)", "color": "#d62728", "marker": "D"},
}
SEEDS = [0, 1, 2]
DUMP_TIMESTEPS = ["10k", "20k", "30k", "40k", "50k"]

LOCAL_DATA_DIR = "/tmp/accel_comparison_data"
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "comparison")

VAE_CHECKPOINT_PATH = "runs/runs/20260227_185835_lr5e-05_lat64_baseline_weighted_recon_model_beta1.0_beta1.0/checkpoints/checkpoint_260000.pkl"
VAE_CONFIG_PATH = "runs/runs/20260227_185835_lr5e-05_lat64_baseline_weighted_recon_model_beta1.0_beta1.0/config.yaml"

matplotlib.rcParams.update({
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})

os.makedirs(PLOT_DIR, exist_ok=True)


def savefig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 1. Download from GCS ───────────────────────────────────────────────────
def download_from_gcs(run_name, seed):
    gcs_path = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/buffer_dumps/{run_name}/{seed}/"
    local_dir = os.path.join(LOCAL_DATA_DIR, run_name, str(seed))
    os.makedirs(local_dir, exist_ok=True)

    if os.path.exists(os.path.join(local_dir, "buffer_dump_final.npz")):
        print(f"  [skip] {run_name}/seed{seed} already downloaded")
        return local_dir

    print(f"  Downloading {run_name}/seed{seed}...")
    result = subprocess.run(
        ["gcloud", "storage", "cp", "--recursive", gcs_path, local_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [WARN] {result.stderr[:200]}")
    return local_dir


def load_buffer_dumps(run_name, seed):
    local_dir = os.path.join(LOCAL_DATA_DIR, run_name, str(seed))
    for sub in [local_dir, os.path.join(local_dir, str(seed))]:
        if os.path.exists(os.path.join(sub, "buffer_dump_final.npz")):
            local_dir = sub
            break

    dumps = {}
    for f in sorted(glob_mod.glob(os.path.join(local_dir, "buffer_dump_*.npz"))):
        tag = os.path.basename(f).replace("buffer_dump_", "").replace(".npz", "")
        dumps[tag] = dict(np.load(f))

    eval_path = os.path.join(local_dir, "buffer_eval.npz")
    eval_data = dict(np.load(eval_path)) if os.path.exists(eval_path) else None
    return dumps, eval_data


print("=" * 60)
print("1. Downloading data from GCS")
print("=" * 60)
for run_name in CONDITIONS:
    print(f"\n--- {run_name} ---")
    for seed in SEEDS:
        download_from_gcs(run_name, seed)

print("\nLoading buffer dumps...")
all_data = {}
for run_name in CONDITIONS:
    all_data[run_name] = {}
    for seed in SEEDS:
        dumps, eval_data = load_buffer_dumps(run_name, seed)
        all_data[run_name][seed] = {"dumps": dumps, "eval": eval_data}
        print(f"  {run_name}/seed{seed}: {len(dumps)} dumps, eval={'yes' if eval_data else 'NO'}")


# ── 2. Sample Efficiency (wandb) ──────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Sample Efficiency (wandb)")
print("=" * 60)

try:
    import wandb
    api = wandb.Api()
    path = f"{WANDB_ENTITY + '/' if WANDB_ENTITY else ''}{WANDB_PROJECT}"
    wandb_runs = api.runs(path)

    rows = []
    for run in wandb_runs:
        run_name = run.config.get("run_name", run.group)
        seed = run.config.get("seed", None)
        if run_name not in CONDITIONS:
            continue
        history = run.history(
            keys=["num_updates", "solve_rate/mean", "return/mean",
                  "level_sampler/mean_score", "level_sampler/score_std",
                  "level_sampler/size", "level_sampler/max_score"],
            pandas=True,
        )
        history = history.dropna(subset=["num_updates"])
        history["run_name"] = run_name
        history["seed"] = seed
        rows.append(history)

    wandb_df = pd.concat(rows, ignore_index=True)
    print(f"Loaded {len(wandb_df)} data points from {len(rows)} wandb runs")

    def plot_metric_with_shading(df, metric, ylabel, title, ax):
        for run_name, cfg in CONDITIONS.items():
            sub = df[df["run_name"] == run_name].sort_values("num_updates")
            grouped = sub.groupby("num_updates")[metric]
            mean = grouped.mean()
            std = grouped.std().fillna(0)
            ax.plot(mean.index, mean.values, color=cfg["color"],
                    label=cfg["label"], linewidth=1.5)
            ax.fill_between(mean.index, (mean - std).values, (mean + std).values,
                            color=cfg["color"], alpha=0.15)
        ax.set_xlabel("PPO Updates")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    plot_metric_with_shading(wandb_df, "solve_rate/mean", "Solve Rate",
                             "Eval Solve Rate (mean +/- std over 3 seeds)", ax=axes[0])
    plot_metric_with_shading(wandb_df, "return/mean", "Return",
                             "Eval Return (mean +/- std over 3 seeds)", ax=axes[1])
    plt.tight_layout()
    savefig(fig, "01_sample_efficiency.png")

    # Regret curves from wandb
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    plot_metric_with_shading(wandb_df, "level_sampler/mean_score", "Mean Buffer Score (PVL)",
                             "Buffer Mean Regret Over Training", ax=axes[0])
    plot_metric_with_shading(wandb_df, "level_sampler/max_score", "Max Buffer Score (PVL)",
                             "Buffer Max Regret Over Training", ax=axes[1])
    plt.tight_layout()
    savefig(fig, "01b_regret_curves.png")

except Exception as e:
    print(f"  [WARN] wandb failed: {e}")


# ── 3. Buffer Score Analysis ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Buffer Score Analysis")
print("=" * 60)

timesteps_to_show = DUMP_TIMESTEPS + ["final"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for ax_idx, ts in enumerate(timesteps_to_show):
    ax = axes.flat[ax_idx]
    for run_name, cfg in CONDITIONS.items():
        all_scores = []
        for seed in SEEDS:
            dumps = all_data[run_name][seed]["dumps"]
            if ts in dumps:
                size = int(dumps[ts].get("size", len(dumps[ts]["scores"])))
                all_scores.extend(dumps[ts]["scores"][:size])
        if all_scores:
            ax.hist(all_scores, bins=50, alpha=0.4, color=cfg["color"],
                    label=cfg["label"], density=True)
    ax.set_title(f"t = {ts}", fontweight="bold")
    ax.set_xlabel("Score (regret)")
    ax.set_ylabel("Density")
    if ax_idx == 0:
        ax.legend(fontsize=7)

for ax_idx in range(len(timesteps_to_show), len(axes.flat)):
    axes.flat[ax_idx].axis("off")

plt.suptitle("Buffer Score Distributions Over Training", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig(fig, "02_score_distributions.png")

# Score evolution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
stat_names = ["Mean Score", "Score Std", "Max Score"]
stat_fns = [np.mean, np.std, np.max]

for ax, stat_name, stat_fn in zip(axes, stat_names, stat_fns):
    for run_name, cfg in CONDITIONS.items():
        xs, ys_mean, ys_std = [], [], []
        for ts in DUMP_TIMESTEPS + ["final"]:
            seed_stats = []
            for seed in SEEDS:
                dumps = all_data[run_name][seed]["dumps"]
                if ts in dumps:
                    size = int(dumps[ts].get("size", len(dumps[ts]["scores"])))
                    scores = dumps[ts]["scores"][:size]
                    seed_stats.append(stat_fn(scores))
            if seed_stats:
                x_val = int(ts.replace("k", "")) if ts != "final" else 55
                xs.append(x_val)
                ys_mean.append(np.mean(seed_stats))
                ys_std.append(np.std(seed_stats))

        xs, ys_mean, ys_std = np.array(xs), np.array(ys_mean), np.array(ys_std)
        ax.plot(xs, ys_mean, color=cfg["color"], marker=cfg["marker"],
                label=cfg["label"], linewidth=1.5)
        ax.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std,
                        color=cfg["color"], alpha=0.15)

    ax.set_xlabel("Training Update (k)")
    ax.set_ylabel(stat_name)
    ax.set_title(stat_name, fontweight="bold")

axes[0].legend(fontsize=8)
plt.suptitle("Buffer Score Evolution", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig(fig, "03_score_evolution.png")


# ── 3b. Wall Count Analysis ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("3b. Wall Count Analysis")
print("=" * 60)

def count_walls(tokens):
    """Count walls per level from token sequences. First 50 tokens are wall slots; non-zero = wall."""
    return (tokens[:, :50] > 0).sum(axis=1)

# Wall count evolution over training
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for run_name, cfg in CONDITIONS.items():
    xs, ys_mean, ys_std = [], [], []
    for ts in DUMP_TIMESTEPS + ["final"]:
        seed_means = []
        for seed in SEEDS:
            dumps = all_data[run_name][seed]["dumps"]
            if ts in dumps:
                size = int(dumps[ts].get("size", len(dumps[ts]["tokens"])))
                tokens = dumps[ts]["tokens"][:size]
                wc = count_walls(tokens)
                seed_means.append(wc.mean())
        if seed_means:
            x_val = int(ts.replace("k", "")) if ts != "final" else 55
            xs.append(x_val)
            ys_mean.append(np.mean(seed_means))
            ys_std.append(np.std(seed_means))

    xs, ys_mean, ys_std = np.array(xs), np.array(ys_mean), np.array(ys_std)
    axes[0].plot(xs, ys_mean, color=cfg["color"], marker=cfg["marker"],
                 label=cfg["label"], linewidth=1.5)
    axes[0].fill_between(xs, ys_mean - ys_std, ys_mean + ys_std,
                         color=cfg["color"], alpha=0.15)

axes[0].set_xlabel("Training Update (k)")
axes[0].set_ylabel("Mean Wall Count")
axes[0].set_title("Mean Wall Count Over Training", fontweight="bold")
axes[0].legend(fontsize=8)

# Wall count distribution at final timestep
for run_name, cfg in CONDITIONS.items():
    all_wc = []
    for seed in SEEDS:
        dumps = all_data[run_name][seed]["dumps"]
        d = dumps.get("final", dumps[sorted(dumps.keys())[-1]] if dumps else None)
        if d is not None:
            size = int(d.get("size", len(d["tokens"])))
            wc = count_walls(d["tokens"][:size])
            all_wc.extend(wc)
    if all_wc:
        axes[1].hist(all_wc, bins=range(0, 52), alpha=0.4, color=cfg["color"],
                     label=cfg["label"], density=True)

axes[1].set_xlabel("Wall Count")
axes[1].set_ylabel("Density")
axes[1].set_title("Final Buffer Wall Count Distribution", fontweight="bold")
axes[1].legend(fontsize=8)
axes[1].axvline(x=50, color="black", linestyle="--", alpha=0.5, label="Max (50)")

plt.tight_layout()
savefig(fig, "03b_wall_count_analysis.png")

# Wall count vs score scatter (final buffers)
fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(5 * len(CONDITIONS), 5))
if len(CONDITIONS) == 1:
    axes = [axes]

for ax, (run_name, cfg) in zip(axes, CONDITIONS.items()):
    all_wc, all_sc = [], []
    for seed in SEEDS:
        dumps = all_data[run_name][seed]["dumps"]
        d = dumps.get("final", dumps[sorted(dumps.keys())[-1]] if dumps else None)
        if d is not None:
            size = int(d.get("size", len(d["tokens"])))
            wc = count_walls(d["tokens"][:size])
            sc = d["scores"][:size]
            all_wc.extend(wc)
            all_sc.extend(sc)
    if all_wc:
        ax.scatter(all_wc, all_sc, alpha=0.15, s=4, color=cfg["color"])
        ax.set_xlabel("Wall Count")
        ax.set_ylabel("Score (regret)")
        ax.set_title(cfg["label"], fontweight="bold")

plt.suptitle("Wall Count vs Score (Final Buffers)", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig(fig, "03c_wall_count_vs_score.png")


# ── 4. Cross-Condition Latent PCA ─────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Cross-Condition Latent PCA")
print("=" * 60)

import jax
import jax.numpy as jnp

vae_ckpt_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), VAE_CHECKPOINT_PATH)
vae_cfg_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), VAE_CONFIG_PATH)
print(f"Loading shared VAE encoder (beta=1.0)...")
vae, vae_params, vae_cfg = load_vae(vae_ckpt_abs, vae_cfg_abs)
print(f"  latent_dim={vae_cfg['latent_dim']}")

latent_data = {}
for run_name in CONDITIONS:
    latent_data[run_name] = {}
    for seed in SEEDS:
        dumps = all_data[run_name][seed]["dumps"]
        d = dumps.get("final", dumps[sorted(dumps.keys())[-1]] if dumps else None)
        if d is None:
            print(f"  [WARN] {run_name}/seed{seed}: no dumps")
            continue
        size = int(d.get("size", len(d["tokens"])))
        tokens = d["tokens"][:size]
        scores = d["scores"][:size]
        latents = encode_tokens(vae, vae_params, tokens, batch_size=512)
        latent_data[run_name][seed] = {"latents": latents, "scores": scores}
        print(f"  Encoded {run_name}/seed{seed}: {latents.shape}")

# Fit PCA
all_latents_list, all_scores_list, all_labels_list = [], [], []
for run_name in CONDITIONS:
    for seed in SEEDS:
        if seed not in latent_data.get(run_name, {}):
            continue
        ld = latent_data[run_name][seed]
        all_latents_list.append(ld["latents"])
        all_scores_list.append(ld["scores"])
        all_labels_list.extend([run_name] * len(ld["latents"]))

combined_latents = np.concatenate(all_latents_list, axis=0)
combined_scores = np.concatenate(all_scores_list, axis=0)
combined_labels = np.array(all_labels_list)

pca = PCA(n_components=2)
projected = pca.fit_transform(combined_latents)
print(f"PCA on {combined_latents.shape[0]} vectors: "
      f"PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for run_name, cfg in CONDITIONS.items():
    mask = combined_labels == run_name
    axes[0].scatter(projected[mask, 0], projected[mask, 1],
                    c=cfg["color"], alpha=0.3, s=6, label=cfg["label"],
                    marker=cfg["marker"])

axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[0].set_title("Final Buffers by Condition (shared VAE latent PCA)")
axes[0].legend(markerscale=3, fontsize=9)

valid = np.isfinite(combined_scores) & (combined_scores > -1e6)
sc = axes[1].scatter(projected[valid, 0], projected[valid, 1],
                     c=combined_scores[valid], cmap="viridis", alpha=0.3, s=6)
plt.colorbar(sc, ax=axes[1], label="Score (regret)")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[1].set_title("Final Buffers by Score")

plt.tight_layout()
savefig(fig, "04_cross_condition_pca.png")


# ── 5. Buffer Evolution PCA ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Buffer Evolution PCA (all seeds)")
print("=" * 60)

# Encode all conditions × all seeds × all timesteps
evolution_data = {}  # evolution_data[run_name][seed][ts] = latents
all_latents_for_pca = []
for run_name in CONDITIONS:
    evolution_data[run_name] = {}
    for seed in SEEDS:
        evolution_data[run_name][seed] = {}
        dumps = all_data[run_name][seed]["dumps"]
        for ts in sorted(dumps.keys()):
            d = dumps[ts]
            size = int(d.get("size", len(d["tokens"])))
            tokens = d["tokens"][:size]
            latents = encode_tokens(vae, vae_params, tokens, batch_size=512)
            evolution_data[run_name][seed][ts] = latents
            all_latents_for_pca.append(latents)
        print(f"  {run_name}/seed{seed}: encoded {len(dumps)} snapshots")

# Fit shared PCA across ALL conditions, seeds, and timesteps
combined_evo_latents = np.concatenate(all_latents_for_pca, axis=0)
pca_evo = PCA(n_components=2)
pca_evo.fit(combined_evo_latents)
print(f"Shared PCA on {combined_evo_latents.shape[0]} vectors: "
      f"PC1={pca_evo.explained_variance_ratio_[0]:.1%}, PC2={pca_evo.explained_variance_ratio_[1]:.1%}")

# Plot: rows=seeds, columns=conditions
n_conds = len(CONDITIONS)
n_seeds = len(SEEDS)
fig, axes = plt.subplots(n_seeds, n_conds, figsize=(5 * n_conds, 5 * n_seeds))
if n_seeds == 1:
    axes = axes[None, :]
if n_conds == 1:
    axes = axes[:, None]

cmap = plt.cm.viridis

for col, (run_name, cfg) in enumerate(CONDITIONS.items()):
    for row, seed in enumerate(SEEDS):
        ax = axes[row, col]
        snapshots = evolution_data[run_name][seed]
        if not snapshots:
            ax.set_title(f"{cfg['label']} seed {seed} (no data)")
            continue

        ts_keys = sorted(snapshots.keys(),
                         key=lambda x: int(x.replace("k", "").replace("final", "999")))
        n_ts = len(ts_keys)

        for i, ts in enumerate(ts_keys):
            proj = pca_evo.transform(snapshots[ts])
            color = cmap(i / max(n_ts - 1, 1))
            ax.scatter(proj[:, 0], proj[:, 1], c=[color], alpha=0.3, s=4, label=ts)

        ax.set_xlabel(f"PC1 ({pca_evo.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca_evo.explained_variance_ratio_[1]:.1%})")
        if row == 0:
            ax.set_title(f"{cfg['label']}", fontweight="bold")
        if col == 0:
            ax.annotate(f"Seed {seed}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 15, 0),
                        xycoords=ax.yaxis.label, textcoords="offset points",
                        fontsize=12, fontweight="bold", ha="right", va="center")
        if row == 0 and col == n_conds - 1:
            ax.legend(markerscale=3, fontsize=7, ncol=2, loc="upper right")

plt.suptitle("Buffer Evolution in Shared VAE Latent Space (PCA)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
savefig(fig, "05_buffer_evolution_pca.png")


# ── 6. Solve Rate Analysis ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Solve Rate Analysis")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

means, stds, labels, colors = [], [], [], []
for run_name, cfg in CONDITIONS.items():
    seed_means = []
    for seed in SEEDS:
        eval_data = all_data[run_name][seed]["eval"]
        if eval_data is not None:
            seed_means.append(eval_data["solve_rates"].mean())
    means.append(np.mean(seed_means) if seed_means else 0)
    stds.append(np.std(seed_means) if seed_means else 0)
    labels.append(cfg["label"])
    colors.append(cfg["color"])

bars = axes[0].bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
axes[0].set_ylabel("Mean Solve Rate")
axes[0].set_title("Buffer Level Solve Rate by Condition", fontweight="bold")
axes[0].set_ylim(0, 1.05)
for bar, m in zip(bars, means):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{m:.1%}", ha="center", fontsize=9)

for run_name, cfg in CONDITIONS.items():
    all_sr = []
    for seed in SEEDS:
        eval_data = all_data[run_name][seed]["eval"]
        if eval_data is not None:
            all_sr.extend(eval_data["solve_rates"])
    if all_sr:
        sorted_sr = np.sort(all_sr)
        cdf = np.arange(1, len(sorted_sr) + 1) / len(sorted_sr)
        axes[1].plot(sorted_sr, cdf, color=cfg["color"], label=cfg["label"], linewidth=1.5)

axes[1].set_xlabel("Solve Rate")
axes[1].set_ylabel("CDF (fraction of levels)")
axes[1].set_title("Solve Rate CDF Across Buffer Levels", fontweight="bold")
axes[1].legend(fontsize=9)

plt.tight_layout()
savefig(fig, "06_solve_rate_analysis.png")

# Hardest levels
print("\n=== Hardest Levels (lowest solve rate) per condition ===\n")
for run_name, cfg in CONDITIONS.items():
    print(f"--- {cfg['label']} ---")
    for seed in SEEDS:
        eval_data = all_data[run_name][seed]["eval"]
        if eval_data is None:
            print(f"  seed {seed}: no eval data")
            continue
        sr = eval_data["solve_rates"]
        unsolved = (sr == 0).sum()
        hard = (sr < 0.5).sum()
        print(f"  seed {seed}: {unsolved} unsolved (0%), {hard} hard (<50%), "
              f"mean={sr.mean():.1%}, min={sr.min():.1%}")
    print()


# ── 7. Diversity Analysis ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. Diversity Analysis")
print("=" * 60)

from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

def diversity_metrics(latents, tokens, n_sample=2000):
    """Compute diversity metrics for a set of levels.

    Returns dict with:
      - mean_pairwise_latent_dist: mean pairwise L2 in latent space (sampled)
      - latent_std: mean per-dim std in latent space
      - convex_hull_volume: volume of convex hull in PC1-PC2 space
      - unique_wall_configs: fraction of unique wall configurations
      - wall_count_entropy: entropy of wall count distribution
      - mean_pairwise_token_hamming: mean pairwise Hamming distance on wall tokens
    """
    n = len(latents)
    # Sample for pairwise distances (O(n^2) otherwise)
    if n > n_sample:
        idx = np.random.choice(n, n_sample, replace=False)
        lat_sample = latents[idx]
        tok_sample = tokens[idx]
    else:
        lat_sample = latents
        tok_sample = tokens

    # Latent space metrics
    pw_latent = pdist(lat_sample, metric="euclidean")
    latent_std = latents.std(axis=0).mean()

    # Convex hull in PCA space (use the global PCA already fitted)
    proj2d = pca.transform(latents)
    try:
        hull = ConvexHull(proj2d)
        hull_vol = hull.volume  # area in 2D
    except Exception:
        hull_vol = 0.0

    # Token-level diversity (compare actual wall indices = grid positions)
    wall_tokens = tokens[:, :50]
    # Unique wall configurations (exact match on sorted wall index sequences)
    unique_configs = len(set(map(tuple, wall_tokens)))
    # Convert to binary wall maps (169 cells) for structural comparison
    wall_maps = np.zeros((min(n_sample, n), 169), dtype=np.uint8)
    for i, wt in enumerate(wall_tokens[:min(n_sample, n)]):
        nonzero = wt[wt > 0]
        wall_maps[i, nonzero.astype(int) - 1] = 1  # 1-based to 0-based
    # Pairwise Hamming on actual grid positions
    pw_hamming = pdist(wall_maps.astype(float), metric="hamming")
    wall_binary = (wall_tokens > 0).astype(np.uint8)

    # Wall count entropy
    wc = wall_binary.sum(axis=1)
    counts, _ = np.histogram(wc, bins=range(0, 52))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    wc_entropy = -(probs * np.log2(probs)).sum()

    return {
        "mean_pw_latent_dist": pw_latent.mean(),
        "latent_std": latent_std,
        "hull_area_pc12": hull_vol,
        "unique_wall_configs": unique_configs,
        "frac_unique": unique_configs / n,
        "mean_pw_hamming": pw_hamming.mean(),
        "wall_count_entropy": wc_entropy,
    }

# Compute diversity for final buffers (per condition, averaged over seeds)
div_results = {}
for run_name in CONDITIONS:
    seed_divs = []
    for seed in SEEDS:
        if seed not in latent_data.get(run_name, {}):
            continue
        ld = latent_data[run_name][seed]
        dumps = all_data[run_name][seed]["dumps"]
        d = dumps.get("final", dumps[sorted(dumps.keys())[-1]] if dumps else None)
        if d is None:
            continue
        size = int(d.get("size", len(d["tokens"])))
        tokens = d["tokens"][:size]
        latents = ld["latents"]
        seed_divs.append(diversity_metrics(latents, tokens))
    div_results[run_name] = seed_divs

# Print diversity table
print("\n=== Diversity Metrics (final buffer, mean over seeds) ===\n")
metric_keys = ["mean_pw_latent_dist", "latent_std", "hull_area_pc12",
               "unique_wall_configs", "frac_unique", "mean_pw_hamming", "wall_count_entropy"]
metric_labels = ["Pairwise Latent Dist", "Latent Std", "Hull Area (PC1-2)",
                 "Unique Wall Configs", "Frac Unique", "Pairwise Hamming", "Wall Count Entropy"]

for run_name, cfg in CONDITIONS.items():
    print(f"--- {cfg['label']} ---")
    if not div_results[run_name]:
        print("  No data")
        continue
    for key, label in zip(metric_keys, metric_labels):
        vals = [d[key] for d in div_results[run_name]]
        print(f"  {label}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    print()

# Plot 1: Diversity metrics bar chart comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plot_metrics = [
    ("mean_pw_latent_dist", "Mean Pairwise Latent Distance"),
    ("latent_std", "Mean Latent Std (per dim)"),
    ("hull_area_pc12", "Convex Hull Area (PC1-PC2)"),
    ("frac_unique", "Fraction Unique Wall Configs"),
    ("mean_pw_hamming", "Mean Pairwise Hamming Distance"),
    ("wall_count_entropy", "Wall Count Entropy (bits)"),
]

for ax, (key, title) in zip(axes.flat, plot_metrics):
    means, stds, clabels, clrs = [], [], [], []
    for run_name, cfg in CONDITIONS.items():
        vals = [d[key] for d in div_results[run_name]]
        means.append(np.mean(vals) if vals else 0)
        stds.append(np.std(vals) if vals else 0)
        clabels.append(cfg["label"])
        clrs.append(cfg["color"])
    bars = ax.bar(clabels, means, yerr=stds, capsize=5, color=clrs, alpha=0.8)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.tick_params(axis='x', labelsize=8)

plt.suptitle("Buffer Diversity Metrics (Final Buffer)", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig(fig, "07_diversity_metrics.png")

# Plot 2: Diversity evolution over training
print("Computing diversity evolution over training (seed 0)...")
div_evolution = {}
for run_name in CONDITIONS:
    div_evolution[run_name] = {}
    dumps = all_data[run_name][0]["dumps"]
    for ts in sorted(dumps.keys()):
        d = dumps[ts]
        size = int(d.get("size", len(d["tokens"])))
        tokens = d["tokens"][:size]
        # Use already-encoded latents from evolution_data if available
        if ts in evolution_data.get(run_name, {}):
            latents = evolution_data[run_name][ts]
        else:
            latents = encode_tokens(vae, vae_params, tokens, batch_size=512)
        div_evolution[run_name][ts] = diversity_metrics(latents, tokens)
    print(f"  {run_name}: {len(dumps)} timesteps")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for ax, (key, title) in zip(axes.flat, plot_metrics):
    for run_name, cfg in CONDITIONS.items():
        xs, ys = [], []
        ts_keys = sorted(div_evolution[run_name].keys(),
                         key=lambda x: int(x.replace("k", "").replace("final", "999")))
        for ts in ts_keys:
            x_val = int(ts.replace("k", "")) if ts != "final" else 55
            xs.append(x_val)
            ys.append(div_evolution[run_name][ts][key])
        ax.plot(xs, ys, color=cfg["color"], marker=cfg["marker"],
                label=cfg["label"], linewidth=1.5)
    ax.set_xlabel("Training Update (k)")
    ax.set_title(title, fontweight="bold", fontsize=10)
    if ax == axes.flat[0]:
        ax.legend(fontsize=7)

plt.suptitle("Buffer Diversity Evolution (seed 0)", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig(fig, "08_diversity_evolution.png")


# ── 8. VAE Search Quality Diagnostics ─────────────────────────────────────
print("\n" + "=" * 60)
print("8. VAE Search Quality Diagnostics")
print("=" * 60)
print("Is the VAE latent space actually a better search space than random mutations?")

# --- 8a. VAE reconstruction quality per condition ---
print("\n--- 8a. Reconstruction quality (encode → decode → token accuracy) ---")

def reconstruct_tokens(vae, vae_params, tokens, batch_size=512):
    """Encode tokens, decode from mean, return reconstructed tokens and per-level accuracy."""
    all_recon = []
    all_acc = []
    for i in range(0, len(tokens), batch_size):
        batch = jnp.array(tokens[i:i + batch_size])
        mean, _ = vae.apply({"params": vae_params}, batch, train=False, method=vae.encode)
        logits = vae.apply({"params": vae_params}, mean, method=vae.decode)  # (B, 52, vocab)
        recon = jnp.argmax(logits, axis=-1)  # (B, 52)
        acc = (recon == batch).mean(axis=1)  # per-level accuracy
        all_recon.append(np.asarray(recon))
        all_acc.append(np.asarray(acc))
    return np.concatenate(all_recon), np.concatenate(all_acc)

recon_data = {}
for run_name in CONDITIONS:
    recon_data[run_name] = {}
    for seed in SEEDS:
        dumps = all_data[run_name][seed]["dumps"]
        d = dumps.get("final", dumps[sorted(dumps.keys())[-1]] if dumps else None)
        if d is None:
            continue
        size = int(d.get("size", len(d["tokens"])))
        tokens = d["tokens"][:size]
        scores = d["scores"][:size]
        recon_tokens, recon_acc = reconstruct_tokens(vae, vae_params, tokens)
        recon_data[run_name][seed] = {
            "tokens": tokens, "scores": scores,
            "recon_tokens": recon_tokens, "accuracy": recon_acc,
        }
    seed_accs = [recon_data[run_name][s]["accuracy"].mean()
                 for s in SEEDS if s in recon_data[run_name]]
    print(f"  {CONDITIONS[run_name]['label']}: "
          f"mean token accuracy = {np.mean(seed_accs):.1%} +/- {np.std(seed_accs):.1%}")

# Plot: reconstruction accuracy distribution per condition
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for run_name, cfg in CONDITIONS.items():
    all_acc = []
    for seed in SEEDS:
        if seed in recon_data[run_name]:
            all_acc.extend(recon_data[run_name][seed]["accuracy"])
    if all_acc:
        axes[0].hist(all_acc, bins=50, alpha=0.4, color=cfg["color"],
                     label=cfg["label"], density=True)

axes[0].set_xlabel("Token Reconstruction Accuracy")
axes[0].set_ylabel("Density")
axes[0].set_title("VAE Reconstruction Accuracy Distribution", fontweight="bold")
axes[0].legend(fontsize=8)

# Plot: reconstruction accuracy vs score (is the VAE worse at representing hard levels?)
for run_name, cfg in CONDITIONS.items():
    all_acc, all_sc = [], []
    for seed in SEEDS:
        if seed in recon_data[run_name]:
            all_acc.extend(recon_data[run_name][seed]["accuracy"])
            all_sc.extend(recon_data[run_name][seed]["scores"])
    if all_acc:
        axes[1].scatter(all_sc, all_acc, alpha=0.1, s=4, color=cfg["color"], label=cfg["label"])

axes[1].set_xlabel("Score (regret)")
axes[1].set_ylabel("Token Reconstruction Accuracy")
axes[1].set_title("Reconstruction Accuracy vs Score", fontweight="bold")
axes[1].legend(markerscale=3, fontsize=8)

plt.tight_layout()
savefig(fig, "09a_reconstruction_quality.png")

# --- 8b. Reconstruction accuracy vs wall count ---
print("\n--- 8b. Reconstruction accuracy vs wall count ---")

fig, ax = plt.subplots(figsize=(10, 6))
for run_name, cfg in CONDITIONS.items():
    all_wc, all_acc = [], []
    for seed in SEEDS:
        if seed in recon_data[run_name]:
            tokens = recon_data[run_name][seed]["tokens"]
            wc = (tokens[:, :50] > 0).sum(axis=1)
            all_wc.extend(wc)
            all_acc.extend(recon_data[run_name][seed]["accuracy"])
    if all_wc:
        # Bin by wall count and plot mean accuracy per bin
        wc_arr, acc_arr = np.array(all_wc), np.array(all_acc)
        bins = np.arange(0, 51, 2)
        bin_idx = np.digitize(wc_arr, bins)
        bin_means = [acc_arr[bin_idx == i].mean() for i in range(1, len(bins))
                     if (bin_idx == i).sum() > 5]
        bin_centers = [bins[i-1] + 1 for i in range(1, len(bins))
                       if (bin_idx == i).sum() > 5]
        ax.plot(bin_centers, bin_means, color=cfg["color"], marker=cfg["marker"],
                label=cfg["label"], linewidth=1.5)

ax.set_xlabel("Wall Count")
ax.set_ylabel("Mean Token Reconstruction Accuracy")
ax.set_title("VAE Reconstruction Quality vs Maze Complexity", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
savefig(fig, "09b_recon_vs_wall_count.png")

# --- 8c. Where do ACCEL high-regret levels sit in the VAE latent space? ---
print("\n--- 8c. ACCEL levels projected into VAE latent space ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

accel_latents_all, accel_scores_all = [], []
cmaes_latents_all, cmaes_scores_all = [], []

for seed in SEEDS:
    if seed in latent_data.get("plain_accel", {}):
        accel_latents_all.append(latent_data["plain_accel"][seed]["latents"])
        accel_scores_all.append(latent_data["plain_accel"][seed]["scores"])
    for cname in ["cmaes_vae_beta1.0", "cmaes_vae_beta1.5", "cmaes_vae_beta2.0"]:
        if seed in latent_data.get(cname, {}):
            cmaes_latents_all.append(latent_data[cname][seed]["latents"])
            cmaes_scores_all.append(latent_data[cname][seed]["scores"])

if accel_latents_all and cmaes_latents_all:
    accel_lat = np.concatenate(accel_latents_all)
    accel_sc = np.concatenate(accel_scores_all)
    cmaes_lat = np.concatenate(cmaes_latents_all)
    cmaes_sc = np.concatenate(cmaes_scores_all)

    accel_proj = pca.transform(accel_lat)
    cmaes_proj = pca.transform(cmaes_lat)

    # Top 10% regret levels from ACCEL
    accel_thresh = np.percentile(accel_sc, 90)
    accel_high = accel_sc >= accel_thresh

    axes[0].scatter(cmaes_proj[:, 0], cmaes_proj[:, 1], c="lightgray", alpha=0.2, s=4, label="CMA-ES (all)")
    axes[0].scatter(accel_proj[~accel_high, 0], accel_proj[~accel_high, 1],
                    c=CONDITIONS["plain_accel"]["color"], alpha=0.2, s=4, label="ACCEL (low regret)")
    axes[0].scatter(accel_proj[accel_high, 0], accel_proj[accel_high, 1],
                    c="red", alpha=0.6, s=12, label=f"ACCEL top 10% regret (>{accel_thresh:.2f})")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].set_title("ACCEL High-Regret Levels vs CMA-ES Coverage", fontweight="bold")
    axes[0].legend(markerscale=3, fontsize=8)

    # Right: distance from ACCEL levels to nearest CMA-ES level in latent space
    from scipy.spatial import cKDTree
    tree = cKDTree(cmaes_lat)
    dists_high, _ = tree.query(accel_lat[accel_high])
    dists_low, _ = tree.query(accel_lat[~accel_high])

    axes[1].hist(dists_high, bins=50, alpha=0.6, color="red", density=True,
                 label=f"ACCEL top 10% regret (n={accel_high.sum()})")
    axes[1].hist(dists_low, bins=50, alpha=0.4, color=CONDITIONS["plain_accel"]["color"],
                 density=True, label=f"ACCEL rest (n={(~accel_high).sum()})")
    axes[1].set_xlabel("L2 Distance to Nearest CMA-ES Level (latent space)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Are ACCEL's Best Levels Reachable by CMA-ES?", fontweight="bold")
    axes[1].legend(fontsize=8)
    print(f"  ACCEL top 10% regret: mean dist to nearest CMA-ES = {dists_high.mean():.3f}")
    print(f"  ACCEL rest:           mean dist to nearest CMA-ES = {dists_low.mean():.3f}")

plt.tight_layout()
savefig(fig, "09c_accel_vs_cmaes_latent_coverage.png")

# --- 8d. Buffer turnover rate between dumps ---
print("\n--- 8d. Buffer turnover rate (fraction of levels replaced between dumps) ---")

def compute_turnover(tokens_prev, tokens_curr, size_prev, size_curr):
    """Fraction of levels in curr buffer that were NOT in prev buffer.
    Compares actual token sequences (full 52-token vectors)."""
    prev_set = set(map(tuple, tokens_prev[:size_prev]))
    curr_set = set(map(tuple, tokens_curr[:size_curr]))
    if len(curr_set) == 0:
        return 0.0
    new_levels = curr_set - prev_set
    return len(new_levels) / len(curr_set)

fig, ax = plt.subplots(figsize=(12, 6))
for run_name, cfg in CONDITIONS.items():
    ts_list = DUMP_TIMESTEPS + ["final"]
    xs, ys_mean, ys_std = [], [], []
    for i in range(1, len(ts_list)):
        prev_ts, curr_ts = ts_list[i - 1], ts_list[i]
        seed_turnovers = []
        for seed in SEEDS:
            dumps = all_data[run_name][seed]["dumps"]
            if prev_ts not in dumps or curr_ts not in dumps:
                continue
            prev_size = int(dumps[prev_ts].get("size", len(dumps[prev_ts]["tokens"])))
            curr_size = int(dumps[curr_ts].get("size", len(dumps[curr_ts]["tokens"])))
            turnover = compute_turnover(
                dumps[prev_ts]["tokens"], dumps[curr_ts]["tokens"],
                prev_size, curr_size,
            )
            seed_turnovers.append(turnover)
        if seed_turnovers:
            x_val = int(curr_ts.replace("k", "")) if curr_ts != "final" else 55
            xs.append(x_val)
            ys_mean.append(np.mean(seed_turnovers))
            ys_std.append(np.std(seed_turnovers))

    if xs:
        xs, ys_mean, ys_std = np.array(xs), np.array(ys_mean), np.array(ys_std)
        ax.plot(xs, ys_mean, color=cfg["color"], marker=cfg["marker"],
                label=cfg["label"], linewidth=1.5)
        ax.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std,
                        color=cfg["color"], alpha=0.15)

ax.set_xlabel("Training Update (k)")
ax.set_ylabel("Turnover Rate (fraction of buffer replaced)")
ax.set_title("Buffer Turnover Between Dumps (10k update intervals)", fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
plt.tight_layout()
savefig(fig, "09d_buffer_turnover.png")

# Print turnover summary
print("\n  Cumulative turnover (final vs 10k):")
for run_name, cfg in CONDITIONS.items():
    seed_turnovers = []
    for seed in SEEDS:
        dumps = all_data[run_name][seed]["dumps"]
        first_ts = "10k" if "10k" in dumps else sorted(dumps.keys())[0]
        last_ts = "final" if "final" in dumps else sorted(dumps.keys())[-1]
        if first_ts in dumps and last_ts in dumps:
            s1 = int(dumps[first_ts].get("size", len(dumps[first_ts]["tokens"])))
            s2 = int(dumps[last_ts].get("size", len(dumps[last_ts]["tokens"])))
            seed_turnovers.append(compute_turnover(
                dumps[first_ts]["tokens"], dumps[last_ts]["tokens"], s1, s2))
    if seed_turnovers:
        print(f"    {cfg['label']}: {np.mean(seed_turnovers):.1%} +/- {np.std(seed_turnovers):.1%}")


# ── 9. Summary Table ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. Summary Table")
print("=" * 60)

summary_rows = []
for run_name, cfg in CONDITIONS.items():
    seed_solve_rates, seed_mean_scores, seed_std_scores = [], [], []
    seed_max_scores, seed_unsolved, seed_sizes = [], [], []

    for seed in SEEDS:
        eval_data = all_data[run_name][seed]["eval"]
        dumps = all_data[run_name][seed]["dumps"]

        if eval_data is not None:
            sr = eval_data["solve_rates"]
            seed_solve_rates.append(sr.mean())
            seed_unsolved.append((sr == 0).sum())

        final_key = "final" if "final" in dumps else (sorted(dumps.keys())[-1] if dumps else None)
        if final_key and final_key in dumps:
            d = dumps[final_key]
            size = int(d.get("size", len(d["scores"])))
            scores = d["scores"][:size]
            seed_mean_scores.append(scores.mean())
            seed_std_scores.append(scores.std())
            seed_max_scores.append(scores.max())
            seed_sizes.append(size)

    # Diversity metrics
    divs = div_results.get(run_name, [])
    pw_lat = [d["mean_pw_latent_dist"] for d in divs] if divs else []
    frac_uniq = [d["frac_unique"] for d in divs] if divs else []
    pw_ham = [d["mean_pw_hamming"] for d in divs] if divs else []

    summary_rows.append({
        "Condition": cfg["label"],
        "Solve Rate": f"{np.mean(seed_solve_rates):.1%} +/- {np.std(seed_solve_rates):.1%}" if seed_solve_rates else "N/A",
        "Mean Score": f"{np.mean(seed_mean_scores):.3f} +/- {np.std(seed_mean_scores):.3f}" if seed_mean_scores else "N/A",
        "Score Std": f"{np.mean(seed_std_scores):.3f}" if seed_std_scores else "N/A",
        "Max Score": f"{np.mean(seed_max_scores):.3f}" if seed_max_scores else "N/A",
        "Unsolved": f"{np.mean(seed_unsolved):.0f} +/- {np.std(seed_unsolved):.0f}" if seed_unsolved else "N/A",
        "Buffer Size": f"{np.mean(seed_sizes):.0f}" if seed_sizes else "N/A",
        "PW Latent Dist": f"{np.mean(pw_lat):.3f}" if pw_lat else "N/A",
        "Frac Unique": f"{np.mean(frac_uniq):.3f}" if frac_uniq else "N/A",
        "PW Hamming": f"{np.mean(pw_ham):.4f}" if pw_ham else "N/A",
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# Save summary as CSV too
csv_path = os.path.join(PLOT_DIR, "summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")


# ── Upload plots to GCS ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Uploading plots to GCS")
print("=" * 60)
gcs_dest = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/comparison_plots/"
result = subprocess.run(
    ["gcloud", "storage", "cp", "--recursive", PLOT_DIR + "/", gcs_dest],
    capture_output=True, text=True,
)
if result.returncode == 0:
    print(f"Uploaded to {gcs_dest}")
else:
    print(f"Upload failed: {result.stderr[:200]}")

print("\nDone! All plots saved to:", PLOT_DIR)
