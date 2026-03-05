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
                  "level_sampler/mean_score", "level_sampler/score_std"],
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
                    c=cfg["color"], alpha=0.3, s=6, label=cfg["label"])

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
print("5. Buffer Evolution PCA")
print("=" * 60)

evolution_data = {}
for run_name in CONDITIONS:
    evolution_data[run_name] = {}
    dumps = all_data[run_name][0]["dumps"]  # seed 0 as representative
    for ts in sorted(dumps.keys()):
        d = dumps[ts]
        size = int(d.get("size", len(d["tokens"])))
        tokens = d["tokens"][:size]
        latents = encode_tokens(vae, vae_params, tokens, batch_size=512)
        evolution_data[run_name][ts] = latents
    print(f"  {run_name}/seed0: encoded {len(dumps)} snapshots")

n_conds = len(CONDITIONS)
fig, axes = plt.subplots(1, n_conds, figsize=(6 * n_conds, 6))
if n_conds == 1:
    axes = [axes]

for ax, (run_name, cfg) in zip(axes, CONDITIONS.items()):
    snapshots = evolution_data[run_name]
    if not snapshots:
        ax.set_title(f"{cfg['label']} (no data)")
        continue

    all_snap_latents = np.concatenate(list(snapshots.values()), axis=0)
    pca_evo = PCA(n_components=2)
    pca_evo.fit(all_snap_latents)

    cmap = plt.cm.viridis
    ts_keys = sorted(snapshots.keys(),
                     key=lambda x: int(x.replace("k", "").replace("final", "999")))
    n_ts = len(ts_keys)

    for i, ts in enumerate(ts_keys):
        proj = pca_evo.transform(snapshots[ts])
        color = cmap(i / max(n_ts - 1, 1))
        ax.scatter(proj[:, 0], proj[:, 1], c=[color], alpha=0.3, s=6, label=ts)

    ax.set_xlabel(f"PC1 ({pca_evo.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_evo.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"{cfg['label']} (seed 0)", fontweight="bold")
    ax.legend(markerscale=3, fontsize=7, ncol=2)

plt.suptitle("Buffer Evolution in VAE Latent Space", fontsize=14, fontweight="bold")
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


# ── 7. Summary Table ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. Summary Table")
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

    summary_rows.append({
        "Condition": cfg["label"],
        "Solve Rate": f"{np.mean(seed_solve_rates):.1%} +/- {np.std(seed_solve_rates):.1%}" if seed_solve_rates else "N/A",
        "Mean Score": f"{np.mean(seed_mean_scores):.3f} +/- {np.std(seed_mean_scores):.3f}" if seed_mean_scores else "N/A",
        "Score Std": f"{np.mean(seed_std_scores):.3f}" if seed_std_scores else "N/A",
        "Max Score": f"{np.mean(seed_max_scores):.3f}" if seed_max_scores else "N/A",
        "Unsolved": f"{np.mean(seed_unsolved):.0f} +/- {np.std(seed_unsolved):.0f}" if seed_unsolved else "N/A",
        "Buffer Size": f"{np.mean(seed_sizes):.0f}" if seed_sizes else "N/A",
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
