"""
evaluate_checkpoints.py
=======================
Unified GPU checkpoint evaluator for the CLUTR MiniGrid VAE.

Evaluates every checkpoint in CONFIG['checkpoint_dir'] on both train
and validation splits and produces:

  runs/<run_id>/eval/
    ranking.csv              <- key metrics, all checkpoints, sorted rank-1 first
    full_metrics.csv         <- every computed column for every checkpoint
    best_checkpoint.json     <- best step + all its metrics, machine-readable
    checkpoints/step_N.json  <- one JSON per checkpoint
    report.txt               <- human-readable report with interpretation guide
    metric_curves.png        <- visual dashboard across training steps

WHY THESE METRICS (grounded in the CLUTR paper)
------------------------------------------------
The VAE is used as a *fixed, pretrained task manifold* for ES-based
curriculum navigation. Two things must be true:

  1. RECONSTRUCTION QUALITY  – the VAE must faithfully encode/decode
     real mazes so the latent space represents real task structure.
     → recon_hamming, recon_valid_structure_pct, recon_unsolvable_pct

  2. GENERATIVE QUALITY       – sampling z ~ N(0,I) must produce valid,
     diverse, solvable mazes. This is what the ES teacher does at runtime.
     → gen_prior_solvable_pct, gen_latent_coverage

  3. LATENT SMOOTHNESS        – nearby z should decode to similar mazes.
     → gen_perturbation_consistency_pct

  4. ELBO COMPONENTS          – reconstruction loss + KL divergence.
     KL collapse (KL → 0) means the VAE ignores the latent space entirely.
     → elbo_kl_divergence, elbo_collapsed_dims_pct

BEST CHECKPOINT SELECTION CRITERION
-------------------------------------
Primary:   gen_prior_solvable_pct  (ES teacher samples from prior at runtime)
Secondary: val_recon_valid_structure_pct
Guard:     elbo_kl_divergence > KL_COLLAPSE_THRESHOLD (avoid posterior collapse)
Tertiary:  gen_latent_coverage (diversity of the manifold)

Usage
-----
    python evaluate_checkpoints.py
    python evaluate_checkpoints.py --config vae_train_config.yml \\
                                   --run_id run_20250101_abc123 \\
                                   --n_eval 2000 \\
                                   --n_prior_samples 1000 \\
                                   --batch_size 256 \\
                                   --steps 5000,10000,15000
"""

import os
import re
import glob
import json
import pickle
import argparse
import importlib
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr

import jax
import jax.numpy as jnp

import train_vae, utils
importlib.reload(train_vae); importlib.reload(utils)
from train_vae import CluttrVAE
from utils import is_reachable_bfs

# ── Constants ──────────────────────────────────────────────────────────────
GRID_SIZE             = 13
KL_COLLAPSE_THRESHOLD = 0.05   # KL below this → posterior collapse


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CLUTR VAE checkpoints")
    p.add_argument("--config",          default="vae_train_config.yml")
    p.add_argument("--run_id",          default=None,
                   help="Run ID subfolder under runs/. If omitted, the latest "
                        "run in CONFIG['checkpoint_dir'] is auto-detected.")
    p.add_argument("--n_eval",          type=int, default=1000,
                   help="# environments evaluated per split per checkpoint")
    p.add_argument("--n_prior_samples", type=int, default=500,
                   help="# z~N(0,I) samples for generative quality metrics")
    p.add_argument("--batch_size",      type=int, default=256,
                   help="JAX batch size — tune to your VRAM")
    p.add_argument("--steps",           type=str, default=None,
                   help="Comma-separated checkpoint steps to eval (default: all)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Path helpers
# ═══════════════════════════════════════════════════════════════════════════

def resolve_run_dir(config: dict, run_id: str | None) -> str:
    """
    Return the run directory: <working_path>/<vae_folder>/runs/<run_id>/

    If run_id is None, auto-detect the most recently modified run folder.
    """
    runs_root = os.path.join(
        config["working_path"], config["vae_folder"], "runs"
    )
    if run_id is not None:
        return os.path.join(runs_root, run_id)

    # Auto-detect latest run
    run_dirs = sorted(
        [d for d in glob.glob(os.path.join(runs_root, "*")) if os.path.isdir(d)],
        key=os.path.getmtime,
    )
    if not run_dirs:
        raise ValueError(
            f"No run directories found under {runs_root}. "
            "Pass --run_id explicitly."
        )
    chosen = run_dirs[-1]
    print(f"[Run] Auto-detected latest run: {os.path.basename(chosen)}")
    return chosen


def resolve_ckpt_dir(config: dict, run_dir: str) -> str:
    """
    Checkpoint directory lives at <run_dir>/checkpoints/.
    """
    candidate = os.path.join(run_dir, "checkpoints")
    if os.path.isdir(candidate):
        return candidate
    # Fallback: try the config path if the standard location doesn't exist
    fallback = os.path.join(
        config["working_path"], config["vae_folder"], config["checkpoint_dir"]
    )
    print(f"[Ckpt] {candidate} not found — using {fallback}")
    return fallback


def resolve_data_paths(config: dict) -> tuple[str, str]:
    """Return (train_path, val_path) as absolute paths."""
    base = os.path.join(config["working_path"], config["vae_folder"])
    train_path = os.path.join(base, config["train_data_path"])
    val_path   = os.path.join(base, config["validation_data_path"])
    return train_path, val_path


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint discovery & loading
# ═══════════════════════════════════════════════════════════════════════════

def discover_checkpoints(checkpoint_dir: str) -> list[tuple[int, str]]:
    """Return sorted list of (step, path) for every checkpoint_*.pkl found."""
    paths = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pkl")))
    result = []
    for p in paths:
        m = re.search(r"checkpoint_(\d+)\.pkl", p)
        if m:
            result.append((int(m.group(1)), p))
    return sorted(result)


def load_params(path: str) -> dict:
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt["params"] if "params" in ckpt else ckpt


# ═══════════════════════════════════════════════════════════════════════════
# JAX forward-pass helpers
# ═══════════════════════════════════════════════════════════════════════════

def reconstruct_dataset(model, params, data: np.ndarray,
                        batch_size: int, rng):
    """
    VAE forward pass over a full dataset split, in batches.
    Returns: originals, reconstructions, means, logvars — all numpy arrays.
    """
    @jax.jit
    def batch_forward(xs, rng_key):
        def fwd(x):
            logits, mean, logvar = model.apply(
                {"params": params}, x[jnp.newaxis, :], rng_key,
                train=False, rngs={"dropout": jax.random.key(0)}
            )
            return jnp.argmax(logits[0], axis=-1), mean[0], logvar[0]
        return jax.vmap(fwd)(xs)

    originals, reconstructions, all_means, all_logvars = [], [], [], []
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        rng, sub = jax.random.split(rng)
        recons, means, logvars = batch_forward(jnp.array(batch), sub)
        originals.append(np.array(batch))
        reconstructions.append(np.array(recons))
        all_means.append(np.array(means))
        all_logvars.append(np.array(logvars))

    return (np.concatenate(originals),
            np.concatenate(reconstructions),
            np.concatenate(all_means),
            np.concatenate(all_logvars))


def decode_batch(model, params, z_batch: np.ndarray,
                 batch_size: int) -> np.ndarray:
    """
    Decode a batch of latent vectors using CluttrVAE.decode().
    z_batch: (N, latent_dim)
    Returns: (N, seq_len) integer token array.
    """
    @jax.jit
    def _decode_jit(z_b):
        # CluttrVAE.decode() accepts (batch, latent_dim) and returns
        # (batch, seq_len, vocab_size)
        logits = model.apply(
            {"params": params}, z_b, method=CluttrVAE.decode
        )
        return jnp.argmax(logits, axis=-1)   # (batch, seq_len)

    results = []
    for start in range(0, len(z_batch), batch_size):
        zb = jnp.array(z_batch[start:start + batch_size])
        results.append(np.array(_decode_jit(zb)))
    return np.concatenate(results, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# METRIC GROUP 1: Reconstruction quality
# ═══════════════════════════════════════════════════════════════════════════

def _active_walls(seq: np.ndarray) -> set:
    w = seq[:-2]
    return set(w[w != 0].tolist())


def reconstruction_metrics(originals: np.ndarray,
                            reconstructions: np.ndarray) -> dict:
    """
    Measures how faithfully the VAE reconstructs real MiniGrid mazes.
    Sequence format: [obs_0, ..., obs_49, goal_idx, agent_idx]
    """
    n   = len(originals)
    acc = dict(hamming=0, agent_acc=0, goal_acc=0,
               a_on_w=0, g_on_w=0, a_on_g=0,
               valid_structure=0, wall_diff=0, unsolvable=0)

    for orig, recon in zip(originals, reconstructions):
        orig  = orig.flatten().astype(int)
        recon = recon.flatten().astype(int)
        a, g  = int(recon[-1]), int(recon[-2])

        if not is_reachable_bfs(GRID_SIZE, walls=recon[:-2],
                                start_idx=a, goal_idx=g):
            acc["unsolvable"] += 1

        acc["hamming"]    += int(np.sum(orig != recon))
        acc["agent_acc"]  += int(orig[-1] == recon[-1])
        acc["goal_acc"]   += int(orig[-2] == recon[-2])

        rw    = _active_walls(recon)
        a_on_w = a in rw
        g_on_w = g in rw
        a_on_g = (a == g)
        acc["a_on_w"] += int(a_on_w)
        acc["g_on_w"] += int(g_on_w)
        acc["a_on_g"] += int(a_on_g)
        if not (a_on_w or g_on_w or a_on_g):
            acc["valid_structure"] += 1
        acc["wall_diff"] += len(_active_walls(recon)) - len(_active_walls(orig))

    return {
        "recon_avg_hamming":            acc["hamming"]         / n,
        "recon_agent_accuracy_pct":    (acc["agent_acc"]        / n) * 100,
        "recon_goal_accuracy_pct":     (acc["goal_acc"]         / n) * 100,
        "recon_valid_structure_pct":   (acc["valid_structure"]  / n) * 100,
        "recon_unsolvable_pct":        (acc["unsolvable"]       / n) * 100,
        "recon_agent_on_wall_pct":     (acc["a_on_w"]           / n) * 100,
        "recon_goal_on_wall_pct":      (acc["g_on_w"]           / n) * 100,
        "recon_avg_wall_count_error":   acc["wall_diff"]        / n,
    }


# ═══════════════════════════════════════════════════════════════════════════
# METRIC GROUP 2: ELBO components
# ═══════════════════════════════════════════════════════════════════════════

def elbo_metrics(originals: np.ndarray, reconstructions: np.ndarray,
                 means: np.ndarray, logvars: np.ndarray) -> dict:
    kl_per_sample = -0.5 * (1 + logvars - means**2 - np.exp(logvars)).sum(axis=-1)
    kl_mean       = float(kl_per_sample.mean())
    kl_per_dim    = -0.5 * (1 + logvars - means**2 - np.exp(logvars)).mean(axis=0)
    collapsed_pct = float((kl_per_dim < 0.01).mean() * 100)
    token_acc     = float(np.mean([
        np.mean(o.flatten() == r.flatten())
        for o, r in zip(originals, reconstructions)
    ])) * 100
    return {
        "elbo_kl_divergence":      kl_mean,
        "elbo_collapsed_dims_pct": collapsed_pct,
        "elbo_token_accuracy_pct": token_acc,
        "elbo_kl_collapsed":       bool(kl_mean < KL_COLLAPSE_THRESHOLD),
    }


# ═══════════════════════════════════════════════════════════════════════════
# METRIC GROUP 3: Generative quality  ← most important for ES/CLUTR
# ═══════════════════════════════════════════════════════════════════════════

def generative_quality_metrics(model, params,
                                n_samples: int, batch_size: int,
                                latent_dim: int, rng) -> dict:
    """
    Probes generative quality by sampling z ~ N(0, I) and decoding via
    CluttrVAE.decode() — exactly what the ES teacher does at runtime.

    Metrics
    -------
    gen_prior_valid_structure_pct   structural validity of prior samples
    gen_prior_solvable_pct          solvability  ★ PRIMARY metric for ES
    gen_prior_avg_wall_count        obstacle density (healthy: 10–40)
    gen_latent_coverage             mean pairwise L2 distance in z-space
    gen_perturbation_consistency_pct decode stability under small z noise
    """
    rng, sub_prior, sub_pert = jax.random.split(rng, 3)

    # ── Sample z ~ N(0, I) and decode ─────────────────────────────────────
    z_prior  = np.array(jax.random.normal(sub_prior, (n_samples, latent_dim)))
    decoded  = decode_batch(model, params, z_prior, batch_size)  # (N, seq_len)

    valid_count = solvable_count = 0
    wall_counts = []
    for seq in decoded:
        seq  = seq.flatten().astype(int)
        a, g = int(seq[-1]), int(seq[-2])
        rw   = _active_walls(seq)
        a_on_w, g_on_w, a_on_g = (a in rw), (g in rw), (a == g)
        if not (a_on_w or g_on_w or a_on_g):
            valid_count += 1
        if is_reachable_bfs(GRID_SIZE, walls=seq[:-2],
                            start_idx=a, goal_idx=g):
            solvable_count += 1
        wall_counts.append(len(rw))

    # ── Latent coverage (diversity) ────────────────────────────────────────
    idx_cov  = np.random.choice(n_samples, min(200, n_samples), replace=False)
    z_cov    = z_prior[idx_cov]
    dists    = np.sqrt(((z_cov[:, None, :] - z_cov[None, :, :]) ** 2).sum(-1))
    mask     = np.triu(np.ones_like(dists, dtype=bool), k=1)
    coverage = float(dists[mask].mean())

    # ── Perturbation consistency (smoothness) ──────────────────────────────
    # Decode z and z + small noise; compare how much the output changes.
    n_pert  = min(200, n_samples)
    z_base  = z_prior[:n_pert]
    z_pert  = z_base + np.array(
        jax.random.normal(sub_pert, z_base.shape)) * 0.1

    dec_base = decode_batch(model, params, z_base, batch_size)
    dec_pert = decode_batch(model, params, z_pert, batch_size)
    consistency = float(np.mean([
        np.mean(o == p) for o, p in zip(dec_base, dec_pert)
    ])) * 100

    # ── Perturbation validity (are perturbed latents still valid mazes?) ────
    # This is critical for latent-space exploration in the next project phase
    pert_valid_count = pert_solvable_count = 0
    for seq in dec_pert:
        seq  = seq.flatten().astype(int)
        a, g = int(seq[-1]), int(seq[-2])
        rw   = _active_walls(seq)
        a_on_w, g_on_w, a_on_g = (a in rw), (g in rw), (a == g)
        if not (a_on_w or g_on_w or a_on_g):
            pert_valid_count += 1
        if is_reachable_bfs(GRID_SIZE, walls=seq[:-2],
                            start_idx=a, goal_idx=g):
            pert_solvable_count += 1

    return {
        "gen_prior_valid_structure_pct":      (valid_count    / n_samples) * 100,
        "gen_prior_solvable_pct":             (solvable_count / n_samples) * 100,
        "gen_prior_avg_wall_count":            float(np.mean(wall_counts)),
        "gen_latent_coverage":                 coverage,
        "gen_perturbation_consistency_pct":   consistency,
        "gen_perturbation_valid_structure_pct": (pert_valid_count / n_pert) * 100,
        "gen_perturbation_solvable_pct":      (pert_solvable_count / n_pert) * 100,
    }


# ═══════════════════════════════════════════════════════════════════════════
# METRIC GROUP 4: Latent space structure
# ═══════════════════════════════════════════════════════════════════════════

def _maze_similarity(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    Structural similarity in [0, 1] between two maze sequences.
    0.6 × Jaccard(wall sets) + 0.2 × agent_match + 0.2 × goal_match
    """
    a = seq_a.flatten().astype(int)
    b = seq_b.flatten().astype(int)
    walls_a = set(a[:-2][a[:-2] != 0].tolist())
    walls_b = set(b[:-2][b[:-2] != 0].tolist())
    union   = walls_a | walls_b
    jaccard = len(walls_a & walls_b) / len(union) if union else 1.0
    return 0.6 * jaccard + 0.2 * float(a[-1] == b[-1]) + 0.2 * float(a[-2] == b[-2])


def latent_structure_metrics(model, params,
                              originals: np.ndarray,
                              means: np.ndarray,
                              batch_size: int, rng) -> dict:
    """
    Tests whether the latent space geometry reflects maze similarity.

    1. NEIGHBOUR CONSISTENCY
       For each z_i, find its nearest latent neighbour z_j.
       Check that maze_i and maze_j are actually similar.
       → struct_neighbour_maze_similarity  (high = good)

    2. INTERPOLATION SMOOTHNESS  (uses CluttrVAE.decode())
       Decode 7 points along line z_a → z_b for 50 random pairs.
       Check monotone decrease in similarity to endpoint A.
       → struct_interp_monotonicity_pct
       → struct_interp_validity_pct

    3. LATENT DISTANCE vs MAZE SIMILARITY CORRELATION
       Spearman rho over 1000 random pairs.
       Target: negative (far in z → structurally different maze).
       → struct_latent_dist_sim_spearman
    """
    n     = len(originals)
    n_sub = min(500, n)
    rng_np = np.random.default_rng(seed=42)
    idx    = rng_np.choice(n, n_sub, replace=False)
    z_sub  = means[idx]
    m_sub  = originals[idx]

    # Pairwise L2 distances in latent space
    diffs    = z_sub[:, None, :] - z_sub[None, :, :]
    l2_dists = np.sqrt((diffs ** 2).sum(-1))
    np.fill_diagonal(l2_dists, np.inf)
    nearest_idxs = l2_dists.argmin(axis=1)

    # 1. Neighbour consistency
    neighbour_sims = [
        _maze_similarity(m_sub[i], m_sub[nearest_idxs[i]])
        for i in range(n_sub)
    ]
    avg_neighbour_sim = float(np.mean(neighbour_sims))

    # 3. Distance–similarity Spearman correlation
    n_pairs = min(1000, n_sub * (n_sub - 1) // 2)
    pi = rng_np.integers(0, n_sub, n_pairs)
    pj = rng_np.integers(0, n_sub, n_pairs)
    pj[pi == pj] = (pj[pi == pj] + 1) % n_sub
    lat_dists  = l2_dists[pi, pj]
    np.fill_diagonal(l2_dists, 0.0)   # restore for later use
    maze_sims  = np.array([
        _maze_similarity(m_sub[i], m_sub[j])
        for i, j in zip(pi, pj)
    ])
    spearman_rho, spearman_p = spearmanr(lat_dists, maze_sims)

    result = {
        "struct_neighbour_maze_similarity":  avg_neighbour_sim,
        "struct_latent_dist_sim_spearman":   float(spearman_rho),
        "struct_latent_dist_sim_spearman_p": float(spearman_p),
    }

    # 2. Interpolation smoothness — decode using CluttrVAE.decode()
    n_interp_pairs = 50
    n_steps        = 7
    alphas         = np.linspace(0, 1, n_steps)
    pi2 = rng_np.integers(0, n_sub, n_interp_pairs)
    pj2 = rng_np.integers(0, n_sub, n_interp_pairs)
    pj2[pi2 == pj2] = (pj2[pi2 == pj2] + 1) % n_sub

    monotone_count = valid_interp = total_interp = 0

    for a_i, b_i in zip(pi2, pj2):
        za = z_sub[a_i]
        zb = z_sub[b_i]
        z_interp = np.stack([(1 - a) * za + a * zb for a in alphas])  # (n_steps, D)

        # True decode of each interpolated z via CluttrVAE.decode()
        decoded = decode_batch(model, params, z_interp, batch_size=n_steps)

        for seq in decoded:
            seq = seq.flatten().astype(int)
            rw  = _active_walls(seq)
            a_cell, g_cell = int(seq[-1]), int(seq[-2])
            total_interp += 1
            if not (a_cell in rw or g_cell in rw or a_cell == g_cell):
                valid_interp += 1

        sims_to_a = [_maze_similarity(decoded[0], decoded[k])
                     for k in range(n_steps)]
        diffs_mono = np.diff(sims_to_a)
        if (diffs_mono > 0.05).sum() <= 1:
            monotone_count += 1

    result.update({
        "struct_interp_monotonicity_pct": (monotone_count / n_interp_pairs) * 100,
        "struct_interp_validity_pct":     (valid_interp   / total_interp)   * 100,
    })
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Per-checkpoint evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_one_checkpoint(step: int, ckpt_path: str,
                             train_data: np.ndarray, val_data: np.ndarray,
                             n_eval: int, n_prior_samples: int,
                             batch_size: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  Checkpoint step {step:,}   ({ckpt_path})")
    print(f"{'='*60}")

    params     = load_params(ckpt_path)
    model      = CluttrVAE()
    latent_dim = params["mean_layer"]["kernel"].shape[-1]
    rng        = jax.random.key(42)
    row        = {"step": step}

    means_val  = None
    recons_val = None

    for split, data in [("train", train_data[:n_eval]),
                        ("val",   val_data[:n_eval])]:
        print(f"  [{split}] {len(data)} envs ...", end=" ", flush=True)
        orig, recon, means, logvars = reconstruct_dataset(
            model, params, data, batch_size, rng
        )
        r = reconstruction_metrics(orig, recon)
        e = elbo_metrics(orig, recon, means, logvars)
        for k, v in {**r, **e}.items():
            row[f"{split}_{k}"] = v
        print(
            f"valid={r['recon_valid_structure_pct']:.1f}% "
            f"unsolvable={r['recon_unsolvable_pct']:.1f}% "
            f"KL={e['elbo_kl_divergence']:.3f}"
        )
        if e["elbo_kl_collapsed"]:
            print(f"  *** KL COLLAPSE WARNING on {split} split ***")
        if split == "val":
            means_val  = means
            recons_val = recon

    # Generative quality — true prior samples decoded via CluttrVAE.decode()
    print(f"  [gen]    {n_prior_samples} prior samples ...", end=" ", flush=True)
    g = generative_quality_metrics(
        model, params, n_prior_samples, batch_size, latent_dim, rng
    )
    row.update(g)
    print(
        f"solvable={g['gen_prior_solvable_pct']:.1f}% "
        f"valid={g['gen_prior_valid_structure_pct']:.1f}% "
        f"coverage={g['gen_latent_coverage']:.2f}"
    )

    # Latent structure metrics
    print(f"  [struct] latent geometry ...", end=" ", flush=True)
    s = latent_structure_metrics(
        model, params,
        np.array(val_data[:n_eval]),
        means_val, batch_size, rng
    )
    row.update(s)
    print(
        f"neighbour_sim={s['struct_neighbour_maze_similarity']:.3f} "
        f"spearman_rho={s['struct_latent_dist_sim_spearman']:.3f} "
        f"interp_valid={s.get('struct_interp_validity_pct', float('nan')):.1f}%"
    )

    return row


# ═══════════════════════════════════════════════════════════════════════════
# Ranking
# ═══════════════════════════════════════════════════════════════════════════

_INTERNAL_COLS = frozenset({
    "_spearman_neg", "elbo_kl_collapsed",
    "val_elbo_kl_collapsed", "train_elbo_kl_collapsed",
})


def rank_checkpoints(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()

    def norm(col):
        if col not in scored.columns:
            return pd.Series(0.0, index=scored.index)
        mn, mx = scored[col].min(), scored[col].max()
        return (scored[col] - mn) / (mx - mn + 1e-8)

    # Spearman rho is negative when good → negate before normalising
    if "struct_latent_dist_sim_spearman" in scored.columns:
        scored["_spearman_neg"] = -scored["struct_latent_dist_sim_spearman"]
    else:
        scored["_spearman_neg"] = 0.0

    scored["composite_score"] = (
        0.35 * norm("gen_prior_solvable_pct")             +
        0.20 * norm("val_recon_valid_structure_pct")       +
        0.15 * norm("gen_latent_coverage")                 +
        0.15 * norm("struct_neighbour_maze_similarity")    +
        0.10 * norm("_spearman_neg")                       +
        0.05 * norm("gen_perturbation_consistency_pct")
    )

    # Heavy penalty for collapsed checkpoints
    if "val_elbo_kl_divergence" in scored.columns:
        collapsed = scored["val_elbo_kl_divergence"] < KL_COLLAPSE_THRESHOLD
        scored.loc[collapsed, "composite_score"] -= 1.0

    scored["rank"] = scored["composite_score"].rank(
        ascending=False, method="min"
    ).astype(int)
    return scored.sort_values("rank").reset_index(drop=True)


def select_best_checkpoint(df: pd.DataFrame) -> tuple:
    not_collapsed = df[
        df.get("val_elbo_kl_divergence", pd.Series([1.0])) > KL_COLLAPSE_THRESHOLD
    ]
    pool = not_collapsed if len(not_collapsed) > 0 else df
    print(f"\n[Selection] {len(pool)}/{len(df)} non-collapsed checkpoints")

    def norm(col):
        if col not in pool.columns:
            return pd.Series(0.0, index=pool.index)
        mn, mx = pool[col].min(), pool[col].max()
        return (pool[col] - mn) / (mx - mn + 1e-8)

    # Spearman rho is negative when good → negate before normalising
    if "struct_latent_dist_sim_spearman" in pool.columns:
        spearman_neg = -pool["struct_latent_dist_sim_spearman"]
        mn, mx = spearman_neg.min(), spearman_neg.max()
        spearman_norm = (spearman_neg - mn) / (mx - mn + 1e-8)
    else:
        spearman_norm = pd.Series(0.0, index=pool.index)

    score = (
        0.30 * norm("gen_prior_solvable_pct")                +
        0.20 * norm("val_recon_valid_structure_pct")         +
        0.15 * norm("gen_latent_coverage")                   +
        0.12 * norm("struct_neighbour_maze_similarity")      +
        0.08 * spearman_norm                                 +
        0.10 * norm("gen_perturbation_solvable_pct")         +
        0.05 * norm("gen_perturbation_consistency_pct")
    )
    criterion = (
        "composite score: 0.30 × prior_solvable_pct "
        "+ 0.20 × val_valid_structure "
        "+ 0.15 × latent_coverage "
        "+ 0.12 × neighbour_maze_sim "
        "+ 0.08 × latent_dist_spearman "
        "+ 0.10 × perturbation_solvable_pct (critical for exploration) "
        "+ 0.05 × perturbation_consistency"
    )
    best_idx = score.idxmax()
    return int(df.loc[best_idx, "step"]), df.loc[best_idx], criterion


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════

def plot_curves(df: pd.DataFrame, best_step: int, output_dir: str):
    PLOTS = [
        ("gen_prior_solvable_pct",             "Prior Solvable % ★ (ES primary)",         True),
        ("val_recon_valid_structure_pct",       "Val Valid Structure %",                    True),
        ("val_elbo_kl_divergence",              "Val KL Divergence (collapse < 0.05)",      True),
        ("struct_neighbour_maze_similarity",    "Latent Neighbour Maze Similarity ★",       True),
        ("struct_latent_dist_sim_spearman",     "Dist–Similarity Spearman ρ (target: <0)",  False),
        ("struct_interp_monotonicity_pct",      "Interpolation Monotonicity %",             True),
        ("struct_interp_validity_pct",          "Interpolated Maze Validity %",             True),
        ("gen_latent_coverage",                 "Latent Coverage (diversity)",              True),
        ("val_recon_avg_hamming",               "Val Avg Hamming ↓",                        False),
        ("val_recon_unsolvable_pct",            "Val Unsolvable Recons % ↓",                False),
    ]
    plots = [(c, l, h) for c, l, h in PLOTS
             if c in df.columns and not df[c].isna().all()]

    n_cols = 2
    n_rows = (len(plots) + 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4 * n_rows))
    axes = axes.flatten()

    # Composite-score bar chart (first panel)
    ax0 = axes[0]
    df_ranked = df.sort_values("rank") if "rank" in df.columns else df
    colors = ["gold" if s == best_step else "steelblue" for s in df_ranked["step"]]
    bars = ax0.barh(
        [f"step {s:,}" for s in df_ranked["step"]],
        df_ranked["composite_score"],
        color=colors, edgecolor="white", height=0.6
    )
    ax0.set_title("Composite Score Ranking  (★ = best)", fontsize=9, fontweight="bold")
    ax0.set_xlabel("Composite Score")
    ax0.invert_yaxis()
    ax0.grid(axis="x", alpha=0.3)
    for bar, score in zip(bars, df_ranked["composite_score"]):
        ax0.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{score:.3f}", va="center", fontsize=7)

    for ax, (col, lbl, hib) in zip(axes[1:], plots):
        color = "steelblue" if hib else "#c0392b"
        ax.plot(df["step"], df[col], "o-", lw=2, ms=5, color=color)
        if best_step in df["step"].values:
            bv = df.loc[df["step"] == best_step, col].values[0]
            ax.axvline(best_step, color="black", ls="--", lw=1.5, alpha=0.7,
                       label=f"Best: step {best_step:,}")
            ax.scatter([best_step], [bv], color="black", s=80, zorder=5)
        ax.set_title(lbl, fontsize=9, fontweight="bold")
        ax.set_xlabel("Training Step", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        if col == "val_elbo_kl_divergence":
            ax.axhline(KL_COLLAPSE_THRESHOLD, color="orange", ls=":", lw=1.5,
                       label="Collapse threshold")

    for ax in axes[1 + len(plots):]:
        ax.set_visible(False)

    plt.suptitle("CLUTR VAE – Checkpoint Evaluation Dashboard",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "metric_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] metric_curves.png -> {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Output saving
# ═══════════════════════════════════════════════════════════════════════════

RANKING_COLS = [
    "rank", "step", "composite_score",
    "gen_prior_solvable_pct", "gen_prior_valid_structure_pct",
    "gen_prior_avg_wall_count", "gen_latent_coverage",
    "gen_perturbation_consistency_pct", "gen_perturbation_valid_structure_pct",
    "gen_perturbation_solvable_pct",
    "val_recon_valid_structure_pct", "val_recon_unsolvable_pct",
    "val_recon_avg_hamming", "val_recon_agent_accuracy_pct",
    "val_recon_goal_accuracy_pct", "val_recon_avg_wall_count_error",
    "val_elbo_kl_divergence", "val_elbo_collapsed_dims_pct",
    "val_elbo_token_accuracy_pct",
    "struct_neighbour_maze_similarity",
    "struct_latent_dist_sim_spearman", "struct_latent_dist_sim_spearman_p",
    "struct_interp_monotonicity_pct", "struct_interp_validity_pct",
    "train_recon_valid_structure_pct", "train_recon_unsolvable_pct",
    "train_elbo_kl_divergence",
]


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c in _INTERNAL_COLS])


def _to_python(v):
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    return v


def save_all_outputs(df: pd.DataFrame, best_step: int,
                     best_row: pd.Series, criterion: str,
                     output_dir: str):
    """
    Saves all output files under output_dir (which should be
    runs/<run_id>/eval/).

    output_dir/
    ├── ranking.csv
    ├── full_metrics.csv
    ├── best_checkpoint.json
    ├── checkpoints/step_N.json
    ├── report.txt
    └── metric_curves.png   (written by plot_curves)
    """
    df_clean    = _clean_df(df)
    ckpt_subdir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_subdir, exist_ok=True)

    # 1. ranking.csv
    rank_cols   = [c for c in RANKING_COLS if c in df_clean.columns]
    ranking_df  = df_clean[rank_cols].sort_values("rank")
    ranking_path = os.path.join(output_dir, "ranking.csv")
    ranking_df.to_csv(ranking_path, index=False, float_format="%.4f")
    print(f"[Save] ranking.csv          -> {ranking_path}")

    # 2. full_metrics.csv
    full_path = os.path.join(output_dir, "full_metrics.csv")
    df_clean.sort_values("rank").to_csv(full_path, index=False, float_format="%.4f")
    print(f"[Save] full_metrics.csv     -> {full_path}")

    # 3. best_checkpoint.json
    best_dict = {"best_step": int(best_step), "selection_criterion": criterion}
    for k, v in best_row.items():
        if k not in _INTERNAL_COLS:
            best_dict[k] = _to_python(v)
    best_json_path = os.path.join(output_dir, "best_checkpoint.json")
    with open(best_json_path, "w") as f:
        json.dump(best_dict, f, indent=2)
    print(f"[Save] best_checkpoint.json -> {best_json_path}")

    # 4. Per-checkpoint JSON files
    for _, row in df_clean.iterrows():
        step      = int(row["step"])
        json_path = os.path.join(ckpt_subdir, f"step_{step}.json")
        with open(json_path, "w") as f:
            json.dump({k: _to_python(v) for k, v in row.items()}, f, indent=2)
    print(f"[Save] checkpoints/step_N.json -> {ckpt_subdir}/  ({len(df_clean)} files)")

    # 5. report.txt
    def fmt(key, suffix=""):
        v = best_row.get(key, None)
        if v is None or (isinstance(v, float) and v != v):
            return "n/a"
        return f"{v:.3f}{suffix}"

    ranking_table = ranking_df.to_string(index=False, float_format=lambda x: f"{x:.3f}")
    console_cols  = [c for c in [
        "rank", "step", "composite_score",
        "gen_prior_solvable_pct",
        "val_recon_valid_structure_pct",
        "val_elbo_kl_divergence",
        "struct_neighbour_maze_similarity",
        "struct_latent_dist_sim_spearman",
    ] if c in ranking_df.columns]
    console_table = ranking_df[console_cols].to_string(
        index=False, float_format=lambda x: f"{x:.3f}")

    report_lines = [
        "CLUTR VAE CHECKPOINT EVALUATION REPORT",
        "=" * 72,
        "",
        f"Output directory: {os.path.abspath(output_dir)}",
        "",
        "FILES",
        "  ranking.csv            Key metrics, all checkpoints, sorted rank-1 first.",
        "  full_metrics.csv       Every computed column for all checkpoints.",
        "  best_checkpoint.json   Best checkpoint metrics, machine-readable.",
        f"  checkpoints/           One JSON per checkpoint ({len(df_clean)} files).",
        "  metric_curves.png      Dashboard plot of all metrics vs training step.",
        "",
        "=" * 72,
        f"BEST CHECKPOINT: step {best_step:,}",
        f"Criterion: {criterion}",
        "",
        "── Generative quality (what ES uses at runtime) ──",
        f"  gen_prior_solvable_pct          {fmt('gen_prior_solvable_pct', '%')}",
        f"  gen_prior_valid_structure_pct   {fmt('gen_prior_valid_structure_pct', '%')}",
        f"  gen_prior_avg_wall_count        {fmt('gen_prior_avg_wall_count')}   (healthy: 10-40)",
        f"  gen_latent_coverage             {fmt('gen_latent_coverage')}",
        f"  gen_perturbation_consistency    {fmt('gen_perturbation_consistency_pct', '%')}",
        "",
        "── Reconstruction quality ────────────────────────",
        f"  val_recon_valid_structure_pct   {fmt('val_recon_valid_structure_pct', '%')}",
        f"  val_recon_unsolvable_pct        {fmt('val_recon_unsolvable_pct', '%')}   (lower is better)",
        f"  val_recon_avg_hamming           {fmt('val_recon_avg_hamming')}",
        f"  val_recon_agent_accuracy_pct    {fmt('val_recon_agent_accuracy_pct', '%')}",
        f"  val_recon_goal_accuracy_pct     {fmt('val_recon_goal_accuracy_pct', '%')}",
        f"  val_recon_avg_wall_count_error  {fmt('val_recon_avg_wall_count_error')}",
        "",
        "── ELBO / latent health ──────────────────────────",
        f"  val_elbo_kl_divergence          {fmt('val_elbo_kl_divergence')}   (must be > {KL_COLLAPSE_THRESHOLD})",
        f"  val_elbo_collapsed_dims_pct     {fmt('val_elbo_collapsed_dims_pct', '%')}",
        f"  val_elbo_token_accuracy_pct     {fmt('val_elbo_token_accuracy_pct', '%')}",
        "",
        "── Latent structure ──────────────────────────────",
        f"  struct_neighbour_maze_sim       {fmt('struct_neighbour_maze_similarity')}",
        f"  struct_latent_dist_spearman     {fmt('struct_latent_dist_sim_spearman')}   (target: negative)",
        f"  struct_latent_dist_spearman_p   {fmt('struct_latent_dist_sim_spearman_p')}",
        f"  struct_interp_monotonicity_pct  {fmt('struct_interp_monotonicity_pct', '%')}",
        f"  struct_interp_validity_pct      {fmt('struct_interp_validity_pct', '%')}",
        "",
        "── Overfitting check (train vs val) ──────────────",
        f"  train_recon_valid_structure_pct {fmt('train_recon_valid_structure_pct', '%')}",
        f"  train_recon_unsolvable_pct      {fmt('train_recon_unsolvable_pct', '%')}",
        f"  train_elbo_kl_divergence        {fmt('train_elbo_kl_divergence')}",
        "",
        "=" * 72,
        "FULL RANKING",
        "",
        ranking_table,
        "",
        "=" * 72,
        "METRIC INTERPRETATION GUIDE",
        "",
        "  ★ gen_prior_solvable_pct",
        "      The ES teacher samples z~N(0,I) and decodes at runtime.",
        "      Unsolvable mazes give no reward signal — this is the primary metric.",
        "",
        "  ★ struct_neighbour_maze_similarity",
        "      Latent nearest-neighbours should be structurally similar mazes.",
        "      Near-zero = random geometry, ES gradient steps are meaningless.",
        "",
        "  ✓ struct_latent_dist_sim_spearman",
        "      Should be NEGATIVE: far in z-space → structurally different mazes.",
        "      Near 0 = no geometric structure, ES estimation unreliable.",
        "",
        "  ✓ struct_interp_monotonicity_pct",
        "      Smooth z_a → z_b transitions decoded by CluttrVAE.decode().",
        "      Low % = discontinuities / holes in the manifold.",
        "",
        f"  ✓ val_elbo_kl_divergence  (must be > {KL_COLLAPSE_THRESHOLD})",
        "      Posterior collapse: if KL→0 the decoder ignores z entirely.",
    ]

    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"[Save] report.txt           -> {report_path}")

    # Console summary
    print(f"\n{'─'*72}")
    print("CHECKPOINT RANKING (see ranking.csv for full table)")
    print(f"{'─'*72}")
    print(console_table)
    print(f"{'─'*72}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Resolve run directory and output path inside it
    run_dir    = resolve_run_dir(config, args.run_id)
    output_dir = os.path.join(run_dir, "eval")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Run]  {run_dir}")
    print(f"[Eval] outputs -> {output_dir}")

    ckpt_dir  = resolve_ckpt_dir(config, run_dir)
    all_ckpts = discover_checkpoints(ckpt_dir)
    print(f"[Ckpt] Found {len(all_ckpts)} checkpoints in {ckpt_dir}")

    if args.steps:
        requested = {int(s.strip()) for s in args.steps.split(",")}
        all_ckpts = [(s, p) for s, p in all_ckpts if s in requested]
        print(f"[Ckpt] Filtered to steps: {[s for s, _ in all_ckpts]}")

    if not all_ckpts:
        raise ValueError(
            f"No checkpoints found in {ckpt_dir}. "
            "Check --run_id and CONFIG paths."
        )

    train_path, val_path = resolve_data_paths(config)
    train_data = np.load(train_path)
    val_data   = np.load(val_path)
    print(f"[Data] train: {train_data.shape}  val: {val_data.shape}")
    print(f"[Eval] {args.n_eval} envs/split, {args.n_prior_samples} prior samples\n")

    all_results = []
    for step, path in tqdm(all_ckpts, desc="Checkpoints"):
        try:
            row = evaluate_one_checkpoint(
                step, path, train_data, val_data,
                args.n_eval, args.n_prior_samples, args.batch_size
            )
        except Exception as e:
            import traceback
            print(f"  [ERROR] step {step}: {e}")
            traceback.print_exc()
            row = {"step": step, "error": str(e)}
        all_results.append(row)

    df = pd.DataFrame(all_results).sort_values("step").reset_index(drop=True)
    df = rank_checkpoints(df)

    best_step, best_row, criterion = select_best_checkpoint(df)
    save_all_outputs(df, best_step, best_row, criterion, output_dir)
    plot_curves(df, best_step, output_dir)

    print(f"\n{'='*60}")
    print(f"  BEST CHECKPOINT: step {best_step:,}")
    print(f"  All results saved to: {output_dir}")
    print(f"{'='*60}\n")
    return best_step, df


if __name__ == "__main__":
    main()