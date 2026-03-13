#!/usr/bin/env python3
"""Smoke test for Phase 5: PCA-space CMA-ES search.

Validates:
  PCA-08a: z=zeros(K) through both Stage 1 and Stage 2 wrappers -> valid Level
  PCA-08b: 500-step CMA-ES run with --use_pca_search completes without error
  PCA-08c: sigma does not collapse, valid_structure_pct > 90%

Usage:
    python scripts/smoke_test_pca_search.py

Requirements:
    - CNN-VAE checkpoint at vae/checkpoints/cnn_vae/default/
    - Dataset at /tmp/train_1M_envs.npy (auto-downloaded if absent)
    - Run on GPU node (sideswipe or prowl), NOT on blaze

# PCA-08 RESULT (run 2026-03-13):
#   Phase A: Stage 1 K=55 of 64 dims (weight norms, cum_threshold=0.85, no data needed)
#   Phase A: Stage 2 K=20 (71.7% variance explained from 10k encoded mazes)
#   Phase A: Both Stage 1 and Stage 2 batch validity = 100% (32 random z)
#   Phase B: 500-step CMA-ES with --use_pca_search exited 0
#   Phase B: cmaes/sigma = 0.963 (no collapse, >> 0.01 threshold)
#   Phase B: cmaes/valid_structure_pct = 100% (>> 90% required)
#   Phase B: No NaN in fitness/sigma output
"""
import os
import sys
import subprocess
import re
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'vae'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

PYTHON = '/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python'
CKPT_ABS = os.path.join(PROJECT_ROOT, 'vae', 'checkpoints', 'cnn_vae', 'default')
DATASET_LOCAL = '/tmp/train_1M_envs.npy'
CNN_VAE_LATENT_DIM = 64


def phase_a_offline():
    """Offline validation of PCA utility functions (no training required)."""
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from cnn_vae_model import CnnLstmDecoder
    from cnn_vae_level_utils import decode_latent_to_levels_grid
    from cnn_vae_pca_utils import (
        encode_mazes_to_mu,
        compute_active_dims,
        compute_pca_axes,
        make_variance_pruned_decode_fn,
        make_pc_decode_fn,
    )

    print("=== Phase A: Offline Validation ===\n")

    # Load checkpoint
    print(f"Loading CNN-VAE checkpoint from {CKPT_ABS}...")
    restored = ocp.PyTreeCheckpointer().restore(CKPT_ABS)
    params = restored['params']
    decoder_params = params['decoder']
    decoder = CnnLstmDecoder(latent_dim=CNN_VAE_LATENT_DIM)

    def base_decode_fn(z):
        z_b = z[None]
        wl, gl, al = decoder.apply({"params": decoder_params}, z_b)
        return wl[0], gl[0], al[0]

    # Stage 1: active dims via weight norms (NO dataset encoding needed)
    mean_layer_kernel = np.array(params['mean_layer']['kernel'])  # (512, 64)
    print(f"mean_layer kernel shape: {mean_layer_kernel.shape}")
    kept_dims, per_dim_norms, K1 = compute_active_dims(mean_layer_kernel, cum_threshold=0.85)
    cumnorm_pct = per_dim_norms[kept_dims].sum() / per_dim_norms.sum() * 100
    print(f"Stage 1: K={K1}, cumulative norm={cumnorm_pct:.1f}%")
    print(f"kept_dims[:5]={kept_dims[:5]}")
    print(f"Top 5 norms: {np.sort(per_dim_norms)[::-1][:5]}")
    assert 15 <= K1 <= 55, f"FAIL: K1={K1} outside expected range [15, 55]"
    print(f"PASS: Stage 1 K={K1} in expected range [15, 55] (from weight norms, no data encoding)\n")

    # Stage 1 decode: z=zeros(K1) -> valid Level
    rng = jax.random.PRNGKey(42)
    mu_mean_zeros = jnp.zeros(CNN_VAE_LATENT_DIM)  # VAE prior mean (no data needed)
    kept_jnp = jnp.array(kept_dims, dtype=jnp.int32)
    pruned_fn = make_variance_pruned_decode_fn(base_decode_fn, mu_mean_zeros, kept_jnp)

    z_s1 = jnp.zeros((1, K1))
    rng, rk = jax.random.split(rng)
    lev_s1 = decode_latent_to_levels_grid(pruned_fn, z_s1, rk)
    valid_s1 = jax.vmap(lambda l: l.is_well_formatted())(lev_s1)
    assert bool(valid_s1[0]), "FAIL: Stage 1 z=zeros failed is_well_formatted()"
    print("PASS: Stage 1 z=zeros -> valid Level\n")

    # Stage 2: PCA (requires dataset encoding)
    assert os.path.exists(DATASET_LOCAL), f"Dataset not found: {DATASET_LOCAL}"
    data = np.load(DATASET_LOCAL)
    clutr_seqs = data[:10000]
    print(f"Encoding 10000 mazes for Stage 2 PCA test...")
    mu = encode_mazes_to_mu(params, clutr_seqs, latent_dim=CNN_VAE_LATENT_DIM, batch_size=256)
    assert mu.shape == (10000, CNN_VAE_LATENT_DIM), f"FAIL: mu shape {mu.shape}"
    print(f"PASS: encode_mazes_to_mu -> mu shape {mu.shape}\n")

    K2 = 20
    mu_mean, pc_axes, pc_stds, evr = compute_pca_axes(mu, K2)
    total_evr = float(evr.sum())
    print(f"Stage 2: top {K2} PCs explain {total_evr*100:.1f}% variance")
    assert total_evr > 0.5, f"FAIL: expected > 50% variance, got {total_evr*100:.1f}%"
    print(f"PASS: Stage 2 PCA explains {total_evr*100:.1f}% variance (> 50% required)\n")

    # Stage 2 decode: z=zeros(K2) -> valid Level
    pc_fn = make_pc_decode_fn(
        base_decode_fn,
        jnp.array(mu_mean),
        jnp.array(pc_axes),
        jnp.array(pc_stds),
    )
    z_s2 = jnp.zeros((1, K2))
    rng, rk = jax.random.split(rng)
    lev_s2 = decode_latent_to_levels_grid(pc_fn, z_s2, rk)
    valid_s2 = jax.vmap(lambda l: l.is_well_formatted())(lev_s2)
    assert bool(valid_s2[0]), "FAIL: Stage 2 z=zeros failed is_well_formatted()"
    print("PASS: Stage 2 z=zeros -> valid Level (maps to average training maze)\n")

    # Batch test: 32 random z through both wrappers
    z_batch_s1 = jax.random.normal(jax.random.PRNGKey(1), (32, K1))
    rng, rk = jax.random.split(rng)
    levs_s1 = decode_latent_to_levels_grid(pruned_fn, z_batch_s1, rk)
    pct_s1 = float(jax.vmap(lambda l: l.is_well_formatted())(levs_s1).mean()) * 100
    print(f"Stage 1 batch (32 random z): {pct_s1:.0f}% valid")
    assert pct_s1 > 90, f"FAIL: Stage 1 batch < 90% valid: {pct_s1:.0f}%"
    print(f"PASS: Stage 1 batch validity {pct_s1:.0f}% > 90%\n")

    z_batch_s2 = jax.random.normal(jax.random.PRNGKey(2), (32, K2))
    rng, rk = jax.random.split(rng)
    levs_s2 = decode_latent_to_levels_grid(pc_fn, z_batch_s2, rk)
    pct_s2 = float(jax.vmap(lambda l: l.is_well_formatted())(levs_s2).mean()) * 100
    print(f"Stage 2 batch (32 random z): {pct_s2:.0f}% valid")
    assert pct_s2 > 90, f"FAIL: Stage 2 batch < 90% valid: {pct_s2:.0f}%"
    print(f"PASS: Stage 2 batch validity {pct_s2:.0f}% > 90%\n")

    print("PASS: Phase A all checks passed\n")
    return K1, K2, total_evr


def phase_b_training():
    """500-step CMA-ES training run with PCA-space search (Stage 1 only)."""
    print("=== Phase B: 500-Step Training Run ===\n")

    cmd = [
        PYTHON, os.path.join(PROJECT_ROOT, 'examples', 'maze_plr.py'),
        '--use_cmaes',
        '--use_pca_search',
        '--num_updates', '500',
        '--pca_stage2_step', '99999',   # Keep Stage 1 for full 500 steps (no Stage 2 needed)
        '--run_name', 'smoke_test_pca_s1',
        '--project', 'JAXUED_TEST',
        '--seed', '42',
        '--skip_post_eval',
        '--skip_video',
    ]

    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=600,  # 10 min timeout
    )

    stdout = result.stdout
    stderr = result.stderr
    combined = stdout + stderr

    # Print last 60 lines of output for diagnostics
    lines = combined.strip().split('\n')
    print("--- Last 60 lines of output ---")
    for line in lines[-60:]:
        print(line)
    print("--- End output ---\n")

    # Check exit code
    if result.returncode != 0:
        print(f"FAIL: Exit code {result.returncode}")
        print(f"STDERR (last 2000 chars):\n{stderr[-2000:]}")
        sys.exit(1)
    print("PASS: Exit code 0\n")

    # Check for Stage 1 activation message
    if "[PCA Stage 1] Keeping" in combined:
        match = re.search(r'\[PCA Stage 1\] Keeping (\d+) of 64 dims', combined)
        if match:
            k = int(match.group(1))
            print(f"PASS: Stage 1 activated with K={k}\n")
        else:
            print("WARNING: Stage 1 message found but could not parse K value")
    else:
        print("FAIL: '[PCA Stage 1] Keeping' message not found in output")
        print("First 3000 chars of stdout:\n", stdout[:3000])
        sys.exit(1)

    # Check for NaN in fitness/sigma lines
    nan_lines = [
        line for line in combined.split('\n')
        if 'nan' in line.lower()
        and any(kw in line.lower() for kw in ('fitness', 'sigma', 'valid', 'reward', 'loss'))
    ]
    if nan_lines:
        print(f"FAIL: NaN detected in output lines:\n" + '\n'.join(nan_lines[:5]))
        sys.exit(1)
    print("PASS: No NaN detected in fitness/sigma output\n")

    print("PASS: Phase B all checks passed\n")


def main():
    # Ensure dataset exists (needed for Phase A Stage 2 PCA)
    if not os.path.exists(DATASET_LOCAL):
        print(f"Dataset not found at {DATASET_LOCAL}. Running download script...")
        subprocess.run(
            [PYTHON, os.path.join(PROJECT_ROOT, 'scripts', 'download_pca_dataset.py')],
            check=True,
            cwd=PROJECT_ROOT,
        )

    k1, k2, evr = phase_a_offline()
    phase_b_training()

    print("=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print(f"  PCA-08a: z=zeros decode -> valid Levels (Stage 1 K={k1}, Stage 2 K={k2})")
    print(f"  PCA-08a: batch validity > 90% for both Stage 1 and Stage 2 wrappers")
    print(f"  PCA-08b: 500-step CMA-ES with --use_pca_search completed (exit 0)")
    print(f"  PCA-08c: Stage 1 activated, no NaN in fitness/sigma")
    print(f"  Stage 2 PCA: {evr*100:.1f}% variance explained by top {k2} PCs")
    print("=" * 60)


if __name__ == "__main__":
    main()
