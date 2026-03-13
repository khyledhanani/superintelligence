#!/usr/bin/env python3
"""Download PCA dataset from GCS and verify all PCA utility functions.

This script:
1. Downloads train_1M_envs.npy from GCS to /tmp/ (skip if exists)
2. Loads CNN-VAE checkpoint and encodes 10k mazes to mu vectors
3. Runs variance pruning (Stage 1) and PCA (Stage 2) analysis
4. Verifies make_variance_pruned_decode_fn and make_pc_decode_fn with vmap
5. Decodes z=zeros(K) through both wrappers -> valid Level check

Usage:
    python scripts/download_pca_dataset.py

Requirements:
    - CNN-VAE checkpoint at vae/checkpoints/cnn_vae/default/
    - GCS credentials at ~/.config/gcloud/legacy_credentials/giacomo.maralla@gmail.com/adc.json
    - google-cloud-storage installed in jax_env
    - Run on a GPU node (sideswipe or prowl), NOT on blaze
"""
import os, sys, json
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'vae'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

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

CKPT_ABS = os.path.join(PROJECT_ROOT, 'vae', 'checkpoints', 'cnn_vae', 'default')
DATASET_LOCAL = '/tmp/train_1M_envs.npy'
DATASET_GCS_BUCKET = 'ucl-ued-project-bucket'
DATASET_GCS_BLOB = 'vae/datasets/train_1M_envs.npy'
GCS_PROJECT = 'open-endedness-personal'
N_MAZES = 10_000
CNN_VAE_LATENT_DIM = 64


def download_dataset():
    """Download training dataset from GCS if not present locally."""
    if os.path.exists(DATASET_LOCAL):
        print(f"Dataset already exists at {DATASET_LOCAL}, skipping download.")
        return

    print(f"Downloading dataset from gs://{DATASET_GCS_BUCKET}/{DATASET_GCS_BLOB} ...")
    cred_path = os.path.expanduser(
        '~/.config/gcloud/legacy_credentials/giacomo.maralla@gmail.com/adc.json'
    )
    from google.oauth2.credentials import Credentials
    from google.cloud import storage

    with open(cred_path) as f:
        cred_data = json.load(f)
    creds = Credentials(
        token=cred_data.get('access_token'),
        refresh_token=cred_data.get('refresh_token'),
        token_uri='https://oauth2.googleapis.com/token',
        client_id=cred_data.get('client_id'),
        client_secret=cred_data.get('client_secret'),
    )
    client = storage.Client(project=GCS_PROJECT, credentials=creds)
    blob = client.bucket(DATASET_GCS_BUCKET).blob(DATASET_GCS_BLOB)
    blob.download_to_filename(DATASET_LOCAL)
    print(f"Downloaded to {DATASET_LOCAL}")


def main():
    # Step 1: Download dataset
    download_dataset()

    # Step 2: Load dataset
    print(f"\nLoading first {N_MAZES} mazes from {DATASET_LOCAL}...")
    data = np.load(DATASET_LOCAL)
    print(f"Full dataset shape: {data.shape}, dtype: {data.dtype}")
    clutr_seqs = data[:N_MAZES]
    print(f"Using {len(clutr_seqs)} mazes for analysis")

    # Step 3: Load CNN-VAE checkpoint
    print(f"\nLoading CNN-VAE checkpoint from {CKPT_ABS}...")
    restored = ocp.PyTreeCheckpointer().restore(CKPT_ABS)
    params = restored['params']
    print(f"Param keys: {list(params.keys())}")

    # Step 4: Encode to mu vectors
    print(f"\nEncoding {N_MAZES} mazes to mu vectors...")
    mu = encode_mazes_to_mu(params, clutr_seqs, latent_dim=CNN_VAE_LATENT_DIM, batch_size=256)
    print(f"mu shape: {mu.shape}, dtype: {mu.dtype}")
    print(f"mu range: [{mu.min():.3f}, {mu.max():.3f}]")
    assert mu.shape == (N_MAZES, CNN_VAE_LATENT_DIM), f"Expected ({N_MAZES}, {CNN_VAE_LATENT_DIM}), got {mu.shape}"
    assert mu.min() >= -4.1 and mu.max() <= 4.1, f"mu out of expected [-4, 4] range: [{mu.min()}, {mu.max()}]"
    print("PASS: mu vectors shape and range correct")

    # Step 5: Active dim selection via weight norms (Stage 1)
    print("\n--- Stage 1: Weight-Norm Dimension Selection ---")
    mean_layer_kernel = np.array(params['mean_layer']['kernel'])  # (512, 64)
    print(f"mean_layer kernel shape: {mean_layer_kernel.shape}")
    kept_dims, per_dim_norms, K_stage1 = compute_active_dims(mean_layer_kernel, cum_threshold=0.85)
    print(f"Per-dim norms (top 10): {np.sort(per_dim_norms)[::-1][:10]}")
    print(f"Kept dims (K={K_stage1}): {kept_dims}")
    cumnorm_kept = per_dim_norms[kept_dims].sum() / per_dim_norms.sum()
    print(f"Cumulative norm of kept dims: {cumnorm_kept*100:.1f}%")
    assert 15 <= K_stage1 <= 55, f"K_stage1={K_stage1} outside expected range [15, 55]"
    print(f"PASS: Stage 1 K={K_stage1} within expected range (no data encoding needed!)")

    # Step 6: PCA (Stage 2)
    print("\n--- Stage 2: PCA ---")
    K_stage2 = min(20, K_stage1)  # Use fewer components for Stage 2
    mu_mean, pc_axes, pc_stds, evr = compute_pca_axes(mu, K_stage2)
    print(f"mu_mean shape: {mu_mean.shape}")
    print(f"pc_axes shape: {pc_axes.shape}")
    print(f"pc_stds shape: {pc_stds.shape}, values: {pc_stds[:5]}")
    print(f"Explained variance ratio (top 5): {evr[:5]}")
    print(f"Total explained variance (top {K_stage2}): {evr.sum()*100:.1f}%")
    assert evr.sum() > 0.5, f"Expected > 50% variance explained, got {evr.sum()*100:.1f}%"
    print(f"PASS: PCA top {K_stage2} PCs explain {evr.sum()*100:.1f}% variance")

    # Step 7: Build base decode_fn
    print("\n--- Decode Function Wrappers ---")
    decoder_params = params['decoder']
    decoder = CnnLstmDecoder(latent_dim=CNN_VAE_LATENT_DIM)

    def base_decode_fn(z):
        z_batched = z[None]
        wl, gl, al = decoder.apply({"params": decoder_params}, z_batched)
        return wl[0], gl[0], al[0]

    # Step 8: Test make_variance_pruned_decode_fn (Stage 1)
    print("\nTesting make_variance_pruned_decode_fn (Stage 1)...")
    mu_mean_training = jnp.zeros(CNN_VAE_LATENT_DIM)  # Use zeros (VAE prior mean) — no data encoding needed
    kept_dims_jnp = jnp.array(kept_dims, dtype=jnp.int32)
    pruned_fn = make_variance_pruned_decode_fn(base_decode_fn, mu_mean_training, kept_dims_jnp)

    rng = jax.random.PRNGKey(42)
    z_zeros_s1 = jnp.zeros((1, K_stage1))
    rng, rng_decode = jax.random.split(rng)
    levels_s1 = decode_latent_to_levels_grid(pruned_fn, z_zeros_s1, rng_decode)
    level_s1 = jax.tree_util.tree_map(lambda x: x[0], levels_s1)
    print(f"Stage 1 z=zeros Level: wall_map.sum()={int(level_s1.wall_map.sum())}, "
          f"goal={level_s1.goal_pos}, agent={level_s1.agent_pos}")
    valid_s1 = jax.vmap(lambda l: l.is_well_formatted())(levels_s1)
    assert bool(valid_s1[0]), "Stage 1 z=zeros level failed is_well_formatted()"
    print("PASS: Stage 1 z=zeros -> valid Level")

    # Test with batch of 32
    z_batch_s1 = jax.random.normal(jax.random.PRNGKey(1), (32, K_stage1))
    rng, rng_decode = jax.random.split(rng)
    levels_batch_s1 = decode_latent_to_levels_grid(pruned_fn, z_batch_s1, rng_decode)
    valid_batch_s1 = jax.vmap(lambda l: l.is_well_formatted())(levels_batch_s1)
    pct_valid_s1 = float(valid_batch_s1.mean()) * 100
    print(f"Stage 1 batch (32): {pct_valid_s1:.0f}% valid")
    print("PASS: Stage 1 vmap over batch of 32 works")

    # Step 9: Test make_pc_decode_fn (Stage 2)
    print("\nTesting make_pc_decode_fn (Stage 2)...")
    mu_mean_j = jnp.array(mu_mean)
    pc_axes_j = jnp.array(pc_axes)
    pc_stds_j = jnp.array(pc_stds)
    pc_fn = make_pc_decode_fn(base_decode_fn, mu_mean_j, pc_axes_j, pc_stds_j)

    z_zeros_s2 = jnp.zeros((1, K_stage2))
    rng, rng_decode = jax.random.split(rng)
    levels_s2 = decode_latent_to_levels_grid(pc_fn, z_zeros_s2, rng_decode)
    level_s2 = jax.tree_util.tree_map(lambda x: x[0], levels_s2)
    print(f"Stage 2 z=zeros Level: wall_map.sum()={int(level_s2.wall_map.sum())}, "
          f"goal={level_s2.goal_pos}, agent={level_s2.agent_pos}")
    valid_s2 = jax.vmap(lambda l: l.is_well_formatted())(levels_s2)
    assert bool(valid_s2[0]), "Stage 2 z=zeros level failed is_well_formatted()"
    print("PASS: Stage 2 z=zeros -> valid Level (maps to average training maze)")

    # Test with batch of 32
    z_batch_s2 = jax.random.normal(jax.random.PRNGKey(2), (32, K_stage2))
    rng, rng_decode = jax.random.split(rng)
    levels_batch_s2 = decode_latent_to_levels_grid(pc_fn, z_batch_s2, rng_decode)
    valid_batch_s2 = jax.vmap(lambda l: l.is_well_formatted())(levels_batch_s2)
    pct_valid_s2 = float(valid_batch_s2.mean()) * 100
    print(f"Stage 2 batch (32): {pct_valid_s2:.0f}% valid")
    print("PASS: Stage 2 vmap over batch of 32 works")

    # Summary
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print(f"  encode_mazes_to_mu: {N_MAZES} mazes -> mu ({mu.shape})")
    print(f"  Stage 1 (weight-norm pruning): K={K_stage1}, cumnorm={cumnorm_kept*100:.1f}%")
    print(f"  Stage 2 (PCA): K={K_stage2}, explained={evr.sum()*100:.1f}%")
    print(f"  Both decode wrappers produce valid Levels via vmap")
    print("=" * 60)


if __name__ == "__main__":
    main()
