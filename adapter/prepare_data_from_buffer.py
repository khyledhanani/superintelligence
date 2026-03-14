"""
Prepare adapter training data from a buffer dump + prior samples + KL dimension eviction.

1. Load buffer levels (tokens + scores) and encode through VAE to get z
2. Sample additional latent vectors from prior N(0,I), decode to tokens, assign regret=0
   (prior samples provide broad coverage; buffer samples provide frontier detail)
3. Apply KL dimension filtering to zero out dead dims
4. Split into train/test sets

Usage:
    python adapter/prepare_data_from_buffer.py \
        --buffer_path /path/to/buffer_dump_10k.npz \
        --vae_checkpoint_path /tmp/vae_beta10/checkpoint_500000.pkl \
        --vae_config_path /tmp/vae_beta10/config.yaml \
        --kl_threshold 0.1 \
        --prior_multiplier 3 \
        --test_split 0.2 \
        --output_path /tmp/adapter_data/train_data.npz
"""
import argparse
import os
import sys
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vae"))
from vae_model import CluttrVAE
from vae_level_utils import decode_latent_to_levels, level_to_tokens


def main():
    parser = argparse.ArgumentParser(description="Prepare adapter data from buffer + prior samples with KL filtering")
    parser.add_argument("--buffer_path", type=str, required=True)
    parser.add_argument("--vae_checkpoint_path", type=str, required=True)
    parser.add_argument("--vae_config_path", type=str, required=True)
    parser.add_argument("--kl_threshold", type=float, default=0.1,
                        help="KL threshold for dim eviction (dims with KL < threshold are zeroed)")
    parser.add_argument("--prior_multiplier", type=int, default=3,
                        help="Sample N * buffer_size latent vectors from prior (default: 3x)")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="Fraction of data to hold out for test set")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # --- Load VAE ---
    print("[1/5] Loading VAE...")
    with open(args.vae_config_path) as f:
        vae_cfg = yaml.safe_load(f)
    vae = CluttrVAE(
        vocab_size=vae_cfg["vocab_size"],
        embed_dim=vae_cfg["embed_dim"],
        latent_dim=vae_cfg["latent_dim"],
        seq_len=vae_cfg["seq_len"],
    )
    with open(args.vae_checkpoint_path, "rb") as f:
        vae_ckpt = pickle.load(f)
    vae_params = vae_ckpt["params"] if isinstance(vae_ckpt, dict) and "params" in vae_ckpt else vae_ckpt
    latent_dim = vae_cfg["latent_dim"]
    print(f"  latent_dim={latent_dim}")

    @jax.jit
    def encode_batch(tokens):
        return vae.apply({"params": vae_params}, tokens, train=False, method=vae.encode)

    def decode_fn(z):
        return vae.apply({"params": vae_params}, z, method=vae.decode)

    @jax.jit
    def decode_to_tokens(z_batch, rng):
        """Decode latent vectors to token sequences via argmax."""
        logits = jax.vmap(decode_fn)(z_batch)  # (B, seq_len, vocab_size)
        return jnp.argmax(logits, axis=-1)  # (B, seq_len)

    # --- Load buffer ---
    print(f"[2/5] Loading buffer from {args.buffer_path}...")
    buf = np.load(args.buffer_path, allow_pickle=True)
    buf_tokens_raw = buf["tokens"]
    buf_scores_raw = buf["scores"]
    buf_size = int(buf["size"]) if "size" in buf else len(buf_tokens_raw)
    buf_tokens = jnp.array(buf_tokens_raw[:buf_size])
    buf_scores = np.array(buf_scores_raw[:buf_size])
    print(f"  {buf_size} buffer levels")
    print(f"  Buffer scores: mean={buf_scores.mean():.4f}, std={buf_scores.std():.4f}, "
          f"min={buf_scores.min():.4f}, max={buf_scores.max():.4f}")

    # Encode buffer to get z
    print("  Encoding buffer levels...")
    buf_means = []
    buf_logvars = []
    for i in range(0, buf_size, args.batch_size):
        end = min(i + args.batch_size, buf_size)
        means, logvars = encode_batch(buf_tokens[i:end])
        buf_means.append(np.array(means))
        buf_logvars.append(np.array(logvars))
    buf_z = np.concatenate(buf_means, axis=0)
    buf_logvar = np.concatenate(buf_logvars, axis=0)

    # --- Sample from prior ---
    n_prior = args.prior_multiplier * buf_size
    print(f"[3/5] Sampling {n_prior} latent vectors from prior N(0,I)...")
    rng, rng_sample = jax.random.split(rng)
    prior_z = np.array(jax.random.normal(rng_sample, (n_prior, latent_dim)))

    # Decode prior samples to get tokens (for reconstruction target)
    print(f"  Decoding prior samples to tokens...")
    prior_tokens = []
    for i in range(0, n_prior, args.batch_size):
        end = min(i + args.batch_size, n_prior)
        rng, rng_dec = jax.random.split(rng)
        tok = decode_to_tokens(jnp.array(prior_z[i:end]), rng_dec)
        prior_tokens.append(np.array(tok))
    prior_tokens = np.concatenate(prior_tokens, axis=0)

    # Prior samples get regret=0 (unknown difficulty, neutral)
    prior_scores = np.zeros(n_prior, dtype=np.float32)
    print(f"  Prior samples: {n_prior} levels with regret=0 (neutral)")

    # Encode prior tokens back through VAE to get consistent z
    # (the prior_z we sampled won't perfectly match decode→encode round-trip)
    print(f"  Re-encoding prior tokens for consistent z...")
    prior_z_enc = []
    prior_logvar_enc = []
    for i in range(0, n_prior, args.batch_size):
        end = min(i + args.batch_size, n_prior)
        means, logvars = encode_batch(jnp.array(prior_tokens[i:end]))
        prior_z_enc.append(np.array(means))
        prior_logvar_enc.append(np.array(logvars))
    prior_z_enc = np.concatenate(prior_z_enc, axis=0)
    prior_logvar_enc = np.concatenate(prior_logvar_enc, axis=0)

    # --- Combine ---
    all_z = np.concatenate([buf_z, prior_z_enc], axis=0)
    all_tokens = np.concatenate([np.array(buf_tokens), prior_tokens], axis=0)
    all_scores = np.concatenate([buf_scores, prior_scores], axis=0)
    all_logvar = np.concatenate([buf_logvar, prior_logvar_enc], axis=0)
    # Track source: 1 = buffer, 0 = prior
    source = np.concatenate([np.ones(buf_size), np.zeros(n_prior)])
    n_total = len(all_z)
    print(f"\n  Combined: {n_total} samples ({buf_size} buffer + {n_prior} prior)")

    # --- KL dimension filtering ---
    print(f"[4/5] KL dimension eviction (threshold={args.kl_threshold})...")
    # Compute KL on BUFFER data only (prior samples are from N(0,I) so KL=0 by construction)
    mean_enc = jnp.array(buf_z)
    logvar_enc = jnp.array(buf_logvar)
    kl_per_dim = -0.5 * (1 + logvar_enc - mean_enc ** 2 - jnp.exp(logvar_enc))
    kl_per_dim = jnp.mean(kl_per_dim, axis=0)

    active_mask = kl_per_dim > args.kl_threshold
    active_dims = np.where(np.array(active_mask))[0]
    n_active = len(active_dims)
    dead_mask = ~np.array(active_mask)
    n_dead = int(dead_mask.sum())
    print(f"  Active dims: {n_active}/{latent_dim} ({n_dead} dead)")
    print(f"  Active indices: {active_dims.tolist()}")
    print(f"  Top KL values: {np.sort(np.array(kl_per_dim))[::-1][:10].tolist()}")

    # Zero out dead dims
    all_z[:, dead_mask] = 0.0

    # --- Train/test split ---
    print(f"[5/5] Splitting into train/test (test_split={args.test_split})...")
    n_test = int(n_total * args.test_split)
    n_train = n_total - n_test

    perm = np.random.RandomState(args.seed).permutation(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    # Count buffer/prior in each split
    train_buf = int(source[train_idx].sum())
    train_prior = n_train - train_buf
    test_buf = int(source[test_idx].sum())
    test_prior = n_test - test_buf
    print(f"  Train: {n_train} ({train_buf} buffer + {train_prior} prior)")
    print(f"  Test:  {n_test} ({test_buf} buffer + {test_prior} prior)")

    # Save train
    train_path = args.output_path
    np.savez(
        train_path,
        z=all_z[train_idx],
        tokens=all_tokens[train_idx],
        regret=all_scores[train_idx],
        source=source[train_idx],
        latent_dim=latent_dim,
        active_dims=active_dims,
        kl_per_dim=np.array(kl_per_dim),
        kl_threshold=args.kl_threshold,
        n_buffer=buf_size,
        n_prior=n_prior,
    )
    print(f"\n  Train data: {train_path} ({n_train} samples)")

    # Save test alongside train
    test_path = train_path.replace(".npz", "_test.npz")
    np.savez(
        test_path,
        z=all_z[test_idx],
        tokens=all_tokens[test_idx],
        regret=all_scores[test_idx],
        source=source[test_idx],
        latent_dim=latent_dim,
        active_dims=active_dims,
    )
    print(f"  Test data:  {test_path} ({n_test} samples)")

    # Summary
    print(f"\n  Total: {n_total} = {buf_size} buffer + {n_prior} prior ({args.prior_multiplier}x)")
    print(f"  Active latent dims: {n_active}/{latent_dim}")
    print(f"  Regret distribution (buffer only): mean={buf_scores.mean():.4f}, std={buf_scores.std():.4f}")


if __name__ == "__main__":
    main()
