"""
Prepare adapter training data from a buffer dump + prior samples + KL dimension eviction.

1. Load buffer levels (tokens + scores) and encode through VAE to get z
2. Sample additional latent vectors from prior N(0,I), decode to levels
3. Score prior levels using agent rollouts (SFL or other score function)
4. Apply KL dimension filtering to zero out dead dims
5. Split into train/test sets

Usage:
    python adapter/prepare_data_from_buffer.py \
        --buffer_path /path/to/buffer_dump_10k.npz \
        --vae_checkpoint_path /tmp/vae_beta10/checkpoint_500000.pkl \
        --vae_config_path /tmp/vae_beta10/config.yaml \
        --agent_checkpoint_path /tmp/agent_checkpoint/40 \
        --num_rollouts 5 \
        --kl_threshold 0.1 \
        --prior_multiplier 3 \
        --test_split 0.2 \
        --output_path /tmp/adapter_data/train_data.npz
"""
import argparse
import os
import sys
import pickle
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import yaml
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState as BaseTrainState
import distrax
import chex
import optax
import orbax.checkpoint as ocp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vae"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from vae_model import CluttrVAE
from vae_level_utils import decode_latent_to_levels, level_to_tokens, tokens_to_level
from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.linen import ResetRNN
from jaxued.environments import Maze


class ActorCritic(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs
        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)
        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)
        embedding = jnp.append(img_embed, dir_embed, axis=-1)
        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)
        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)
        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))


def evaluate_rnn(rng, env, env_params, train_state, init_hstate, init_obs, init_env_state, max_episode_length):
    """Run RNN agent on levels, return (states, rewards, episode_lengths)."""
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]

    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)
        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)
        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)
        next_mask = mask & ~done
        episode_length += mask
        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (state, reward)

    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (rng, init_hstate, init_obs, init_env_state,
         jnp.zeros(num_levels, dtype=bool), jnp.ones(num_levels, dtype=bool),
         jnp.zeros(num_levels, dtype=jnp.int32)),
        None, length=max_episode_length,
    )
    return states, rewards, episode_lengths


def score_levels_sfl(rng, env, env_params, train_state, levels, num_rollouts, rollout_batch_size=256):
    """Score levels using SFL = p * (1 - p) from multiple rollouts.

    Processes levels in batches to avoid OOM.
    """
    n_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]
    max_steps = env_params.max_steps_in_episode

    @jax.jit
    def _score_batch(rng, batch_levels):
        n_batch = jax.tree_util.tree_flatten(batch_levels)[0][0].shape[0]

        def single_rollout(rng_roll):
            rng_r, rng_e = jax.random.split(rng_roll)
            init_obs, init_env_state = jax.vmap(env.reset_to_level, (0, 0, None))(
                jax.random.split(rng_r, n_batch), batch_levels, env_params)
            _, rewards, _ = evaluate_rnn(
                rng_e, env, env_params, train_state,
                ActorCritic.initialize_carry((n_batch,)),
                init_obs, init_env_state, max_steps)
            return (rewards.sum(axis=0) > 0).astype(jnp.float32)

        rollout_rngs = jax.random.split(rng, num_rollouts)
        successes = jax.vmap(single_rollout)(rollout_rngs)  # (num_rollouts, n_batch)
        p = successes.mean(axis=0)
        return p * (1 - p)

    all_scores = []
    for i in range(0, n_levels, rollout_batch_size):
        end = min(i + rollout_batch_size, n_levels)
        batch = jax.tree_util.tree_map(lambda x: x[i:end], levels)
        rng, rng_batch = jax.random.split(rng)
        scores = _score_batch(rng_batch, batch)
        all_scores.append(np.array(scores))
        print(f"    Scored {end}/{n_levels} levels...")

    return np.concatenate(all_scores, axis=0)


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
    parser.add_argument("--agent_checkpoint_path", type=str, default=None,
                        help="Path to orbax agent checkpoint dir (e.g. /tmp/agent/40). "
                             "Required for scoring prior samples via rollouts.")
    parser.add_argument("--num_rollouts", type=int, default=5,
                        help="Number of rollouts per level for SFL scoring")
    parser.add_argument("--agent_view_size", type=int, default=5)
    parser.add_argument("--rollout_batch_size", type=int, default=256,
                        help="Batch size for agent rollouts (reduce if OOM)")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.agent_checkpoint_path is None:
        print("WARNING: No --agent_checkpoint_path provided. Prior samples will get score=0.")
        print("         For proper scoring, provide the agent checkpoint.")

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

    # --- Score prior samples ---
    if args.agent_checkpoint_path is not None:
        print(f"  Scoring prior samples with agent rollouts ({args.num_rollouts} rollouts each)...")

        # Set up environment
        env = Maze(max_height=13, max_width=13, agent_view_size=args.agent_view_size, normalize_obs=True)
        env_params = env.default_params

        # Load agent checkpoint (orbax)
        network = ActorCritic(action_dim=env.action_space(env_params).n)
        # Init network to get param structure
        rng, rng_init = jax.random.split(rng)
        init_obs_dummy, _ = env.reset_to_level(
            jax.random.PRNGKey(0),
            jax.tree_util.tree_map(lambda x: x[0], decode_latent_to_levels(decode_fn, jnp.zeros((1, latent_dim)), jax.random.PRNGKey(0))),
            env_params,
        )
        init_x = jax.tree_util.tree_map(lambda x: x[None, None, ...], (init_obs_dummy, jnp.zeros((), dtype=bool)))
        network_params = network.init(rng_init, init_x, ActorCritic.initialize_carry((1,)))

        # Create a minimal train state for evaluate_rnn
        tx = optax.sgd(1e-3)  # dummy optimizer, not used
        agent_train_state = BaseTrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx,
        )

        # Restore checkpoint params
        print(f"  Loading agent from {args.agent_checkpoint_path}...")
        ckpt_mgr = ocp.CheckpointManager(os.path.dirname(args.agent_checkpoint_path))
        step = int(os.path.basename(args.agent_checkpoint_path))
        restored = ckpt_mgr.restore(step)
        agent_params = restored["params"] if isinstance(restored, dict) and "params" in restored else restored.params
        agent_train_state = agent_train_state.replace(params=agent_params)

        # Decode prior z to levels (not tokens — we need Level dataclass for rollouts)
        print(f"  Decoding {n_prior} prior samples to levels for rollout...")
        rng, rng_decode = jax.random.split(rng)
        prior_levels_list = []
        for i in range(0, n_prior, args.batch_size):
            end = min(i + args.batch_size, n_prior)
            rng_decode, rng_dec_batch = jax.random.split(rng_decode)
            lvls = decode_latent_to_levels(decode_fn, jnp.array(prior_z[i:end]), rng_dec_batch)
            prior_levels_list.append(lvls)
        # Stack all level batches
        prior_levels = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *prior_levels_list)

        # Score with SFL
        rng, rng_score = jax.random.split(rng)
        prior_scores = score_levels_sfl(
            rng_score, env, env_params, agent_train_state, prior_levels,
            args.num_rollouts, args.rollout_batch_size,
        )
        print(f"  Prior SFL scores: mean={prior_scores.mean():.4f}, std={prior_scores.std():.4f}, "
              f"max={prior_scores.max():.4f}, nonzero={np.count_nonzero(prior_scores)}/{n_prior}")
    else:
        prior_scores = np.zeros(n_prior, dtype=np.float32)
        print(f"  Prior samples: {n_prior} levels with score=0 (no agent provided)")

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
