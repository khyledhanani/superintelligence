"""
Phase 1: Collect (z, tokens, regret) training data for the adapter.

Samples latent vectors from the VAE prior N(0,I), decodes them to levels,
rolls out the trained student agent, and computes regret scores.
Optionally merges with existing CMA-ES buffer data.

Usage:
    python adapter/collect_data.py \
        --agent_checkpoint /path/to/accel/checkpoints/run_name/seed/models \
        --vae_checkpoint_path /tmp/vae_beta10/checkpoint_500000.pkl \
        --vae_config_path /tmp/vae_beta10/config.yaml \
        --n_prior_samples 15000 \
        --buffer_path /path/to/buffer_dump.npz \
        --output_path /tmp/adapter_data/train_data.npz \
        --score_function maxmc \
        --batch_size 256
"""
import argparse
import os
import sys
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml
import orbax.checkpoint as ocp

# Add parent dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vae"))

from vae_model import CluttrVAE
from vae_level_utils import decode_latent_to_levels, level_to_tokens, tokens_to_level, repair_tokens
from jaxued.environments.underspecified_env import EnvParams
from jaxued.environments.maze import Maze, Level, make_level_generator
from jaxued.wrappers import AutoReplayWrapper


def load_vae(vae_checkpoint_path, vae_config_path):
    """Load frozen VAE and return (decode_fn, encode_fn, vae_cfg)."""
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

    def decode_fn(z):
        return vae.apply({"params": vae_params}, z, method=vae.decode)

    def encode_fn(tokens):
        return vae.apply({"params": vae_params}, tokens, train=False, method=vae.encode)

    return decode_fn, encode_fn, vae_cfg


def load_agent(agent_checkpoint_path, env, env_params, agent_view_size=5):
    """Load trained agent from orbax checkpoint. Returns (train_state, network)."""
    from flax import linen as nn
    from flax.linen.initializers import constant, orthogonal
    import distrax
    import optax

    # Import ActorCritic and ResetRNN from maze_plr
    # We redefine here to avoid circular imports
    class ResetRNN(nn.Module):
        cell: nn.RNNCellBase

        @nn.compact
        def __call__(self, inputs, initial_carry=None):
            x, resets = inputs
            carry = self.variable("carry", "hidden", lambda: initial_carry).value
            carry = jax.tree_util.tree_map(
                lambda c, ic: jnp.where(resets[:, None], ic, c), carry, initial_carry
            )
            carry, y = self.cell(carry, x)
            self.variable("carry", "hidden").value = carry
            return carry, y

    class ActorCritic(nn.Module):
        action_dim: int

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

    network = ActorCritic(action_dim=env.action_space(env_params).n)

    # Load checkpoint
    ws_manager = ocp.CheckpointManager(
        agent_checkpoint_path,
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    ws_step = ws_manager.latest_step()
    ws_ckpt = ws_manager.restore(ws_step)
    agent_params = ws_ckpt["params"] if isinstance(ws_ckpt, dict) and "params" in ws_ckpt else ws_ckpt.params
    print(f"[Agent] Loaded checkpoint from step {ws_step}")

    return agent_params, network, ActorCritic


def rollout_levels(rng, levels, agent_params, network, ActorCritic, env, env_params, max_steps=250):
    """Roll out the agent on a batch of levels and return (rewards_sum, max_returns, values, advantages).

    Args:
        levels: batched Level with leading dimension N
        Returns per-level regret scores.
    """
    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]

    # Reset env to these levels
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(env.reset_to_level, (0, 0, None))(
        jax.random.split(rng_reset, num_levels), levels, env_params
    )

    # Run evaluation rollout
    init_hstate = ActorCritic.initialize_carry((num_levels,))

    def step(carry, _):
        rng, hstate, obs, state, done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, value = network.apply({"params": agent_params}, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)
        value = value.squeeze(0)

        next_obs, next_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)

        return (rng, hstate, next_obs, next_state, done), (reward, done, value)

    rng, rng_rollout = jax.random.split(rng)
    _, (rewards, dones, values) = jax.lax.scan(
        step,
        (rng_rollout, init_hstate, init_obs, init_env_state, jnp.zeros(num_levels, dtype=bool)),
        None,
        length=max_steps,
    )

    # rewards: (T, N), dones: (T, N), values: (T, N)
    return rewards, dones, values


def compute_maxmc_regret(rewards, dones, values):
    """Compute MaxMC regret: max(cumulative_return) - V(s0).

    Simple proxy: total episode return (sum of rewards until first done).
    MaxMC = max over time of (discounted return - value estimate).
    For simplicity, use: regret = max(0, optimal_return - achieved_return).
    """
    # Sum rewards per level
    total_returns = rewards.sum(axis=0)  # (N,)
    # Max returns: maximum cumulative reward achievable
    max_returns = jnp.maximum(total_returns, 0.0)
    # Initial value estimate
    v0 = values[0]  # (N,)
    # Regret = how much better the agent could have done
    regret = jnp.maximum(max_returns - v0, 0.0)
    return regret


def compute_mna_regret(rewards, dones, values):
    """MNA: mean of clipped negative advantages over trajectory."""
    # Compute TD advantages
    gamma = 0.995
    gae_lambda = 0.95

    # Simple GAE computation
    T, N = rewards.shape
    advantages = jnp.zeros(N)
    gae = jnp.zeros(N)

    # Reverse scan for GAE
    def gae_step(carry, t_data):
        gae = carry
        reward, done, value, next_value = t_data
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return gae, gae

    # Append zero value at end
    next_values = jnp.concatenate([values[1:], jnp.zeros((1, N))], axis=0)
    _, all_advantages = jax.lax.scan(
        gae_step,
        jnp.zeros(N),
        (rewards[::-1], dones[::-1], values[::-1], next_values[::-1]),
    )
    all_advantages = all_advantages[::-1]  # (T, N)

    # MNA = mean(-min(advantages, 0))
    neg_adv = -jnp.minimum(all_advantages, 0.0)
    # Only count steps before first done
    mask = jnp.cumprod(1 - dones.astype(jnp.float32), axis=0)
    mna = (neg_adv * mask).sum(axis=0) / jnp.maximum(mask.sum(axis=0), 1.0)
    return mna


def compute_sfl_scores(rng, levels, agent_params, network, ActorCritic, env, env_params,
                       max_steps=250, n_rollouts=10):
    """SFL: success_rate * (1 - success_rate). Requires multiple rollouts."""
    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]
    all_successes = []

    for i in range(n_rollouts):
        rng, rng_roll = jax.random.split(rng)
        rewards, dones, _ = rollout_levels(
            rng_roll, levels, agent_params, network, ActorCritic, env, env_params, max_steps
        )
        success = (rewards.sum(axis=0) > 0).astype(jnp.float32)
        all_successes.append(success)

    all_successes = jnp.stack(all_successes)  # (n_rollouts, N)
    p = all_successes.mean(axis=0)
    return p * (1 - p)


def main():
    parser = argparse.ArgumentParser(description="Collect (z, tokens, regret) training data for adapter")
    parser.add_argument("--agent_checkpoint", type=str, required=True,
                        help="Path to orbax checkpoint dir (e.g., .../models)")
    parser.add_argument("--vae_checkpoint_path", type=str, required=True)
    parser.add_argument("--vae_config_path", type=str, required=True)
    parser.add_argument("--n_prior_samples", type=int, default=15000,
                        help="Number of latent vectors to sample from prior N(0,I)")
    parser.add_argument("--buffer_path", type=str, default=None,
                        help="Path to buffer_dump.npz to merge with prior samples")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save the collected data (.npz)")
    parser.add_argument("--score_function", type=str, default="maxmc",
                        choices=["maxmc", "mna", "sfl"],
                        help="Which regret measure to use")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for rollouts (limited by GPU memory)")
    parser.add_argument("--max_steps", type=int, default=250,
                        help="Max episode length for rollouts")
    parser.add_argument("--n_sfl_rollouts", type=int, default=10,
                        help="Number of rollouts per level for SFL scoring")
    parser.add_argument("--agent_view_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # --- Load VAE ---
    print("[1/4] Loading VAE...")
    decode_fn, encode_fn, vae_cfg = load_vae(args.vae_checkpoint_path, args.vae_config_path)
    latent_dim = vae_cfg["latent_dim"]
    print(f"  latent_dim={latent_dim}")

    # --- Load agent ---
    print("[2/4] Loading agent...")
    env = Maze(max_height=13, max_width=13, agent_view_size=args.agent_view_size, normalize_obs=True)
    eval_env = env  # unwrapped for reset_to_level
    env_params = env.default_params

    agent_params, network, ActorCriticCls = load_agent(
        args.agent_checkpoint, env, env_params, args.agent_view_size
    )

    # --- Sample latent vectors ---
    print(f"[3/4] Sampling {args.n_prior_samples} latent vectors from prior...")
    rng, rng_sample = jax.random.split(rng)
    z_prior = jax.random.normal(rng_sample, (args.n_prior_samples, latent_dim))

    # Decode to levels and get tokens
    all_z = []
    all_tokens = []
    all_regret = []

    # JIT the rollout
    @jax.jit
    def _rollout_batch(rng, levels):
        return rollout_levels(rng, levels, agent_params, network, ActorCriticCls, eval_env, env_params, args.max_steps)

    @jax.jit
    def _decode_batch(z_batch, rng):
        return decode_latent_to_levels(decode_fn, z_batch, rng)

    @jax.jit
    def _get_tokens(levels):
        return jax.vmap(level_to_tokens)(levels)

    # Process prior samples in batches
    n_batches = (args.n_prior_samples + args.batch_size - 1) // args.batch_size
    print(f"  Processing {n_batches} batches of size {args.batch_size}...")
    t0 = time.time()

    for i in range(n_batches):
        start = i * args.batch_size
        end = min(start + args.batch_size, args.n_prior_samples)
        z_batch = z_prior[start:end]

        # Decode latent -> levels
        rng, rng_decode, rng_roll = jax.random.split(rng, 3)
        levels = _decode_batch(z_batch, rng_decode)
        tokens = _get_tokens(levels)

        # Rollout agent
        rewards, dones, values = _rollout_batch(rng_roll, levels)

        # Compute regret
        if args.score_function == "maxmc":
            regret = compute_maxmc_regret(rewards, dones, values)
        elif args.score_function == "mna":
            regret = compute_mna_regret(rewards, dones, values)
        elif args.score_function == "sfl":
            regret = compute_sfl_scores(
                rng_roll, levels, agent_params, network, ActorCriticCls,
                eval_env, env_params, args.max_steps, args.n_sfl_rollouts
            )

        all_z.append(np.array(z_batch))
        all_tokens.append(np.array(tokens))
        all_regret.append(np.array(regret))

        if (i + 1) % 10 == 0 or i == n_batches - 1:
            elapsed = time.time() - t0
            print(f"  Batch {i+1}/{n_batches} done ({elapsed:.1f}s elapsed)")

    # --- Merge with buffer if provided ---
    if args.buffer_path and os.path.exists(args.buffer_path):
        print(f"[3b] Merging with buffer from {args.buffer_path}...")
        buf = np.load(args.buffer_path, allow_pickle=True)
        buf_tokens = buf["tokens"]  # (N, 52)
        buf_scores = buf["scores"]  # (N,)
        buf_size = int(buf["size"]) if "size" in buf else len(buf_tokens)
        buf_tokens = buf_tokens[:buf_size]
        buf_scores = buf_scores[:buf_size]

        # Encode buffer tokens to get z
        print(f"  Encoding {buf_size} buffer levels...")
        buf_z = []
        for j in range(0, buf_size, args.batch_size):
            end_j = min(j + args.batch_size, buf_size)
            means, _ = encode_fn(jnp.array(buf_tokens[j:end_j]))
            buf_z.append(np.array(means))
        buf_z = np.concatenate(buf_z, axis=0)

        all_z.append(buf_z)
        all_tokens.append(buf_tokens)
        all_regret.append(buf_scores)
        print(f"  Added {buf_size} buffer levels (total will be {sum(len(x) for x in all_z)})")

    # --- Save ---
    all_z = np.concatenate(all_z, axis=0)
    all_tokens = np.concatenate(all_tokens, axis=0)
    all_regret = np.concatenate(all_regret, axis=0)

    print(f"[4/4] Saving {len(all_z)} samples to {args.output_path}...")
    print(f"  z shape: {all_z.shape}")
    print(f"  tokens shape: {all_tokens.shape}")
    print(f"  regret stats: mean={all_regret.mean():.4f}, std={all_regret.std():.4f}, "
          f"min={all_regret.min():.4f}, max={all_regret.max():.4f}")

    np.savez(
        args.output_path,
        z=all_z,
        tokens=all_tokens,
        regret=all_regret,
        score_function=args.score_function,
        latent_dim=latent_dim,
    )
    print("Done.")


if __name__ == "__main__":
    main()
