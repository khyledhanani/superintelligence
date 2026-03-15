"""
Collect agent trajectories (position sequences) on random/buffer mazes.

For each level, records:
  - trajectory: (max_steps,) position sequence [y*13 + x + 1, padded with 0]
  - regret: scalar MaxMC or other score
  - episode_length: int
  - reached_goal: bool
  - tokens: (52,) level tokens (for cross-referencing with level VAE)

Usage:
    python trajectory/collect_trajectories.py \
        --agent_checkpoint /path/to/models \
        --vae_checkpoint_path /path/to/vae.pkl \
        --vae_config_path /path/to/config.yaml \
        --n_levels 10000 \
        --output_path trajectory/data/trajectories.npz \
        --source prior
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vae"))

from vae_model import CluttrVAE
from vae_level_utils import decode_latent_to_levels, level_to_tokens
from jaxued.environments.maze import Maze, Level
from jaxued.wrappers import AutoReplayWrapper

from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax


GRID_SIZE = 13

from jaxued.linen import ResetRNN


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


def rollout_with_trajectories(rng, levels, agent_params, network, env, env_params, max_steps=250):
    """Roll out agent on levels and record position trajectories.

    Returns:
        positions: (max_steps, N) int32 — 1-based position indices (0 = padding after done)
        rewards: (max_steps, N) float32
        dones: (max_steps, N) bool
        values: (max_steps, N) float32
    """
    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]

    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(env.reset_to_level, (0, 0, None))(
        jax.random.split(rng_reset, num_levels), levels, env_params
    )

    init_hstate = ActorCritic.initialize_carry((num_levels,))

    def step(carry, _):
        rng, hstate, obs, state, done, ever_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, value = network.apply({"params": agent_params}, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)
        value = value.squeeze(0)

        # Record position BEFORE step (current position)
        # 1-based index: y * 13 + x + 1, masked to 0 after episode ends
        pos = state.agent_pos  # (N, 2) = (x, y)
        pos_idx = pos[:, 1] * GRID_SIZE + pos[:, 0] + 1  # (N,) 1-based
        pos_idx = jnp.where(ever_done, 0, pos_idx)  # mask after done

        next_obs, next_state, reward, next_done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)

        # Track if episode has ever ended
        ever_done = ever_done | next_done

        return (rng, hstate, next_obs, next_state, next_done, ever_done), (pos_idx, reward, next_done, value)

    rng, rng_rollout = jax.random.split(rng)
    _, (positions, rewards, dones, values) = jax.lax.scan(
        step,
        (rng_rollout, init_hstate, init_obs, init_env_state,
         jnp.zeros(num_levels, dtype=bool), jnp.zeros(num_levels, dtype=bool)),
        None,
        length=max_steps,
    )

    return positions, rewards, dones, values


def compute_maxmc_regret(rewards, dones, values):
    """Proper MaxMC regret using jaxued.utils."""
    from jaxued.utils import compute_max_returns, max_mc
    max_returns = compute_max_returns(dones, rewards)
    return max_mc(dones, values, max_returns, incomplete_value=0.0)


def main():
    parser = argparse.ArgumentParser(description="Collect agent trajectories for trajectory VAE")
    parser.add_argument("--agent_checkpoint", type=str, required=True,
                        help="Path to orbax checkpoint dir (e.g., .../models)")
    parser.add_argument("--vae_checkpoint_path", type=str, default=None,
                        help="VAE checkpoint (needed for --source prior)")
    parser.add_argument("--vae_config_path", type=str, default=None,
                        help="VAE config (needed for --source prior)")
    parser.add_argument("--buffer_path", type=str, default=None,
                        help="Buffer dump .npz (needed for --source buffer)")
    parser.add_argument("--n_levels", type=int, default=10000,
                        help="Number of levels to collect trajectories on")
    parser.add_argument("--source", type=str, default="prior",
                        choices=["prior", "buffer", "random"],
                        help="Level source: prior (VAE decode), buffer (PLR buffer), random (env generator)")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--agent_view_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.seed)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # --- Load environment ---
    env = Maze(max_height=13, max_width=13, agent_view_size=args.agent_view_size, normalize_obs=True)
    env_params = env.default_params

    # --- Load agent ---
    print("[1/3] Loading agent...")
    network = ActorCritic(action_dim=env.action_space(env_params).n)
    ws_manager = ocp.CheckpointManager(
        args.agent_checkpoint,
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    ws_step = ws_manager.latest_step()
    ws_ckpt = ws_manager.restore(ws_step)
    agent_params = ws_ckpt["params"] if isinstance(ws_ckpt, dict) and "params" in ws_ckpt else ws_ckpt.params
    # Unwrap double-nested {"params": {"params": ...}} from orbax
    if isinstance(agent_params, dict) and "params" in agent_params:
        agent_params = agent_params["params"]
    print(f"  Loaded checkpoint from step {ws_step}")

    # --- Generate levels ---
    print(f"[2/3] Generating {args.n_levels} levels from source='{args.source}'...")

    if args.source == "prior":
        assert args.vae_checkpoint_path and args.vae_config_path, "Need --vae_checkpoint_path and --vae_config_path for prior source"
        with open(args.vae_config_path) as f:
            vae_cfg = yaml.safe_load(f)
        vae = CluttrVAE(
            vocab_size=vae_cfg["vocab_size"], embed_dim=vae_cfg["embed_dim"],
            latent_dim=vae_cfg["latent_dim"], seq_len=vae_cfg["seq_len"],
        )
        with open(args.vae_checkpoint_path, "rb") as f:
            vae_ckpt = pickle.load(f)
        vae_params = vae_ckpt["params"] if isinstance(vae_ckpt, dict) and "params" in vae_ckpt else vae_ckpt
        decode_fn = lambda z: vae.apply({"params": vae_params}, z, method=vae.decode)

        def get_levels_batch(rng, batch_size):
            rng_z, rng_dec = jax.random.split(rng)
            z = jax.random.normal(rng_z, (batch_size, vae_cfg["latent_dim"]))
            return decode_latent_to_levels(decode_fn, z, rng_dec)

    elif args.source == "buffer":
        assert args.buffer_path, "Need --buffer_path for buffer source"
        from vae_level_utils import tokens_to_level
        buf = np.load(args.buffer_path, allow_pickle=True)
        buf_tokens = buf["tokens"]
        buf_size = int(buf["size"]) if "size" in buf else len(buf_tokens)
        buf_tokens = jnp.array(buf_tokens[:buf_size])
        buf_scores = np.array(buf["scores"][:buf_size])
        # Use exactly the buffer levels, override n_levels
        args.n_levels = buf_size
        print(f"  Buffer has {buf_size} levels, using all of them")
        print(f"  Buffer scores: mean={buf_scores.mean():.4f}, std={buf_scores.std():.4f}, "
              f"min={buf_scores.min():.4f}, max={buf_scores.max():.4f}")

        # Pre-convert all buffer levels
        all_buf_levels = jax.vmap(tokens_to_level)(buf_tokens)

        def get_levels_batch(rng, batch_size):
            """Not used for buffer — we slice directly in the collection loop."""
            raise RuntimeError("Should not be called for buffer source")

    elif args.source == "random":
        from jaxued.environments.maze import make_level_generator
        level_gen = make_level_generator(env, env_params)

        def get_levels_batch(rng, batch_size):
            return jax.vmap(level_gen)(jax.random.split(rng, batch_size))

    # --- JIT rollout ---
    @jax.jit
    def _rollout_batch(rng, levels):
        return rollout_with_trajectories(rng, levels, agent_params, network, env, env_params, args.max_steps)

    @jax.jit
    def _get_tokens(levels):
        return jax.vmap(level_to_tokens)(levels)

    # --- Collect ---
    print(f"[3/3] Collecting trajectories...")
    all_trajectories = []
    all_regret = []
    all_tokens = []
    all_episode_lengths = []
    all_reached_goal = []

    n_batches = (args.n_levels + args.batch_size - 1) // args.batch_size
    t0 = time.time()

    for i in range(n_batches):
        start = i * args.batch_size
        bs = min(args.batch_size, args.n_levels - start)
        rng, rng_levels, rng_roll = jax.random.split(rng, 3)

        if args.source == "buffer":
            levels = jax.tree_util.tree_map(lambda x: x[start:start+bs], all_buf_levels)
        else:
            levels = get_levels_batch(rng_levels, bs)
        tokens = _get_tokens(levels)
        positions, rewards, dones, values = _rollout_batch(rng_roll, levels)

        # positions: (max_steps, N) -> transpose to (N, max_steps)
        positions_np = np.array(positions.T)
        rewards_np = np.array(rewards)
        dones_np = np.array(dones)

        # Get regret: use buffer's own scores if available, otherwise compute
        if args.source == "buffer":
            regret = buf_scores[start:start+bs]
        else:
            regret = np.array(compute_maxmc_regret(rewards, dones, values))

        # Episode length: first done index + 1 (or max_steps if never done)
        first_done = np.argmax(dones_np, axis=0)  # (N,)
        never_done = ~np.any(dones_np, axis=0)
        ep_lengths = np.where(never_done, args.max_steps, first_done + 1)

        # Reached goal: any positive reward
        reached = np.any(rewards_np > 0, axis=0)

        all_trajectories.append(positions_np)
        all_regret.append(regret)
        all_tokens.append(np.array(tokens))
        all_episode_lengths.append(ep_lengths)
        all_reached_goal.append(reached)

        if (i + 1) % 10 == 0 or i == n_batches - 1:
            elapsed = time.time() - t0
            print(f"  Batch {i+1}/{n_batches} ({elapsed:.1f}s)")

    # --- Concatenate and save ---
    trajectories = np.concatenate(all_trajectories, axis=0)  # (N, max_steps)
    regret = np.concatenate(all_regret, axis=0)
    tokens = np.concatenate(all_tokens, axis=0)
    episode_lengths = np.concatenate(all_episode_lengths, axis=0)
    reached_goal = np.concatenate(all_reached_goal, axis=0)

    print(f"\nCollected {len(trajectories)} trajectories")
    print(f"  Trajectory shape: {trajectories.shape}")
    print(f"  Regret: mean={regret.mean():.4f}, std={regret.std():.4f}, min={regret.min():.4f}, max={regret.max():.4f}")
    print(f"  Episode length: mean={episode_lengths.mean():.1f}, std={episode_lengths.std():.1f}")
    print(f"  Reached goal: {reached_goal.sum()}/{len(reached_goal)} ({reached_goal.mean()*100:.1f}%)")

    np.savez(
        args.output_path,
        trajectories=trajectories,
        regret=regret,
        tokens=tokens,
        episode_lengths=episode_lengths,
        reached_goal=reached_goal,
        max_steps=args.max_steps,
        source=args.source,
    )
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
