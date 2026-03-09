"""
latent_perturbation_diagnostic.py
=================================
Diagnostic: Does latent proximity imply similar agent difficulty?

1. Load a trained agent checkpoint (Orbax) and a VAE checkpoint
2. Sample 10 random levels from the agent's replay buffer
3. For each level: encode through VAE, perturb z in 10 random directions, decode
4. Evaluate the agent on original + perturbations (11 levels per base)
5. Report whether nearby latent mazes produce similar agent returns/solve rates

Usage:
    python latent_perturbation_diagnostic.py \
        --agent_checkpoint_dir checkpoints/run_name/seed_0 \
        --vae_checkpoint_path /path/to/checkpoint_N.pkl \
        --vae_config_path /path/to/vae_train_config.yml \
        --perturbation_scale 0.5 \
        --num_base_levels 10 \
        --num_perturbations 10
"""

import os
import sys
import pickle
import argparse
import json

import numpy as np
import jax
import jax.numpy as jnp
import yaml

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import orbax.checkpoint as ocp
import distrax
import flax.linen as nn
from flax.training.train_state import TrainState as BaseTrainState
from flax import core, struct

from jaxued.environments.maze import Maze, Level
from jaxued.environments.maze.env import EnvParams
from jaxued.environments.maze.util import make_level_generator

# --- Import VAE ---
try:
    from vae_model import CluttrVAE
except ImportError:
    import train_vae
    from train_vae import CluttrVAE


# ═══════════════════════════════════════════════════════════════════════════
# Token <-> Level conversion (from vae_level_utils.py)
# ═══════════════════════════════════════════════════════════════════════════

GRID_SIZE = 13
VOCAB_SIZE = 170
SEQ_LEN = 52
MAX_WALLS = 50


def tokens_to_level(tokens):
    """Convert a 52-token VAE sequence to a Level dataclass."""
    agent_idx = tokens[-1]
    goal_idx = tokens[-2]
    wall_tokens = tokens[:-2]

    num_cells = GRID_SIZE * GRID_SIZE
    wall_map_flat = jnp.zeros(num_cells, dtype=jnp.bool_)
    wall_idx_0 = jnp.clip(wall_tokens - 1, 0, num_cells - 1)
    valid_walls = wall_tokens > 0
    wall_map_flat = wall_map_flat.at[wall_idx_0].set(valid_walls)
    wall_map = wall_map_flat.reshape(GRID_SIZE, GRID_SIZE)

    agent_0 = jnp.clip(agent_idx - 1, 0, num_cells - 1)
    agent_pos = jnp.array([agent_0 % GRID_SIZE, agent_0 // GRID_SIZE], dtype=jnp.uint32)

    goal_0 = jnp.clip(goal_idx - 1, 0, num_cells - 1)
    goal_pos = jnp.array([goal_0 % GRID_SIZE, goal_0 // GRID_SIZE], dtype=jnp.uint32)

    wall_map = wall_map.at[agent_pos[1], agent_pos[0]].set(False)
    wall_map = wall_map.at[goal_pos[1], goal_pos[0]].set(False)

    return Level(
        wall_map=wall_map,
        goal_pos=goal_pos,
        agent_pos=agent_pos,
        agent_dir=jnp.array(0, dtype=jnp.uint8),
        width=GRID_SIZE,
        height=GRID_SIZE,
    )


def level_to_tokens(level):
    """Convert a Level dataclass to a 52-token VAE sequence."""
    wall_map = level.wall_map
    wall_flat = wall_map.reshape(-1)

    indices_1based = jnp.arange(1, GRID_SIZE * GRID_SIZE + 1)
    wall_indices = jnp.where(wall_flat, indices_1based, 0)
    wall_indices = jnp.sort(wall_indices)[::-1][:MAX_WALLS]
    wall_indices = jnp.sort(wall_indices)

    goal_idx = level.goal_pos[1] * GRID_SIZE + level.goal_pos[0] + 1
    agent_idx = level.agent_pos[1] * GRID_SIZE + level.agent_pos[0] + 1

    return jnp.concatenate([wall_indices, jnp.array([goal_idx, agent_idx])]).astype(jnp.int32)


def repair_tokens(tokens):
    """Ensure decoded tokens are valid."""
    tokens = jnp.clip(tokens, 0, VOCAB_SIZE - 1).astype(jnp.int32)
    goal = jnp.clip(tokens[-2], 1, VOCAB_SIZE - 1)
    agent = jnp.clip(tokens[-1], 1, VOCAB_SIZE - 1)
    agent = jnp.where(goal == agent, (agent % (VOCAB_SIZE - 1)) + 1, agent)
    walls = tokens[:-2]
    walls = jnp.where(walls == goal, 0, walls)
    walls = jnp.where(walls == agent, 0, walls)
    walls = jnp.sort(walls)
    return jnp.concatenate([walls, jnp.array([goal, agent])])


# ═══════════════════════════════════════════════════════════════════════════
# Agent model (must match maze_plr.py architecture)
# ═══════════════════════════════════════════════════════════════════════════

DIR_TO_VEC = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]], dtype=jnp.int32)


class ResetRNN(nn.Module):
    cell: nn.Module

    @nn.compact
    def __call__(self, inputs, initial_carry):
        seq, resets = inputs
        def step_fn(carry, x):
            inp, reset = x
            carry = jax.tree_util.tree_map(
                lambda c, i: jnp.where(reset[..., None], i, c),
                carry, initial_carry,
            )
            carry, y = self.cell(carry, inp)
            return carry, y

        return nn.scan(
            step_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0, out_axes=0,
        )(initial_carry, (seq, resets))


class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs
        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(
            5, kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.constant(0.0), name="scalar_embed"
        )(dir_embed)

        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))(
            (embedding, dones), initial_carry=hidden,
        )

        actor_mean = nn.Dense(
            32, kernel_init=nn.initializers.orthogonal(2),
            bias_init=nn.initializers.constant(0.0), name="actor0"
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0), name="actor1"
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            32, kernel_init=nn.initializers.orthogonal(2),
            bias_init=nn.initializers.constant(0.0), name="critic0"
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0), name="critic1"
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, 256)
        )


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_agent_on_levels(env, env_params, agent_params, network, levels, rng,
                              max_steps=250):
    """Evaluate the agent on a batch of levels. Returns per-level cumulative rewards."""
    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]

    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(env.reset_to_level, (0, 0, None))(
        jax.random.split(rng_reset, num_levels), levels, env_params,
    )

    init_hstate = ActorCritic.initialize_carry((num_levels,))

    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = network.apply(agent_params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)

        next_mask = mask & ~done
        episode_length = episode_length + mask.astype(jnp.int32)

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (reward, mask)

    init_done = jnp.zeros(num_levels, dtype=jnp.bool_)
    init_mask = jnp.ones(num_levels, dtype=jnp.bool_)
    init_episode_length = jnp.zeros(num_levels, dtype=jnp.int32)

    rng, rng_eval = jax.random.split(rng)
    (_, _, _, _, _, _, episode_lengths), (rewards, masks) = jax.lax.scan(
        step,
        (rng_eval, init_hstate, init_obs, init_env_state, init_done, init_mask, init_episode_length),
        None,
        length=max_steps,
    )

    cum_rewards = (rewards * masks).sum(axis=0)
    solved = cum_rewards > 0  # agent gets reward for reaching goal

    return cum_rewards, solved, episode_lengths


# ═══════════════════════════════════════════════════════════════════════════
# VAE encode / decode
# ═══════════════════════════════════════════════════════════════════════════

def load_vae(vae_config_path, vae_checkpoint_path):
    """Load VAE model and params."""
    with open(vae_config_path) as f:
        vae_cfg = yaml.safe_load(f)

    # Try vae_model.py style (constructor args) first, fall back to train_vae.py style (global CONFIG)
    try:
        model = CluttrVAE(
            vocab_size=vae_cfg['vocab_size'],
            embed_dim=vae_cfg['embed_dim'],
            latent_dim=vae_cfg['latent_dim'],
            seq_len=vae_cfg['seq_len'],
        )
    except TypeError:
        # train_vae.py version uses global CONFIG, no constructor args
        import train_vae
        train_vae.CONFIG['latent_dim'] = vae_cfg['latent_dim']
        train_vae.CONFIG['embed_dim'] = vae_cfg['embed_dim']
        train_vae.CONFIG['vocab_size'] = vae_cfg['vocab_size']
        train_vae.CONFIG['seq_len'] = vae_cfg['seq_len']
        model = CluttrVAE()

    with open(vae_checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt

    return model, params, vae_cfg


def vae_encode(model, params, tokens):
    """Encode a batch of token sequences to latent means."""
    @jax.jit
    def _encode(params, tokens):
        mean, logvar = model.apply(
            {"params": params}, tokens, train=False,
            method=CluttrVAE.encode,
            rngs={"dropout": jax.random.key(0)},
        )
        return mean
    return _encode(params, tokens)


def vae_decode(model, params, z):
    """Decode latent vectors to token sequences."""
    @jax.jit
    def _decode(params, z):
        logits = model.apply({"params": params}, z, method=CluttrVAE.decode)
        return jnp.argmax(logits, axis=-1)
    return _decode(params, z)


# ═══════════════════════════════════════════════════════════════════════════
# Main diagnostic
# ═══════════════════════════════════════════════════════════════════════════

def load_agent_checkpoint(checkpoint_dir, env, env_params, network):
    """Load agent params from an Orbax checkpoint."""
    models_dir = os.path.join(checkpoint_dir, 'models')
    checkpoint_manager = ocp.CheckpointManager(
        models_dir,
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    step = checkpoint_manager.latest_step()
    print(f"[Agent] Loading checkpoint step {step} from {models_dir}")
    loaded = checkpoint_manager.restore(step)
    return loaded['params'], step


def extract_buffer_levels(checkpoint_dir):
    """Load the sampler state from Orbax and extract buffer levels + scores."""
    models_dir = os.path.join(checkpoint_dir, 'models')
    checkpoint_manager = ocp.CheckpointManager(
        models_dir,
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    step = checkpoint_manager.latest_step()
    loaded = checkpoint_manager.restore(step)

    # The sampler is stored in the train state
    sampler = loaded['sampler']
    size = int(sampler['size'])
    scores = np.array(sampler['scores'][:size])
    print(f"[Buffer] {size} levels in buffer, score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Extract levels (batched Level struct)
    levels = jax.tree_util.tree_map(lambda x: x[:size], sampler['levels'])
    return levels, scores, size


def load_buffer_dump(dump_path):
    """Load a buffer dump .npz or .npy file (from CMA-ES runs).

    .npz format: tokens (N,52), scores (N,), timestamps (N,), size, update_num
    .npy format: tokens only (N,52) — no scores available
    """
    if dump_path.endswith('.npy'):
        tokens = np.load(dump_path)
        scores = np.zeros(len(tokens))  # no scores in .npy
        print(f"[Buffer] Loaded {len(tokens)} levels from {dump_path} (.npy, no scores)")
        return tokens, scores
    else:
        data = np.load(dump_path)
        tokens = data['tokens']      # (N, 52)
        scores = data['scores']      # (N,)
        size = int(data['size'])
        print(f"[Buffer] Loaded {size} levels from {dump_path}")
        print(f"  Score range: [{scores[:size].min():.3f}, {scores[:size].max():.3f}]")
        return tokens[:size], scores[:size]


def run_diagnostic(args):
    print("=" * 70)
    print("  LATENT PERTURBATION DIAGNOSTIC")
    print("  Does latent proximity imply similar agent difficulty?")
    print("=" * 70)

    # --- Setup environment ---
    env = Maze(max_height=13, max_width=13, agent_view_size=args.agent_view_size,
               normalize_obs=True)
    env_params = env.default_params

    # --- Load VAE ---
    print(f"\n[VAE] Loading from {args.vae_checkpoint_path}")
    vae_model, vae_params, vae_cfg = load_vae(args.vae_config_path, args.vae_checkpoint_path)
    latent_dim = vae_cfg['latent_dim']
    print(f"[VAE] latent_dim={latent_dim}")

    # --- Load agent ---
    network = ActorCritic(action_dim=env.action_space(env_params).n)

    if args.agent_params_pkl:
        # Direct pickle loading
        print(f"\n[Agent] Loading params from {args.agent_params_pkl}")
        with open(args.agent_params_pkl, "rb") as f:
            agent_data = pickle.load(f)
        agent_params = agent_data['params'] if isinstance(agent_data, dict) and 'params' in agent_data else agent_data
    else:
        # Orbax checkpoint
        print(f"\n[Agent] Loading from {args.agent_checkpoint_dir}")
        agent_params, agent_step = load_agent_checkpoint(
            args.agent_checkpoint_dir, env, env_params, network
        )

    # --- Load buffer levels ---
    if args.buffer_dump_path:
        buffer_tokens, buffer_scores = load_buffer_dump(args.buffer_dump_path)
        buffer_from_tokens = True
    elif args.agent_checkpoint_dir:
        buffer_levels, buffer_scores, buffer_size = extract_buffer_levels(args.agent_checkpoint_dir)
        # Convert to tokens for VAE encoding
        buffer_tokens = np.array(jax.vmap(level_to_tokens)(buffer_levels))
        buffer_from_tokens = True
    else:
        raise ValueError("Provide either --buffer_dump_path or --agent_checkpoint_dir")

    # --- Sample base levels ---
    rng = jax.random.PRNGKey(args.seed)
    n_base = args.num_base_levels
    n_pert = args.num_perturbations
    eps = args.perturbation_scale

    rng, rng_sample = jax.random.split(rng)
    indices = np.random.default_rng(args.seed).choice(len(buffer_tokens), n_base, replace=False)
    base_tokens = jnp.array(buffer_tokens[indices])
    base_scores = buffer_scores[indices]
    print(f"\n[Sample] {n_base} base levels selected (buffer indices: {indices})")
    print(f"  Buffer scores: {base_scores}")

    # --- Encode base levels through VAE ---
    base_z = np.array(vae_encode(vae_model, vae_params, base_tokens))  # (n_base, latent_dim)
    print(f"[Encode] z shape: {base_z.shape}, mean norm: {np.linalg.norm(base_z, axis=1).mean():.2f}")

    # --- Generate perturbations ---
    rng, rng_pert = jax.random.split(rng)
    all_results = []

    for i in range(n_base):
        z_base = base_z[i]  # (latent_dim,)

        # Generate random perturbation directions
        rng_pert, rng_dir = jax.random.split(rng_pert)
        directions = jax.random.normal(rng_dir, (n_pert, latent_dim))
        directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

        # Create perturbed z vectors
        z_perturbed = z_base[None, :] + eps * directions  # (n_pert, latent_dim)

        # Stack original + perturbations
        z_all = jnp.concatenate([z_base[None, :], z_perturbed], axis=0)  # (1 + n_pert, latent_dim)

        # Decode all
        decoded_tokens = np.array(vae_decode(vae_model, vae_params, z_all))  # (1+n_pert, 52)

        # Repair tokens and convert to levels
        repaired = jax.vmap(repair_tokens)(jnp.array(decoded_tokens))
        levels = jax.vmap(tokens_to_level)(repaired)

        # Compute L2 distances from original
        l2_dists = np.linalg.norm(np.array(z_all[1:]) - np.array(z_all[0:1]), axis=1)

        # Compute token similarity (Jaccard on walls)
        orig_walls = set(np.array(repaired[0][:-2])[np.array(repaired[0][:-2]) > 0].tolist())
        token_sims = []
        for j in range(1, len(repaired)):
            pert_walls = set(np.array(repaired[j][:-2])[np.array(repaired[j][:-2]) > 0].tolist())
            union = orig_walls | pert_walls
            jaccard = len(orig_walls & pert_walls) / len(union) if union else 1.0
            token_sims.append(jaccard)

        # Evaluate agent on all levels
        rng, rng_eval = jax.random.split(rng)
        cum_rewards, solved, ep_lengths = evaluate_agent_on_levels(
            env, env_params, agent_params, network, levels, rng_eval,
            max_steps=env_params.max_steps_in_episode,
        )

        cum_rewards = np.array(cum_rewards)
        solved = np.array(solved)
        ep_lengths = np.array(ep_lengths)

        result = {
            "base_idx": int(indices[i]),
            "base_buffer_score": float(base_scores[i]),
            "base_reward": float(cum_rewards[0]),
            "base_solved": bool(solved[0]),
            "base_ep_length": int(ep_lengths[0]),
            "pert_rewards": cum_rewards[1:].tolist(),
            "pert_solved": solved[1:].tolist(),
            "pert_ep_lengths": ep_lengths[1:].tolist(),
            "pert_l2_dists": l2_dists.tolist(),
            "pert_wall_jaccard": token_sims,
            "reward_std": float(cum_rewards[1:].std()),
            "solve_rate": float(solved[1:].mean()),
            "reward_correlation_with_dist": float(
                np.corrcoef(l2_dists, cum_rewards[1:])[0, 1]
            ) if n_pert > 2 else 0.0,
        }
        all_results.append(result)

        print(f"\n--- Base level {i+1}/{n_base} (buffer idx {indices[i]}) ---")
        print(f"  Buffer score: {base_scores[i]:.3f}")
        print(f"  Original:  reward={cum_rewards[0]:.2f}  solved={solved[0]}")
        print(f"  Perturbed: reward={cum_rewards[1:].mean():.2f} +/- {cum_rewards[1:].std():.2f}")
        print(f"  Perturbed: solve rate={solved[1:].mean():.0%}")
        print(f"  Wall Jaccard (orig vs pert): {np.mean(token_sims):.3f} +/- {np.std(token_sims):.3f}")
        print(f"  L2 distances: {l2_dists.mean():.3f} +/- {l2_dists.std():.3f}")
        print(f"  Reward-distance correlation: {result['reward_correlation_with_dist']:.3f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    all_base_rewards = [r["base_reward"] for r in all_results]
    all_pert_rewards = [np.mean(r["pert_rewards"]) for r in all_results]
    all_pert_stds = [r["reward_std"] for r in all_results]
    all_solve_rates = [r["solve_rate"] for r in all_results]
    all_jaccards = [np.mean(r["pert_wall_jaccard"]) for r in all_results]
    all_correlations = [r["reward_correlation_with_dist"] for r in all_results]

    print(f"\n  Across {n_base} base levels with {n_pert} perturbations each (eps={eps}):")
    print(f"  Mean base reward:          {np.mean(all_base_rewards):.3f}")
    print(f"  Mean perturbed reward:     {np.mean(all_pert_rewards):.3f}")
    print(f"  Mean perturbation std:     {np.mean(all_pert_stds):.3f}")
    print(f"  Mean perturbation solve%:  {np.mean(all_solve_rates):.1%}")
    print(f"  Mean wall Jaccard:         {np.mean(all_jaccards):.3f}")
    print(f"  Mean reward-dist corr:     {np.mean(all_correlations):.3f}")
    print()

    # Interpretation
    if np.mean(all_pert_stds) < 0.5 * np.std(all_base_rewards + all_pert_rewards):
        print("  -> LOW variance within perturbation neighborhoods")
        print("     Nearby latent points produce similar difficulty.")
        print("     ES gradient estimation should be meaningful.")
    else:
        print("  -> HIGH variance within perturbation neighborhoods")
        print("     Nearby latent points produce varying difficulty.")
        print("     ES gradient steps may be unreliable.")

    if np.mean(all_jaccards) > 0.5:
        print("  -> HIGH wall overlap between original and perturbations")
        print("     VAE latent perturbations produce structurally similar mazes.")
    else:
        print("  -> LOW wall overlap between original and perturbations")
        print("     VAE latent perturbations produce structurally different mazes.")

    # Save results
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump({
                "config": {
                    "num_base_levels": n_base,
                    "num_perturbations": n_pert,
                    "perturbation_scale": eps,
                    "seed": args.seed,
                    "vae_checkpoint": args.vae_checkpoint_path,
                    "agent_checkpoint": args.agent_checkpoint_dir or args.agent_params_pkl,
                },
                "summary": {
                    "mean_base_reward": float(np.mean(all_base_rewards)),
                    "mean_pert_reward": float(np.mean(all_pert_rewards)),
                    "mean_pert_std": float(np.mean(all_pert_stds)),
                    "mean_solve_rate": float(np.mean(all_solve_rates)),
                    "mean_wall_jaccard": float(np.mean(all_jaccards)),
                    "mean_reward_dist_corr": float(np.mean(all_correlations)),
                },
                "per_level": all_results,
            }, f, indent=2)
        print(f"\n[Save] Results -> {args.output_path}")

    return all_results


def parse_args():
    p = argparse.ArgumentParser(description="Latent perturbation diagnostic")
    p.add_argument("--agent_checkpoint_dir", type=str, default=None,
                   help="Path to agent checkpoint dir (contains models/ and config.json)")
    p.add_argument("--agent_params_pkl", type=str, default=None,
                   help="Direct path to pickled agent params (alternative to Orbax)")
    p.add_argument("--buffer_dump_path", type=str, default=None,
                   help="Path to buffer_dump.npz (tokens + scores)")
    p.add_argument("--vae_checkpoint_path", type=str, required=True,
                   help="Path to VAE checkpoint .pkl")
    p.add_argument("--vae_config_path", type=str, default="vae_train_config.yml",
                   help="Path to VAE config YAML")
    p.add_argument("--perturbation_scale", type=float, default=0.5,
                   help="Epsilon for latent perturbation (L2 magnitude)")
    p.add_argument("--num_base_levels", type=int, default=10,
                   help="Number of base levels to sample from buffer")
    p.add_argument("--num_perturbations", type=int, default=10,
                   help="Number of perturbation directions per base level")
    p.add_argument("--agent_view_size", type=int, default=5,
                   help="Agent view size (must match training config)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_path", type=str, default="perturbation_diagnostic.json",
                   help="Path to save JSON results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_diagnostic(args)
