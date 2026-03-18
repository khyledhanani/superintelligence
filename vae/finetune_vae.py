"""
Fine-tune a pre-trained CluttrVAE on a prescribed dataset (e.g., PLR buffer archive).

Supports:
  - Multi-device data parallelism (TPU pods, multi-GPU)
  - Differential learning rates (encoder vs decoder)
  - Linear warmup + cosine decay schedule
  - Anchor regularization to prevent latent drift
  - Weighted reconstruction loss (wall vs agent/goal tokens)
  - Periodic validation on held-out data
  - Checkpoint saving (best + periodic + final)
  - wandb logging

Usage:
  python vae/finetune_vae.py \
      --checkpoint_path /tmp/vae_beta10/checkpoint_500000.pkl \
      --config_path /tmp/vae_beta10/config.yaml \
      --train_data path/to/level_archive_final.npz \
      --val_data /path/to/new_val_20k_envs.npy \
      --num_steps 2000 --batch_size 128 \
      --encoder_lr 5e-5 --decoder_lr 2e-5 \
      --anchor_weight 0.1
"""
import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training import train_state
from functools import partial
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from vae_model import CluttrVAE


# ─── SFL scoring ─────────────────────────────────────────────────────────────

def score_levels_sfl(agent_params, tokens, agent_checkpoint_path,
                     num_rollouts=5, batch_size=32, max_episode_length=250,
                     agent_view_size=5, seed=0):
    """Score levels by SFL learnability using a trained agent.

    Loads the agent, converts tokens to levels, runs rollouts, returns p*(1-p).

    Args:
        agent_params: Flax params dict for ActorCritic (already loaded).
        tokens: (N, 52) int32 token array.
        agent_checkpoint_path: unused if agent_params provided (kept for API compat).
        num_rollouts: number of eval rollouts per level.
        batch_size: levels per batch for rollouts.
        max_episode_length: max steps per episode.
        agent_view_size: observation window size.
        seed: random seed.

    Returns:
        scores: (N,) float32 array of SFL scores in [0, 0.25].
    """
    import sys, os
    # Add examples/ to path for ActorCritic, and parent for jaxued
    examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
    sys.path.insert(0, examples_dir)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import chex
    import distrax
    import flax.linen as nn
    from flax.linen.initializers import orthogonal, constant
    from jaxued.environments import Maze
    from vae_level_utils import tokens_to_level

    # Inline ActorCritic definition (avoids importing the huge maze_plr.py)
    class ResetRNN(nn.Module):
        cell: nn.Module
        @nn.compact
        def __call__(self, inputs, initial_carry):
            embedding, dones = inputs
            hidden = self.cell.initialize_carry(jax.random.PRNGKey(0), embedding.shape)
            hidden = jax.tree_util.tree_map(
                lambda new, old: jnp.where(dones[..., None], new, old),
                initial_carry, hidden)
            return self.cell(hidden, embedding)

    class ActorCritic(nn.Module):
        action_dim: int
        @nn.compact
        def __call__(self, inputs, hidden):
            obs, dones = inputs
            img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
            img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
            img_embed = nn.relu(img_embed)
            dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
            dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)),
                                 bias_init=constant(0.0), name="scalar_embed")(dir_embed)
            embedding = jnp.append(img_embed, dir_embed, axis=-1)
            hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))(
                (embedding, dones), initial_carry=hidden)
            actor_mean = nn.Dense(32, kernel_init=orthogonal(2),
                                  bias_init=constant(0.0), name="actor0")(embedding)
            actor_mean = nn.relu(actor_mean)
            actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                                  bias_init=constant(0.0), name="actor1")(actor_mean)
            pi = distrax.Categorical(logits=actor_mean)
            critic = nn.Dense(32, kernel_init=orthogonal(2),
                              bias_init=constant(0.0), name="critic0")(embedding)
            critic = nn.relu(critic)
            critic = nn.Dense(1, kernel_init=orthogonal(1.0),
                              bias_init=constant(0.0), name="critic1")(critic)
            return hidden, pi, jnp.squeeze(critic, axis=-1)

        @staticmethod
        def initialize_carry(batch_dims):
            return nn.OptimizedLSTMCell(features=256).initialize_carry(
                jax.random.PRNGKey(0), (*batch_dims, 256))

    # Setup
    env = Maze(max_height=13, max_width=13, agent_view_size=agent_view_size, normalize_obs=True)
    env_params = env.default_params

    # Wrap agent params for apply
    class FakeTrainState:
        def __init__(self, params):
            self.params = params
            self.apply_fn = ActorCritic(action_dim=5).apply

    fake_ts = FakeTrainState(agent_params)

    def evaluate_rnn_batch(rng, levels_batch, n_levels):
        """Run one rollout on a batch of levels, return success mask."""
        rng_reset, rng_eval = jax.random.split(rng)
        init_obs, init_env_state = jax.vmap(env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, n_levels), levels_batch, env_params)
        init_hstate = ActorCritic.initialize_carry((n_levels,))

        def step(carry, _):
            rng, hstate, obs, state, done, mask = carry
            rng, rng_action, rng_step = jax.random.split(rng, 3)
            x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
            hstate, pi, _ = fake_ts.apply_fn(fake_ts.params, x, hstate)
            action = pi.sample(seed=rng_action).squeeze(0)
            obs, next_state, reward, done, _ = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(rng_step, n_levels), state, action, env_params)
            next_mask = mask & ~done
            return (rng, hstate, obs, next_state, done, next_mask), reward

        (_, _, _, _, _, _), rewards = jax.lax.scan(
            step,
            (rng_eval, init_hstate, init_obs, init_env_state,
             jnp.zeros(n_levels, dtype=bool), jnp.ones(n_levels, dtype=bool)),
            None, length=max_episode_length)
        return (rewards.sum(axis=0) > 0).astype(jnp.float32)

    # Convert tokens to levels
    N = len(tokens)
    print(f"[SFL Scoring] Scoring {N} levels with {num_rollouts} rollouts each...")

    # Process in batches
    all_scores = []
    rng = jax.random.PRNGKey(seed)

    # JIT the rollout for a fixed batch size
    eval_jit = jax.jit(evaluate_rnn_batch, static_argnums=(2,))

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        actual_bs = end - start
        batch_tokens = jnp.array(tokens[start:end])

        # Pad to batch_size if needed
        if actual_bs < batch_size:
            pad = batch_size - actual_bs
            batch_tokens = jnp.concatenate([batch_tokens, jnp.zeros((pad, 52), dtype=jnp.int32)])

        batch_levels = jax.vmap(tokens_to_level)(batch_tokens)

        # Run num_rollouts
        successes = []
        for r in range(num_rollouts):
            rng, rng_r = jax.random.split(rng)
            success = eval_jit(rng_r, batch_levels, batch_size)
            successes.append(success)

        all_successes = jnp.stack(successes, axis=0)  # (num_rollouts, batch_size)
        p = all_successes.mean(axis=0)
        batch_scores = p * (1 - p)  # SFL score

        # Trim padding
        all_scores.append(np.array(batch_scores[:actual_bs]))

        if (start // batch_size) % 50 == 0:
            print(f"  [{start}/{N}] batch mean score: {float(batch_scores[:actual_bs].mean()):.4f}")

    scores = np.concatenate(all_scores)
    print(f"[SFL Scoring] Done. Score stats: mean={scores.mean():.4f}, "
          f"std={scores.std():.4f}, max={scores.max():.4f}, "
          f"nonzero={int((scores > 0).sum())}/{N}")
    return scores


# ─── Data loading ────────────────────────────────────────────────────────────

def load_train_data(path):
    """Load training data. Supports .npz (archive) or .npy (raw tokens).

    Returns dict with 'tokens' and optional metadata fields.
    """
    if path.endswith(".npz"):
        data = np.load(path)
        result = {"tokens": data["tokens"]}  # (N, 52)
        for key in ["scores", "peak_scores", "first_inserted",
                     "last_inserted", "last_replayed", "replay_count"]:
            if key in data:
                result[key] = data[key]
        return result
    elif path.endswith(".npy"):
        return {"tokens": np.load(path)}
    else:
        raise ValueError(f"Unsupported file format: {path}")


def compute_sample_weights(train_data, mode="uniform", recency_decay=0.9999):
    """Compute per-level sampling probabilities for fine-tuning.

    Modes:
      - uniform:  all levels equally likely
      - score:    proportional to SFL score (higher = more likely)
      - recency:  exponential decay from last_replayed step
      - combined: score * recency_decay^(max_step - last_replayed)
    """
    n = len(train_data["tokens"])

    if mode == "uniform" or "scores" not in train_data:
        return np.ones(n) / n

    scores = train_data["scores"].copy()
    scores = np.clip(scores, 0, None)  # SFL scores are non-negative

    if mode == "score":
        w = scores + 1e-8
        return w / w.sum()

    # For recency modes, use last_replayed (fall back to last_inserted)
    if "last_replayed" in train_data:
        last_active = train_data["last_replayed"].copy()
        # Levels never replayed: use last_inserted instead
        never_replayed = last_active < 0
        if "last_inserted" in train_data:
            last_active[never_replayed] = train_data["last_inserted"][never_replayed]
        else:
            last_active[never_replayed] = 0
    elif "last_inserted" in train_data:
        last_active = train_data["last_inserted"]
    else:
        return np.ones(n) / n

    max_step = last_active.max()
    recency = recency_decay ** (max_step - last_active)

    if mode == "recency":
        w = recency + 1e-8
        return w / w.sum()

    if mode == "combined":
        w = (scores + 1e-8) * recency
        return w / w.sum()

    raise ValueError(f"Unknown sample_weighting mode: {mode}")


def load_val_data(path):
    """Load validation data (.npy of tokens)."""
    if path is None:
        return None
    tokens = np.load(path)
    # If shape is (N, 52), it's already tokens; otherwise might need slicing
    if tokens.ndim == 2 and tokens.shape[1] == 52:
        return tokens
    raise ValueError(f"Expected (N, 52) array, got shape {tokens.shape}")


# ─── Learning rate schedule ──────────────────────────────────────────────────

def create_lr_schedule(peak_lr, warmup_steps, total_steps):
    """Linear warmup then cosine decay to 0."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=peak_lr,
        transition_steps=warmup_steps)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=max(total_steps - warmup_steps, 1))
    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps])


def create_optimizer(params, encoder_lr, decoder_lr, warmup_steps, total_steps, max_grad_norm=1.0):
    """Create optimizer with differential LR for encoder vs decoder."""
    enc_schedule = create_lr_schedule(encoder_lr, warmup_steps, total_steps)
    dec_schedule = create_lr_schedule(decoder_lr, warmup_steps, total_steps)

    # Partition params into encoder and decoder
    def label_fn(path, _):
        path_str = "/".join(str(p) for p in path)
        if any(k in path_str for k in ["enc_", "embed", "mean_layer", "logvar_layer"]):
            return "encoder"
        return "decoder"

    param_labels = jax.tree_util.tree_map_with_path(label_fn, params)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.multi_transform(
            transforms={
                "encoder": optax.adam(enc_schedule),
                "decoder": optax.adam(dec_schedule),
            },
            param_labels=param_labels,
        ),
    )
    return optimizer, param_labels


# ─── Loss function ───────────────────────────────────────────────────────────

def compute_loss(params, pretrained_params, batch, rng, model, config):
    """VAE loss with anchor regularization.

    Returns: (total_loss, aux_dict)
    """
    rng_z, rng_drop = jax.random.split(rng)
    logits, mean, logvar = model.apply(
        {"params": params}, batch, rng_z, train=True,
        rngs={"dropout": rng_drop})

    # Reconstruction loss: weighted cross-entropy
    labels_onehot = jax.nn.one_hot(batch, num_classes=config["vocab_size"])
    per_token_loss = optax.softmax_cross_entropy(logits, labels_onehot)  # (B, 52)

    weights = jnp.ones_like(per_token_loss)
    weights = weights.at[:, :-2].set(config["wall_weight"])
    weights = weights.at[:, -2:].set(config["agent_goal_weight"])
    weights = weights / jnp.mean(weights, axis=-1, keepdims=True)

    recon_loss = jnp.mean(per_token_loss * weights)

    # KL divergence
    kl_loss = -0.5 * jnp.mean(
        jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1))

    # Anchor regularization: penalize encoder drift from pretrained
    anchor_loss = 0.0
    if config["anchor_weight"] > 0 and pretrained_params is not None:
        # Compute pretrained means for the same batch (no grad)
        pretrained_mean, _ = model.apply(
            {"params": pretrained_params}, batch, train=False, method=model.encode)
        anchor_loss = jnp.mean(jnp.sum((mean - pretrained_mean) ** 2, axis=-1))

    total_loss = (config["recon_weight"] * recon_loss
                  + config["beta"] * kl_loss
                  + config["anchor_weight"] * anchor_loss)

    aux = {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "anchor_loss": anchor_loss,
        "mean_abs_mean": jnp.mean(jnp.abs(mean)),
        "mean_logvar": jnp.mean(logvar),
    }
    return total_loss, aux


# ─── Sharded training/eval steps ─────────────────────────────────────────────

def make_train_step(model, config_tuple):
    """Create a sharded train step function."""
    @jax.jit
    def _train_step(state, pretrained_params, batch, rng):
        config = dict(config_tuple)
        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, aux), grads = grad_fn(state.params, pretrained_params, batch, rng, model, config)
        # Average gradients across devices (already done by jit with sharding)
        grads = jax.lax.pmean(grads, axis_name="batch")
        state = state.apply_gradients(grads=grads)
        # Average aux metrics across devices
        aux = jax.lax.pmean(aux, axis_name="batch")
        return state, aux
    return _train_step


def make_eval_step(model, config_tuple):
    """Create a sharded eval step function."""
    @jax.jit
    def _eval_step(params, pretrained_params, batch, rng):
        config = dict(config_tuple)
        _, aux = compute_loss(params, pretrained_params, batch, rng, model, config)
        rng_z, _ = jax.random.split(rng)
        logits, _, _ = model.apply({"params": params}, batch, rng_z, train=False)
        preds = jnp.argmax(logits, axis=-1)
        token_acc = jnp.mean(preds == batch)
        aux["token_accuracy"] = token_acc
        return aux
    return _eval_step


# ─── Checkpoint utilities ────────────────────────────────────────────────────

def save_checkpoint(params, config, step, path):
    # Unreplicate params (get from first device) if sharded
    params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"params": params, "config": config, "step": step}, f)
    print(f"[Checkpoint] Saved: {path}")


def load_pretrained(checkpoint_path, config_path):
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = CluttrVAE(
        vocab_size=cfg.get("vocab_size", 170),
        embed_dim=cfg.get("embed_dim", 300),
        latent_dim=cfg.get("latent_dim", 64),
        seq_len=cfg.get("seq_len", 52),
    )

    params = ckpt["params"]
    return model, params, cfg


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained CluttrVAE")

    # Required
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to pre-trained VAE .pkl checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to pre-trained VAE config.yaml")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data (.npz archive or .npy tokens)")

    # Optional data
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation data (.npy tokens)")

    # Training hyperparameters
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Global batch size (split across all devices)")
    parser.add_argument("--encoder_lr", type=float, default=5e-5)
    parser.add_argument("--decoder_lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Loss weights
    parser.add_argument("--beta", type=float, default=10.0,
                        help="KL weight (default: match pre-training beta)")
    parser.add_argument("--recon_weight", type=float, default=1000.0,
                        help="Reconstruction loss weight")
    parser.add_argument("--wall_weight", type=float, default=1.0)
    parser.add_argument("--agent_goal_weight", type=float, default=5.0)
    parser.add_argument("--anchor_weight", type=float, default=0.1,
                        help="Anchor regularization weight (0 to disable)")
    parser.add_argument("--freeze_decoder", action="store_true",
                        help="Freeze decoder params (only fine-tune encoder)")
    parser.add_argument("--sample_weighting", type=str, default="uniform",
                        choices=["uniform", "score", "recency", "combined"],
                        help="How to weight level sampling: uniform, score (SFL), "
                             "recency (last replayed), combined (score * recency)")
    parser.add_argument("--recency_decay", type=float, default=0.9999,
                        help="Exponential decay rate for recency weighting per step gap")

    # SFL scoring (optional — score levels with a trained agent before fine-tuning)
    parser.add_argument("--agent_checkpoint", type=str, default=None,
                        help="Path to Orbax agent checkpoint dir for SFL scoring. "
                             "If provided, scores extra levels with agent rollouts before training.")
    parser.add_argument("--extra_train_data", type=str, default=None,
                        help="Path to additional training data (.npy or .npz) to sample from. "
                             "Levels are scored with --agent_checkpoint and merged with --train_data.")
    parser.add_argument("--sfl_rollouts", type=int, default=5,
                        help="Number of rollouts per level for SFL scoring")
    parser.add_argument("--sfl_batch_size", type=int, default=32,
                        help="Batch size for SFL rollout scoring")
    parser.add_argument("--sfl_max_levels", type=int, default=100000,
                        help="Max extra levels to sample and score (0 = use all extra data)")

    # Saving
    parser.add_argument("--output_dir", type=str, default="vae/finetune_checkpoints",
                        help="Directory to save fine-tuned checkpoints")
    parser.add_argument("--save_freq", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_freq", type=int, default=100,
                        help="Evaluate on val set every N steps")

    # Logging
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="wandb project name (None to disable)")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # ── Device setup ──
    n_devices = jax.device_count()
    print(f"[Fine-tune] JAX devices: {n_devices} x {jax.devices()[0].platform}")

    # Batch size must be divisible by device count
    if args.batch_size % n_devices != 0:
        old_bs = args.batch_size
        args.batch_size = (args.batch_size // n_devices) * n_devices
        print(f"[Fine-tune] Adjusted batch_size {old_bs} -> {args.batch_size} (divisible by {n_devices})")

    # Create mesh for data parallelism
    mesh = Mesh(np.array(jax.devices()), axis_names=("batch",))
    data_sharding = NamedSharding(mesh, P("batch"))       # shard batch dim
    replicated_sharding = NamedSharding(mesh, P())         # replicate params

    # ── Setup ──
    print(f"[Fine-tune] Loading pre-trained VAE from {args.checkpoint_path}")
    model, pretrained_params, pretrained_cfg = load_pretrained(
        args.checkpoint_path, args.config_path)

    print(f"[Fine-tune] Loading training data from {args.train_data}")
    train_data = load_train_data(args.train_data)
    train_tokens = train_data["tokens"]
    print(f"[Fine-tune] Training set: {len(train_tokens)} levels")

    # ── Optional: score extra levels with agent and merge ──
    if args.agent_checkpoint:
        import orbax.checkpoint as ocp
        print(f"[Fine-tune] Loading agent from {args.agent_checkpoint} for SFL scoring...")
        ws_manager = ocp.CheckpointManager(
            args.agent_checkpoint,
            item_handlers=ocp.StandardCheckpointHandler(),
        )
        ws_step = ws_manager.latest_step()
        try:
            ws_ckpt = ws_manager.restore(ws_step)
        except (ValueError, KeyError) as e:
            if "Topology mismatch" in str(e) or "was not found" in str(e):
                print(f"[Fine-tune] Topology mismatch (GPU→TPU) — loading arrays directly...")
                import tensorstore as ts
                import os
                ckpt_dir = os.path.join(args.agent_checkpoint, str(ws_step), "default")
                def _load_tree(path):
                    """Recursively load zarr arrays from Orbax checkpoint as numpy."""
                    meta_path = os.path.join(path, ".zarray")
                    if os.path.exists(meta_path):
                        spec = {"driver": "zarr", "kvstore": {"driver": "file", "path": path}}
                        return np.array(ts.open(spec).result().read().result())
                    result = {}
                    for name in sorted(os.listdir(path)):
                        sub = os.path.join(path, name)
                        if os.path.isdir(sub):
                            val = _load_tree(sub)
                            if val is not None:
                                result[name] = val
                    return result if result else None
                ws_ckpt = _load_tree(ckpt_dir)
                if ws_ckpt is None:
                    raise RuntimeError(f"Could not load checkpoint from {ckpt_dir}")
                print(f"[Fine-tune] Loaded checkpoint as numpy arrays (keys: {list(ws_ckpt.keys()) if isinstance(ws_ckpt, dict) else type(ws_ckpt)})")
            else:
                raise
        agent_params = ws_ckpt["params"] if isinstance(ws_ckpt, dict) and "params" in ws_ckpt else ws_ckpt.params
        print(f"[Fine-tune] Agent loaded from step {ws_step}")

        # Buffer already has SFL scores
        n_buffer = len(train_tokens)
        buffer_scores = train_data.get("scores", None)
        if buffer_scores is not None:
            print(f"[Fine-tune] Buffer has existing SFL scores for {n_buffer} levels "
                  f"(mean={buffer_scores.mean():.4f}, nonzero={int((buffer_scores > 0).sum())})")

        # Load and score extra levels from general training set
        if args.extra_train_data:
            extra_data = load_train_data(args.extra_train_data)
            extra_tokens_all = extra_data["tokens"]
            n_extra_available = len(extra_tokens_all)

            # Sample up to sfl_max_levels extra levels
            max_extra = args.sfl_max_levels if args.sfl_max_levels > 0 else n_extra_available
            max_extra = min(max_extra, n_extra_available)
            rng_sub = np.random.default_rng(args.seed)
            if n_extra_available > max_extra:
                extra_idx = rng_sub.choice(n_extra_available, size=max_extra, replace=False)
                extra_tokens = extra_tokens_all[extra_idx]
            else:
                extra_tokens = extra_tokens_all
            print(f"[Fine-tune] Scoring {len(extra_tokens)}/{n_extra_available} extra levels from {args.extra_train_data}")

            extra_scores = score_levels_sfl(
                agent_params=agent_params,
                tokens=extra_tokens,
                agent_checkpoint_path=args.agent_checkpoint,
                num_rollouts=args.sfl_rollouts,
                batch_size=args.sfl_batch_size,
                seed=args.seed,
            )

            # Merge: buffer levels (with existing scores) + scored extra levels
            merged_tokens = np.concatenate([train_tokens, extra_tokens], axis=0)
            if buffer_scores is None:
                buffer_scores = np.zeros(n_buffer, dtype=np.float32)
            merged_scores = np.concatenate([buffer_scores, extra_scores])

            train_data["tokens"] = merged_tokens
            train_data["scores"] = merged_scores
            # Clear recency metadata (doesn't apply to extra levels)
            for key in ["last_replayed", "last_inserted", "first_inserted", "replay_count"]:
                train_data.pop(key, None)

            train_tokens = merged_tokens
            n_scored_extra = int((extra_scores > 0).sum())
            print(f"[Fine-tune] Merged: {n_buffer} buffer + {len(extra_tokens)} extra "
                  f"= {len(merged_tokens)} total levels "
                  f"({n_scored_extra} extra with nonzero SFL score)")
        else:
            # No extra data — just re-score the buffer levels if no existing scores
            if buffer_scores is None:
                print(f"[Fine-tune] No existing scores — scoring {n_buffer} buffer levels")
                sfl_scores = score_levels_sfl(
                    agent_params=agent_params,
                    tokens=train_tokens,
                    agent_checkpoint_path=args.agent_checkpoint,
                    num_rollouts=args.sfl_rollouts,
                    batch_size=args.sfl_batch_size,
                    seed=args.seed,
                )
                train_data["scores"] = sfl_scores

        # Use score-based weighting
        if args.sample_weighting == "uniform":
            print("[Fine-tune] Switching sample_weighting to 'score' (SFL scores available)")
            args.sample_weighting = "score"

    # Compute sampling weights
    sample_probs = compute_sample_weights(train_data, mode=args.sample_weighting,
                                          recency_decay=args.recency_decay)
    if args.sample_weighting != "uniform":
        effective_n = 1.0 / (sample_probs ** 2).sum()
        print(f"[Fine-tune] Sample weighting: {args.sample_weighting} "
              f"(effective N = {effective_n:.0f} / {len(train_tokens)})")

    val_tokens = load_val_data(args.val_data)
    if val_tokens is not None:
        print(f"[Fine-tune] Validation set: {len(val_tokens)} levels")

    # ── wandb ──
    if args.wandb_project:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"finetune_{os.path.basename(args.train_data)}",
            config=vars(args),
        )

    # ── Optimizer with differential LR ──
    dec_lr = 0.0 if args.freeze_decoder else args.decoder_lr
    if args.freeze_decoder:
        print("[Fine-tune] Decoder FROZEN — only encoder will be updated")
    optimizer, param_labels = create_optimizer(
        pretrained_params,
        encoder_lr=args.encoder_lr,
        decoder_lr=dec_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.num_steps,
        max_grad_norm=args.max_grad_norm,
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=pretrained_params,
        tx=optimizer,
    )

    # Shard state (params replicated, opt state replicated)
    state = jax.device_put(state, replicated_sharding)

    # Freeze pretrained params for anchor loss
    frozen_pretrained = jax.device_put(pretrained_params, replicated_sharding)

    # Config tuple for jit (must be hashable)
    loss_config = tuple(sorted({
        "vocab_size": pretrained_cfg.get("vocab_size", 170),
        "recon_weight": args.recon_weight,
        "beta": args.beta,
        "wall_weight": args.wall_weight,
        "agent_goal_weight": args.agent_goal_weight,
        "anchor_weight": args.anchor_weight,
    }.items()))

    # Create sharded step functions
    @partial(jax.jit, in_shardings=(replicated_sharding, replicated_sharding, data_sharding, replicated_sharding),
             out_shardings=(replicated_sharding, replicated_sharding))
    def sharded_train_step(state, pretrained_params, batch, rng):
        config = dict(loss_config)
        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, aux), grads = grad_fn(state.params, pretrained_params, batch, rng, model, config)
        state = state.apply_gradients(grads=grads)
        return state, aux

    @partial(jax.jit, in_shardings=(replicated_sharding, replicated_sharding, data_sharding, replicated_sharding),
             out_shardings=replicated_sharding)
    def sharded_eval_step(params, pretrained_params, batch, rng):
        config = dict(loss_config)
        _, aux = compute_loss(params, pretrained_params, batch, rng, model, config)
        rng_z, _ = jax.random.split(rng)
        logits, _, _ = model.apply({"params": params}, batch, rng_z, train=False)
        preds = jnp.argmax(logits, axis=-1)
        token_acc = jnp.mean(preds == batch)
        aux["token_accuracy"] = token_acc
        return aux

    def shard_batch(batch_np):
        """Place a numpy batch onto devices with data sharding."""
        return jax.device_put(jnp.array(batch_np), data_sharding)

    # ── Training loop ──
    rng = jax.random.PRNGKey(args.seed)
    best_val_loss = float("inf")

    print(f"\n[Fine-tune] Starting: {args.num_steps} steps, "
          f"encoder_lr={args.encoder_lr}, decoder_lr={args.decoder_lr}, "
          f"beta={args.beta}, anchor={args.anchor_weight}")
    print(f"[Fine-tune] Global batch_size={args.batch_size} "
          f"(per-device={args.batch_size // n_devices})")
    print(f"[Fine-tune] Output dir: {args.output_dir}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    # Save config
    with open(os.path.join(args.output_dir, "finetune_config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    t0 = time.time()
    for step in range(1, args.num_steps + 1):
        rng, rng_step = jax.random.split(rng)

        # Sample batch (weighted if configured)
        idx = np.random.choice(len(train_tokens), size=args.batch_size, p=sample_probs)
        batch = shard_batch(train_tokens[idx])

        state, aux = sharded_train_step(state, frozen_pretrained, batch, rng_step)

        # Logging
        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            print(f"[Step {step:5d}/{args.num_steps}] "
                  f"loss={float(aux['total_loss']):.4f} "
                  f"recon={float(aux['recon_loss']):.4f} "
                  f"kl={float(aux['kl_loss']):.4f} "
                  f"anchor={float(aux['anchor_loss']):.4f} "
                  f"({steps_per_sec:.1f} steps/s)")

            if args.wandb_project:
                import wandb
                wandb.log({
                    "train/total_loss": float(aux["total_loss"]),
                    "train/recon_loss": float(aux["recon_loss"]),
                    "train/kl_loss": float(aux["kl_loss"]),
                    "train/anchor_loss": float(aux["anchor_loss"]),
                    "train/mean_abs_mean": float(aux["mean_abs_mean"]),
                    "train/mean_logvar": float(aux["mean_logvar"]),
                    "train/step": step,
                }, step=step)

        # Validation
        if val_tokens is not None and step % args.eval_freq == 0:
            rng, rng_val = jax.random.split(rng)
            val_bs = min(1024, len(val_tokens))
            val_bs = (val_bs // n_devices) * n_devices  # ensure divisible
            val_idx = np.random.randint(0, len(val_tokens), size=(val_bs,))
            val_batch = shard_batch(val_tokens[val_idx])
            val_aux = sharded_eval_step(state.params, frozen_pretrained, val_batch, rng_val)

            val_loss = float(val_aux["total_loss"])
            val_acc = float(val_aux["token_accuracy"])
            print(f"  [Val] loss={val_loss:.4f} "
                  f"recon={float(val_aux['recon_loss']):.4f} "
                  f"kl={float(val_aux['kl_loss']):.4f} "
                  f"acc={val_acc:.4f}")

            if args.wandb_project:
                import wandb
                wandb.log({
                    "val/total_loss": val_loss,
                    "val/recon_loss": float(val_aux["recon_loss"]),
                    "val/kl_loss": float(val_aux["kl_loss"]),
                    "val/token_accuracy": val_acc,
                    "val/anchor_loss": float(val_aux["anchor_loss"]),
                }, step=step)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    state.params, vars(args), step,
                    os.path.join(args.output_dir, "checkpoint_best.pkl"))

        # Periodic checkpoint
        if step % args.save_freq == 0:
            save_checkpoint(
                state.params, vars(args), step,
                os.path.join(args.output_dir, f"checkpoint_{step}.pkl"))

    # Final checkpoint
    save_checkpoint(
        state.params, vars(args), args.num_steps,
        os.path.join(args.output_dir, "checkpoint_final.pkl"))

    print(f"\n[Fine-tune] Done. {args.num_steps} steps in {time.time() - t0:.1f}s")
    print(f"[Fine-tune] Best val loss: {best_val_loss:.4f}")
    print(f"[Fine-tune] Checkpoints in: {args.output_dir}")

    if args.wandb_project:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
