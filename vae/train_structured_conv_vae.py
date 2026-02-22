#!/usr/bin/env python3
"""
Train a structured 3-head convolutional VAE on 13x13 maze grids.

Reconstruction heads:
- wall map (Bernoulli field): BCE + soft Dice
- goal location (categorical over 169 cells): CE with wall masking
- agent location (categorical over 169 cells): CE with wall masking
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml
from flax import linen as nn
from flax.training import train_state
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "structured_conv_vae_config.yml",
    )
    return parser.parse_args()


class StructuredConvGridVAE(nn.Module):
    latent_dim: int
    n_diff_dims: int = 0  # 0 = disabled (plain VAE). >0 = disentangled mode.

    @nn.compact
    def __call__(self, x, z_rng):
        # Encoder
        h = nn.Conv(32, (3, 3), strides=(2, 2), padding="SAME")(x)  # 13 -> 7
        h = nn.relu(h)
        h = nn.Conv(64, (3, 3), strides=(2, 2), padding="SAME")(h)  # 7 -> 4
        h = nn.relu(h)
        h = nn.Conv(128, (3, 3), strides=(2, 2), padding="SAME")(h)  # 4 -> 2
        h = nn.relu(h)
        h = h.reshape((h.shape[0], -1))
        h = nn.Dense(256)(h)
        h = nn.relu(h)

        stats = nn.Dense(self.latent_dim * 2)(h)
        mean, logvar = jnp.split(stats, 2, axis=-1)
        mean = jnp.tanh(mean) * 4.0

        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(z_rng, mean.shape)
        z = mean + eps * std

        # Difficulty prediction head (only when n_diff_dims > 0)
        # Operates on z_diff = z[:, :n_diff_dims] which is forced to encode
        # curriculum difficulty via a supervised regression loss during training.
        if self.n_diff_dims > 0:
            z_diff = z[:, : self.n_diff_dims]
            dh = nn.Dense(32, name="diff_h1")(z_diff)
            dh = nn.relu(dh)
            diff_pred = nn.Dense(1, name="diff_out")(dh)[..., 0]  # (B,) logit
        else:
            diff_pred = jnp.zeros(z.shape[0])

        # Decoder trunk
        d = nn.Dense(4 * 4 * 128)(z)
        d = nn.relu(d)
        d = d.reshape((d.shape[0], 4, 4, 128))

        d = jax.image.resize(d, (d.shape[0], 7, 7, d.shape[-1]), method="nearest")
        d = nn.Conv(96, (3, 3), padding="SAME")(d)
        d = nn.relu(d)

        d = jax.image.resize(d, (d.shape[0], 13, 13, d.shape[-1]), method="nearest")
        d = nn.Conv(64, (3, 3), padding="SAME")(d)
        d = nn.relu(d)

        # Structured heads
        wall_logits = nn.Conv(1, (1, 1), padding="SAME", name="wall_head")(d)[..., 0]
        goal_logits_map = nn.Conv(1, (1, 1), padding="SAME", name="goal_head")(d)[..., 0]
        agent_logits_map = nn.Conv(1, (1, 1), padding="SAME", name="agent_head")(d)[..., 0]

        return wall_logits, goal_logits_map, agent_logits_map, mean, logvar, diff_pred


def save_checkpoint(ckpt_dir: Path, state: train_state.TrainState, step):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint_{step}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"params": state.params, "step": step}, f)


def load_checkpoint(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    raw_step = data.get("step", 0)
    try:
        step = int(raw_step)
    except (TypeError, ValueError):
        step = 0
    return data["params"], step


def get_beta(
    step: int,
    *,
    schedule: str,
    beta_max: float,
    anneal_steps: int,
    cycle_steps: int,
    warmup_frac: float,
    ae_pretrain_steps: int,
) -> jnp.ndarray:
    if step < ae_pretrain_steps:
        return jnp.array(0.0, dtype=jnp.float32)

    offset_step = step - ae_pretrain_steps
    if schedule == "linear":
        progress = jnp.minimum(1.0, offset_step / max(1, anneal_steps))
        return beta_max * progress
    if schedule == "cyclical":
        cyc = max(1, cycle_steps)
        warm = max(1, int(warmup_frac * cyc))
        phase = offset_step % cyc
        progress = jnp.minimum(1.0, phase / warm)
        return beta_max * progress
    raise ValueError(f"Unsupported kl schedule: {schedule}")


def prepare_targets(batch: jnp.ndarray):
    wall_true = batch[..., 0]
    goal_true = batch[..., 1].reshape((batch.shape[0], -1))
    agent_true = batch[..., 2].reshape((batch.shape[0], -1))

    goal_idx = jnp.argmax(goal_true, axis=-1).astype(jnp.int32)
    agent_idx = jnp.argmax(agent_true, axis=-1).astype(jnp.int32)
    wall_flat_mask = (wall_true.reshape((batch.shape[0], -1)) > 0.5)
    return wall_true, goal_idx, agent_idx, wall_flat_mask


def masked_logits(logits_flat: jnp.ndarray, wall_mask: jnp.ndarray) -> jnp.ndarray:
    # Prevent goal/agent probability mass on wall cells.
    neg_inf = jnp.full_like(logits_flat, -1e9)
    return jnp.where(wall_mask, neg_inf, logits_flat)


def soft_dice_loss(probs: jnp.ndarray, target: jnp.ndarray, eps: float) -> jnp.ndarray:
    inter = jnp.sum(probs * target, axis=(1, 2))
    denom = jnp.sum(probs, axis=(1, 2)) + jnp.sum(target, axis=(1, 2))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - jnp.mean(dice)


def soft_tversky_loss(
    probs: jnp.ndarray,
    target: jnp.ndarray,
    alpha: float,
    beta: float,
    eps: float,
) -> jnp.ndarray:
    tp = jnp.sum(probs * target, axis=(1, 2))
    fp = jnp.sum(probs * (1.0 - target), axis=(1, 2))
    fn = jnp.sum((1.0 - probs) * target, axis=(1, 2))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - jnp.mean(tversky)


def wall_iou_from_logits(wall_logits: jnp.ndarray, wall_true: jnp.ndarray, threshold: float) -> jnp.ndarray:
    wall_pred = jax.nn.sigmoid(wall_logits) >= threshold
    wall_true_b = wall_true > 0.5
    inter = jnp.sum(jnp.logical_and(wall_pred, wall_true_b), axis=(1, 2)).astype(jnp.float32)
    union = jnp.sum(jnp.logical_or(wall_pred, wall_true_b), axis=(1, 2)).astype(jnp.float32)
    return jnp.mean(inter / jnp.maximum(union, 1.0))


def run_training(config: dict):
    work = Path(config["working_path"])
    vae_folder = Path(config["vae_folder"])
    dataset_path = work / vae_folder / config["dataset_path"]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    npz = np.load(dataset_path)
    x = npz["grids"].astype(np.float32)
    if x.ndim != 4 or x.shape[1:] != (13, 13, 3):
        raise ValueError(f"Unexpected grid shape {x.shape}, expected [N,13,13,3]")

    # Difficulty scores — present in PLR datasets, optional for random-maze datasets.
    n_diff_dims = int(config.get("n_diff_dims", 0))
    has_scores = "scores" in npz and n_diff_dims > 0
    if has_scores:
        raw_scores = npz["scores"].astype(np.float32)
        # Rank-normalise to [0,1] so all difficulty levels are equally represented.
        # This avoids the heavy right-skew of raw PVL scores (mean≈0.03, max≈0.6).
        order = raw_scores.argsort()
        norm_scores = np.empty_like(raw_scores)
        norm_scores[order] = np.linspace(0.0, 1.0, len(raw_scores))
        print(f"Scores loaded and rank-normalised. raw range: "
              f"[{raw_scores.min():.4f}, {raw_scores.max():.4f}]")
    else:
        norm_scores = np.zeros(len(x), dtype=np.float32)
        if n_diff_dims > 0:
            print("WARNING: n_diff_dims > 0 but no 'scores' key in dataset — "
                  "difficulty supervision disabled.")
            n_diff_dims = 0

    n = len(x)
    rng_np = np.random.default_rng(config["seed"])
    perm = rng_np.permutation(n)
    n_val = int(config["val_fraction"] * n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    x_train = x[train_idx]
    x_val = x[val_idx]
    s_train = norm_scores[train_idx]
    s_val = norm_scores[val_idx]

    print(f"Loaded dataset: {x.shape}")
    print(f"Train/Val split: {x_train.shape} / {x_val.shape}")

    wall_pos_rate = float(x_train[..., 0].mean())
    print(f"Wall positive rate: {wall_pos_rate:.4f}")
    print(f"Disentangled dims (n_diff_dims): {n_diff_dims}")

    model = StructuredConvGridVAE(
        latent_dim=int(config["latent_dim"]),
        n_diff_dims=n_diff_dims,
    )
    key = jax.random.PRNGKey(int(config["seed"]))
    key, init_key, z_key = jax.random.split(key, 3)

    start_step = 0
    if config.get("resume_path"):
        params, start_step = load_checkpoint(Path(config["resume_path"]))
        print(f"Resuming from step {start_step}")
    else:
        params = model.init(init_key, jnp.zeros((1, 13, 13, 3), dtype=jnp.float32), z_key)["params"]

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(float(config["learning_rate"])),
    )

    wall_pos_weight = float(config.get("wall_pos_weight", 1.0))
    if wall_pos_weight <= 0:
        raise ValueError("wall_pos_weight must be > 0")

    wall_bce_weight = float(config.get("wall_bce_weight", 1.0))
    wall_dice_weight = float(config.get("wall_dice_weight", 1.0))
    wall_tversky_weight = float(config.get("wall_tversky_weight", 0.0))
    wall_tversky_alpha = float(config.get("wall_tversky_alpha", 0.5))
    wall_tversky_beta = float(config.get("wall_tversky_beta", 0.5))
    wall_count_weight = float(config.get("wall_count_weight", 0.0))
    goal_ce_weight = float(config.get("goal_ce_weight", 1.0))
    agent_ce_weight = float(config.get("agent_ce_weight", 1.0))
    overlap_penalty_weight = float(config.get("overlap_penalty_weight", 0.0))
    # Disentanglement loss weights (only active when n_diff_dims > 0)
    diff_loss_weight = float(config.get("diff_loss_weight", 0.0)) if n_diff_dims > 0 else 0.0
    mi_penalty_weight = float(config.get("mi_penalty_weight", 0.0)) if n_diff_dims > 0 else 0.0
    wall_focus_steps = int(config.get("wall_focus_steps", 0))
    wall_focus_entity_scale = float(config.get("wall_focus_entity_scale", 1.0))
    wall_focus_kl_scale = float(config.get("wall_focus_kl_scale", 1.0))

    kl_schedule = str(config.get("kl_schedule", "cyclical"))
    beta_max = float(config.get("beta_max", 0.05))
    kl_free_bits = float(config.get("kl_free_bits_per_dim", 0.01))
    cycle_steps = int(config.get("kl_cycle_steps", max(1, int(config["num_steps"]) // 4)))
    warmup_frac = float(config.get("kl_warmup_frac", 0.5))
    ae_pretrain_steps = int(config.get("ae_pretrain_steps", 0))
    wall_threshold = float(config.get("wall_threshold", 0.5))
    dice_eps = float(config.get("dice_eps", 1e-6))

    print(
        "Loss weights: "
        f"wall_bce={wall_bce_weight}, wall_dice={wall_dice_weight}, "
        f"wall_tversky={wall_tversky_weight}, wall_count={wall_count_weight}, "
        f"goal_ce={goal_ce_weight}, agent_ce={agent_ce_weight}, overlap={overlap_penalty_weight}"
    )
    print(
        f"Wall focus: steps={wall_focus_steps}, entity_scale={wall_focus_entity_scale}, "
        f"kl_scale={wall_focus_kl_scale}"
    )
    print(
        f"KL schedule={kl_schedule}, beta_max={beta_max}, free_bits/dim={kl_free_bits}, "
        f"cycle_steps={cycle_steps}, warmup_frac={warmup_frac}, ae_pretrain_steps={ae_pretrain_steps}"
    )
    if n_diff_dims > 0:
        print(f"Disentanglement: diff_loss_weight={diff_loss_weight}, mi_penalty_weight={mi_penalty_weight}")

    @jax.jit
    def train_step(state, batch, scores, z_rng, kl_weight, entity_scale):
        def loss_fn(params):
            wall_logits, goal_logits_map, agent_logits_map, mean, logvar, diff_pred = model.apply({"params": params}, batch, z_rng)

            wall_true, goal_idx, agent_idx, wall_mask = prepare_targets(batch)
            goal_logits = goal_logits_map.reshape((batch.shape[0], -1))
            agent_logits = agent_logits_map.reshape((batch.shape[0], -1))
            goal_logits_m = masked_logits(goal_logits, wall_mask)
            agent_logits_m = masked_logits(agent_logits, wall_mask)

            wall_bce = optax.sigmoid_binary_cross_entropy(wall_logits, wall_true)
            wall_bce = wall_bce * (1.0 + wall_true * (wall_pos_weight - 1.0))
            wall_bce = jnp.mean(wall_bce)

            wall_probs = jax.nn.sigmoid(wall_logits)
            wall_dice = soft_dice_loss(wall_probs, wall_true, dice_eps)
            wall_tversky = soft_tversky_loss(
                wall_probs,
                wall_true,
                wall_tversky_alpha,
                wall_tversky_beta,
                dice_eps,
            )
            wall_true_count = jnp.mean(wall_true, axis=(1, 2))
            wall_pred_count = jnp.mean(wall_probs, axis=(1, 2))
            wall_count_l1 = jnp.mean(jnp.abs(wall_pred_count - wall_true_count))
            wall_loss = (
                wall_bce_weight * wall_bce
                + wall_dice_weight * wall_dice
                + wall_tversky_weight * wall_tversky
                + wall_count_weight * wall_count_l1
            )

            goal_ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(goal_logits_m, goal_idx))
            agent_ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(agent_logits_m, agent_idx))

            goal_probs = jax.nn.softmax(goal_logits_m, axis=-1)
            agent_probs = jax.nn.softmax(agent_logits_m, axis=-1)
            overlap = jnp.mean(jnp.sum(goal_probs * agent_probs, axis=-1))

            recon = wall_loss + entity_scale * (
                goal_ce_weight * goal_ce
                + agent_ce_weight * agent_ce
                + overlap_penalty_weight * overlap
            )

            kl_per_dim = -0.5 * (1 + logvar - jnp.square(mean) - jnp.exp(logvar))
            kl_per_dim = jnp.maximum(kl_per_dim, kl_free_bits)
            kl = jnp.mean(jnp.sum(kl_per_dim, axis=-1))

            # Difficulty regression loss: MSE between sigmoid(diff_pred) and
            # rank-normalised PVL score. Active only when n_diff_dims > 0.
            diff_loss = jnp.mean(jnp.square(jax.nn.sigmoid(diff_pred) - scores))

            # MI decorrelation: penalise covariance between z_diff and z_style.
            # Uses the posterior mean (not the sample) for a stable estimate.
            if n_diff_dims > 0 and n_diff_dims < mean.shape[-1]:
                z_diff_m = mean[:, :n_diff_dims]
                z_style_m = mean[:, n_diff_dims:]
                z_diff_c = z_diff_m - z_diff_m.mean(0)
                z_style_c = z_style_m - z_style_m.mean(0)
                cov = jnp.einsum("bi,bj->ij", z_diff_c, z_style_c) / (mean.shape[0] - 1)
                mi_penalty = jnp.sum(jnp.square(cov))
            else:
                mi_penalty = jnp.array(0.0)

            total = (
                recon
                + kl_weight * kl
                + diff_loss_weight * diff_loss
                + mi_penalty_weight * mi_penalty
            )

            goal_pred = jnp.argmax(goal_logits_m, axis=-1)
            agent_pred = jnp.argmax(agent_logits_m, axis=-1)
            goal_acc = jnp.mean((goal_pred == goal_idx).astype(jnp.float32))
            agent_acc = jnp.mean((agent_pred == agent_idx).astype(jnp.float32))
            same_cell_rate = jnp.mean((goal_pred == agent_pred).astype(jnp.float32))
            wall_iou = wall_iou_from_logits(wall_logits, wall_true, wall_threshold)
            mu_abs = jnp.mean(jnp.abs(mean))
            logvar_mean = jnp.mean(logvar)
            diff_pred_prob = jax.nn.sigmoid(diff_pred)
            diff_mae = jnp.mean(jnp.abs(diff_pred_prob - scores))

            aux = (
                recon,
                kl,
                wall_bce,
                wall_dice,
                wall_tversky,
                wall_count_l1,
                goal_ce,
                agent_ce,
                overlap,
                wall_iou,
                goal_acc,
                agent_acc,
                same_cell_rate,
                mu_abs,
                logvar_mean,
                diff_loss,
                diff_mae,
                mi_penalty,
            )
            return total, aux

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, aux

    @jax.jit
    def eval_step(params, batch, scores, z_rng):
        wall_logits, goal_logits_map, agent_logits_map, mean, logvar, diff_pred = model.apply({"params": params}, batch, z_rng)

        wall_true, goal_idx, agent_idx, wall_mask = prepare_targets(batch)
        goal_logits = goal_logits_map.reshape((batch.shape[0], -1))
        agent_logits = agent_logits_map.reshape((batch.shape[0], -1))
        goal_logits_m = masked_logits(goal_logits, wall_mask)
        agent_logits_m = masked_logits(agent_logits, wall_mask)

        wall_bce = optax.sigmoid_binary_cross_entropy(wall_logits, wall_true)
        wall_bce = wall_bce * (1.0 + wall_true * (wall_pos_weight - 1.0))
        wall_bce = jnp.mean(wall_bce)
        wall_probs = jax.nn.sigmoid(wall_logits)
        wall_dice = soft_dice_loss(wall_probs, wall_true, dice_eps)
        wall_tversky = soft_tversky_loss(
            wall_probs,
            wall_true,
            wall_tversky_alpha,
            wall_tversky_beta,
            dice_eps,
        )
        wall_true_count = jnp.mean(wall_true, axis=(1, 2))
        wall_pred_count = jnp.mean(wall_probs, axis=(1, 2))
        wall_count_l1 = jnp.mean(jnp.abs(wall_pred_count - wall_true_count))
        wall_loss = (
            wall_bce_weight * wall_bce
            + wall_dice_weight * wall_dice
            + wall_tversky_weight * wall_tversky
            + wall_count_weight * wall_count_l1
        )

        goal_ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(goal_logits_m, goal_idx))
        agent_ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(agent_logits_m, agent_idx))
        goal_probs = jax.nn.softmax(goal_logits_m, axis=-1)
        agent_probs = jax.nn.softmax(agent_logits_m, axis=-1)
        overlap = jnp.mean(jnp.sum(goal_probs * agent_probs, axis=-1))

        recon = wall_loss + goal_ce_weight * goal_ce + agent_ce_weight * agent_ce + overlap_penalty_weight * overlap

        kl_per_dim = -0.5 * (1 + logvar - jnp.square(mean) - jnp.exp(logvar))
        kl_per_dim = jnp.maximum(kl_per_dim, kl_free_bits)
        kl = jnp.mean(jnp.sum(kl_per_dim, axis=-1))

        diff_pred_prob = jax.nn.sigmoid(diff_pred)
        diff_loss = jnp.mean(jnp.square(diff_pred_prob - scores))
        diff_mae = jnp.mean(jnp.abs(diff_pred_prob - scores))

        goal_pred = jnp.argmax(goal_logits_m, axis=-1)
        agent_pred = jnp.argmax(agent_logits_m, axis=-1)
        goal_acc = jnp.mean((goal_pred == goal_idx).astype(jnp.float32))
        agent_acc = jnp.mean((agent_pred == agent_idx).astype(jnp.float32))
        same_cell_rate = jnp.mean((goal_pred == agent_pred).astype(jnp.float32))
        wall_iou = wall_iou_from_logits(wall_logits, wall_true, wall_threshold)
        mu_abs = jnp.mean(jnp.abs(mean))
        logvar_mean = jnp.mean(logvar)

        return (
            recon,
            kl,
            wall_bce,
            wall_dice,
            wall_tversky,
            wall_count_l1,
            goal_ce,
            agent_ce,
            overlap,
            wall_iou,
            goal_acc,
            agent_acc,
            same_cell_rate,
            mu_abs,
            logvar_mean,
            diff_loss,
            diff_mae,
        )

    ckpt_dir = work / vae_folder / config["checkpoint_dir"]
    plot_path = work / vae_folder / "structured_conv_training_metrics.png"
    history_path = work / vae_folder / "structured_conv_training_history.json"

    history = {
        "recon": [],
        "kl": [],
        "wall_iou": [],
        "wall_bce": [],
        "wall_dice": [],
        "wall_tversky": [],
        "wall_count_l1": [],
        "goal_acc": [],
        "agent_acc": [],
        "same_cell_rate": [],
        "diff_loss": [],
        "diff_mae": [],
        "mi_penalty": [],
        "val_recon": [],
        "val_kl": [],
        "val_wall_iou": [],
        "val_wall_bce": [],
        "val_wall_dice": [],
        "val_wall_tversky": [],
        "val_wall_count_l1": [],
        "val_goal_acc": [],
        "val_agent_acc": [],
        "val_same_cell_rate": [],
        "val_diff_loss": [],
        "val_diff_mae": [],
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    progress = tqdm(range(start_step, int(config["num_steps"])), desc="Structured Conv VAE")

    for step in progress:
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (int(config["batch_size"]),), 0, len(x_train))
        idx_np = np.array(idx)
        batch = x_train[idx_np]
        scores_batch = s_train[idx_np]

        key, z_rng = jax.random.split(key)
        beta = get_beta(
            step,
            schedule=kl_schedule,
            beta_max=beta_max,
            anneal_steps=int(config["anneal_steps"]),
            cycle_steps=cycle_steps,
            warmup_frac=warmup_frac,
            ae_pretrain_steps=ae_pretrain_steps,
        )
        if step < wall_focus_steps:
            entity_scale = jnp.array(wall_focus_entity_scale, dtype=jnp.float32)
            beta = beta * wall_focus_kl_scale
        else:
            entity_scale = jnp.array(1.0, dtype=jnp.float32)

        state, _loss, aux = train_step(state, jnp.array(batch), jnp.array(scores_batch), z_rng, beta, entity_scale)
        (
            recon,
            kl,
            wall_bce,
            wall_dice,
            wall_tversky,
            wall_count_l1,
            goal_ce,
            agent_ce,
            overlap,
            wall_iou,
            goal_acc,
            agent_acc,
            same_cell_rate,
            mu_abs,
            logvar_mean,
            diff_loss,
            diff_mae,
            mi_penalty,
        ) = aux

        history["recon"].append(float(recon))
        history["kl"].append(float(kl))
        history["wall_iou"].append(float(wall_iou))
        history["wall_bce"].append(float(wall_bce))
        history["wall_dice"].append(float(wall_dice))
        history["wall_tversky"].append(float(wall_tversky))
        history["wall_count_l1"].append(float(wall_count_l1))
        history["goal_acc"].append(float(goal_acc))
        history["agent_acc"].append(float(agent_acc))
        history["same_cell_rate"].append(float(same_cell_rate))
        history["diff_loss"].append(float(diff_loss))
        history["diff_mae"].append(float(diff_mae))
        history["mi_penalty"].append(float(mi_penalty))

        if step % int(config["plot_freq"]) == 0:
            key, ev = jax.random.split(key)
            vbs = min(int(config["val_eval_size"]), len(x_val))
            vsel = rng_np.choice(len(x_val), size=vbs, replace=False)
            vbatch = jnp.array(x_val[vsel])
            vscores = jnp.array(s_val[vsel])
            (
                v_recon,
                v_kl,
                v_wall_bce,
                v_wall_dice,
                v_wall_tversky,
                v_wall_count_l1,
                v_goal_ce,
                v_agent_ce,
                v_overlap,
                v_wall_iou,
                v_goal_acc,
                v_agent_acc,
                v_same_cell_rate,
                v_mu_abs,
                v_logvar_mean,
                v_diff_loss,
                v_diff_mae,
            ) = eval_step(state.params, vbatch, vscores, ev)

            history["val_recon"].append(float(v_recon))
            history["val_kl"].append(float(v_kl))
            history["val_wall_iou"].append(float(v_wall_iou))
            history["val_wall_bce"].append(float(v_wall_bce))
            history["val_wall_dice"].append(float(v_wall_dice))
            history["val_wall_tversky"].append(float(v_wall_tversky))
            history["val_wall_count_l1"].append(float(v_wall_count_l1))
            history["val_goal_acc"].append(float(v_goal_acc))
            history["val_agent_acc"].append(float(v_agent_acc))
            history["val_same_cell_rate"].append(float(v_same_cell_rate))
            history["val_diff_loss"].append(float(v_diff_loss))
            history["val_diff_mae"].append(float(v_diff_mae))

            postfix = {
                "Recon": f"{float(recon):.4f}",
                "KL": f"{float(kl):.4f}",
                "WallIoU": f"{float(wall_iou):.3f}",
                "GoalAcc": f"{float(goal_acc):.3f}",
                "AgentAcc": f"{float(agent_acc):.3f}",
                "ValRecon": f"{float(v_recon):.4f}",
                "ValWallIoU": f"{float(v_wall_iou):.3f}",
                "ValGoalAcc": f"{float(v_goal_acc):.3f}",
                "ValAgentAcc": f"{float(v_agent_acc):.3f}",
                "WallBCE": f"{float(wall_bce):.3f}",
                "WallDice": f"{float(wall_dice):.3f}",
                "WallT": f"{float(wall_tversky):.3f}",
                "WallCnt": f"{float(wall_count_l1):.3f}",
                "Beta": f"{float(beta):.3f}",
                "MuAbs": f"{float(mu_abs):.3f}",
                "LV": f"{float(logvar_mean):.3f}",
                "ValMu": f"{float(v_mu_abs):.3f}",
                "ValLV": f"{float(v_logvar_mean):.3f}",
            }
            if n_diff_dims > 0:
                postfix["DiffMAE"] = f"{float(diff_mae):.3f}"
                postfix["ValDiffMAE"] = f"{float(v_diff_mae):.3f}"
                postfix["MI"] = f"{float(mi_penalty):.3f}"
            progress.set_postfix(postfix)

            # Plot losses
            axes[0].clear()
            axes[0].plot(history["recon"], label="Train Recon", color="blue")
            axes[0].plot(history["kl"], label="Train KL", color="red", alpha=0.7)
            if history["val_recon"]:
                x_val_pts = np.linspace(0, len(history["recon"]) - 1, len(history["val_recon"]))
                axes[0].plot(x_val_pts, history["val_recon"], label="Val Recon", color="green")
                axes[0].plot(x_val_pts, history["val_kl"], label="Val KL", color="orange", alpha=0.8)
            axes[0].set_yscale("log")
            axes[0].set_title(f"Step {step} | Beta {float(beta):.3f}")
            axes[0].legend(loc="best")

            # Plot structural metrics
            axes[1].clear()
            axes[1].plot(history["wall_iou"], label="Train Wall IoU", color="#4c72b0")
            axes[1].plot(history["goal_acc"], label="Train Goal Acc", color="#55a868")
            axes[1].plot(history["agent_acc"], label="Train Agent Acc", color="#c44e52")
            if history["val_wall_iou"]:
                x_val_pts = np.linspace(0, len(history["recon"]) - 1, len(history["val_wall_iou"]))
                axes[1].plot(x_val_pts, history["val_wall_iou"], label="Val Wall IoU", color="#8172b3")
                axes[1].plot(x_val_pts, history["val_goal_acc"], label="Val Goal Acc", color="#64b5cd")
                axes[1].plot(x_val_pts, history["val_agent_acc"], label="Val Agent Acc", color="#ccb974")
            axes[1].set_ylim(0.0, 1.0)
            axes[1].set_title("Structured Reconstruction Metrics")
            axes[1].legend(loc="best", ncol=3, fontsize=8)

            fig.tight_layout()
            fig.savefig(plot_path)

        if step % int(config["save_freq"]) == 0 and step > 0:
            save_checkpoint(ckpt_dir, state, step)

    save_checkpoint(ckpt_dir, state, "final")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    plt.close(fig)
    print(f"Training complete. Checkpoints: {ckpt_dir}")
    print(f"Training plot: {plot_path}")
    print(f"Training history: {history_path}")

    # After training: compute z_diff centroids per difficulty decile for the decoder.
    # These allow the decoder to target a specific difficulty without gradient-based
    # optimisation at decode time — just look up the nearest centroid.
    if n_diff_dims > 0:
        print("\nComputing z_diff centroids across difficulty deciles ...")
        n_deciles = 10
        encode_fn = jax.jit(lambda params, batch, rng: model.apply({"params": params}, batch, rng)[3])  # returns mean
        all_means = []
        batch_size_enc = 512
        dummy_rng = jax.random.PRNGKey(0)
        for start in range(0, len(x), batch_size_enc):
            end = min(start + batch_size_enc, len(x))
            bx = jnp.array(x[start:end])
            mu = encode_fn(state.params, bx, dummy_rng)
            all_means.append(np.array(mu))
        all_means = np.concatenate(all_means, axis=0)  # (N, latent_dim)
        z_diff_all = all_means[:, :n_diff_dims]         # (N, n_diff_dims)

        centroids = {}
        for d in range(n_deciles):
            lo = d / n_deciles
            hi = (d + 1) / n_deciles
            mask = (norm_scores >= lo) & (norm_scores < hi)
            if not mask.any():
                mask = np.abs(norm_scores - (lo + hi) / 2) < 0.15
            centroid = z_diff_all[mask].mean(axis=0).tolist() if mask.any() else [0.0] * n_diff_dims
            centroids[f"decile_{d}"] = centroid

        # Also save a fine-grained lookup: 100 percentile points
        fine_targets = np.linspace(0.0, 1.0, 100)
        fine_centroids = []
        for t in fine_targets:
            # Weighted average of z_diff using a Gaussian kernel around target
            w = np.exp(-0.5 * ((norm_scores - t) / 0.05) ** 2)
            if w.sum() < 1e-6:
                w = np.exp(-0.5 * ((norm_scores - t) / 0.15) ** 2)
            w = w / w.sum()
            fine_centroids.append((w[:, None] * z_diff_all).sum(axis=0).tolist())

        stats_path = work / vae_folder / "zdiff_centroids.json"
        with open(stats_path, "w") as f:
            json.dump({
                "n_diff_dims": n_diff_dims,
                "latent_dim": int(config["latent_dim"]),
                "decile_centroids": centroids,
                "fine_targets": fine_targets.tolist(),
                "fine_centroids": fine_centroids,
            }, f, indent=2)
        print(f"z_diff centroids saved → {stats_path}")


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    run_training(config)


if __name__ == "__main__":
    main()
