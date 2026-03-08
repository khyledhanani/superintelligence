#!/usr/bin/env python3
"""Train task-aware Maze beta-VAE.

Stage 1 (default): static-only supervision.
Stage 2: dynamic curriculum supervision enabled from dataset fields
`p_ema` and `success_obs_count`.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml
from flax.training import train_state
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from es.maze_ae import MazeTaskAwareVAE, compute_structural_targets_from_grids


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def weighted_bce(logits: jnp.ndarray, targets: jnp.ndarray, pos_weight: float) -> jnp.ndarray:
    loss = (
        pos_weight * targets * jax.nn.softplus(-logits)
        + (1.0 - targets) * jax.nn.softplus(logits)
    )
    return loss.mean()


def dice_loss(probs: jnp.ndarray, targets: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    intersection = (probs * targets).sum(axis=(1, 2))
    denom = probs.sum(axis=(1, 2)) + targets.sum(axis=(1, 2))
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


def huber_mean(pred: jnp.ndarray, target: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    err = pred - target
    abs_err = jnp.abs(err)
    quad = jnp.minimum(abs_err, delta)
    lin = abs_err - quad
    return (0.5 * quad ** 2 + delta * lin).mean()


def pairwise_metric_loss(z: jnp.ndarray, y: jnp.ndarray, y_weights: jnp.ndarray) -> jnp.ndarray:
    # z distances
    z_diff = z[:, None, :] - z[None, :, :]
    z_dist = jnp.sqrt(jnp.sum(z_diff ** 2, axis=-1) + 1e-8)

    # weighted semantic distances
    yw = y * y_weights[None, :]
    y_diff = yw[:, None, :] - yw[None, :, :]
    y_dist = jnp.sqrt(jnp.sum(y_diff ** 2, axis=-1) + 1e-8)

    b = z.shape[0]
    mask = jnp.triu(jnp.ones((b, b), dtype=jnp.float32), k=1)
    diff_sq = (z_dist - y_dist) ** 2
    denom = jnp.maximum(mask.sum(), 1.0)
    return (diff_sq * mask).sum() / denom


def compute_loss(
    params: dict,
    batch: dict,
    rng: jax.Array,
    config: dict,
) -> tuple[jnp.ndarray, dict]:
    grids = batch["grids"]
    static_targets = batch["static_targets"]
    p_ema = batch["p_ema"]
    success_obs_count = batch["success_obs_count"]

    b, h, w, _ = grids.shape

    wall_targets = grids[:, :, :, 0]
    goal_targets = grids[:, :, :, 1].reshape(b, -1).argmax(axis=-1)
    agent_targets = grids[:, :, :, 2].reshape(b, -1).argmax(axis=-1)

    out = MazeTaskAwareVAE(
        latent_dim=config["latent_dim"],
        height=config["height"],
        width=config["width"],
    ).apply({"params": params}, grids, z_rng=rng, deterministic=False)

    z = out["z"]
    mean = out["mean"]
    logvar = out["logvar"]
    wall_logits = out["wall_logits"]
    goal_logits = out["goal_logits"]
    agent_logits = out["agent_logits"]

    wall_probs = jax.nn.sigmoid(wall_logits)
    l_wall_bce = weighted_bce(wall_logits, wall_targets, config["wall_pos_weight"])
    l_wall_dice = dice_loss(wall_probs, wall_targets)
    l_wall = config["wall_bce_weight"] * l_wall_bce + config["wall_dice_weight"] * l_wall_dice

    l_goal = optax.softmax_cross_entropy_with_integer_labels(goal_logits, goal_targets).mean()
    l_agent = optax.softmax_cross_entropy_with_integer_labels(agent_logits, agent_targets).mean()
    l_pos = config["goal_ce_weight"] * l_goal + config["agent_ce_weight"] * l_agent

    goal_p = jax.nn.softmax(goal_logits)
    agent_p = jax.nn.softmax(agent_logits)
    l_overlap = config["overlap_penalty_weight"] * (goal_p * agent_p).sum(axis=-1).mean()
    l_recon = l_wall + l_pos + l_overlap

    l_kl = -0.5 * jnp.mean(jnp.sum(1.0 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1))

    # Static targets: s1..s7
    solvable_target = static_targets[:, 0]
    static_reg_target = static_targets[:, 1:]

    l_static = (
        optax.sigmoid_binary_cross_entropy(out["solvable_logit"], solvable_target).mean()
        + huber_mean(out["static_reg"], static_reg_target)
    )

    # Dynamic targets: c1 = p_ema, c2 = p*(1-p)
    c1 = p_ema
    c2 = p_ema * (1.0 - p_ema)
    w_dyn = jnp.minimum(1.0, success_obs_count / float(config["dynamic_confidence_ref"]))
    curr_bce = optax.sigmoid_binary_cross_entropy(out["p_logit"], c1)
    curr_huber = optax.huber_loss(out["learnability"], c2, delta=1.0)
    l_curr = (w_dyn * (curr_bce + curr_huber)).mean()

    y = jnp.concatenate([static_targets, c1[:, None], c2[:, None]], axis=-1)
    y_weights = jnp.asarray(config["metric_y_weights"], dtype=jnp.float32)
    if config["train_stage"] == "stage1":
        # Stage 1 must stay task-agnostic: never let dynamic labels shape latent geometry.
        y_weights = y_weights.at[7:].set(0.0)
    l_metric = pairwise_metric_loss(z, y, y_weights)

    l_valid = optax.sigmoid_binary_cross_entropy(out["valid_logit"], solvable_target).mean()

    lambda_c = 0.0 if config["train_stage"] == "stage1" else config["lambda_curriculum"]

    total = (
        l_recon
        + config["beta"] * l_kl
        + config["lambda_static"] * l_static
        + lambda_c * l_curr
        + config["lambda_metric"] * l_metric
        + config["lambda_valid"] * l_valid
    )

    return total, {
        "total": total,
        "recon": l_recon,
        "kl": l_kl,
        "static": l_static,
        "curr": l_curr,
        "metric": l_metric,
        "valid": l_valid,
        "wall_bce": l_wall_bce,
        "wall_dice": l_wall_dice,
        "goal_ce": l_goal,
        "agent_ce": l_agent,
        "z_std": z.std(),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def compute_val_metrics(params: dict, val_batch: dict, config: dict) -> dict:
    grids = val_batch["grids"]
    static_targets = val_batch["static_targets"]
    p_ema = val_batch["p_ema"]

    b, h, w, _ = grids.shape
    out = MazeTaskAwareVAE(
        latent_dim=config["latent_dim"],
        height=h,
        width=w,
    ).apply({"params": params}, grids, deterministic=True)

    wall_targets = grids[:, :, :, 0]
    goal_targets = grids[:, :, :, 1].reshape(b, -1).argmax(axis=-1)
    agent_targets = grids[:, :, :, 2].reshape(b, -1).argmax(axis=-1)

    wall_pred = jax.nn.sigmoid(out["wall_logits"]) > config["wall_threshold"]
    tp = (wall_pred & (wall_targets > 0.5)).sum(axis=(1, 2)).astype(float)
    fp = (wall_pred & (wall_targets < 0.5)).sum(axis=(1, 2)).astype(float)
    fn = (~wall_pred & (wall_targets > 0.5)).sum(axis=(1, 2)).astype(float)
    wall_iou = (tp / (tp + fp + fn + 1e-6)).mean()

    goal_pred = out["goal_logits"].argmax(axis=-1)
    agent_pred = out["agent_logits"].argmax(axis=-1)
    goal_acc = (goal_pred == goal_targets).mean()
    agent_acc = (agent_pred == agent_targets).mean()

    p_pred = jax.nn.sigmoid(out["p_logit"])
    p_mae = jnp.abs(p_pred - p_ema).mean()

    return {
        "val/wall_iou": wall_iou,
        "val/goal_acc": goal_acc,
        "val/agent_acc": agent_acc,
        "val/p_mae": p_mae,
        "val/solvable_bce": optax.sigmoid_binary_cross_entropy(out["solvable_logit"], static_targets[:, 0]).mean(),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _compute_static_targets_np(grids_np: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    fn = jax.jit(compute_structural_targets_from_grids)
    outs: list[np.ndarray] = []
    n = len(grids_np)
    for i in range(0, n, chunk_size):
        g = jnp.asarray(grids_np[i : i + chunk_size], dtype=jnp.float32)
        outs.append(np.asarray(fn(g), dtype=np.float32))
    return np.concatenate(outs, axis=0)


def _load_dataset(dataset_path: str) -> dict:
    d = np.load(dataset_path)
    grids = d["grids"].astype(np.float32)

    if "static_targets" in d.files:
        static_targets = d["static_targets"].astype(np.float32)
    else:
        static_targets = _compute_static_targets_np(grids)

    p_ema = d["p_ema"].astype(np.float32) if "p_ema" in d.files else np.zeros((len(grids),), dtype=np.float32)
    success_obs_count = (
        d["success_obs_count"].astype(np.float32)
        if "success_obs_count" in d.files
        else np.zeros((len(grids),), dtype=np.float32)
    )

    return {
        "grids": grids,
        "static_targets": static_targets,
        "p_ema": p_ema,
        "success_obs_count": success_obs_count,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(state: train_state.TrainState, step: int | str, checkpoint_dir: str) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"params": state.params, "step": step}, f)
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    raw_step = data.get("step", 0)
    if isinstance(raw_step, (int, np.integer)):
        step = int(raw_step)
    elif isinstance(raw_step, str):
        step = int(raw_step) if raw_step.isdigit() else 0
    else:
        try:
            step = int(raw_step)
        except (TypeError, ValueError):
            step = 0
    return data["params"], step


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_training(config: dict) -> None:
    base = os.path.join(config["working_path"], config["vae_folder"])
    dataset_path = os.path.join(base, config["dataset_path"])
    checkpoint_dir = os.path.join(base, config["checkpoint_dir"])
    plot_path = os.path.join(base, config["checkpoint_dir"], "training_metrics.png")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Loading dataset: {dataset_path}")
    ds = _load_dataset(dataset_path)
    grids_np = ds["grids"]
    static_np = ds["static_targets"]
    p_ema_np = ds["p_ema"]
    obs_count_np = ds["success_obs_count"]

    n = len(grids_np)
    print(f"  Dataset size: {n} levels")

    rng = jax.random.PRNGKey(config["seed"])
    perm = np.random.RandomState(int(config["seed"])).permutation(n)
    n_val = max(1, int(n * config["val_fraction"]))

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    val_sel = val_idx[: min(len(val_idx), config["val_eval_size"])]
    val_batch = {
        "grids": jnp.asarray(grids_np[val_sel]),
        "static_targets": jnp.asarray(static_np[val_sel]),
        "p_ema": jnp.asarray(p_ema_np[val_sel]),
        "success_obs_count": jnp.asarray(obs_count_np[val_sel]),
    }

    model = MazeTaskAwareVAE(
        latent_dim=config["latent_dim"],
        height=config["height"],
        width=config["width"],
    )

    rng, rng_init, rng_z = jax.random.split(rng, 3)
    dummy_grid = jnp.zeros((1, config["height"], config["width"], 3), dtype=jnp.float32)
    params = model.init(rng_init, dummy_grid, z_rng=rng_z, deterministic=False)["params"]

    start_step = 0
    if config.get("resume_path"):
        params, start_step = load_checkpoint(config["resume_path"])
        print(f"  Resuming from step {start_step}")

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(config["learning_rate"], eps=1e-8),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, batch, rng):
        (loss, metrics), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            state.params, batch, rng, config
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, metrics

    def run_val(p):
        m = compute_val_metrics(p, val_batch, config)
        return {k: float(v) for k, v in m.items()}

    history: dict[str, list] = {k: [] for k in [
        "total",
        "recon",
        "kl",
        "static",
        "curr",
        "metric",
        "valid",
        "z_std",
        "val/wall_iou",
        "val/goal_acc",
        "val/agent_acc",
        "val/p_mae",
        "val/solvable_bce",
    ]}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    print("\nStarting training...")
    batch_size = config["batch_size"]
    n_train = len(train_idx)
    bar = tqdm(range(start_step, config["num_steps"]), desc="MazeTaskAwareVAE")

    for step in bar:
        rng, rng_train = jax.random.split(rng)
        idx = np.random.choice(n_train, batch_size, replace=False)
        ti = train_idx[idx]

        batch = {
            "grids": jnp.asarray(grids_np[ti]),
            "static_targets": jnp.asarray(static_np[ti]),
            "p_ema": jnp.asarray(p_ema_np[ti]),
            "success_obs_count": jnp.asarray(obs_count_np[ti]),
        }

        state, loss, metrics = train_step(state, batch, rng_train)

        for k in ["total", "recon", "kl", "static", "curr", "metric", "valid", "z_std"]:
            history[k].append(float(metrics[k]))

        if step % config["plot_freq"] == 0:
            val_m = run_val(state.params)
            for k, v in val_m.items():
                history[k].append(v)

            bar.set_postfix({
                "loss": f"{float(loss):.3f}",
                "val_iou": f"{val_m['val/wall_iou']:.3f}",
                "val_goal": f"{val_m['val/goal_acc']:.3f}",
                "z_std": f"{float(metrics['z_std']):.3f}",
                "curr": f"{float(metrics['curr']):.3f}",
            })

            plots = [
                ("total", "Total"),
                ("kl", "KL"),
                ("curr", "Curriculum"),
                ("val/wall_iou", "Val Wall IoU"),
                ("val/goal_acc", "Val Goal Acc"),
                ("z_std", "Latent std"),
            ]
            for ax, (k, title) in zip(axes, plots):
                ax.clear()
                ax.plot(history[k])
                ax.set_title(title)
            fig.suptitle(f"Step {step} | stage={config['train_stage']}")
            fig.tight_layout()
            fig.savefig(plot_path, dpi=80)

        if step % config["save_freq"] == 0 and step > start_step:
            save_checkpoint(state, step, checkpoint_dir)

    save_checkpoint(state, "final", checkpoint_dir)
    plt.close(fig)
    print(f"\nTraining complete. Checkpoints in: {checkpoint_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "maze_ae_config.yml"))

    override_keys = [
        ("--working_path", str, None),
        ("--dataset_path", str, None),
        ("--checkpoint_dir", str, None),
        ("--resume_path", str, None),
        ("--seed", int, None),
        ("--num_steps", int, None),
        ("--batch_size", int, None),
        ("--learning_rate", float, None),
        ("--latent_dim", int, None),
        ("--beta", float, None),
        ("--lambda_static", float, None),
        ("--lambda_curriculum", float, None),
        ("--lambda_metric", float, None),
        ("--lambda_valid", float, None),
        ("--dynamic_confidence_ref", float, None),
        ("--save_freq", int, None),
        ("--plot_freq", int, None),
        ("--train_stage", str, None),
    ]
    for flag, typ, default in override_keys:
        p.add_argument(flag, type=typ, default=default)

    args = vars(p.parse_args())
    cfg_path = args.pop("config")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for k, v in args.items():
        if v is not None:
            config[k] = v

    # Backward-compatible defaults for reconstruction terms.
    config.setdefault("wall_pos_weight", 7.0)
    config.setdefault("wall_bce_weight", 1.0)
    config.setdefault("wall_dice_weight", 1.5)
    config.setdefault("goal_ce_weight", 2.0)
    config.setdefault("agent_ce_weight", 2.0)
    config.setdefault("overlap_penalty_weight", 0.5)

    config.setdefault("beta", 0.05)
    config.setdefault("lambda_static", 1.0)
    config.setdefault("lambda_curriculum", 0.5)
    config.setdefault("lambda_metric", 0.2)
    config.setdefault("lambda_valid", 0.5)
    config.setdefault("dynamic_confidence_ref", 20.0)
    config.setdefault("metric_y_weights", [1.0] * 9)
    config.setdefault("train_stage", "stage1")
    if len(config["metric_y_weights"]) != 9:
        raise ValueError("metric_y_weights must have length 9 for y=[s1..s7,c1,c2].")

    return config


if __name__ == "__main__":
    cfg = parse_args()
    run_training(cfg)
