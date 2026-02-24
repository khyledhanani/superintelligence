#!/usr/bin/env python3
"""
Train a grid-based CNN Autoencoder with SIGReg on PLR maze levels.

Run from the project root:
    python vae/train_maze_ae.py [--config vae/maze_ae_config.yml]

Or with overrides:
    python vae/train_maze_ae.py --num_steps 50000 --sigreg_weight 0.1
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml
from flax.training import train_state
from tqdm.auto import tqdm

# Make sure project root is on path so we can import es.*
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from es.maze_ae import MazeAE, MazeDecoder, sigreg_loss


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def weighted_bce(logits: jnp.ndarray, targets: jnp.ndarray, pos_weight: float) -> jnp.ndarray:
    """Binary cross-entropy with per-positive class weight.

    Args:
        logits:     (B, H, W) raw logits.
        targets:    (B, H, W) float32 in {0, 1}.
        pos_weight: Weight applied to positive (wall) terms.

    Returns:
        Scalar mean loss.
    """
    loss = (
        pos_weight * targets * jax.nn.softplus(-logits)
        + (1.0 - targets) * jax.nn.softplus(logits)
    )
    return loss.mean()


def dice_loss(probs: jnp.ndarray, targets: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Soft Dice loss averaged over the batch."""
    intersection = (probs * targets).sum(axis=(1, 2))
    denom = probs.sum(axis=(1, 2)) + targets.sum(axis=(1, 2))
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


def compute_loss(
    params: dict,
    grids: jnp.ndarray,
    rng: jax.Array,
    config: dict,
    sigreg_scale: float,
) -> tuple[jnp.ndarray, dict]:
    """Full loss function.

    Args:
        params:        MazeAE parameters.
        grids:         (B, H, W, 3) float32 input grids.
        rng:           PRNG key for SIGReg direction sampling.
        config:        Config dict.
        sigreg_scale:  Multiplier on sigreg_weight (0 during warmup, 1 after).

    Returns:
        (total_loss, metrics_dict)
    """
    B, H, W, _ = grids.shape

    # Target channels
    wall_targets  = grids[:, :, :, 0]                                   # (B, H, W)
    goal_targets  = grids[:, :, :, 1].reshape(B, -1).argmax(axis=-1)   # (B,) int
    agent_targets = grids[:, :, :, 2].reshape(B, -1).argmax(axis=-1)   # (B,) int

    z, wall_logits, goal_logits, agent_logits = MazeAE(
        latent_dim=config['latent_dim'],
        height=config['height'],
        width=config['width'],
    ).apply({'params': params}, grids)

    # ---- Wall losses ----
    wall_probs = jax.nn.sigmoid(wall_logits)
    l_wall_bce  = weighted_bce(wall_logits, wall_targets, config['wall_pos_weight'])
    l_wall_dice = dice_loss(wall_probs, wall_targets)
    l_wall = config['wall_bce_weight'] * l_wall_bce + config['wall_dice_weight'] * l_wall_dice

    # ---- Position losses ----
    l_goal  = optax.softmax_cross_entropy_with_integer_labels(goal_logits,  goal_targets ).mean()
    l_agent = optax.softmax_cross_entropy_with_integer_labels(agent_logits, agent_targets).mean()
    l_pos = config['goal_ce_weight'] * l_goal + config['agent_ce_weight'] * l_agent

    # ---- Overlap penalty: penalise P(goal) * P(agent) at same cell ----
    goal_p  = jax.nn.softmax(goal_logits)
    agent_p = jax.nn.softmax(agent_logits)
    l_overlap = config['overlap_penalty_weight'] * (goal_p * agent_p).sum(axis=-1).mean()

    # ---- SIGReg ----
    l_sigreg = sigreg_loss(
        z, rng,
        n_directions=config['sigreg_directions'],
        n_t=config['sigreg_t_points'],
        t_max=config['sigreg_t_max'],
    )
    l_sigreg_weighted = sigreg_scale * config['sigreg_weight'] * l_sigreg

    total = l_wall + l_pos + l_overlap + l_sigreg_weighted

    metrics = {
        'total':      total,
        'wall_bce':   l_wall_bce,
        'wall_dice':  l_wall_dice,
        'goal_ce':    l_goal,
        'agent_ce':   l_agent,
        'overlap':    l_overlap,
        'sigreg':     l_sigreg,
        'z_mean':     z.mean(),
        'z_std':      z.std(),
    }
    return total, metrics


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def compute_val_metrics(
    params: dict,
    val_grids: jnp.ndarray,
    config: dict,
) -> dict:
    """Run decoder on validation grids and compute reconstruction metrics."""
    B, H, W, _ = val_grids.shape
    _, wall_logits, goal_logits, agent_logits = MazeAE(
        latent_dim=config['latent_dim'],
        height=H,
        width=W,
    ).apply({'params': params}, val_grids)

    wall_targets  = val_grids[:, :, :, 0]
    goal_targets  = val_grids[:, :, :, 1].reshape(B, -1).argmax(axis=-1)
    agent_targets = val_grids[:, :, :, 2].reshape(B, -1).argmax(axis=-1)

    # Wall IoU
    wall_pred = jax.nn.sigmoid(wall_logits) > config['wall_threshold']
    tp = (wall_pred & (wall_targets > 0.5)).sum(axis=(1, 2)).astype(float)
    fp = (wall_pred & (wall_targets < 0.5)).sum(axis=(1, 2)).astype(float)
    fn = (~wall_pred & (wall_targets > 0.5)).sum(axis=(1, 2)).astype(float)
    wall_iou = (tp / (tp + fp + fn + 1e-6)).mean()

    # Goal / agent accuracy (argmax)
    goal_pred  = goal_logits.argmax(axis=-1)
    agent_pred = agent_logits.argmax(axis=-1)
    goal_acc  = (goal_pred  == goal_targets ).mean()
    agent_acc = (agent_pred == agent_targets).mean()

    return {
        'val/wall_iou':   float(wall_iou),
        'val/goal_acc':   float(goal_acc),
        'val/agent_acc':  float(agent_acc),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: train_state.TrainState, step: int | str, checkpoint_dir: str) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'checkpoint_{step}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'params': state.params, 'step': step}, f)
    print(f'  Saved checkpoint: {path}')


def load_checkpoint(path: str) -> tuple[dict, int]:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['params'], data.get('step', 0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(config: dict) -> None:
    # ---- Paths ----
    base = os.path.join(config['working_path'], config['vae_folder'])
    dataset_path  = os.path.join(base, config['dataset_path'])
    checkpoint_dir = os.path.join(base, config['checkpoint_dir'])
    plot_path     = os.path.join(base, config['checkpoint_dir'], 'training_metrics.png')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- Data ----
    print(f'Loading dataset: {dataset_path}')
    data = np.load(dataset_path)
    grids_np = data['grids'].astype(np.float32)   # (N, H, W, 3)
    N = len(grids_np)
    print(f'  Dataset size: {N} levels, grid shape: {grids_np.shape[1:]}')

    rng = jax.random.PRNGKey(config['seed'])

    # Train / val split
    rng, rng_split = jax.random.split(rng)
    perm = np.random.RandomState(int(config['seed'])).permutation(N)
    n_val = max(1, int(N * config['val_fraction']))
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]
    val_grids = jnp.array(grids_np[val_idx[:config['val_eval_size']]])
    print(f'  Train: {len(train_idx)}, Val: {n_val}')

    # ---- Model init ----
    model = MazeAE(
        latent_dim=config['latent_dim'],
        height=config['height'],
        width=config['width'],
    )
    rng, rng_init, rng_z = jax.random.split(rng, 3)
    dummy_grid = jnp.zeros((1, config['height'], config['width'], 3))
    params = model.init(rng_init, dummy_grid)['params']

    start_step = 0
    if config.get('resume_path'):
        params, start_step = load_checkpoint(config['resume_path'])
        print(f'  Resuming from step {start_step}')

    # ---- Optimiser: Adam with cosine decay ----
    lr_schedule = optax.cosine_decay_schedule(
        init_value=config['learning_rate'],
        decay_steps=config['num_steps'] - config.get('lr_decay_steps', int(config['num_steps'] * 0.8)),
        alpha=0.1,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(config['learning_rate'], eps=1e-8),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    # ---- JIT-compile train step ----
    @jax.jit
    def train_step(state, batch, rng, sigreg_scale):
        (loss, metrics), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            state.params, batch, rng, config, sigreg_scale
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, metrics

    val_metrics_jit = jax.jit(lambda p: compute_val_metrics(p, val_grids, config))

    # ---- History for plots ----
    history: dict[str, list] = {k: [] for k in [
        'total', 'wall_bce', 'wall_dice', 'goal_ce', 'agent_ce', 'sigreg',
        'z_std', 'val/wall_iou', 'val/goal_acc', 'val/agent_acc',
    ]}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    print('\nStarting training...')
    batch_size  = config['batch_size']
    n_train     = len(train_idx)
    warmup_steps = config.get('sigreg_warmup_steps', 2000)
    bar = tqdm(range(start_step, config['num_steps']), desc='MazeAE')

    for step in bar:
        rng, rng_batch, rng_sigreg = jax.random.split(rng, 3)

        # Sample batch
        idx = np.random.choice(n_train, batch_size, replace=False)
        batch = jnp.array(grids_np[train_idx[idx]])

        # SIGReg warmup: linearly ramp in over warmup_steps
        sigreg_scale = float(min(1.0, (step - start_step) / max(warmup_steps, 1)))

        state, loss, metrics = train_step(state, batch, rng_sigreg, sigreg_scale)

        # Record
        for k in ['total', 'wall_bce', 'wall_dice', 'goal_ce', 'agent_ce', 'sigreg', 'z_std']:
            history[k].append(float(metrics[k]))

        if step % config['plot_freq'] == 0:
            val_m = val_metrics_jit(state.params)
            for k, v in val_m.items():
                history[k].append(v)

            bar.set_postfix({
                'loss': f"{float(loss):.3f}",
                'wall_iou': f"{val_m['val/wall_iou']:.3f}",
                'goal_acc': f"{val_m['val/goal_acc']:.3f}",
                'z_std':    f"{float(metrics['z_std']):.3f}",
                'sigreg':   f"{float(metrics['sigreg']):.3f}",
            })

            # Plot
            plot_keys = [
                ('total',          'Total loss'),
                ('wall_bce',       'Wall BCE'),
                ('sigreg',         'SIGReg'),
                ('val/wall_iou',   'Val Wall IoU'),
                ('val/goal_acc',   'Val Goal Acc'),
                ('z_std',          'Latent std (target: 1.0)'),
            ]
            for ax, (k, title) in zip(axes, plot_keys):
                ax.clear()
                ax.plot(history[k])
                ax.set_title(title)
                ax.set_xlabel('step / plot_freq')
            fig.suptitle(f'Step {step} | sigreg_scale={sigreg_scale:.2f}')
            fig.tight_layout()
            fig.savefig(plot_path, dpi=80)

        if step % config['save_freq'] == 0 and step > start_step:
            save_checkpoint(state, step, checkpoint_dir)

    save_checkpoint(state, 'final', checkpoint_dir)
    plt.close(fig)
    print(f'\nTraining complete. Checkpoints in: {checkpoint_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--config', type=str,
                   default=os.path.join(os.path.dirname(__file__), 'maze_ae_config.yml'))

    # Optional overrides for every config key
    override_keys = [
        ('--working_path',         str,   None),
        ('--dataset_path',         str,   None),
        ('--checkpoint_dir',       str,   None),
        ('--resume_path',          str,   None),
        ('--seed',                 int,   None),
        ('--num_steps',            int,   None),
        ('--batch_size',           int,   None),
        ('--learning_rate',        float, None),
        ('--latent_dim',           int,   None),
        ('--wall_pos_weight',      float, None),
        ('--wall_bce_weight',      float, None),
        ('--wall_dice_weight',     float, None),
        ('--goal_ce_weight',       float, None),
        ('--agent_ce_weight',      float, None),
        ('--overlap_penalty_weight', float, None),
        ('--sigreg_weight',        float, None),
        ('--sigreg_directions',    int,   None),
        ('--sigreg_t_points',      int,   None),
        ('--sigreg_t_max',         float, None),
        ('--sigreg_warmup_steps',  int,   None),
        ('--save_freq',            int,   None),
        ('--plot_freq',            int,   None),
        ('--val_eval_size',        int,   None),
    ]
    for flag, typ, default in override_keys:
        p.add_argument(flag, type=typ, default=default)

    args = vars(p.parse_args())
    config_path = args.pop('config')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    for k, v in args.items():
        if v is not None:
            config[k] = v

    return config


if __name__ == '__main__':
    cfg = parse_args()
    print('Config:')
    for k, v in cfg.items():
        print(f'  {k}: {v}')
    run_training(cfg)
