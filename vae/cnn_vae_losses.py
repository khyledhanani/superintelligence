"""Loss functions for CNN+LSTM VAE maze reconstruction.

Phase 1 note: This file defines loss functions for testing and reference only.
Phase 2 will import from this file in the training loop.

Wall masking ensures goal and agent heads cannot assign probability to wall
positions, preventing impossible maze configurations.

Exports:
    apply_wall_mask       -- Mask goal/agent logits at wall positions
    wall_bce_dice_loss    -- Binary cross-entropy + soft Dice for wall prediction
    goal_ce_loss          -- Cross-entropy for goal position (wall-masked)
    agent_ce_loss         -- Cross-entropy for agent position (wall-masked)
"""

import jax
import jax.numpy as jnp
from typing import Tuple


def apply_wall_mask(wall_mask_2d: jnp.ndarray, logits_2d: jnp.ndarray) -> jnp.ndarray:
    """Mask goal or agent logits at wall positions.

    Sets logits to -1e9 at wall positions, ensuring that after softmax
    the probability at wall positions is ~0.

    Args:
        wall_mask_2d: (B, 13, 13) binary wall mask (1=wall, 0=free).
                      Use ground truth targets during training to prevent
                      self-referential masking that causes loss spikes.
        logits_2d:    (B, 13, 13) goal or agent logits

    Returns:
        (B, 169) masked logits with -1e9 at wall positions
    """
    B = wall_mask_2d.shape[0]
    wall_mask = (wall_mask_2d > 0.5).reshape(B, -1)   # (B, 169) bool, True=wall
    logits_flat = logits_2d.reshape(B, -1)             # (B, 169)
    return jnp.where(wall_mask, jnp.full_like(logits_flat, -1e9), logits_flat)


def wall_bce_dice_loss(wall_logits: jnp.ndarray, wall_targets: jnp.ndarray,
                       pos_weight: float = 1.0) -> jnp.ndarray:
    """Wall prediction loss: numerically stable BCE + soft Dice.

    BCE is computed from logits (not probabilities) for numerical stability.
    Soft Dice penalises the overlap between predicted and target wall masks.

    Args:
        wall_logits:  (B, 13, 13) raw logits from decoder wall_head
        wall_targets: (B, 13, 13) binary float32 ground truth (1=wall, 0=free)
        pos_weight:   Weight multiplier for positive (wall) class in BCE.
                      Set to ~3.17 (= 0.76/0.24) to balance 24% wall / 76% free.

    Returns:
        Scalar loss = weighted_BCE + soft_dice (mean over batch)
    """
    B = wall_logits.shape[0]

    # Numerically stable sigmoid BCE from logits:
    # BCE = max(l, 0) - l*y + log(1 + exp(-|l|))
    bce_per_element = (
        jnp.maximum(wall_logits, 0)
        - wall_logits * wall_targets
        + jnp.log1p(jnp.exp(-jnp.abs(wall_logits)))
    )

    # Positive class weighting: wall cells (y=1) get pos_weight, free cells get 1.0
    weight = jnp.where(wall_targets > 0.5, pos_weight, 1.0)
    bce = jnp.mean(weight * bce_per_element)

    # Soft Dice loss (smooth=1 for numerical stability)
    smooth = 1.0
    pred = jax.nn.sigmoid(wall_logits).reshape(B, -1)   # (B, 169)
    tgt = wall_targets.reshape(B, -1)                    # (B, 169)
    intersection = jnp.sum(pred * tgt, axis=-1)          # (B,)
    dice = 1 - (2 * intersection + smooth) / (
        jnp.sum(pred, axis=-1) + jnp.sum(tgt, axis=-1) + smooth
    )
    soft_dice = jnp.mean(dice)

    return bce + soft_dice


def goal_ce_loss(
    wall_mask: jnp.ndarray,
    goal_logits: jnp.ndarray,
    goal_targets: jnp.ndarray,
) -> jnp.ndarray:
    """Cross-entropy loss for goal position prediction with wall masking.

    Wall positions are masked to -1e9 before softmax so the model cannot
    place the goal on a wall cell.

    Args:
        wall_mask:    (B, 13, 13) binary wall mask (1=wall, 0=free).
                      Use ground truth targets during training.
        goal_logits:  (B, 13, 13) raw goal logits from decoder goal_head
        goal_targets: (B,) int32 cell indices (0-indexed, 0-168)

    Returns:
        Scalar cross-entropy loss (mean over batch)
    """
    B = wall_mask.shape[0]
    masked = apply_wall_mask(wall_mask, goal_logits)               # (B, 169)
    log_probs = jax.nn.log_softmax(masked, axis=-1)                # (B, 169)
    return -jnp.mean(log_probs[jnp.arange(B), goal_targets])


def agent_ce_loss(
    wall_mask: jnp.ndarray,
    agent_logits: jnp.ndarray,
    agent_targets: jnp.ndarray,
) -> jnp.ndarray:
    """Cross-entropy loss for agent position prediction with wall masking.

    Wall positions are masked to -1e9 before softmax so the model cannot
    place the agent on a wall cell.

    Args:
        wall_mask:     (B, 13, 13) binary wall mask (1=wall, 0=free).
                       Use ground truth targets during training.
        agent_logits:  (B, 13, 13) raw agent logits from decoder agent_head
        agent_targets: (B,) int32 cell indices (0-indexed, 0-168)

    Returns:
        Scalar cross-entropy loss (mean over batch)
    """
    B = wall_mask.shape[0]
    masked = apply_wall_mask(wall_mask, agent_logits)              # (B, 169)
    log_probs = jax.nn.log_softmax(masked, axis=-1)                # (B, 169)
    return -jnp.mean(log_probs[jnp.arange(B), agent_targets])
