"""Masked-maze JEPA proof of concept with discrete latent commands.

This module is intentionally small and offline-only. It answers one narrow
question: can a model discover a reusable discrete command vocabulary from
static mazes alone by predicting masked-out structure?

The setup is:
1. Sample a random rectangular mask on a maze.
2. Encode the visible context and the hidden target region.
3. Infer a discrete command from the pair.
4. Predict the hidden target embedding from the context + command.
5. Optionally decode masked content for a cheap qualitative metric.

This is not wired into the environment loop yet.
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


Array = jax.Array


def l2_normalize(x: Array, eps: float = 1e-6) -> Array:
    sq_norm = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
    inv_norm = jax.lax.rsqrt(jnp.maximum(sq_norm, eps * eps))
    return x * inv_norm


def _gaussian_kernel(x: Array, y: Array, gamma: float) -> Array:
    diff = x[:, None] - y[None, :]
    return jnp.exp(-gamma * diff * diff)


def sigreg_loss(
    embeddings: Array,
    rng: Array,
    num_projections: int = 16,
    kernel_gamma: float = 1.0,
) -> Array:
    """Sketched isotropic-Gaussian regularizer via sliced Gaussian-kernel MMD.

    This approximates SIGReg by projecting the embedding batch onto random unit
    directions and matching each projected 1D empirical distribution to N(0, 1).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected (B,D) embeddings, got {embeddings.shape}")
    batch_size, dim = embeddings.shape
    if batch_size < 2:
        return jnp.array(0.0, dtype=embeddings.dtype)

    proj_rng, ref_rng = jax.random.split(rng)
    directions = jax.random.normal(proj_rng, (num_projections, dim), dtype=embeddings.dtype)
    directions = directions / jnp.maximum(
        jnp.linalg.norm(directions, axis=-1, keepdims=True), 1e-6
    )

    projected = embeddings @ directions.T  # (B, M)
    reference = jax.random.normal(ref_rng, (batch_size, num_projections), dtype=embeddings.dtype)

    def _mmd_1d(x: Array, y: Array) -> Array:
        k_xx = _gaussian_kernel(x, x, kernel_gamma)
        k_yy = _gaussian_kernel(y, y, kernel_gamma)
        k_xy = _gaussian_kernel(x, y, kernel_gamma)
        return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

    return jax.vmap(_mmd_1d, in_axes=(1, 1))(projected, reference).mean()


def sample_rect_masks(
    rng: Array,
    batch_size: int,
    height: int,
    width: int,
    min_size: int = 4,
    max_size: int = 7,
) -> Array:
    """Sample binary rectangular masks with shape (B, H, W, 1)."""
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}")
    if max_size < min_size:
        raise ValueError(f"max_size must be >= min_size, got {max_size} < {min_size}")
    if max_size > min(height, width):
        raise ValueError(
            f"max_size must fit inside grid, got max_size={max_size} for {(height, width)}"
        )

    rng_h, rng_w, rng_y, rng_x = jax.random.split(rng, 4)
    rect_h = jax.random.randint(rng_h, (batch_size,), minval=min_size, maxval=max_size + 1)
    rect_w = jax.random.randint(rng_w, (batch_size,), minval=min_size, maxval=max_size + 1)

    max_y = jnp.maximum(height - rect_h + 1, 1)
    max_x = jnp.maximum(width - rect_w + 1, 1)
    y0 = jax.random.randint(rng_y, (batch_size,), minval=0, maxval=max_y)
    x0 = jax.random.randint(rng_x, (batch_size,), minval=0, maxval=max_x)

    rows = jnp.arange(height)[None, :, None]
    cols = jnp.arange(width)[None, None, :]
    in_rows = (rows >= y0[:, None, None]) & (rows < (y0 + rect_h)[:, None, None])
    in_cols = (cols >= x0[:, None, None]) & (cols < (x0 + rect_w)[:, None, None])
    return (in_rows & in_cols)[..., None].astype(jnp.float32)


def sample_balanced_multi_block_masks(
    rng: Array,
    batch_size: int,
    height: int,
    width: int,
    block_height: int = 4,
    block_width: int = 4,
    num_regions: int = 2,
) -> Array:
    """Sample fixed-area masks from non-overlapping balanced anchor cells.

    The anchor grid is evenly spread across the maze so that mask geometry is
    fixed and position coverage is much more uniform than random rectangle
    sampling. Selecting multiple anchor cells yields disjoint masked regions.
    """
    if block_height < 1 or block_width < 1:
        raise ValueError(f"block size must be >= 1, got {(block_height, block_width)}")
    if block_height > height or block_width > width:
        raise ValueError(
            f"block size {(block_height, block_width)} does not fit inside {(height, width)}"
        )

    n_rows = max(1, height // block_height)
    n_cols = max(1, width // block_width)
    anchor_y = jnp.round(jnp.linspace(0, height - block_height, n_rows)).astype(jnp.int32)
    anchor_x = jnp.round(jnp.linspace(0, width - block_width, n_cols)).astype(jnp.int32)
    grid_y, grid_x = jnp.meshgrid(anchor_y, anchor_x, indexing="ij")
    anchors = jnp.stack([grid_y.reshape(-1), grid_x.reshape(-1)], axis=-1)
    n_anchors = int(anchors.shape[0])

    if num_regions < 1:
        raise ValueError(f"num_regions must be >= 1, got {num_regions}")
    if num_regions > n_anchors:
        raise ValueError(
            f"num_regions={num_regions} exceeds available anchor cells={n_anchors}"
        )

    def _single_mask(key: Array) -> Array:
        idx = jax.random.choice(key, n_anchors, shape=(num_regions,), replace=False)
        chosen = anchors[idx]
        mask = jnp.zeros((height, width), dtype=jnp.float32)
        block = jnp.ones((block_height, block_width), dtype=jnp.float32)

        def _place(i: int, cur: Array) -> Array:
            y0 = chosen[i, 0]
            x0 = chosen[i, 1]
            return jax.lax.dynamic_update_slice(cur, block, (y0, x0))

        return jax.lax.fori_loop(0, num_regions, _place, mask)

    keys = jax.random.split(rng, batch_size)
    return jax.vmap(_single_mask)(keys)[..., None]


def sample_masks(
    rng: Array,
    batch_size: int,
    height: int,
    width: int,
    mask_mode: str = "rect",
    min_size: int = 4,
    max_size: int = 7,
    block_height: int = 4,
    block_width: int = 4,
    num_regions: int = 2,
) -> Array:
    if mask_mode == "rect":
        return sample_rect_masks(
            rng,
            batch_size=batch_size,
            height=height,
            width=width,
            min_size=min_size,
            max_size=max_size,
        )
    if mask_mode == "balanced_blocks":
        return sample_balanced_multi_block_masks(
            rng,
            batch_size=batch_size,
            height=height,
            width=width,
            block_height=block_height,
            block_width=block_width,
            num_regions=num_regions,
        )
    raise ValueError(f"Unknown mask_mode={mask_mode!r}")


def split_context_and_target(grids: Array, masks: Array) -> tuple[Array, Array]:
    """Return masked context and masked target grids."""
    context = grids * (1.0 - masks)
    target = grids * masks
    return context, target


class MazeMaskEncoder(nn.Module):
    embed_dim: int = 64

    @nn.compact
    def __call__(self, x: Array) -> Array:
        h = nn.relu(nn.Conv(32, (3, 3), padding="SAME")(x))
        h = nn.relu(nn.Conv(64, (3, 3), strides=(2, 2), padding="SAME")(h))
        h = nn.relu(nn.Conv(128, (3, 3), strides=(2, 2), padding="SAME")(h))
        h = h.mean(axis=(1, 2))
        h = nn.relu(nn.Dense(128)(h))
        h = nn.Dense(self.embed_dim)(h)
        return l2_normalize(h)


class CommandInferenceNet(nn.Module):
    num_codes: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, context_embed: Array, target_embed: Array) -> Array:
        h = jnp.concatenate([context_embed, target_embed], axis=-1)
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        return nn.Dense(self.num_codes)(h)


class CommandPredictor(nn.Module):
    embed_dim: int = 64
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, context_embed: Array, code_embed: Array) -> Array:
        h = jnp.concatenate([context_embed, code_embed], axis=-1)
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        h = nn.Dense(self.embed_dim)(h)
        return l2_normalize(h)


class MaskedPatchDecoder(nn.Module):
    height: int = 13
    width: int = 13
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, context_embed: Array, code_embed: Array) -> tuple[Array, Array, Array]:
        h = jnp.concatenate([context_embed, code_embed], axis=-1)
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        wall_logits = nn.Dense(self.height * self.width)(h).reshape((-1, self.height, self.width))
        goal_logits = nn.Dense(self.height * self.width)(h)
        agent_logits = nn.Dense(self.height * self.width)(h)
        return wall_logits, goal_logits, agent_logits


def _straight_through_gumbel(logits: Array, temperature: float, rng: Array) -> Array:
    uniform = jax.random.uniform(rng, logits.shape, minval=1e-6, maxval=1.0 - 1e-6)
    gumbel = -jnp.log(-jnp.log(uniform))
    y_soft = jax.nn.softmax((logits + gumbel) / temperature, axis=-1)
    y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), logits.shape[-1], dtype=logits.dtype)
    return y_soft + jax.lax.stop_gradient(y_hard - y_soft)


class MaskedMazeJepaPoc(nn.Module):
    height: int = 13
    width: int = 13
    embed_dim: int = 64
    num_codes: int = 16
    code_dim: int = 32
    gumbel_temperature: float = 0.7

    @nn.compact
    def __call__(self, grids: Array, masks: Array, deterministic: bool = False) -> dict[str, Array]:
        if grids.ndim != 4 or grids.shape[-1] != 3:
            raise ValueError(f"Expected (B,H,W,3) grids, got {grids.shape}")
        if masks.shape != grids.shape[:3] + (1,):
            raise ValueError(f"Expected masks of shape {grids.shape[:3] + (1,)}, got {masks.shape}")

        context, target = split_context_and_target(grids, masks)
        encoder = MazeMaskEncoder(embed_dim=self.embed_dim, name="encoder")
        context_embed = encoder(jnp.concatenate([context, masks], axis=-1))
        target_embed = encoder(jnp.concatenate([target, masks], axis=-1))

        code_logits = CommandInferenceNet(
            num_codes=self.num_codes,
            hidden_dim=max(128, self.embed_dim * 2),
            name="inference",
        )(context_embed, jax.lax.stop_gradient(target_embed))
        code_probs = jax.nn.softmax(code_logits, axis=-1)

        if deterministic:
            code_assign = jax.nn.one_hot(
                jnp.argmax(code_logits, axis=-1), self.num_codes, dtype=grids.dtype
            )
        else:
            code_assign = _straight_through_gumbel(
                code_logits, self.gumbel_temperature, self.make_rng("gumbel")
            )

        codebook = self.param(
            "codebook",
            nn.initializers.normal(stddev=0.02),
            (self.num_codes, self.code_dim),
        )
        code_embed = code_assign @ codebook

        pred_embed = CommandPredictor(
            embed_dim=self.embed_dim,
            hidden_dim=max(128, self.embed_dim * 2),
            name="predictor",
        )(context_embed, code_embed)
        wall_logits, goal_logits, agent_logits = MaskedPatchDecoder(
            height=self.height,
            width=self.width,
            hidden_dim=max(256, self.embed_dim * 4),
            name="decoder",
        )(context_embed, code_embed)

        return {
            "context_embed": context_embed,
            "target_embed": target_embed,
            "code_logits": code_logits,
            "code_probs": code_probs,
            "code_assign": code_assign,
            "code_indices": jnp.argmax(code_assign, axis=-1),
            "code_embed": code_embed,
            "pred_embed": pred_embed,
            "wall_logits": wall_logits,
            "goal_logits": goal_logits,
            "agent_logits": agent_logits,
        }


def compute_masked_jepa_loss(
    params: dict[str, Any],
    model: MaskedMazeJepaPoc,
    grids: Array,
    masks: Array,
    rng: Array,
    cfg: dict[str, float],
    deterministic: bool = False,
) -> tuple[Array, dict[str, Array]]:
    """Compute proof-of-concept loss and metrics."""
    out = model.apply({"params": params}, grids, masks, deterministic=deterministic, rngs={"gumbel": rng})

    target_embed = jax.lax.stop_gradient(out["target_embed"])
    pred_embed = out["pred_embed"]
    cosine = jnp.sum(pred_embed * target_embed, axis=-1)
    jepa_loss = (1.0 - cosine).mean()

    mask_2d = masks[..., 0]
    mask_sum = jnp.maximum(mask_2d.sum(), 1.0)

    wall_targets = grids[..., 0]
    wall_bce = optax.sigmoid_binary_cross_entropy(out["wall_logits"], wall_targets)
    wall_loss = (wall_bce * mask_2d).sum() / mask_sum

    flat_mask = mask_2d.reshape((grids.shape[0], -1))
    goal_targets = grids[..., 1].reshape((grids.shape[0], -1)).argmax(axis=-1)
    agent_targets = grids[..., 2].reshape((grids.shape[0], -1)).argmax(axis=-1)
    goal_present = flat_mask[jnp.arange(grids.shape[0]), goal_targets]
    agent_present = flat_mask[jnp.arange(grids.shape[0]), agent_targets]

    goal_ce = optax.softmax_cross_entropy_with_integer_labels(out["goal_logits"], goal_targets)
    goal_loss = (goal_ce * goal_present).sum() / jnp.maximum(goal_present.sum(), 1.0)

    agent_ce = optax.softmax_cross_entropy_with_integer_labels(out["agent_logits"], agent_targets)
    agent_loss = (agent_ce * agent_present).sum() / jnp.maximum(agent_present.sum(), 1.0)

    code_probs = out["code_probs"]
    eps = 1e-6
    usage = code_probs.mean(axis=0)
    sig_rng_pred, sig_rng_target = jax.random.split(rng)
    sigreg = 0.5 * (
        sigreg_loss(
            pred_embed,
            sig_rng_pred,
            num_projections=int(cfg.get("sigreg_num_projections", 16)),
            kernel_gamma=float(cfg.get("sigreg_kernel_gamma", 1.0)),
        )
        + sigreg_loss(
            target_embed,
            sig_rng_target,
            num_projections=int(cfg.get("sigreg_num_projections", 16)),
            kernel_gamma=float(cfg.get("sigreg_kernel_gamma", 1.0)),
        )
    )

    total = (
        float(cfg.get("jepa_weight", 1.0)) * jepa_loss
        + float(cfg.get("wall_weight", 1.0)) * wall_loss
        + float(cfg.get("goal_weight", 0.2)) * goal_loss
        + float(cfg.get("agent_weight", 0.2)) * agent_loss
        + float(cfg.get("sigreg_weight", 1.0)) * sigreg
    )

    wall_pred = (jax.nn.sigmoid(out["wall_logits"]) > 0.5).astype(jnp.float32)
    wall_acc = (((wall_pred == wall_targets).astype(jnp.float32)) * mask_2d).sum() / mask_sum
    goal_acc = ((jnp.argmax(out["goal_logits"], axis=-1) == goal_targets).astype(jnp.float32) * goal_present).sum() / jnp.maximum(goal_present.sum(), 1.0)
    agent_acc = ((jnp.argmax(out["agent_logits"], axis=-1) == agent_targets).astype(jnp.float32) * agent_present).sum() / jnp.maximum(agent_present.sum(), 1.0)

    metrics = {
        "total": total,
        "jepa": jepa_loss,
        "wall": wall_loss,
        "goal": goal_loss,
        "agent": agent_loss,
        "sigreg": sigreg,
        "wall_acc_masked": wall_acc,
        "goal_acc_masked": goal_acc,
        "agent_acc_masked": agent_acc,
        "code_perplexity": jnp.exp(-jnp.sum(usage * jnp.log(usage + eps))),
        "num_active_codes": (usage > (1.0 / model.num_codes) * 0.5).sum().astype(jnp.float32),
        "mean_mask_fraction": mask_2d.mean(),
    }
    return total, metrics


def summarize_code_assignments(
    outputs: dict[str, Array],
    grids: Array,
    masks: Array,
    num_codes: int,
) -> list[dict[str, float]]:
    """Summarize how each discovered code is being used on a batch."""
    code_idx = outputs["code_indices"]
    mask_2d = masks[..., 0]
    masked_walls = (grids[..., 0] * mask_2d).sum(axis=(1, 2))
    mask_area = jnp.maximum(mask_2d.sum(axis=(1, 2)), 1.0)
    wall_density = masked_walls / mask_area
    goal_targets = grids[..., 1].reshape((grids.shape[0], -1)).argmax(axis=-1)
    agent_targets = grids[..., 2].reshape((grids.shape[0], -1)).argmax(axis=-1)
    flat_mask = mask_2d.reshape((grids.shape[0], -1))
    goal_in_mask = flat_mask[jnp.arange(grids.shape[0]), goal_targets]
    agent_in_mask = flat_mask[jnp.arange(grids.shape[0]), agent_targets]

    y_coords = jnp.arange(mask_2d.shape[1], dtype=jnp.float32)[None, :, None]
    x_coords = jnp.arange(mask_2d.shape[2], dtype=jnp.float32)[None, None, :]
    center_y = (mask_2d * y_coords).sum(axis=(1, 2)) / mask_area
    center_x = (mask_2d * x_coords).sum(axis=(1, 2)) / mask_area

    rows: list[dict[str, float]] = []
    for code in range(num_codes):
        sel = code_idx == code
        count = float(sel.sum())
        if count == 0:
            rows.append(
                {
                    "code": float(code),
                    "count": 0.0,
                    "mean_masked_wall_density": 0.0,
                    "goal_in_mask_rate": 0.0,
                    "agent_in_mask_rate": 0.0,
                    "mean_mask_center_y": 0.0,
                    "mean_mask_center_x": 0.0,
                }
            )
            continue
        rows.append(
            {
                "code": float(code),
                "count": count,
                "mean_masked_wall_density": float(wall_density[sel].mean()),
                "goal_in_mask_rate": float(goal_in_mask[sel].mean()),
                "agent_in_mask_rate": float(agent_in_mask[sel].mean()),
                "mean_mask_center_y": float(center_y[sel].mean()),
                "mean_mask_center_x": float(center_x[sel].mean()),
            }
        )
    return rows
