"""Local patch JEPA proof of concept for unsupervised maze rewrite primitives.

This experiment discards whole-maze masking and instead learns a discrete
vocabulary of local completion modes:

1. Sample a local wall patch from a maze.
2. Hide a fixed center block and reveal only the surrounding boundary context.
3. Infer a discrete code that explains the hidden center.
4. Predict the target embedding and decode the hidden center block.

The patch lives in a local coordinate frame with optional dihedral
augmentation, so the codebook has a better chance of capturing reusable local
topological edits rather than absolute maze position.
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from vae.masked_jepa_poc import l2_normalize, sigreg_loss


Array = jax.Array


def center_block_mask(patch_size: int, center_size: int) -> np.ndarray:
    if patch_size % 2 == 0:
        raise ValueError(f"patch_size must be odd, got {patch_size}")
    if center_size % 2 == 0:
        raise ValueError(f"center_size must be odd, got {center_size}")
    if center_size >= patch_size:
        raise ValueError(f"center_size must be < patch_size, got {center_size} >= {patch_size}")
    mask = np.zeros((patch_size, patch_size, 1), dtype=np.float32)
    start = (patch_size - center_size) // 2
    end = start + center_size
    mask[start:end, start:end, 0] = 1.0
    return mask


def extract_wall_patches_np(
    grids: np.ndarray,
    center_y: np.ndarray,
    center_x: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """Extract wall-only local patches centered on (center_y, center_x)."""
    if grids.ndim != 4 or grids.shape[-1] < 1:
        raise ValueError(f"Expected (N,H,W,C) grids, got {grids.shape}")
    radius = patch_size // 2
    walls = np.asarray(grids[..., 0], dtype=np.float32)
    patches = np.empty((len(center_y), patch_size, patch_size, 1), dtype=np.float32)
    for i, (y, x) in enumerate(zip(center_y, center_x, strict=False)):
        patches[i, :, :, 0] = walls[i, y - radius : y + radius + 1, x - radius : x + radius + 1]
    return patches


def apply_random_dihedral_np(rng: np.random.Generator, patches: np.ndarray) -> np.ndarray:
    out = np.empty_like(patches)
    for i in range(len(patches)):
        patch = patches[i]
        rot_k = int(rng.integers(0, 4))
        flip = int(rng.integers(0, 2))
        view = np.rot90(patch, k=rot_k, axes=(0, 1))
        if flip:
            view = np.flip(view, axis=1)
        out[i] = view
    return out


def sample_wall_patch_batch_np(
    grids: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
    patch_size: int = 9,
    augment_dihedral: bool = True,
) -> np.ndarray:
    """Sample wall-only local patches from maze grids."""
    n, h, w, _ = grids.shape
    radius = patch_size // 2
    if h <= 2 * radius or w <= 2 * radius:
        raise ValueError(f"patch_size={patch_size} too large for grid shape {(h, w)}")

    replace = n < batch_size
    maze_idx = rng.choice(n, size=batch_size, replace=replace)
    center_y = rng.integers(radius, h - radius, size=batch_size)
    center_x = rng.integers(radius, w - radius, size=batch_size)
    patches = extract_wall_patches_np(grids[maze_idx], center_y, center_x, patch_size=patch_size)
    if augment_dihedral:
        patches = apply_random_dihedral_np(rng, patches)
    return patches


def split_context_and_target(patches: Array, center_mask: Array) -> tuple[Array, Array]:
    context = patches * (1.0 - center_mask)
    target = patches * center_mask
    return context, target


class LocalPatchEncoder(nn.Module):
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


class PrimitiveInferenceNet(nn.Module):
    num_codes: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, context_embed: Array, target_embed: Array) -> Array:
        h = jnp.concatenate([context_embed, target_embed], axis=-1)
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        return nn.Dense(self.num_codes)(h)


class PrimitivePredictor(nn.Module):
    embed_dim: int = 64
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, context_embed: Array, code_embed: Array) -> Array:
        h = jnp.concatenate([context_embed, code_embed], axis=-1)
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        h = nn.Dense(self.embed_dim)(h)
        return l2_normalize(h)


class CenterBlockDecoder(nn.Module):
    center_size: int = 3
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, context_embed: Array, code_embed: Array) -> Array:
        h = jnp.concatenate([context_embed, code_embed], axis=-1)
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        return nn.Dense(self.center_size * self.center_size)(h).reshape(
            (-1, self.center_size, self.center_size)
        )


def _straight_through_gumbel(logits: Array, temperature: float, rng: Array) -> Array:
    uniform = jax.random.uniform(rng, logits.shape, minval=1e-6, maxval=1.0 - 1e-6)
    gumbel = -jnp.log(-jnp.log(uniform))
    y_soft = jax.nn.softmax((logits + gumbel) / temperature, axis=-1)
    y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), logits.shape[-1], dtype=logits.dtype)
    return y_soft + jax.lax.stop_gradient(y_hard - y_soft)


class LocalPatchJepaPoc(nn.Module):
    patch_size: int = 9
    center_size: int = 3
    embed_dim: int = 64
    num_codes: int = 16
    code_dim: int = 32
    gumbel_temperature: float = 0.7

    def setup(self) -> None:
        hidden_dim = max(128, self.embed_dim * 2)
        self.codebook = self.param(
            "codebook",
            nn.initializers.normal(stddev=0.02),
            (self.num_codes, self.code_dim),
        )
        self.encoder_net = LocalPatchEncoder(embed_dim=self.embed_dim, name="encoder")
        self.inference_net = PrimitiveInferenceNet(
            num_codes=self.num_codes,
            hidden_dim=hidden_dim,
            name="inference",
        )
        self.predictor_net = PrimitivePredictor(
            embed_dim=self.embed_dim,
            hidden_dim=hidden_dim,
            name="predictor",
        )
        self.decoder_net = CenterBlockDecoder(
            center_size=self.center_size,
            hidden_dim=hidden_dim,
            name="decoder",
        )

    def _center_mask(self, dtype: jnp.dtype) -> Array:
        return jnp.asarray(center_block_mask(self.patch_size, self.center_size), dtype=dtype)

    def _code_embed_from_assign(self, code_assign: Array) -> Array:
        return code_assign @ self.codebook

    def _encode_context_target(self, patches: Array) -> tuple[Array, Array, Array]:
        center_mask = self._center_mask(patches.dtype)
        context, target = split_context_and_target(patches, center_mask)
        context_embed = self.encoder_net(context)
        target_embed = self.encoder_net(target)
        return center_mask, context_embed, target_embed

    def decode_with_code(self, patches: Array, code_indices: Array) -> dict[str, Array]:
        if patches.ndim != 4 or patches.shape[-1] != 1:
            raise ValueError(f"Expected (B,P,P,1) patches, got {patches.shape}")
        if patches.shape[1] != self.patch_size or patches.shape[2] != self.patch_size:
            raise ValueError(
                f"Expected patch spatial size {(self.patch_size, self.patch_size)}, got {patches.shape[1:3]}"
            )
        center_mask, context_embed, _ = self._encode_context_target(patches)
        context = patches * (1.0 - center_mask)
        code_assign = jax.nn.one_hot(code_indices, self.num_codes, dtype=patches.dtype)
        code_embed = self._code_embed_from_assign(code_assign)
        pred_embed = self.predictor_net(context_embed, code_embed)
        center_logits = self.decoder_net(context_embed, code_embed)
        center_probs = jax.nn.sigmoid(center_logits)
        center_confidence = jnp.mean(jnp.abs(center_probs - 0.5) * 2.0, axis=(1, 2))
        return {
            "context": context,
            "center_mask": center_mask,
            "context_embed": context_embed,
            "code_assign": code_assign,
            "code_embed": code_embed,
            "pred_embed": pred_embed,
            "center_logits": center_logits,
            "center_probs": center_probs,
            "center_confidence": center_confidence,
            "code_indices": code_indices.astype(jnp.int32),
        }

    def decode_all_codes(self, patches: Array) -> dict[str, Array]:
        if patches.ndim != 4 or patches.shape[-1] != 1:
            raise ValueError(f"Expected (B,P,P,1) patches, got {patches.shape}")
        if patches.shape[1] != self.patch_size or patches.shape[2] != self.patch_size:
            raise ValueError(
                f"Expected patch spatial size {(self.patch_size, self.patch_size)}, got {patches.shape[1:3]}"
            )

        center_mask, context_embed, _ = self._encode_context_target(patches)
        context = patches * (1.0 - center_mask)
        code_embeds = self.codebook

        def decode_for_code(code_embed: Array) -> Array:
            code_embed_batch = jnp.broadcast_to(code_embed[None, :], (patches.shape[0], self.code_dim))
            return self.decoder_net(context_embed, code_embed_batch)

        all_center_logits = jax.vmap(decode_for_code, in_axes=0, out_axes=0)(code_embeds)
        all_center_logits = jnp.swapaxes(all_center_logits, 0, 1)
        all_center_probs = jax.nn.sigmoid(all_center_logits)
        return {
            "context": context,
            "center_mask": center_mask,
            "context_embed": context_embed,
            "all_center_logits": all_center_logits,
            "all_center_probs": all_center_probs,
        }

    def __call__(self, patches: Array, deterministic: bool = False) -> dict[str, Array]:
        if patches.ndim != 4 or patches.shape[-1] != 1:
            raise ValueError(f"Expected (B,P,P,1) patches, got {patches.shape}")
        if patches.shape[1] != self.patch_size or patches.shape[2] != self.patch_size:
            raise ValueError(
                f"Expected patch spatial size {(self.patch_size, self.patch_size)}, got {patches.shape[1:3]}"
            )

        center_mask, context_embed, target_embed = self._encode_context_target(patches)
        context, target = split_context_and_target(patches, center_mask)

        code_logits = self.inference_net(context_embed, jax.lax.stop_gradient(target_embed))
        code_probs = jax.nn.softmax(code_logits, axis=-1)
        if deterministic:
            code_assign = jax.nn.one_hot(
                jnp.argmax(code_logits, axis=-1), self.num_codes, dtype=patches.dtype
            )
        else:
            code_assign = _straight_through_gumbel(
                code_logits, self.gumbel_temperature, self.make_rng("gumbel")
            )
        code_embed = self._code_embed_from_assign(code_assign)
        pred_embed = self.predictor_net(context_embed, code_embed)
        center_logits = self.decoder_net(context_embed, code_embed)

        return {
            "context": context,
            "target": target,
            "center_mask": center_mask,
            "context_embed": context_embed,
            "target_embed": target_embed,
            "code_logits": code_logits,
            "code_probs": code_probs,
            "code_assign": code_assign,
            "code_indices": jnp.argmax(code_assign, axis=-1),
            "code_embed": code_embed,
            "pred_embed": pred_embed,
            "center_logits": center_logits,
        }


def compute_local_patch_jepa_loss(
    params: dict[str, Any],
    model: LocalPatchJepaPoc,
    patches: Array,
    rng: Array,
    cfg: dict[str, float],
    deterministic: bool = False,
) -> tuple[Array, dict[str, Array]]:
    out = model.apply({"params": params}, patches, deterministic=deterministic, rngs={"gumbel": rng})

    target_embed = jax.lax.stop_gradient(out["target_embed"])
    pred_embed = out["pred_embed"]
    cosine = jnp.sum(pred_embed * target_embed, axis=-1)
    jepa_loss = (1.0 - cosine).mean()

    start = (model.patch_size - model.center_size) // 2
    end = start + model.center_size
    center_targets = patches[:, start:end, start:end, 0]
    center_bce = optax.sigmoid_binary_cross_entropy(out["center_logits"], center_targets).mean()

    counterfactual_weight = float(cfg.get("counterfactual_weight", 0.0))
    counterfactual_margin = float(cfg.get("counterfactual_margin", 0.05))
    diversity_weight = float(cfg.get("diversity_weight", 0.0))
    diversity_margin = float(cfg.get("diversity_margin", 0.1))
    counterfactual_loss = jnp.asarray(0.0, dtype=patches.dtype)
    diversity_loss = jnp.asarray(0.0, dtype=patches.dtype)
    mean_pairwise_code_l1 = jnp.asarray(0.0, dtype=patches.dtype)
    positive_center_bce = center_bce
    hardest_negative_center_bce = center_bce
    if counterfactual_weight > 0.0 or diversity_weight > 0.0:
        all_decode = model.apply({"params": params}, patches, method=LocalPatchJepaPoc.decode_all_codes)
        per_code_center_bce = optax.sigmoid_binary_cross_entropy(
            all_decode["all_center_logits"],
            center_targets[:, None, :, :],
        ).mean(axis=(2, 3))
        all_center_probs = all_decode["all_center_probs"]
        code_indices = jax.lax.stop_gradient(out["code_indices"])
        batch_indices = jnp.arange(patches.shape[0])
        positive_center_bce = per_code_center_bce[batch_indices, code_indices]
        negative_mask = jax.nn.one_hot(code_indices, model.num_codes, dtype=patches.dtype) > 0.5
        negative_bce = jnp.where(negative_mask, jnp.inf, per_code_center_bce)
        hardest_negative_center_bce = negative_bce.min(axis=-1)
        if counterfactual_weight > 0.0:
            counterfactual_loss = jnp.maximum(
                0.0,
                counterfactual_margin + positive_center_bce - hardest_negative_center_bce,
            ).mean()
        if diversity_weight > 0.0:
            prob_diffs = jnp.abs(all_center_probs[:, :, None, :, :] - all_center_probs[:, None, :, :, :])
            pairwise_l1 = prob_diffs.mean(axis=(3, 4))
            pair_mask = jnp.triu(jnp.ones((model.num_codes, model.num_codes), dtype=patches.dtype), k=1)
            masked_pairwise_l1 = pairwise_l1 * pair_mask[None, :, :]
            num_pairs = model.num_codes * (model.num_codes - 1) / 2.0
            mean_pairwise_code_l1 = masked_pairwise_l1.sum(axis=(1, 2)) / max(num_pairs, 1.0)
            diversity_loss = jnp.maximum(0.0, diversity_margin - mean_pairwise_code_l1).mean()

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
        + float(cfg.get("center_weight", 1.0)) * center_bce
        + float(cfg.get("sigreg_weight", 0.2)) * sigreg
        + counterfactual_weight * counterfactual_loss
        + diversity_weight * diversity_loss
    )

    center_pred = (jax.nn.sigmoid(out["center_logits"]) > 0.5).astype(jnp.float32)
    center_exact = jnp.all(center_pred == center_targets, axis=(1, 2)).mean()
    center_acc = (center_pred == center_targets).astype(jnp.float32).mean()

    code_probs = out["code_probs"]
    eps = 1e-6
    usage = code_probs.mean(axis=0)
    per_sample_entropy = -jnp.sum(code_probs * jnp.log(code_probs + eps), axis=-1).mean()
    usage_kl_uniform = jnp.sum(usage * (jnp.log(usage + eps) + jnp.log(model.num_codes)))
    metrics = {
        "total": total,
        "jepa": jepa_loss,
        "center": center_bce,
        "sigreg": sigreg,
        "counterfactual": counterfactual_loss,
        "diversity": diversity_loss,
        "mean_pairwise_code_l1": mean_pairwise_code_l1.mean(),
        "positive_center_bce": positive_center_bce.mean(),
        "hardest_negative_center_bce": hardest_negative_center_bce.mean(),
        "entropy": per_sample_entropy,
        "usage_kl_uniform": usage_kl_uniform,
        "center_acc": center_acc,
        "center_exact_match": center_exact,
        "code_perplexity": jnp.exp(-jnp.sum(usage * jnp.log(usage + eps))),
        "num_active_codes": (usage > (1.0 / model.num_codes) * 0.5).sum().astype(jnp.float32),
        "mean_center_wall_density": center_targets.mean(),
    }
    total = total + float(cfg.get("entropy_weight", 0.01)) * per_sample_entropy
    total = total + float(cfg.get("usage_weight", 0.05)) * usage_kl_uniform
    metrics["total"] = total
    return total, metrics


def center_pattern_ids_np(
    patches: np.ndarray,
    center_size: int,
) -> np.ndarray:
    start = (patches.shape[1] - center_size) // 2
    end = start + center_size
    center = (patches[:, start:end, start:end, 0] > 0.5).astype(np.int32).reshape(len(patches), -1)
    bit_weights = (1 << np.arange(center.shape[1], dtype=np.int32))
    return (center * bit_weights[None, :]).sum(axis=1).astype(np.int32)


def _dihedral_variants_bits(grid2d: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    base = grid2d
    for rot_k in range(4):
        rot = np.rot90(base, k=rot_k)
        variants.append(rot)
        variants.append(np.flip(rot, axis=1))
    return variants


def canonical_center_pattern_ids_np(
    patches: np.ndarray,
    center_size: int,
) -> np.ndarray:
    start = (patches.shape[1] - center_size) // 2
    end = start + center_size
    center = (patches[:, start:end, start:end, 0] > 0.5).astype(np.int32)
    bit_weights = (1 << np.arange(center_size * center_size, dtype=np.int32))
    ids = np.empty((len(patches),), dtype=np.int32)
    for i, grid in enumerate(center):
        variant_ids = []
        for variant in _dihedral_variants_bits(grid):
            flat = variant.reshape(-1)
            variant_ids.append(int((flat * bit_weights).sum()))
        ids[i] = min(variant_ids)
    return ids


def summarize_local_code_assignments(
    code_indices: np.ndarray,
    patches: np.ndarray,
    num_codes: int,
    center_size: int,
) -> list[dict[str, float]]:
    raw_ids = center_pattern_ids_np(patches, center_size=center_size)
    canonical_ids = canonical_center_pattern_ids_np(patches, center_size=center_size)
    start = (patches.shape[1] - center_size) // 2
    end = start + center_size
    center = patches[:, start:end, start:end, 0]
    density = center.mean(axis=(1, 2))

    rows: list[dict[str, float]] = []
    for code in range(num_codes):
        sel = code_indices == code
        count = int(sel.sum())
        if count == 0:
            rows.append(
                {
                    "code": float(code),
                    "count": 0.0,
                    "mean_center_wall_density": 0.0,
                    "top_raw_pattern_id": 0.0,
                    "top_raw_pattern_prob": 0.0,
                    "top_canonical_pattern_id": 0.0,
                    "top_canonical_pattern_prob": 0.0,
                }
            )
            continue

        raw_vals, raw_counts = np.unique(raw_ids[sel], return_counts=True)
        can_vals, can_counts = np.unique(canonical_ids[sel], return_counts=True)
        raw_top = int(raw_vals[np.argmax(raw_counts)])
        can_top = int(can_vals[np.argmax(can_counts)])
        rows.append(
            {
                "code": float(code),
                "count": float(count),
                "mean_center_wall_density": float(density[sel].mean()),
                "top_raw_pattern_id": float(raw_top),
                "top_raw_pattern_prob": float(raw_counts.max() / count),
                "top_canonical_pattern_id": float(can_top),
                "top_canonical_pattern_prob": float(can_counts.max() / count),
            }
        )
    return rows
