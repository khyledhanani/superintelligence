from pathlib import Path
import sys

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vae.masked_jepa_poc import (
    MaskedMazeJepaPoc,
    compute_masked_jepa_loss,
    sample_balanced_multi_block_masks,
    sample_rect_masks,
)


def test_sample_rect_masks_are_nonempty_and_batched():
    masks = sample_rect_masks(
        jax.random.PRNGKey(0),
        batch_size=8,
        height=13,
        width=13,
        min_size=4,
        max_size=6,
    )
    assert masks.shape == (8, 13, 13, 1)
    assert masks.dtype == jnp.float32
    assert jnp.all(masks.sum(axis=(1, 2, 3)) > 0)


def test_masked_jepa_forward_and_loss_are_finite():
    key = jax.random.PRNGKey(0)
    grids = jnp.zeros((4, 13, 13, 3), dtype=jnp.float32)
    grids = grids.at[:, 2, 2, 2].set(1.0)
    grids = grids.at[:, 10, 10, 1].set(1.0)
    grids = grids.at[:, 5:8, 6, 0].set(1.0)

    masks = sample_rect_masks(
        jax.random.PRNGKey(1),
        batch_size=4,
        height=13,
        width=13,
        min_size=4,
        max_size=6,
    )

    model = MaskedMazeJepaPoc(height=13, width=13, embed_dim=32, num_codes=8, code_dim=16)
    params = model.init({"params": key, "gumbel": jax.random.PRNGKey(2)}, grids, masks)["params"]

    outputs = model.apply({"params": params}, grids, masks, deterministic=True)
    assert outputs["context_embed"].shape == (4, 32)
    assert outputs["target_embed"].shape == (4, 32)
    assert outputs["code_probs"].shape == (4, 8)
    assert outputs["wall_logits"].shape == (4, 13, 13)
    assert outputs["goal_logits"].shape == (4, 169)
    assert outputs["agent_logits"].shape == (4, 169)

    loss_cfg = {
        "jepa_weight": 1.0,
        "wall_weight": 1.0,
        "goal_weight": 0.2,
        "agent_weight": 0.2,
        "sigreg_weight": 0.2,
        "sigreg_num_projections": 8,
        "sigreg_kernel_gamma": 1.0,
    }
    total, metrics = compute_masked_jepa_loss(
        params=params,
        model=model,
        grids=grids,
        masks=masks,
        rng=jax.random.PRNGKey(3),
        cfg=loss_cfg,
        deterministic=False,
    )
    assert jnp.isfinite(total)
    for value in metrics.values():
        assert jnp.isfinite(value)


def test_sample_balanced_multi_block_masks_have_fixed_area():
    masks = sample_balanced_multi_block_masks(
        jax.random.PRNGKey(4),
        batch_size=8,
        height=13,
        width=13,
        block_height=4,
        block_width=4,
        num_regions=2,
    )
    areas = masks.sum(axis=(1, 2, 3))
    assert masks.shape == (8, 13, 13, 1)
    assert jnp.all(areas == 32.0)
