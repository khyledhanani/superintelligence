from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vae.local_patch_jepa_poc import (
    LocalPatchJepaPoc,
    center_block_mask,
    compute_local_patch_jepa_loss,
    extract_wall_patches_np,
)


def test_extract_wall_patches_np_preserves_local_center():
    grids = np.zeros((1, 13, 13, 3), dtype=np.float32)
    grids[0, 6, 6, 0] = 1.0
    grids[0, 6, 7, 0] = 1.0
    patches = extract_wall_patches_np(grids, np.array([6]), np.array([6]), patch_size=5)
    assert patches.shape == (1, 5, 5, 1)
    assert float(patches[0, 2, 2, 0]) == 1.0
    assert float(patches[0, 2, 3, 0]) == 1.0


def test_center_block_mask_has_expected_area():
    mask = center_block_mask(patch_size=9, center_size=3)
    assert mask.shape == (9, 9, 1)
    assert float(mask.sum()) == 9.0


def test_local_patch_jepa_forward_and_loss_are_finite():
    key = jax.random.PRNGKey(0)
    patches = jnp.zeros((4, 9, 9, 1), dtype=jnp.float32)
    patches = patches.at[:, 3:6, 4, 0].set(1.0)

    model = LocalPatchJepaPoc(
        patch_size=9,
        center_size=3,
        embed_dim=32,
        num_codes=8,
        code_dim=16,
    )
    params = model.init({"params": key, "gumbel": jax.random.PRNGKey(1)}, patches)["params"]

    outputs = model.apply({"params": params}, patches, deterministic=True)
    assert outputs["context_embed"].shape == (4, 32)
    assert outputs["target_embed"].shape == (4, 32)
    assert outputs["code_probs"].shape == (4, 8)
    assert outputs["center_logits"].shape == (4, 3, 3)

    total, metrics = compute_local_patch_jepa_loss(
        params=params,
        model=model,
        patches=patches,
        rng=jax.random.PRNGKey(2),
        cfg={
            "jepa_weight": 1.0,
            "center_weight": 1.0,
            "sigreg_weight": 0.2,
            "sigreg_num_projections": 8,
            "sigreg_kernel_gamma": 1.0,
            "counterfactual_weight": 0.1,
            "counterfactual_margin": 0.05,
        },
        deterministic=False,
    )
    assert jnp.isfinite(total)
    for value in metrics.values():
        assert jnp.isfinite(value)


def test_local_patch_jepa_can_force_decode_specific_code():
    key = jax.random.PRNGKey(0)
    patches = jnp.zeros((3, 9, 9, 1), dtype=jnp.float32)
    patches = patches.at[:, 4, 3:6, 0].set(1.0)

    model = LocalPatchJepaPoc(
        patch_size=9,
        center_size=3,
        embed_dim=32,
        num_codes=8,
        code_dim=16,
    )
    params = model.init({"params": key, "gumbel": jax.random.PRNGKey(1)}, patches)["params"]
    outputs = model.apply(
        {"params": params},
        patches,
        jnp.array([1, 3, 5], dtype=jnp.int32),
        method=LocalPatchJepaPoc.decode_with_code,
    )
    assert outputs["center_logits"].shape == (3, 3, 3)
    assert outputs["center_probs"].shape == (3, 3, 3)
    assert outputs["center_confidence"].shape == (3,)
    assert outputs["code_indices"].tolist() == [1, 3, 5]
    assert jnp.all(jnp.isfinite(outputs["center_logits"]))
    assert jnp.all(jnp.isfinite(outputs["center_confidence"]))

    all_outputs = model.apply({"params": params}, patches, method=LocalPatchJepaPoc.decode_all_codes)
    assert all_outputs["all_center_logits"].shape == (3, 8, 3, 3)
    assert all_outputs["all_center_probs"].shape == (3, 8, 3, 3)
    assert jnp.all(jnp.isfinite(all_outputs["all_center_logits"]))
