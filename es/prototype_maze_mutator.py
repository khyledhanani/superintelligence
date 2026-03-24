"""JAX-friendly local maze mutation using a discrete prototype library.

The library is learned self-supervised from static maze patches:
- observe a local wall context
- choose a compatible center prototype
- apply that prototype as a local wall edit

Unlike the JEPA decoder experiments, the selected code cannot be ignored:
the chosen prototype is written directly into the wall map.
"""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np


Array = jax.Array


def load_context_prototype_library(path: str) -> dict[str, np.ndarray | int]:
    data = np.load(path, allow_pickle=False)
    return {
        "patch_size": int(np.asarray(data["patch_size"]).item()),
        "center_size": int(np.asarray(data["center_size"]).item()),
        "num_codes": int(np.asarray(data["num_codes"]).item()),
        "context_ids": np.asarray(data["context_ids"], dtype=np.int64),
        "context_code_probs": np.asarray(data["context_code_probs"], dtype=np.float32),
        "global_probs": np.asarray(data["global_probs"], dtype=np.float32),
        "prototypes": np.asarray(data["prototypes"], dtype=np.float32),
        "prototype_ids": np.asarray(data["prototype_ids"], dtype=np.int64),
    }


def library_to_jax_arrays(library: dict[str, Any]) -> dict[str, Array | int]:
    return {
        "patch_size": int(library["patch_size"]),
        "center_size": int(library["center_size"]),
        "num_codes": int(library["num_codes"]),
        "context_ids": jnp.asarray(library["context_ids"], dtype=jnp.int32),
        "context_code_probs": jnp.asarray(library["context_code_probs"], dtype=jnp.float32),
        "global_probs": jnp.asarray(library["global_probs"], dtype=jnp.float32),
        "prototypes": jnp.asarray(library["prototypes"], dtype=jnp.float32),
        "prototype_ids": jnp.asarray(library["prototype_ids"], dtype=jnp.int32),
    }


def _bit_ids(binary_bits: Array) -> Array:
    weights = (1 << jnp.arange(binary_bits.shape[-1], dtype=jnp.int32))[None, :]
    return (binary_bits.astype(jnp.int32) * weights).sum(axis=-1)


def _context_and_center_indices(patch_size: int, center_size: int) -> tuple[np.ndarray, np.ndarray]:
    start = (patch_size - center_size) // 2
    end = start + center_size
    flat_idx = np.arange(patch_size * patch_size, dtype=np.int32).reshape(patch_size, patch_size)
    context_mask = np.ones((patch_size, patch_size), dtype=bool)
    context_mask[start:end, start:end] = False
    center_mask = ~context_mask
    return flat_idx[context_mask], flat_idx[center_mask]


def valid_site_coordinates(height: int, width: int, patch_size: int) -> np.ndarray:
    radius = patch_size // 2
    ys, xs = np.meshgrid(
        np.arange(radius, height - radius, dtype=np.int32),
        np.arange(radius, width - radius, dtype=np.int32),
        indexing="ij",
    )
    return np.stack([ys.reshape(-1), xs.reshape(-1)], axis=-1)


def _lookup_context_probs(
    context_ids: Array,
    table_ids: Array,
    table_probs: Array,
    global_probs: Array,
) -> Array:
    flat_ids = context_ids.reshape(-1)
    idx = jnp.searchsorted(table_ids, flat_ids, side="left")
    idx = jnp.clip(idx, 0, table_ids.shape[0] - 1)
    found = table_ids[idx] == flat_ids
    probs = jnp.where(found[:, None], table_probs[idx], global_probs[None, :])
    return probs.reshape(context_ids.shape + (global_probs.shape[0],))


def _extract_site_patches(wall_maps: Array, site_coords: Array, patch_size: int) -> Array:
    radius = patch_size // 2

    def extract_single(wall_map: Array, coord: Array) -> Array:
        y = coord[0] - radius
        x = coord[1] - radius
        return jax.lax.dynamic_slice(wall_map, (y, x), (patch_size, patch_size))

    def extract_for_map(wall_map: Array) -> Array:
        return jax.vmap(lambda coord: extract_single(wall_map, coord))(site_coords)

    return jax.vmap(extract_for_map)(wall_maps)


def _nearest_prototype_codes(center_bits: Array, prototypes: Array) -> Array:
    proto_bits = (prototypes > 0.5).astype(jnp.float32)
    dists = jnp.abs(center_bits[:, :, None, :] - proto_bits[None, None, :, :]).sum(axis=-1)
    return jnp.argmin(dists, axis=-1).astype(jnp.int32)


def mutate_wall_maps_with_prototypes(
    rng: chex.PRNGKey,
    wall_maps: Array,
    goal_pos: Array,
    agent_pos: Array,
    library: dict[str, Array | int],
    deterministic: bool = False,
    prototype_wall_bias: float = 0.0,
) -> tuple[Array, dict[str, Array]]:
    patch_size = int(library["patch_size"])
    center_size = int(library["center_size"])
    num_codes = int(library["num_codes"])
    wall_maps = wall_maps.astype(jnp.bool_)
    batch_size, height, width = wall_maps.shape
    site_coords = jnp.asarray(valid_site_coordinates(height, width, patch_size), dtype=jnp.int32)
    context_idx_np, center_idx_np = _context_and_center_indices(patch_size, center_size)
    context_idx = jnp.asarray(context_idx_np, dtype=jnp.int32)
    center_idx = jnp.asarray(center_idx_np, dtype=jnp.int32)

    patches = _extract_site_patches(wall_maps, site_coords, patch_size=patch_size)
    flat_patches = patches.reshape((batch_size, site_coords.shape[0], patch_size * patch_size))
    context_bits = flat_patches[:, :, context_idx].astype(jnp.float32)
    center_bits = flat_patches[:, :, center_idx].astype(jnp.float32)
    context_ids = _bit_ids(context_bits.reshape((-1, context_bits.shape[-1]))).reshape(context_bits.shape[:2])
    context_probs = _lookup_context_probs(
        context_ids,
        library["context_ids"],
        library["context_code_probs"],
        library["global_probs"],
    )
    current_codes = _nearest_prototype_codes(center_bits, library["prototypes"])

    goal_pos = goal_pos.astype(jnp.int32)
    agent_pos = agent_pos.astype(jnp.int32)
    center_radius = center_size // 2
    site_y = site_coords[:, 0][None, :]
    site_x = site_coords[:, 1][None, :]
    blocks_goal = (
        (jnp.abs(goal_pos[:, 1:2] - site_y) <= center_radius)
        & (jnp.abs(goal_pos[:, 0:1] - site_x) <= center_radius)
    )
    blocks_agent = (
        (jnp.abs(agent_pos[:, 1:2] - site_y) <= center_radius)
        & (jnp.abs(agent_pos[:, 0:1] - site_x) <= center_radius)
    )
    site_allowed = ~(blocks_goal | blocks_agent)

    same_mask = jax.nn.one_hot(current_codes, num_codes, dtype=jnp.float32)
    alt_probs = context_probs * (1.0 - same_mask)
    alt_mass = alt_probs.sum(axis=-1)
    site_scores = jnp.where(site_allowed, alt_mass, 0.0)
    any_allowed = site_allowed.any(axis=-1, keepdims=True)
    site_scores = jnp.where(any_allowed, site_scores, jnp.ones_like(site_scores))

    rng_site, rng_code = jax.random.split(rng)
    if deterministic:
        site_idx = jnp.argmax(site_scores, axis=-1).astype(jnp.int32)
    else:
        gumbel = jax.random.gumbel(rng_site, site_scores.shape)
        site_idx = jnp.argmax(jnp.log(jnp.maximum(site_scores, 1e-8)) + gumbel, axis=-1).astype(jnp.int32)

    batch_idx = jnp.arange(batch_size)
    chosen_alt_probs = alt_probs[batch_idx, site_idx]
    chosen_current = current_codes[batch_idx, site_idx]
    fallback = library["global_probs"][None, :] * (1.0 - jax.nn.one_hot(chosen_current, num_codes, dtype=jnp.float32))
    chosen_alt_probs = jnp.where(
        chosen_alt_probs.sum(axis=-1, keepdims=True) > 0,
        chosen_alt_probs,
        fallback,
    )
    if prototype_wall_bias != 0.0:
        proto_wall_frac = library["prototypes"].mean(axis=-1)
        proto_wall_frac = proto_wall_frac - proto_wall_frac.mean()
        chosen_alt_probs = chosen_alt_probs * jnp.exp(float(prototype_wall_bias) * proto_wall_frac[None, :])
    chosen_alt_probs = chosen_alt_probs / jnp.maximum(chosen_alt_probs.sum(axis=-1, keepdims=True), 1e-8)

    if deterministic:
        chosen_code = jnp.argmax(chosen_alt_probs, axis=-1).astype(jnp.int32)
    else:
        gumbel_code = jax.random.gumbel(rng_code, chosen_alt_probs.shape)
        chosen_code = jnp.argmax(jnp.log(jnp.maximum(chosen_alt_probs, 1e-8)) + gumbel_code, axis=-1).astype(jnp.int32)

    prototypes = (library["prototypes"] > 0.5).astype(jnp.bool_).reshape((num_codes, center_size, center_size))
    chosen_proto = prototypes[chosen_code]
    chosen_sites = site_coords[site_idx]

    def apply_one(
        wall_map: Array,
        proto: Array,
        site: Array,
        goal: Array,
        agent: Array,
    ) -> Array:
        y0 = site[0] - center_radius
        x0 = site[1] - center_radius
        updated = jax.lax.dynamic_update_slice(wall_map, proto, (y0, x0))
        updated = updated.at[goal[1], goal[0]].set(False)
        updated = updated.at[agent[1], agent[0]].set(False)
        return updated

    child_walls = jax.vmap(apply_one)(
        wall_maps,
        chosen_proto,
        chosen_sites,
        goal_pos,
        agent_pos,
    )
    info = {
        "site_index": site_idx,
        "site_y": chosen_sites[:, 0],
        "site_x": chosen_sites[:, 1],
        "current_code": chosen_current,
        "chosen_code": chosen_code,
        "chosen_alt_mass": alt_mass[batch_idx, site_idx],
    }
    return child_walls, info


def mutate_levels_with_prototypes(
    rng: chex.PRNGKey,
    levels: Any,
    library: dict[str, Array | int],
    deterministic: bool = False,
    prototype_wall_bias: float = 0.0,
) -> tuple[Any, dict[str, Array]]:
    child_walls, info = mutate_wall_maps_with_prototypes(
        rng,
        levels.wall_map,
        levels.goal_pos,
        levels.agent_pos,
        library,
        deterministic=deterministic,
        prototype_wall_bias=prototype_wall_bias,
    )
    child_levels = levels.replace(wall_map=child_walls) if hasattr(levels, "replace") else type(levels)(
        wall_map=child_walls,
        goal_pos=levels.goal_pos,
        agent_pos=levels.agent_pos,
        agent_dir=levels.agent_dir,
        width=levels.width,
        height=levels.height,
    )
    return child_levels, info
