#!/usr/bin/env python
"""Verification script for Phase 2: Grid Adapter (GRID-01 through GRID-09).

Run from project root:
    python scripts/test_grid_adapter.py

Exits 0 on all PASS, exits 1 on first FAIL.
Requires: jax_env conda environment, CNN-VAE checkpoint at vae/checkpoints/cnn_vae/
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from vae.cnn_vae_model import CnnLstmDecoder
from vae.cnn_vae_level_utils import decode_latent_to_levels_grid, GRID_SIZE

# ── Checkpoint load ───────────────────────────────────────────────────────────
CKPT_DEFAULT_ABS = os.path.join(PROJECT_ROOT, 'vae/checkpoints/cnn_vae/default')
print(f"Loading checkpoint from: {CKPT_DEFAULT_ABS}")
checkpointer = ocp.PyTreeCheckpointer()
restored = checkpointer.restore(CKPT_DEFAULT_ABS)
decoder_params = restored['params']['decoder']
print("Checkpoint loaded.")

# ── decode_fn closure (unbatched: z (64,) -> logits each (13,13)) ─────────────
decoder = CnnLstmDecoder(latent_dim=64)
def decode_fn(z):
    z_batched = z[None]   # (64,) -> (1, 64)
    wl, gl, al = decoder.apply({'params': decoder_params}, z_batched)
    return wl[0], gl[0], al[0]   # (1,13,13) -> (13,13)

# ── Test 1: GRID-01/09 — z=zeros decode produces valid Level with correct dtypes ──
print("\n--- GRID-01 / GRID-09: z=zeros decode, dtype check, is_well_formatted ---")
z_zeros = jnp.zeros((1, 64))
rng = jax.random.PRNGKey(0)
levels = decode_latent_to_levels_grid(decode_fn, z_zeros, rng)

# dtype checks (GRID-01 implies bool wall_map)
assert levels.wall_map.dtype == jnp.bool_, f"FAIL wall_map dtype: {levels.wall_map.dtype}"
assert levels.goal_pos.dtype == jnp.uint32, f"FAIL goal_pos dtype: {levels.goal_pos.dtype}"
assert levels.agent_pos.dtype == jnp.uint32, f"FAIL agent_pos dtype: {levels.agent_pos.dtype}"
assert levels.agent_dir.dtype == jnp.uint8, f"FAIL agent_dir dtype: {levels.agent_dir.dtype}"
print("PASS GRID-01: wall_map dtype bool (sigmoid > 0.5 produces bool)")

# is_well_formatted check (GRID-09) — must use vmap on batched Level
valid = jax.vmap(lambda l: l.is_well_formatted())(levels)
assert bool(valid[0]), f"FAIL GRID-09: z=zeros level not well formatted"
print("PASS GRID-09: z=zeros level passes is_well_formatted()")

# ── Test 2: GRID-02/03/04 visual check — to_str() shows non-wall placements ──
print("\n--- GRID-02 / GRID-03 / GRID-04: visual coordinate check (to_str) ---")
single_level = jax.tree_util.tree_map(lambda x: x[0], levels)
level_str = single_level.to_str()
print(level_str)
# Assert goal and agent appear in the string (G for goal, directional char for agent)
assert 'G' in level_str, "FAIL: no G (goal) found in to_str() output"
# Agent is one of >, <, ^, v
agent_chars = set('><^v')
assert any(c in level_str for c in agent_chars), "FAIL: no agent direction char in to_str()"
print("PASS GRID-02/03/04: goal G and agent visible in to_str() at non-wall positions")

# ── Test 3: GRID-07 — agent direction is uint8 in range 0-3 ──────────────────
print("\n--- GRID-07: agent direction range ---")
assert int(single_level.agent_dir) in range(4), f"FAIL agent_dir out of range: {single_level.agent_dir}"
print(f"PASS GRID-07: agent_dir = {int(single_level.agent_dir)} (0-3 uint8)")

# ── Test 4: GRID-05/06/09 — 1000 random-z batch: 0 collisions, 0 wall placements ──
print("\n--- GRID-05 / GRID-06 / GRID-08 / GRID-09: 1000-sample batch test ---")
z_batch = jax.random.normal(jax.random.PRNGKey(42), (1000, 64))

# GRID-08: run under jit — decode_fn is a Python callable, must use static_argnums=(0,)
jitted_fn = jax.jit(decode_latent_to_levels_grid, static_argnums=(0,))
levels_batch = jitted_fn(decode_fn, z_batch, jax.random.PRNGKey(1))
print("PASS GRID-08: jax.jit(decode_latent_to_levels_grid) compiled and ran 1000 samples")

# GRID-09: all levels valid
valid_batch = jax.vmap(lambda l: l.is_well_formatted())(levels_batch)
n_invalid = int(jnp.sum(~valid_batch))
assert n_invalid == 0, f"FAIL GRID-09: {n_invalid}/1000 levels failed is_well_formatted()"
print(f"PASS GRID-09: 1000/1000 levels pass is_well_formatted()")

# GRID-05: no goal-agent collisions
# goal_flat = y*13 + x = goal_pos[:,1]*13 + goal_pos[:,0]
goal_flat  = levels_batch.goal_pos[:, 1].astype(jnp.int32) * GRID_SIZE + levels_batch.goal_pos[:, 0].astype(jnp.int32)
agent_flat = levels_batch.agent_pos[:, 1].astype(jnp.int32) * GRID_SIZE + levels_batch.agent_pos[:, 0].astype(jnp.int32)
n_collisions = int(jnp.sum(goal_flat == agent_flat))
assert n_collisions == 0, f"FAIL GRID-05: {n_collisions} goal-agent collisions in 1000 samples"
print(f"PASS GRID-05: 0 goal-agent collisions in 1000 samples")

# GRID-06: no wall at goal position
wall_at_goal = jax.vmap(lambda l: l.wall_map[l.goal_pos[1], l.goal_pos[0]])(levels_batch)
n_wall_goal = int(jnp.sum(wall_at_goal))
assert n_wall_goal == 0, f"FAIL GRID-06: {n_wall_goal} wall placements at goal in 1000 samples"

# GRID-06: no wall at agent position
wall_at_agent = jax.vmap(lambda l: l.wall_map[l.agent_pos[1], l.agent_pos[0]])(levels_batch)
n_wall_agent = int(jnp.sum(wall_at_agent))
assert n_wall_agent == 0, f"FAIL GRID-06: {n_wall_agent} wall placements at agent in 1000 samples"
print(f"PASS GRID-06: 0 wall placements at goal or agent in 1000 samples")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ALL GRID-01..09 REQUIREMENTS PASSED")
print("="*60)
sys.exit(0)
