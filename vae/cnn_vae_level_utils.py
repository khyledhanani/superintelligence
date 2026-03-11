"""Grid-based decode utilities for CNN-LSTM VAE maze decoder.

All functions are jittable and vmappable. This is the Phase 2 adapter
for CNN-VAE (CnnLstmDecoder) output. The existing vae_level_utils.py
token-based decoder for CluttrVAE is unchanged.

Output coordinate system:
    wall_map[y, x] — boolean (13, 13), y=row, x=col
    positions are (x, y) = (col, row)
    flat index (0-based): row = flat_idx // 13, col = flat_idx % 13
"""
import jax
import jax.numpy as jnp
from jaxued.environments.maze import Level

GRID_SIZE = 13


def _decode_single_z(decode_fn, z, rng):
    """Decode a single latent vector to a Level.

    Args:
        decode_fn: Pure function z (latent_dim,) -> (wall_logits, goal_logits,
                   agent_logits) each (GRID_SIZE, GRID_SIZE). Must handle
                   UNBATCHED input (single sample, no leading batch dim).
        z: (latent_dim,) latent vector.
        rng: PRNGKey for agent direction randomization.

    Returns:
        Level with:
            wall_map:  (GRID_SIZE, GRID_SIZE) bool
            goal_pos:  (2,) uint32  [x=col, y=row]
            agent_pos: (2,) uint32  [x=col, y=row]
            agent_dir: () uint8, value in 0..3
    """
    wall_logits, goal_logits, agent_logits = decode_fn(z)  # each (13, 13)

    # GRID-01: wall map from sigmoid threshold at 0.5 (equivalent to logit > 0)
    wall_map = jax.nn.sigmoid(wall_logits) > 0.5           # (13, 13) bool

    # GRID-02: mask logits at wall positions — set to -1e9 to exclude from argmax
    wall_mask_flat = wall_map.flatten()                     # (169,)
    goal_logits_masked = jnp.where(wall_mask_flat, -1e9, goal_logits.flatten())
    agent_logits_masked = jnp.where(wall_mask_flat, -1e9, agent_logits.flatten())

    # GRID-03: goal position from masked argmax — x=col, y=row
    goal_flat = jnp.argmax(goal_logits_masked).astype(jnp.uint32)
    goal_pos = jnp.array([goal_flat % GRID_SIZE, goal_flat // GRID_SIZE], dtype=jnp.uint32)

    # GRID-04: agent position from masked argmax — x=col, y=row
    agent_flat = jnp.argmax(agent_logits_masked).astype(jnp.uint32)

    # GRID-05: collision resolution — shift agent +1 (wrap) if same flat index as goal
    # jnp.where is JIT-compatible; Python if/else on traced values is NOT
    agent_flat = jnp.where(
        goal_flat == agent_flat,
        (agent_flat + 1) % (GRID_SIZE * GRID_SIZE),
        agent_flat,
    )
    agent_pos = jnp.array([agent_flat % GRID_SIZE, agent_flat // GRID_SIZE], dtype=jnp.uint32)

    # GRID-06: clear walls at goal and agent positions (defensive — ensures no wall at placement)
    # Access pattern: wall_map[y, x] = wall_map[row, col]
    wall_map = wall_map.at[goal_pos[1], goal_pos[0]].set(False)
    wall_map = wall_map.at[agent_pos[1], agent_pos[0]].set(False)

    # GRID-07: randomize agent direction uniformly in {0, 1, 2, 3}
    agent_dir = jax.random.randint(rng, (), 0, 4).astype(jnp.uint8)

    return Level(
        wall_map=wall_map,
        goal_pos=goal_pos,
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        width=GRID_SIZE,
        height=GRID_SIZE,
    )


def decode_latent_to_levels_grid(decode_fn, z_batch, rng):
    """Decode a batch of latent vectors to a batch of Levels.

    Drop-in replacement for decode_latent_to_levels() from vae_level_utils.py
    for use with CnnLstmDecoder (3-channel grid output) instead of CluttrVAE
    (token sequence output). Signature is identical.

    Args:
        decode_fn: Pure function z (latent_dim,) -> (wall_logits, goal_logits,
                   agent_logits) each (GRID_SIZE, GRID_SIZE). Must handle
                   UNBATCHED input (single sample). See Pattern 3 in RESEARCH.md
                   for the correct decode_fn closure using CnnLstmDecoder.
        z_batch: (N, latent_dim) latent vectors.
        rng: PRNGKey.

    Returns:
        Batched Level — each field has leading dimension N:
            wall_map:  (N, GRID_SIZE, GRID_SIZE) bool
            goal_pos:  (N, 2) uint32
            agent_pos: (N, 2) uint32
            agent_dir: (N,) uint8

    Validation:
        Use jax.vmap(lambda l: l.is_well_formatted())(levels) to check validity.
        Do NOT call levels.is_well_formatted() directly on a batched Level —
        it raises a shape broadcast error.
    """
    # GRID-08: JIT-compatible via jax.vmap over single-sample function
    N = z_batch.shape[0]
    rngs = jax.random.split(rng, N)
    return jax.vmap(_decode_single_z, in_axes=(None, 0, 0))(decode_fn, z_batch, rngs)
