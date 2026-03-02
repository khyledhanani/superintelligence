"""
Pure JAX functions for converting between VAE token sequences and Level dataclass.
All functions are jittable and vmappable.

Token format (52 tokens):
  [0-padded wall indices (1-based, sorted) ..., goal_idx (1-based), agent_idx (1-based)]

Level coordinate system:
  wall_map[y, x] — boolean (13, 13)
  positions are (x, y) = (col, row)
  1-based index → row = (idx-1) // 13, col = (idx-1) % 13
"""
import jax
import jax.numpy as jnp
from jaxued.environments.maze import Level

GRID_SIZE = 13
VOCAB_SIZE = 170
SEQ_LEN = 52
MAX_WALLS = 50  # first 50 tokens are wall slots


def repair_tokens(tokens):
    """JAX-jittable repair of a decoded token sequence.

    Ensures: tokens in [0, 169], goal != agent, no wall at agent/goal, walls sorted.
    If agent or goal sits on a wall, that wall is removed (not the agent/goal).
    """
    tokens = jnp.clip(tokens, 0, VOCAB_SIZE - 1).astype(jnp.int32)

    goal = jnp.clip(tokens[-2], 1, VOCAB_SIZE - 1)
    agent = jnp.clip(tokens[-1], 1, VOCAB_SIZE - 1)
    # If agent == goal, shift agent by 1 (wrap around in valid range)
    agent = jnp.where(goal == agent, (agent % (VOCAB_SIZE - 1)) + 1, agent)

    walls = tokens[:-2]
    # Zero out any wall that coincides with agent or goal
    walls = jnp.where(walls == goal, 0, walls)
    walls = jnp.where(walls == agent, 0, walls)
    # Sort so padding zeros come first
    walls = jnp.sort(walls)

    return jnp.concatenate([walls, jnp.array([goal, agent])])


def tokens_to_level(tokens):
    """Convert a 52-token VAE sequence to a Level dataclass.

    Args:
        tokens: (52,) int32 array.

    Returns:
        Level with wall_map (13,13), goal_pos (2,), agent_pos (2,), etc.
    """
    agent_idx = tokens[-1]   # 1-based
    goal_idx = tokens[-2]    # 1-based
    wall_tokens = tokens[:-2]  # (50,)

    # Build wall_map from 1-based indices
    num_cells = GRID_SIZE * GRID_SIZE
    wall_map_flat = jnp.zeros(num_cells, dtype=jnp.bool_)
    # Convert 1-based to 0-based, clip for safety
    wall_idx_0 = jnp.clip(wall_tokens - 1, 0, num_cells - 1)
    valid_walls = wall_tokens > 0
    wall_map_flat = wall_map_flat.at[wall_idx_0].set(valid_walls)
    wall_map = wall_map_flat.reshape(GRID_SIZE, GRID_SIZE)

    # Convert 1-based index to (x, y) = (col, row)
    agent_0 = jnp.clip(agent_idx - 1, 0, num_cells - 1)
    agent_pos = jnp.array([agent_0 % GRID_SIZE, agent_0 // GRID_SIZE], dtype=jnp.uint32)

    goal_0 = jnp.clip(goal_idx - 1, 0, num_cells - 1)
    goal_pos = jnp.array([goal_0 % GRID_SIZE, goal_0 // GRID_SIZE], dtype=jnp.uint32)

    # Clear wall at agent and goal positions (defensive)
    wall_map = wall_map.at[agent_pos[1], agent_pos[0]].set(False)
    wall_map = wall_map.at[goal_pos[1], goal_pos[0]].set(False)

    return Level(
        wall_map=wall_map,
        goal_pos=goal_pos,
        agent_pos=agent_pos,
        agent_dir=jnp.array(0, dtype=jnp.uint8),
        width=GRID_SIZE,
        height=GRID_SIZE,
    )


def level_to_tokens(level):
    """Convert a Level dataclass to a 52-token VAE sequence.

    Inverse of tokens_to_level(). Output format:
      [50 wall indices (1-based, sorted, 0-padded), goal_idx (1-based), agent_idx (1-based)]

    Args:
        level: Level with wall_map (13,13), goal_pos (2,), agent_pos (2,).

    Returns:
        (52,) int32 array in VAE dataset token format.
    """
    wall_map = level.wall_map  # (13, 13) bool
    wall_flat = wall_map.reshape(-1)  # (169,)

    # 1-based indices where walls exist, 0 where not
    indices_1based = jnp.arange(1, GRID_SIZE * GRID_SIZE + 1)  # 1..169
    wall_indices = jnp.where(wall_flat, indices_1based, 0)  # (169,)

    # Sort descending to get non-zero first, take top MAX_WALLS, then sort ascending
    wall_indices = jnp.sort(wall_indices)[::-1][:MAX_WALLS]
    wall_indices = jnp.sort(wall_indices)  # zeros first, then wall indices ascending

    # Agent and goal as 1-based indices: idx = y * 13 + x + 1
    goal_idx = level.goal_pos[1] * GRID_SIZE + level.goal_pos[0] + 1
    agent_idx = level.agent_pos[1] * GRID_SIZE + level.agent_pos[0] + 1

    return jnp.concatenate([wall_indices, jnp.array([goal_idx, agent_idx])]).astype(jnp.int32)


def _decode_single(decode_fn, z, rng):
    """Decode a single latent vector to a Level."""
    logits = decode_fn(z)                  # (seq_len, vocab_size)
    tokens = jnp.argmax(logits, axis=-1)   # (seq_len,)
    tokens = repair_tokens(tokens)
    level = tokens_to_level(tokens)
    # Randomize agent direction
    agent_dir = jax.random.randint(rng, (), 0, 4).astype(jnp.uint8)
    level = level.replace(agent_dir=agent_dir)
    return level


def decode_latent_to_levels(decode_fn, z_batch, rng):
    """Decode a batch of latent vectors to a batch of Levels.

    Args:
        decode_fn: Pure function z (latent_dim,) -> logits (seq_len, vocab_size).
                   Must handle single (unbatched) input.
        z_batch: (N, latent_dim) latent vectors.
        rng: PRNGKey.

    Returns:
        Batched Level (each field has leading dimension N).
    """
    N = z_batch.shape[0]
    rngs = jax.random.split(rng, N)
    return jax.vmap(_decode_single, in_axes=(None, 0, 0))(decode_fn, z_batch, rngs)
