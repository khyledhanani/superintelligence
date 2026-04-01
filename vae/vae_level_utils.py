"""
Pure JAX functions for converting between VAE token sequences and Level dataclass.
All functions are jittable and vmappable.

Token format (variable length depending on grid size):
  [0-padded wall indices (1-based, sorted) ..., goal_idx (1-based), agent_idx (1-based)]

Level coordinate system:
  wall_map[y, x] — boolean (grid_size, grid_size)
  positions are (x, y) = (col, row)
  1-based index → row = (idx-1) // grid_size, col = (idx-1) % grid_size

Default constants match the 13x13 VAE. For 21x21, pass grid_size=21,
vocab_size=442, seq_len=152, max_walls=150 (from the VAE config).
"""
import jax
import jax.numpy as jnp
from jaxued.environments.maze import Level

# 13x13 defaults (backward compatible)
GRID_SIZE = 13
VOCAB_SIZE = 170
SEQ_LEN = 52
MAX_WALLS = 50  # first 50 tokens are wall slots


def repair_tokens(tokens, vocab_size=VOCAB_SIZE):
    """JAX-jittable repair of a decoded token sequence.

    Ensures: tokens in [0, vocab_size-1], goal != agent, no wall at agent/goal, walls sorted.
    If agent or goal sits on a wall, that wall is removed (not the agent/goal).
    """
    tokens = jnp.clip(tokens, 0, vocab_size - 1).astype(jnp.int32)

    goal = jnp.clip(tokens[-2], 1, vocab_size - 1)
    agent = jnp.clip(tokens[-1], 1, vocab_size - 1)
    # If agent == goal, shift agent by 1 (wrap around in valid range)
    agent = jnp.where(goal == agent, (agent % (vocab_size - 1)) + 1, agent)

    walls = tokens[:-2]
    # Zero out any wall that coincides with agent or goal
    walls = jnp.where(walls == goal, 0, walls)
    walls = jnp.where(walls == agent, 0, walls)
    # Sort so padding zeros come first
    walls = jnp.sort(walls)

    return jnp.concatenate([walls, jnp.array([goal, agent])])


def tokens_to_level(tokens, grid_size=GRID_SIZE):
    """Convert a token sequence to a Level dataclass.

    Args:
        tokens: (seq_len,) int32 array.
        grid_size: Grid dimension (default 13).

    Returns:
        Level with wall_map (grid_size, grid_size), goal_pos (2,), agent_pos (2,), etc.
    """
    agent_idx = tokens[-1]   # 1-based
    goal_idx = tokens[-2]    # 1-based
    wall_tokens = tokens[:-2]  # (max_walls,)

    # Build wall_map from 1-based indices
    num_cells = grid_size * grid_size
    wall_map_flat = jnp.zeros(num_cells, dtype=jnp.bool_)
    # Convert 1-based to 0-based, clip for safety
    wall_idx_0 = jnp.clip(wall_tokens - 1, 0, num_cells - 1)
    valid_walls = wall_tokens > 0
    wall_map_flat = wall_map_flat.at[wall_idx_0].set(valid_walls)
    wall_map = wall_map_flat.reshape(grid_size, grid_size)

    # Convert 1-based index to (x, y) = (col, row)
    agent_0 = jnp.clip(agent_idx - 1, 0, num_cells - 1)
    agent_pos = jnp.array([agent_0 % grid_size, agent_0 // grid_size], dtype=jnp.uint32)

    goal_0 = jnp.clip(goal_idx - 1, 0, num_cells - 1)
    goal_pos = jnp.array([goal_0 % grid_size, goal_0 // grid_size], dtype=jnp.uint32)

    # Clear wall at agent and goal positions (defensive)
    wall_map = wall_map.at[agent_pos[1], agent_pos[0]].set(False)
    wall_map = wall_map.at[goal_pos[1], goal_pos[0]].set(False)

    return Level(
        wall_map=wall_map,
        goal_pos=goal_pos,
        agent_pos=agent_pos,
        agent_dir=jnp.array(0, dtype=jnp.uint8),
        width=grid_size,
        height=grid_size,
    )


def level_to_tokens(level, grid_size=GRID_SIZE, max_walls=MAX_WALLS):
    """Convert a Level dataclass to a token sequence.

    Inverse of tokens_to_level(). Output format:
      [max_walls wall indices (1-based, sorted, 0-padded), goal_idx (1-based), agent_idx (1-based)]

    Args:
        level: Level with wall_map (grid_size, grid_size), goal_pos (2,), agent_pos (2,).
        grid_size: Grid dimension (default 13).
        max_walls: Number of wall token slots (default 50).

    Returns:
        (max_walls + 2,) int32 array in VAE dataset token format.
    """
    wall_map = level.wall_map  # (grid_size, grid_size) bool
    wall_flat = wall_map.reshape(-1)  # (grid_size*grid_size,)

    # 1-based indices where walls exist, 0 where not
    indices_1based = jnp.arange(1, grid_size * grid_size + 1)
    wall_indices = jnp.where(wall_flat, indices_1based, 0)

    # Sort descending to get non-zero first, take top max_walls, then sort ascending
    wall_indices = jnp.sort(wall_indices)[::-1][:max_walls]
    wall_indices = jnp.sort(wall_indices)  # zeros first, then wall indices ascending

    # Agent and goal as 1-based indices: idx = y * grid_size + x + 1
    goal_idx = level.goal_pos[1] * grid_size + level.goal_pos[0] + 1
    agent_idx = level.agent_pos[1] * grid_size + level.agent_pos[0] + 1

    return jnp.concatenate([wall_indices, jnp.array([goal_idx, agent_idx])]).astype(jnp.int32)


def _decode_single(decode_fn, z, rng, grid_size=GRID_SIZE, vocab_size=VOCAB_SIZE):
    """Decode a single latent vector to a Level."""
    logits = decode_fn(z)                  # (seq_len, vocab_size)
    tokens = jnp.argmax(logits, axis=-1)   # (seq_len,)
    tokens = repair_tokens(tokens, vocab_size=vocab_size)
    level = tokens_to_level(tokens, grid_size=grid_size)
    # Randomize agent direction
    agent_dir = jax.random.randint(rng, (), 0, 4).astype(jnp.uint8)
    level = level.replace(agent_dir=agent_dir)
    return level


def decode_latent_to_levels(decode_fn, z_batch, rng, grid_size=GRID_SIZE, vocab_size=VOCAB_SIZE):
    """Decode a batch of latent vectors to a batch of Levels.

    Args:
        decode_fn: Pure function z (latent_dim,) -> logits (seq_len, vocab_size).
                   Must handle single (unbatched) input.
        z_batch: (N, latent_dim) latent vectors.
        rng: PRNGKey.
        grid_size: Grid dimension (default 13).
        vocab_size: VAE vocabulary size (default 170).

    Returns:
        Batched Level (each field has leading dimension N).
    """
    N = z_batch.shape[0]
    rngs = jax.random.split(rng, N)
    from functools import partial
    _decode = partial(_decode_single, grid_size=grid_size, vocab_size=vocab_size)
    return jax.vmap(_decode, in_axes=(None, 0, 0))(decode_fn, z_batch, rngs)
