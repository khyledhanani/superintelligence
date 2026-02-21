"""
Environment bridge: convert CLUTTR 52-element integer sequences to jaxued Maze Level objects,
and check solvability via flood-fill.

Coordinate conventions:
    CLUTTR: 1-indexed linear on 13x13 grid.
        idx -> row = (idx-1) // 13,  col = (idx-1) % 13
    Maze Level: positions are [x, y] = [col, row].
        wall_map indexed as wall_map[row, col]  (i.e. wall_map[y, x])
"""

import jax
import jax.numpy as jnp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jaxued.environments.maze.level import Level


# ---------------------------------------------------------------------------
# CLUTTR sequence -> Maze Level
# ---------------------------------------------------------------------------

def cluttr_sequence_to_level(seq, rng_key, height=13, width=13):
    """Convert a single 52-element CLUTTR sequence to a jaxued Maze Level.

    JIT and vmap compatible. For batch conversion use:
        jax.vmap(cluttr_sequence_to_level)(seqs, rng_keys)

    Args:
        seq: (52,) integer array. [obstacles(50), goal_idx(1), agent_idx(1)].
             Indices are 1-indexed linear positions on a height x width grid.
             0 = padding (no obstacle).
        rng_key: PRNG key for random agent direction.
        height: Grid height (default 13).
        width: Grid width (default 13).

    Returns:
        Level dataclass with wall_map, goal_pos, agent_pos, agent_dir, width, height.
    """
    obstacles = seq[:50]
    goal_idx = seq[50]
    agent_idx = seq[51]

    # Build wall_map via scatter: convert 1-indexed linear -> (row, col)
    obs_rows = (obstacles - 1) // width
    obs_cols = (obstacles - 1) % width
    obs_valid = obstacles > 0
    flat_idx = obs_rows * width + obs_cols
    flat_wall = jnp.zeros(height * width, dtype=jnp.bool_)
    flat_wall = flat_wall.at[flat_idx].max(obs_valid)
    wall_map = flat_wall.reshape(height, width)

    # Positions: Level convention is [x, y] = [col, row]
    goal_col = (goal_idx - 1) % width
    goal_row = (goal_idx - 1) // width
    goal_pos = jnp.array([goal_col, goal_row], dtype=jnp.uint32)

    agent_col = (agent_idx - 1) % width
    agent_row = (agent_idx - 1) // width
    agent_pos = jnp.array([agent_col, agent_row], dtype=jnp.uint32)

    # Safety: clear walls at agent/goal positions (repair should handle this already)
    wall_map = wall_map.at[goal_row, goal_col].set(False)
    wall_map = wall_map.at[agent_row, agent_col].set(False)

    # Random agent direction: 0=right, 1=down, 2=left, 3=up
    agent_dir = jax.random.randint(rng_key, (), 0, 4).astype(jnp.uint8)

    return Level(
        wall_map=wall_map,
        goal_pos=goal_pos,
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# Solvability checker: flood-fill with fori_loop (vmap + JIT compatible)
# ---------------------------------------------------------------------------

def flood_fill_solvable(wall_map, agent_pos, goal_pos):
    """Check if goal is reachable from agent via 4-connected flood fill.

    Uses fori_loop with H*W iterations (upper bound on shortest path).
    This is vmap and JIT compatible, unlike MazeSolved's while_loop.

    Ignores agent direction (4-connected reachability). Since the agent can
    turn freely, a 4-connected path implies a direction-aware path exists.

    Args:
        wall_map: (H, W) boolean array. True = wall.
        agent_pos: [x, y] = [col, row] (Level convention).
        goal_pos: [x, y] = [col, row] (Level convention).

    Returns:
        Boolean scalar: True if goal is reachable from agent.
    """
    H, W = wall_map.shape
    reachable = jnp.zeros((H, W), dtype=jnp.bool_)
    # Agent position: wall_map[y, x] = wall_map[row, col]
    reachable = reachable.at[agent_pos[1], agent_pos[0]].set(True)

    def expand(_, reachable):
        # Shift in 4 directions, clearing wrapped edges
        up = jnp.roll(reachable, -1, axis=0).at[-1, :].set(False)
        down = jnp.roll(reachable, 1, axis=0).at[0, :].set(False)
        left = jnp.roll(reachable, -1, axis=1).at[:, -1].set(False)
        right = jnp.roll(reachable, 1, axis=1).at[:, 0].set(False)
        return reachable | ((up | down | left | right) & ~wall_map)

    reachable = jax.lax.fori_loop(0, H * W, expand, reachable)
    return reachable[goal_pos[1], goal_pos[0]]


def bfs_path_length(wall_map, agent_pos, goal_pos):
    """Compute the shortest navigable path length from agent to goal.

    Uses a BFS-style distance relaxation via fori_loop so it is vmap and
    JIT compatible. Returns H*W (= 169 for a 13x13 grid) when the goal is
    unreachable, which acts as a sentinel value.

    Args:
        wall_map: (H, W) boolean array. True = wall.
        agent_pos: [x, y] = [col, row] (Level convention).
        goal_pos:  [x, y] = [col, row] (Level convention).

    Returns:
        Integer scalar: shortest path length in steps, or H*W if unreachable.
    """
    H, W = wall_map.shape
    INF = H * W
    dist = jnp.full((H, W), INF, dtype=jnp.int32)
    dist = dist.at[agent_pos[1], agent_pos[0]].set(0)

    def relax(_, dist):
        up    = jnp.roll(dist, -1, axis=0).at[-1, :].set(INF)
        down  = jnp.roll(dist,  1, axis=0).at[ 0, :].set(INF)
        left  = jnp.roll(dist, -1, axis=1).at[:, -1].set(INF)
        right = jnp.roll(dist,  1, axis=1).at[:,  0].set(INF)
        nbr_min = jnp.minimum(jnp.minimum(up, down), jnp.minimum(left, right)) + 1
        return jnp.where(wall_map, INF, jnp.minimum(dist, nbr_min))

    dist = jax.lax.fori_loop(0, H * W, relax, dist)
    return dist[goal_pos[1], goal_pos[0]]
