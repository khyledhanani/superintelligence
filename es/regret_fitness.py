"""
Regret-based fitness for ES environment evolution.

Uses a frozen ACCEL agent to evaluate environments via MaxMC regret
(ACCEL-inspired proxy from jaxued, not the exact ACCEL PVL metric).

Pipeline:
    CLUTTR sequences -> Maze Levels -> solvability check ->
    agent rollout (AutoReplayWrapper) -> MaxMC regret -> fitness scores

JIT notes:
    env and network are Python objects kept STATIC outside JIT.
    Only rng, sequences, agent_params flow as traced arrays.
"""

import jax
import jax.numpy as jnp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jaxued.utils import compute_max_returns, max_mc
from env_bridge import cluttr_sequence_to_level, flood_fill_solvable
from agent_loader import ActorCritic


# ---------------------------------------------------------------------------
# Complexity filter
# ---------------------------------------------------------------------------

def compute_complexity_mask(sequences, min_obstacles=5, min_distance=3, inner_dim=13):
    """Mask out trivially simple environments.

    Args:
        sequences: (pop_size, 52) CLUTTR sequences.
        min_obstacles: Minimum number of non-zero obstacle tokens.
        min_distance: Minimum Manhattan distance between agent and goal.
        inner_dim: Grid width/height (13 for CLUTTR).

    Returns:
        Boolean mask, shape (pop_size,). True = complex enough.
    """
    obs_count = jnp.sum(sequences[:, :50] > 0, axis=1)
    goal_idx = sequences[:, 50]
    agent_idx = sequences[:, 51]
    goal_row = (goal_idx - 1) // inner_dim
    goal_col = (goal_idx - 1) % inner_dim
    agent_row = (agent_idx - 1) // inner_dim
    agent_col = (agent_idx - 1) % inner_dim
    manhattan = jnp.abs(goal_row - agent_row) + jnp.abs(goal_col - agent_col)
    return (obs_count >= min_obstacles) & (manhattan >= min_distance)


# ---------------------------------------------------------------------------
# Agent rollout on a batch of levels
# ---------------------------------------------------------------------------

def rollout_agent_on_levels(rng, env, env_params, agent_params, network, levels,
                            num_steps=256, deterministic=True):
    """Roll out a frozen agent on a batch of levels, collecting trajectory data.

    Uses AutoReplayWrapper so the agent replays the same level after each
    episode end, giving multiple episodes per rollout for better MaxMC estimation.

    Args:
        rng: PRNG key.
        env: Maze wrapped in AutoReplayWrapper (Python object, NOT jitted).
        env_params: EnvParams (max_steps_in_episode=250).
        agent_params: Frozen network parameters dict.
        network: ActorCritic instance (Python object, NOT jitted).
        levels: Batched Level pytree, shape (pop_size, ...).
        num_steps: Total rollout steps (256 = ACCEL default).
        deterministic: If True use argmax policy; if False sample actions.

    Returns:
        rewards: (num_steps, pop_size)
        values:  (num_steps, pop_size) — critic V(s_t) at each step
        dones:   (num_steps, pop_size)
    """
    pop_size = jax.tree_util.tree_flatten(levels)[0][0].shape[0]

    # Reset all envs to their respective levels (vmapped)
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(rng_reset, pop_size), levels, env_params
    )

    init_hstate = ActorCritic.initialize_carry((pop_size,))

    def step_fn(carry, _):
        rng, hstate, obs, state, done = carry
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        # Agent inference: add leading (1, ...) batch dim for RNN
        # ActorCritic.__call__ expects inputs=(obs, dones), hidden
        x = jax.tree_util.tree_map(lambda a: a[None, ...], (obs, done))
        hstate, pi, value = network.apply(agent_params, x, hstate)

        # Action selection: deterministic (argmax) or stochastic (sample)
        action = jax.lax.cond(
            deterministic,
            lambda: jnp.argmax(pi.logits, axis=-1).squeeze(0),
            lambda: pi.sample(seed=rng_act).squeeze(0),
        )
        value = value.squeeze(0)

        # Env step (vmapped over pop_size)
        next_obs, next_state, reward, next_done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, pop_size), state, action, env_params)

        return (rng, hstate, next_obs, next_state, next_done), (reward, value, next_done)

    _, (rewards, values, dones) = jax.lax.scan(
        step_fn,
        (rng, init_hstate, init_obs, init_state, jnp.zeros(pop_size, dtype=jnp.bool_)),
        None,
        length=num_steps,
    )

    return rewards, values, dones


# ---------------------------------------------------------------------------
# Rollout wrapper that also emits agent positions
# ---------------------------------------------------------------------------

def rollout_agent_on_levels_with_positions(rng, env, env_params, agent_params, network, levels,
                                           num_steps=256, deterministic=True):
    """Thin wrapper around rollout_agent_on_levels that also emits agent_pos from the scan.

    Backward-compatible: existing callers of rollout_agent_on_levels are unaffected.

    Args:
        rng: PRNG key.
        env: Maze wrapped in AutoReplayWrapper (Python object, NOT jitted).
        env_params: EnvParams (max_steps_in_episode=250).
        agent_params: Frozen network parameters dict.
        network: ActorCritic instance (Python object, NOT jitted).
        levels: Batched Level pytree, shape (pop_size, ...).
        num_steps: Total rollout steps (256 = ACCEL default).
        deterministic: If True use argmax policy; if False sample actions.

    Returns:
        rewards:         (num_steps, pop_size)
        values:          (num_steps, pop_size) — critic V(s_t) at each step
        dones:           (num_steps, pop_size)
        agent_positions: (num_steps, pop_size, 2) — [col, row] at each step
    """
    pop_size = jax.tree_util.tree_flatten(levels)[0][0].shape[0]

    # Reset all envs to their respective levels (vmapped)
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(rng_reset, pop_size), levels, env_params
    )

    init_hstate = ActorCritic.initialize_carry((pop_size,))

    def step_fn(carry, _):
        rng, hstate, obs, state, done = carry
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        # Agent inference: add leading (1, ...) batch dim for RNN
        # ActorCritic.__call__ expects inputs=(obs, dones), hidden
        # agent_params is the full Flax variable dict {'params': {...}} — pass directly
        x = jax.tree_util.tree_map(lambda a: a[None, ...], (obs, done))
        hstate, pi, value = network.apply(agent_params, x, hstate)

        # Action selection: deterministic (argmax) or stochastic (sample)
        action = jax.lax.cond(
            deterministic,
            lambda: jnp.argmax(pi.logits, axis=-1).squeeze(0),
            lambda: pi.sample(seed=rng_act).squeeze(0),
        )
        value = value.squeeze(0)

        # Env step (vmapped over pop_size)
        next_obs, next_state, reward, next_done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, pop_size), state, action, env_params)

        # Emit agent_pos ONLY (not full EnvState — avoids OOM from maze_map)
        # next_state is AutoReplayState; agent_pos lives in next_state.env_state.agent_pos
        return (rng, hstate, next_obs, next_state, next_done), (reward, value, next_done, next_state.env_state.agent_pos)

    _, (rewards, values, dones, agent_positions) = jax.lax.scan(
        step_fn,
        (rng, init_hstate, init_obs, init_state, jnp.zeros(pop_size, dtype=jnp.bool_)),
        None,
        length=num_steps,
    )

    return rewards, values, dones, agent_positions


# ---------------------------------------------------------------------------
# Behavior signature extractor
# ---------------------------------------------------------------------------

# TODO: EXPERIMENTAL v1 -- behavior signature design is NOT final. See .planning/DECISIONS.md for design rationale and planned revisit criteria.
def extract_behavior_signature(agent_positions, num_steps, grid_h=13, grid_w=13):
    """Extract an L1-normalized visit-count histogram over the maze grid.

    Given a trajectory of agent positions from a rollout, returns a fixed-length
    histogram representing the fraction of steps the agent spent in each grid cell.
    This is the core primitive for all ES diversity mechanisms (NS-ES, SV-CMA-ES).

    Coordinate convention:
        agent_positions[..., 0] = col (x)
        agent_positions[..., 1] = row (y)
        Linear cell index = row * grid_w + col

    Args:
        agent_positions: (num_steps, pop_size, 2) int32 array of [col, row] positions.
        num_steps: Number of rollout steps (used for documentation; shape inferred from array).
        grid_h: Grid height in cells (default 13 for CLUTTR maze).
        grid_w: Grid width in cells (default 13 for CLUTTR maze).

    Returns:
        hist: (pop_size, grid_h * grid_w) float32 array, L1-normalized so each row
              sums to 1.0 (or 0.0 if no steps recorded).

    Usage:
        dummy_positions = jnp.zeros((4, 2, 2), dtype=jnp.int32)
        lowered = jax.jit(extract_behavior_signature).lower(dummy_positions, 4)
        compiled = lowered.compile()  # must succeed without error
        sig = compiled(dummy_positions, 4)  # shape (2, 169)
    """
    num_cells = grid_h * grid_w  # 169
    col = agent_positions[..., 0].astype(jnp.int32)
    row = agent_positions[..., 1].astype(jnp.int32)
    cell_idx = row * grid_w + col
    one_hot = jax.nn.one_hot(cell_idx, num_classes=num_cells, dtype=jnp.float32)
    hist = one_hot.sum(axis=0)  # (pop_size, num_cells)
    total = hist.sum(axis=-1, keepdims=True)
    hist = hist / jnp.maximum(total, 1.0)
    return hist


# ---------------------------------------------------------------------------
# Regret-based fitness function
# ---------------------------------------------------------------------------

def regret_fitness(rng, sequences, agent_params, network, env, env_params,
                   num_steps=256, deterministic=True,
                   min_obstacles=5, min_distance=3, inner_dim=13):
    """Compute MaxMC regret-based fitness for a batch of CLUTTR sequences.

    Returns NEGATED regret because evosax MINIMIZES (we want to MAXIMIZE regret).
    Unsolvable environments receive an adaptive penalty (max observed regret + margin).

    Note: This is an ACCEL-inspired regret proxy via max_mc from jaxued,
    not the exact ACCEL metric (which uses PVL/estimated regret).

    Args:
        rng: PRNG key.
        sequences: (pop_size, 52) integer arrays (decoded + repaired CLUTTR sequences).
        agent_params: Frozen network parameters.
        network: ActorCritic instance (static Python object).
        env: AutoReplayWrapper(Maze) instance (static Python object).
        env_params: EnvParams.
        num_steps: Rollout length per level.
        deterministic: If True, use argmax policy for stable fitness signal.
        min_obstacles: Complexity threshold (minimum non-zero obstacle tokens).
        min_distance: Complexity threshold (minimum agent-goal Manhattan distance).
        inner_dim: Grid width/height (13 for CLUTTR).

    Returns:
        fitness: (pop_size,) — negated regret (lower = better for CMA-ES).
        info: Dict with regret/validity diagnostics.
    """
    pop_size = sequences.shape[0]

    # 1. Convert CLUTTR sequences to Maze Levels
    rng, rng_dirs = jax.random.split(rng)
    dir_keys = jax.random.split(rng_dirs, pop_size)
    levels = jax.vmap(cluttr_sequence_to_level)(sequences, dir_keys)

    # 2. Solvability check (used for masking results, not filtering rollout)
    solvable = jax.vmap(flood_fill_solvable)(
        levels.wall_map, levels.agent_pos, levels.goal_pos
    )
    complex_enough = compute_complexity_mask(
        sequences, min_obstacles=min_obstacles, min_distance=min_distance, inner_dim=inner_dim
    )
    valid = solvable & complex_enough

    # 3. Rollout ALL levels (unsolvable ones produce zero reward; no significant waste)
    rng, rng_roll = jax.random.split(rng)
    rewards, values, dones = rollout_agent_on_levels(
        rng_roll, env, env_params, agent_params, network, levels, num_steps,
        deterministic=deterministic,
    )

    # 4. MaxMC regret (ACCEL-inspired proxy)
    max_returns = compute_max_returns(dones, rewards)
    regret = max_mc(dones, values, max_returns, incomplete_value=0.0)

    # 5. Adaptive penalty with edge-case guard for all-invalid generations
    any_solvable = solvable.sum() > 0
    any_valid = valid.sum() > 0
    max_observed = jnp.where(
        any_valid,
        jnp.max(jnp.where(valid, regret, -jnp.inf)),
        jnp.max(regret),  # fallback when all are filtered out
    )
    penalty = max_observed + 1.0

    # Negate regret for valid envs (CMA-ES minimizes), penalize filtered envs.
    fitness = jnp.where(valid, -regret, penalty)

    info = {
        'regret': regret,
        'solvable': solvable,
        'complex_enough': complex_enough,
        'valid': valid,
        'max_returns': max_returns,
        'any_solvable': any_solvable,
        'any_valid': any_valid,
    }
    return fitness, info
