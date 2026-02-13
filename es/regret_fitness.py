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
        hstate, pi, value = network.apply({'params': agent_params}, x, hstate)

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
