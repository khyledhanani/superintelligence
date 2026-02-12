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
                   num_steps=256, deterministic=True):
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

    Returns:
        fitness: (pop_size,) — negated regret (lower = better for CMA-ES).
        info: Dict with 'regret', 'solvable', 'max_returns', 'any_solvable'.
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

    # 3. Rollout ALL levels (unsolvable ones produce zero reward; no significant waste)
    rng, rng_roll = jax.random.split(rng)
    rewards, values, dones = rollout_agent_on_levels(
        rng_roll, env, env_params, agent_params, network, levels, num_steps,
        deterministic=deterministic,
    )

    # 4. MaxMC regret (ACCEL-inspired proxy)
    max_returns = compute_max_returns(dones, rewards)
    regret = max_mc(dones, values, max_returns, incomplete_value=0.0)

    # 5. Adaptive penalty with edge-case guard for all-unsolvable generations
    any_solvable = solvable.sum() > 0
    max_observed = jnp.where(
        any_solvable,
        jnp.max(jnp.where(solvable, regret, -jnp.inf)),
        1.0,  # default if all unsolvable
    )
    penalty = max_observed + 1.0

    # Negate regret for solvable (CMA-ES minimizes), penalize unsolvable
    fitness = jnp.where(solvable, -regret, penalty)

    info = {
        'regret': regret,
        'solvable': solvable,
        'max_returns': max_returns,
        'any_solvable': any_solvable,
    }
    return fitness, info
