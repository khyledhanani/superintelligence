"""Agent evaluator: load trained agent and roll out on candidate mazes.

Provides the trajectory data needed by the decision gate.
Wraps the checkpoint loading and rollout logic from cross_evaluate.py
and plot_metrics_demo.py into a reusable class.

Usage (direct params — preferred for live training integration):
    evaluator = AgentEvaluator(apply_fn=train_state.apply_fn,
                               params=train_state.params,
                               env_params=env_params)
    trajectory = evaluator.evaluate_level(level)

Usage (checkpoint loading — backward compat for test_generator.py):
    evaluator = AgentEvaluator.from_checkpoint("gcs_artifacts/agent/cmaes_vae_beta2.0_seed0_198")
    trajectory = evaluator.evaluate_level(level)
"""

import os
import sys
import logging
from typing import Optional, List

import numpy as np
import jax
import jax.numpy as jnp

# Ensure project paths are available
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [
    os.path.join(_project_root, "src"),
    os.path.join(_project_root, "examples"),
    os.path.join(_project_root, "vae"),
    _project_root,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from jaxued.environments import Maze
from jaxued.environments.maze import Level
from jaxued.wrappers import AutoReplayWrapper
from maze_plr import ActorCritic

logger = logging.getLogger(__name__)


class AgentEvaluator:
    """Evaluates a trained agent on maze levels.

    Accepts live policy params directly (no file I/O). Call update_params()
    at each injection event to refresh the policy weights and invalidate the
    cached JIT rollout function.

    Usage (direct params — preferred for live training integration):
        evaluator = AgentEvaluator(apply_fn=train_state.apply_fn,
                                   params=train_state.params,
                                   env_params=env_params)
        evaluator.update_params(new_params)  # call at each injection event
        trajectory = evaluator.evaluate_level(level)

    Usage (checkpoint loading — backward compat for test_generator.py):
        evaluator = AgentEvaluator.from_checkpoint("path/to/checkpoint")
    """

    def __init__(
        self,
        apply_fn,
        params,
        env_params,
        num_steps: int = 250,
        seed: int = 42,
    ):
        """Construct AgentEvaluator with direct policy params.

        Args:
            apply_fn: Policy network apply function (train_state.apply_fn)
            params: Policy network parameters (train_state.params)
            env_params: Environment parameters for rollout
            num_steps: Max rollout steps per episode
            seed: Random seed for rollouts
        """
        self.apply_fn = apply_fn
        self.params = params
        self.env_params = env_params
        self.num_steps = num_steps
        self.rng = jax.random.PRNGKey(seed)

        # Build a clean eval env (no wrappers that interfere)
        self.eval_env = Maze(
            max_height=13, max_width=13,
            agent_view_size=5, normalize_obs=True,
        )
        # Override env_params with clean eval defaults
        self.env_params = self.eval_env.default_params

        self._rollout_fn_cache = {}  # {num_levels: jit_fn}

        logger.info("AgentEvaluator constructed")

    def update_params(self, params) -> None:
        """Refresh policy params. Call at each injection event.

        Forces rollout fn rebuild so next evaluate uses current policy.
        The old JIT-compiled function is discarded; JAX will retrace on
        next call to _build_rollout_fn (traces once per num_levels shape).

        Args:
            params: Updated policy network parameters (train_state.params)
        """
        self.params = params
        self._rollout_fn_cache = {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        checkpoint_step: int = -1,
        num_steps: int = 250,
        seed: int = 42,
    ) -> "AgentEvaluator":
        """Load agent from checkpoint directory (backward compat for test_generator.py).

        Args:
            checkpoint_dir: Path to agent checkpoint directory
            checkpoint_step: Specific checkpoint step (-1 for latest)
            num_steps: Max rollout steps per episode
            seed: Random seed for rollouts

        Returns:
            AgentEvaluator instance with params loaded from checkpoint

        Raises:
            RuntimeError: If checkpoint loading fails
        """
        from cross_evaluate import load_agent

        checkpoint_dir = os.path.abspath(checkpoint_dir)
        logger.info(f"Loading agent from {checkpoint_dir}...")
        train_state, config, env, env_params = load_agent(
            checkpoint_dir, checkpoint_step
        )

        if train_state is None:
            raise RuntimeError(f"Failed to load agent from {checkpoint_dir}")

        return cls(
            apply_fn=train_state.apply_fn,
            params=train_state.params,
            env_params=env_params,
            num_steps=num_steps,
            seed=seed,
        )

    def _build_rollout_fn(self, num_levels: int):
        """Build a JIT-compiled rollout function for a given batch size.

        Caches the compiled function by num_levels. When update_params() is called,
        self._rollout_fn is set to None, forcing a rebuild with fresh params on
        the next evaluate call.

        Args:
            num_levels: Number of levels in the evaluation batch
        """
        # Return cached fn if already compiled for this batch size
        if num_levels in self._rollout_fn_cache:
            return self._rollout_fn_cache[num_levels]

        eval_env = self.eval_env
        env_params = self.env_params
        # Snapshot params into local var — captured in JIT closure at trace time
        params = self.params
        apply_fn = self.apply_fn
        num_steps = self.num_steps

        @jax.jit
        def rollout(rng, levels):
            rng, rng_reset = jax.random.split(rng)
            init_obs, init_env_state = jax.vmap(
                eval_env.reset_to_level, (0, 0, None)
            )(jax.random.split(rng_reset, num_levels), levels, env_params)

            init_hstate = ActorCritic.initialize_carry((num_levels,))

            def step(carry, _):
                rng, hstate, obs, state, done = carry
                rng, rng_action, rng_step = jax.random.split(rng, 3)

                agent_pos = state.agent_pos

                # Network forward pass
                x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
                hstate, pi, value = apply_fn(
                    params, x, hstate
                )
                action = pi.sample(seed=rng_action).squeeze(0)
                value = value.squeeze(0)
                entropy = pi.entropy().squeeze(0)

                # Step environment
                next_obs, next_state, reward, next_done, _ = jax.vmap(
                    eval_env.step, in_axes=(0, 0, 0, None)
                )(jax.random.split(rng_step, num_levels), state, action, env_params)

                # Extract LSTM hidden state (carry_h, the output; not carry_c, the cell)
                hstate_h = hstate[1]  # (N, 256)

                carry = (rng, hstate, next_obs, next_state, next_done)
                return carry, (obs.image, action, reward, done, agent_pos, value, entropy, hstate_h)

            _, traj = jax.lax.scan(
                step,
                (rng, init_hstate, init_obs, init_env_state,
                 jnp.zeros(num_levels, dtype=bool)),
                None,
                length=num_steps,
            )
            return traj

        self._rollout_fn_cache[num_levels] = rollout
        return rollout

    def evaluate_level(self, level) -> dict:
        """Evaluate the agent on a single Level and return trajectory data.

        Args:
            level: A Level object (wall_map, goal_pos, agent_pos, agent_dir, width, height)

        Returns:
            Dict with keys: observations, positions, values, dones, actions, rewards
            All arrays have shape (T, ...) — single level, no batch dim.
        """
        # Use Level.stack to handle int fields (width, height) correctly
        batched_level = Level.stack([level])
        results = self._evaluate_batch(batched_level, 1)
        # Remove batch dimension
        return {k: v[:, 0] if v.ndim > 1 else v for k, v in results.items()}

    def evaluate_level_multi_rollout(self, level, n_rollouts: int = 100) -> dict:
        """Evaluate the agent on a single level with multiple rollouts.

        Runs n_rollouts independent rollouts (different RNG seeds) in a single
        batched call. Returns the trajectory with the highest episode return,
        giving a better estimate of the agent's best-case performance for
        regret computation.

        Args:
            level: A Level object
            n_rollouts: Number of independent rollouts (default: 100)

        Returns:
            Dict with keys matching evaluate_level(), plus:
                best_return: float — highest episode return across all rollouts
                solve_rate: float — fraction of rollouts that solved the level
                all_returns: (n_rollouts,) array of per-rollout returns
        """
        # Duplicate the level n_rollouts times
        batched_levels = Level.stack([level] * n_rollouts)
        results = self._evaluate_batch(batched_levels, n_rollouts)

        # Compute per-rollout episode returns
        rewards = results["rewards"]      # (T, N)
        dones = results["dones"]          # (T, N)

        # For each rollout, sum rewards up to first done
        returns = np.zeros(n_rollouts)
        solved = np.zeros(n_rollouts, dtype=bool)
        for i in range(n_rollouts):
            done_idx = np.where(dones[:, i])[0]
            if len(done_idx) > 0:
                end = done_idx[0] + 1
                solved[i] = True
            else:
                end = len(rewards)
            returns[i] = float(np.sum(rewards[:end, i]))

        # Pick the rollout with highest return
        best_idx = int(np.argmax(returns))
        traj = {
            "observations": results["observations"][:, best_idx],
            "positions": results["positions"][:, best_idx],
            "values": results["values"][:, best_idx],
            "dones": results["dones"][:, best_idx],
            "actions": results["actions"][:, best_idx],
            "rewards": results["rewards"][:, best_idx],
            "entropy": results["entropy"][:, best_idx],
            "hstates": results["hstates"][:, best_idx],
            "best_return": float(returns[best_idx]),
            "solve_rate": float(np.mean(solved)),
            "all_returns": returns,
        }
        return traj

    def evaluate_levels(self, levels: list) -> List[dict]:
        """Evaluate the agent on multiple Level objects.

        Args:
            levels: List of Level objects

        Returns:
            List of trajectory dicts, one per level
        """
        batched_levels = Level.stack(levels)
        n = len(levels)
        results = self._evaluate_batch(batched_levels, n)

        # Split batch into per-level dicts
        trajectories = []
        for i in range(n):
            traj = {
                "observations": results["observations"][:, i],
                "positions": results["positions"][:, i],
                "values": results["values"][:, i],
                "dones": results["dones"][:, i],
                "actions": results["actions"][:, i],
                "rewards": results["rewards"][:, i],
                "entropy": results["entropy"][:, i],
                "hstates": results["hstates"][:, i],
            }
            trajectories.append(traj)
        return trajectories

    def _evaluate_batch(self, batched_levels, num_levels: int) -> dict:
        """Run the agent on a batch of levels.

        Args:
            batched_levels: Batched Level pytree (each field has leading dim num_levels)
            num_levels: Number of levels in batch

        Returns:
            Dict with arrays of shape (T, num_levels, ...)
        """
        rollout_fn = self._build_rollout_fn(num_levels)

        self.rng, rng_eval = jax.random.split(self.rng)
        obs_images, actions, rewards, dones, positions, values, entropy, hstates = rollout_fn(
            rng_eval, batched_levels
        )

        return {
            "observations": np.asarray(obs_images),   # (T, N, H, W, C)
            "actions": np.asarray(actions),            # (T, N)
            "rewards": np.asarray(rewards),            # (T, N)
            "dones": np.asarray(dones),                # (T, N)
            "positions": np.asarray(positions),        # (T, N, 2)
            "values": np.asarray(values),              # (T, N)
            "entropy": np.asarray(entropy),            # (T, N)
            "hstates": np.asarray(hstates),            # (T, N, 256) LSTM hidden states
        }
