"""Agent evaluator: load trained agent and roll out on candidate mazes.

Provides the trajectory data needed by the decision gate.
Wraps the checkpoint loading and rollout logic from cross_evaluate.py
and plot_metrics_demo.py into a reusable class.
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
from cross_evaluate import load_agent

logger = logging.getLogger(__name__)


class AgentEvaluator:
    """Loads a trained agent checkpoint and evaluates it on maze levels.

    Usage:
        evaluator = AgentEvaluator("gcs_artifacts/agent/cmaes_vae_beta2.0_seed0_198")
        trajectory = evaluator.evaluate_level(level)
        trajectories = evaluator.evaluate_levels([level1, level2, ...])
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_step: int = -1,
        num_steps: int = 250,
        seed: int = 42,
    ):
        """Load agent from checkpoint.

        Args:
            checkpoint_dir: Path to agent checkpoint directory
            checkpoint_step: Specific checkpoint step (-1 for latest)
            num_steps: Max rollout steps per episode
            seed: Random seed for rollouts
        """
        self.num_steps = num_steps
        self.rng = jax.random.PRNGKey(seed)

        # Orbax requires absolute paths
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        logger.info(f"Loading agent from {checkpoint_dir}...")
        self.train_state, self.config, self.env, self.env_params = load_agent(
            checkpoint_dir, checkpoint_step=checkpoint_step
        )

        if self.train_state is None:
            raise RuntimeError(f"Failed to load agent from {checkpoint_dir}")

        # Build a clean eval env (no wrappers that interfere)
        self.eval_env = Maze(
            max_height=13, max_width=13,
            agent_view_size=5, normalize_obs=True,
        )
        self.env_params = self.eval_env.default_params

        logger.info("Agent loaded successfully")
        self._rollout_fn = None

    def _build_rollout_fn(self, num_levels: int):
        """Build a JIT-compiled rollout function for a given batch size."""
        eval_env = self.eval_env
        env_params = self.env_params
        train_state = self.train_state
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
                hstate, pi, value = train_state.apply_fn(
                    train_state.params, x, hstate
                )
                action = pi.sample(seed=rng_action).squeeze(0)
                value = value.squeeze(0)
                entropy = pi.entropy().squeeze(0)

                # Step environment
                next_obs, next_state, reward, next_done, _ = jax.vmap(
                    eval_env.step, in_axes=(0, 0, 0, None)
                )(jax.random.split(rng_step, num_levels), state, action, env_params)

                carry = (rng, hstate, next_obs, next_state, next_done)
                return carry, (obs.image, action, reward, done, agent_pos, value, entropy)

            _, traj = jax.lax.scan(
                step,
                (rng, init_hstate, init_obs, init_env_state,
                 jnp.zeros(num_levels, dtype=bool)),
                None,
                length=num_steps,
            )
            return traj

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
        obs_images, actions, rewards, dones, positions, values, entropy = rollout_fn(
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
        }
