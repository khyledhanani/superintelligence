import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from examples import maze_coevo_accel, maze_dr, maze_paired, maze_plr  # noqa: E402
from jaxued.environments import Maze  # noqa: E402
from jaxued.environments.maze.level import Level  # noqa: E402
from jaxued.environments.maze.perfect_maze import generate_perfect_maze_levels  # noqa: E402
from jaxued.environments.underspecified_env import EnvState, Observation, UnderspecifiedEnv  # noqa: E402


ALGO_SPECS = {
    "coevo_accel": {
        "module": maze_coevo_accel,
        "params_path": ("params",),
    },
    "dr": {
        "module": maze_dr,
        "params_path": ("params",),
    },
    "plr": {
        "module": maze_plr,
        "params_path": ("params",),
    },
    "paired": {
        "module": maze_paired,
        "params_path": ("pro_train_state", "params"),
    },
}


def _tree_get(tree: dict[str, Any], path: tuple[str, ...]) -> Any:
    value = tree
    for key in path:
        value = value[key]
    return value


def _normalize_params(params: Any) -> Any:
    if isinstance(params, dict) and set(params.keys()) == {"params"}:
        return params["params"]
    return params


def _restore_checkpoint(models_dir: str, step: int) -> Any:
    last_error: Exception | None = None
    for builder in (
        lambda: ocp.CheckpointManager(models_dir, item_handlers=ocp.StandardCheckpointHandler()),
        lambda: ocp.CheckpointManager(models_dir, ocp.PyTreeCheckpointer()),
    ):
        try:
            manager = builder()
            restore_step = manager.latest_step() if step < 0 else step
            if restore_step is None:
                raise ValueError(f"No checkpoints found in {models_dir}.")
            return manager.restore(restore_step), restore_step
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise RuntimeError(f"Failed to restore checkpoint from {models_dir}: {last_error}") from last_error


def _load_policy(algo: str, checkpoint_directory: str, checkpoint_step: int, num_envs: int, obs: Observation):
    if algo not in ALGO_SPECS:
        raise ValueError(f"Unsupported algo {algo}. Expected one of {sorted(ALGO_SPECS)}.")

    spec = ALGO_SPECS[algo]
    checkpoint_directory = os.path.abspath(checkpoint_directory)
    config_path = os.path.join(checkpoint_directory, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    network = spec["module"].ActorCritic(7)
    init_dones = jnp.zeros((1, num_envs), dtype=jnp.bool_)
    init_obs = jax.tree_util.tree_map(lambda x: x[None, ...], obs)
    params_template = network.init(
        jax.random.PRNGKey(0),
        (init_obs, init_dones),
        spec["module"].ActorCritic.initialize_carry((num_envs,)),
    )

    restored, restore_step = _restore_checkpoint(os.path.join(checkpoint_directory, "models"), checkpoint_step)
    params = _normalize_params(_tree_get(restored, spec["params_path"]))

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params_template["params"],
        tx=optax.identity(),
    ).replace(params=params)
    return config, train_state, restore_step


def evaluate_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
    deterministic: bool,
):
    def eval_step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)
        x = jax.tree_util.tree_map(lambda value: value[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn({"params": train_state.params}, x, hstate)
        sampled_action = pi.mode() if deterministic else pi.sample(seed=rng_action)
        action = sampled_action.squeeze(0)
        next_obs, next_state, reward, next_done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(rng_step, action.shape[0]),
            state,
            action,
            env_params,
        )
        next_mask = mask & (~done)
        episode_length = episode_length + next_mask
        return (rng, hstate, next_obs, next_state, next_done, next_mask, episode_length), (state, reward)

    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        eval_step,
        (
            rng,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros((init_env_state.agent_dir.shape[0],), dtype=jnp.bool_),
            jnp.ones((init_env_state.agent_dir.shape[0],), dtype=jnp.bool_),
            jnp.zeros((init_env_state.agent_dir.shape[0],), dtype=jnp.uint32),
        ),
        None,
        length=max_episode_length,
    )
    return states, rewards, episode_lengths


def run_eval(args):
    args.checkpoint_directory = os.path.abspath(args.checkpoint_directory)
    args.output_dir = os.path.abspath(args.output_dir)
    maze_seeds = [args.seed + i for i in range(args.num_mazes)]
    levels_list = generate_perfect_maze_levels(size=args.maze_size, seeds=maze_seeds)
    levels = Level.stack(levels_list)

    env = Maze(
        max_height=args.maze_size,
        max_width=args.maze_size,
        agent_view_size=args.agent_view_size,
        normalize_obs=True,
    )
    env_params = env.default_params.replace(max_steps_in_episode=args.max_steps)

    rng_reset = jax.random.PRNGKey(args.seed)
    init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(rng_reset, args.num_mazes),
        levels,
        env_params,
    )

    config, train_state, restore_step = _load_policy(
        algo=args.algo,
        checkpoint_directory=args.checkpoint_directory,
        checkpoint_step=args.checkpoint_step,
        num_envs=args.num_mazes,
        obs=init_obs,
    )

    rng_eval = jax.random.PRNGKey(args.seed + 10_000)
    states, rewards, episode_lengths = evaluate_rnn(
        rng=rng_eval,
        env=env,
        env_params=env_params,
        train_state=train_state,
        init_hstate=ALGO_SPECS[args.algo]["module"].ActorCritic.initialize_carry((args.num_mazes,)),
        init_obs=init_obs,
        init_env_state=init_env_state,
        max_episode_length=args.max_steps,
        deterministic=args.deterministic,
    )

    mask = jnp.arange(args.max_steps)[:, None] < episode_lengths[None, :]
    returns = (rewards * mask).sum(axis=0)
    solved = returns > 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "algo": args.algo,
        "checkpoint_directory": args.checkpoint_directory,
        "checkpoint_step": int(restore_step),
        "maze_size": args.maze_size,
        "num_mazes": args.num_mazes,
        "max_steps": args.max_steps,
        "deterministic": bool(args.deterministic),
        "solve_rate": float(np.asarray(solved, dtype=np.float32).mean()),
        "mean_return": float(np.asarray(returns).mean()),
        "mean_episode_length": float(np.asarray(episode_lengths).mean()),
        "agent_view_size": int(args.agent_view_size),
        "train_run_name": config.get("run_name"),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        output_dir / "results.npz",
        states=np.asarray(states),
        rewards=np.asarray(rewards),
        returns=np.asarray(returns),
        solved=np.asarray(solved),
        episode_lengths=np.asarray(episode_lengths),
        maze_seeds=np.asarray(maze_seeds, dtype=np.int32),
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=sorted(ALGO_SPECS), required=True)
    parser.add_argument("--checkpoint_directory", type=str, required=True)
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--maze_size", type=int, default=51)
    parser.add_argument("--num_mazes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=5202)
    parser.add_argument("--agent_view_size", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    summary = run_eval(parse_args())
    print(json.dumps(summary, indent=2))
