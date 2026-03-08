"""Co-evolutionary ACCEL: online level model retraining on the evolving replay buffer.

Three components co-evolve:
  1. Agent (PPO, unchanged from ACCEL)
  2. Replay buffer (levels evolving toward higher regret)
  3. Level model (MazeTaskAwareVAE retrained periodically on buffer contents)

The key insight: using a fixed offline VAE (trained on random mazes) means latent
perturbations are no better than random tile edits. By retraining the VAE online
on replay buffer contents, the latent space increasingly captures the structure of
high-regret mazes — making mutations semantically meaningful.

Inside-JIT: `jax.lax.cond` branches between latent mutation (coevo) and random
  mutation (minimax) depending on `level_model_initialized`.
Outside-JIT: retrain the VAE every `coevo_retrain_every` eval cycles.
"""

from __future__ import annotations

import json
import os
import sys
import time
from enum import IntEnum
from typing import Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax import core, struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState as BaseTrainState

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import distrax
import flax.linen as nn
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator, make_level_mutator_minimax
from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.level_sampler import LevelSampler
from jaxued.linen import ResetRNN
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from jaxued.wrappers import AutoReplayWrapper

from es.maze_ae import (
    MazeEncoder,
    MazeDecoder,
    MazeTaskAwareVAE,
    compute_structural_targets_from_grids,
    decode_maze_latents,
    encode_maze_levels,
    extract_maze_decoder_params,
    extract_maze_encoder_params,
    maze_level_to_grid,
)
from es.online_level_model import (
    build_dataset_with_reservoir,
    build_reservoir_dataset,
    compute_pca_from_latents,
    encode_batch_np,
    extract_buffer_grids,
    init_level_model_state,
    retrain_level_model,
)

# Re-use PPO utilities defined at module scope in maze_plr
from examples.maze_plr import (
    ActorCritic,
    compute_gae,
    compute_score,
    rollout_success_from_rewards,
    sample_trajectories_rnn,
    update_actor_critic_rnn,
    update_success_ema,
    setup_checkpointing,
    evaluate_rnn,
)


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------

class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1


class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # Logging counters
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    num_uphill: int
    num_compared: int
    uphill_last_fraction: chex.Array
    # Level batches for logging/visualization
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_inds: chex.Array = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    # Co-evolutionary level model state (all live in TrainState so JIT sees them)
    level_model_params: core.FrozenDict = struct.field(pytree_node=True)
    level_model_initialized: chex.Array = struct.field(pytree_node=True)   # bool scalar
    level_model_pca_eigvecs: chex.Array = struct.field(pytree_node=True)   # (D, D)
    level_model_pca_eigvals: chex.Array = struct.field(pytree_node=True)   # (D,)


def train_state_to_log_dict(train_state: TrainState, level_sampler: LevelSampler) -> dict:
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)
    log = {
        "level_sampler/size": sampler["size"],
        "level_sampler/episode_count": sampler["episode_count"],
        "level_sampler/max_score": sampler["scores"].max(),
        "level_sampler/weighted_score": (sampler["scores"] * level_sampler.level_weights(sampler)).sum(),
        "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
        "mutation/num_mutations": train_state.num_mutation_updates,
        "mutation/last_uphill_fraction": train_state.uphill_last_fraction,
        "mutation/cumulative_uphill_fraction": (
            train_state.num_uphill / jnp.maximum(train_state.num_compared, 1)
        ),
        "coevo/model_initialized": train_state.level_model_initialized.astype(jnp.int32),
    }
    return {
        "log": log,
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config=None, project: str = "COEVO_ACCEL"):
    tags = ["ACCEL", "COEVO"]
    run = wandb.init(config=config, project=project, group=config.get("run_name", "coevo"), tags=tags)
    config = wandb.config

    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    for prefix in ["solve_rate", "level_sampler", "agent", "return", "eval_ep_lengths",
                   "mutation", "coevo"]:
        wandb.define_metric(f"{prefix}/*", step_metric="num_updates")

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------
    env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    eval_env = env
    height, width = env.max_height, env.max_width

    sample_random_level = make_level_generator(height, width, config["n_walls"])
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    mutate_level = make_level_mutator_minimax(100)

    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config["topk_k"]},
        duplicate_check=config["buffer_duplicate_check"],
    )

    latent_dim: int = int(config["level_model_latent_dim"])

    # ------------------------------------------------------------------
    # Level model initialization (outside JIT)
    # ------------------------------------------------------------------
    rng_global = jax.random.PRNGKey(config["seed"])
    rng_global, rng_lm_init, rng_reservoir = jax.random.split(rng_global, 3)

    lm_cfg = dict(config)  # wandb.config supports dict(config)
    level_model_params_np, level_model_opt_state = init_level_model_state(
        rng_lm_init, lm_cfg, height, width
    )

    # Reservoir of random levels (used to prevent latent collapse during retraining)
    print(f"Building reservoir of {config['coevo_reservoir_size']} random levels...")
    reservoir = build_reservoir_dataset(
        sample_random_level, rng_reservoir, int(config["coevo_reservoir_size"]), height, width
    )
    print("Reservoir ready.")

    # ------------------------------------------------------------------
    # TrainState creation
    # ------------------------------------------------------------------
    @jax.jit
    def create_train_state(rng: chex.PRNGKey) -> TrainState:
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["num_minibatches"] * config["epoch_ppo"]))
                / config["num_updates"]
            )
            return config["lr"] * frac

        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0
            ),
            obs,
        )
        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        network = ActorCritic(env.action_space(env_params).n)
        network_params = network.init(rng, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )

        pholder_level = sample_random_level(jax.random.PRNGKey(0))
        sampler = level_sampler.initialize(
            pholder_level,
            {
                "max_return": -jnp.inf,
                "success_ema": jnp.array(0.0, dtype=jnp.float32),
                "success_obs_count": jnp.array(0.0, dtype=jnp.float32),
            },
        )
        pholder_batch = jax.tree_util.tree_map(
            lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0),
            pholder_level,
        )
        pholder_inds = jnp.zeros((config["num_train_envs"],), dtype=jnp.int32)

        # Freeze level model params as a pytree so JIT can trace them
        lm_frozen = core.freeze(level_model_params_np)
        D = latent_dim

        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            num_uphill=0,
            num_compared=0,
            uphill_last_fraction=jnp.array(0.0, dtype=jnp.float32),
            dr_last_level_batch=pholder_batch,
            replay_last_level_batch=pholder_batch,
            replay_last_level_inds=pholder_inds,
            mutation_last_level_batch=pholder_batch,
            level_model_params=lm_frozen,
            level_model_initialized=jnp.array(False),
            level_model_pca_eigvecs=jnp.eye(D, dtype=jnp.float32),
            level_model_pca_eigvals=jnp.ones(D, dtype=jnp.float32),
        )

    # ------------------------------------------------------------------
    # Training loop body (JIT-compiled)
    # ------------------------------------------------------------------
    num_train_envs: int = int(config["num_train_envs"])
    coevo_sigma: float = float(config["coevo_sigma"])
    coevo_mutation_mode: str = str(config["coevo_mutation_mode"])
    coevo_mutation_prob: float = float(config["coevo_mutation_prob"])
    decode_temperature: float = float(config["coevo_decode_temperature"])
    num_edits: int = int(config["num_edits"])

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """Single ACCEL update step: DR, replay, or mutate."""

        def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
            sampler = train_state.sampler
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, num_train_envs))
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, num_train_envs), new_levels, env_params
            )
            (
                (rng, train_state, _, _, _, last_value),
                (obs, actions, rewards, dones, log_probs, values, _),
            ) = sample_trajectories_rnn(
                rng, env, env_params, train_state,
                ActorCritic.initialize_carry((num_train_envs,)),
                init_obs, init_env_state, num_train_envs, config["num_steps"],
            )
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = compute_max_returns(dones, rewards)
            scores = compute_score(config, dones, values, max_returns, advantages)
            success = rollout_success_from_rewards(rewards)
            sampler, _ = level_sampler.insert_batch(
                sampler, new_levels, scores,
                {"max_return": max_returns, "success_ema": success, "success_obs_count": jnp.ones_like(success)},
            )
            (rng, train_state), losses = update_actor_critic_rnn(
                rng, train_state,
                ActorCritic.initialize_carry((num_train_envs,)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                num_train_envs, config["num_steps"], config["num_minibatches"],
                config["epoch_ppo"], config["clip_eps"], config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": new_levels.wall_map.sum() / num_train_envs,
                "coevo_used": jnp.array(0, dtype=jnp.int32),
            }
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=new_levels,
            )
            return (rng, train_state), metrics

        def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):
            sampler = train_state.sampler
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, num_train_envs)
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, num_train_envs), levels, env_params
            )
            (
                (rng, train_state, _, _, _, last_value),
                (obs, actions, rewards, dones, log_probs, values, _),
            ) = sample_trajectories_rnn(
                rng, env, env_params, train_state,
                ActorCritic.initialize_carry((num_train_envs,)),
                init_obs, init_env_state, num_train_envs, config["num_steps"],
            )
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            level_extras = level_sampler.get_levels_extra(sampler, level_inds)
            max_returns = jnp.maximum(level_extras["max_return"], compute_max_returns(dones, rewards))
            scores = compute_score(config, dones, values, max_returns, advantages)
            success = rollout_success_from_rewards(rewards)
            new_success_ema, new_success_obs_count = update_success_ema(
                level_extras["success_ema"],
                level_extras["success_obs_count"],
                success,
                float(config["success_ema_alpha"]),
            )
            sampler = level_sampler.update_batch(
                sampler, level_inds, scores,
                {"max_return": max_returns, "success_ema": new_success_ema, "success_obs_count": new_success_obs_count},
            )
            (rng, train_state), losses = update_actor_critic_rnn(
                rng, train_state,
                ActorCritic.initialize_carry((num_train_envs,)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                num_train_envs, config["num_steps"], config["num_minibatches"],
                config["epoch_ppo"], config["clip_eps"], config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=True,
            )
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": levels.wall_map.sum() / num_train_envs,
                "coevo_used": jnp.array(0, dtype=jnp.int32),
            }
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
                replay_last_level_inds=level_inds,
            )
            return (rng, train_state), metrics

        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            """ACCEL mutation step with inside-JIT coevo/random branch."""
            sampler = train_state.sampler
            parent_levels = train_state.replay_last_level_batch
            parent_level_inds = train_state.replay_last_level_inds
            parent_scores = train_state.sampler["scores"][parent_level_inds]

            rng, rng_decide, rng_mutate, rng_reset = jax.random.split(rng, 4)

            # --- Coevo mutation branch (inside JIT) ---
            def _coevo_mutate(rng_m: chex.PRNGKey) -> Level:
                parent_grids = jax.vmap(
                    lambda wm, gp, ap: maze_level_to_grid(wm, gp, ap, height, width)
                )(parent_levels.wall_map, parent_levels.goal_pos, parent_levels.agent_pos)

                enc_params = train_state.level_model_params["MazeEncoder_0"]
                mean, _ = MazeEncoder(latent_dim=latent_dim, variational=True).apply(
                    {"params": enc_params}, parent_grids
                )
                z = mean  # (B, D)

                rng_noise, rng_dec = jax.random.split(rng_m)
                noise = jax.random.normal(rng_noise, z.shape)

                if coevo_mutation_mode == "pca":
                    scaled = (
                        noise
                        * jnp.sqrt(jnp.maximum(train_state.level_model_pca_eigvals, 1e-8))
                        * coevo_sigma
                    )
                    # train_state.level_model_pca_eigvecs: (D, D), rows = PCs
                    # Project scaled perturbation back to original latent space
                    z_child = z + scaled @ train_state.level_model_pca_eigvecs  # (B, D)
                else:
                    # Gaussian (default)
                    z_child = z + coevo_sigma * noise

                dec_params = train_state.level_model_params["MazeDecoder_0"]
                return decode_maze_latents(
                    dec_params, z_child,
                    jax.random.split(rng_dec, num_train_envs),
                    temperature=decode_temperature,
                    height=height,
                    width=width,
                )

            # --- Random minimax mutation branch ---
            def _random_mutate(rng_m: chex.PRNGKey) -> Level:
                return jax.vmap(mutate_level, (0, 0, None))(
                    jax.random.split(rng_m, num_train_envs),
                    parent_levels,
                    num_edits,
                )

            # Decide: use coevo if model initialized AND random draw < coevo_mutation_prob
            use_coevo = train_state.level_model_initialized & (
                jax.random.uniform(rng_decide) < coevo_mutation_prob
            )
            child_levels = jax.lax.cond(use_coevo, _coevo_mutate, _random_mutate, rng_mutate)

            # Reset child levels and collect trajectories
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, num_train_envs), child_levels, env_params
            )
            (
                (rng, train_state, _, _, _, last_value),
                (obs, actions, rewards, dones, log_probs, values, _),
            ) = sample_trajectories_rnn(
                rng, env, env_params, train_state,
                ActorCritic.initialize_carry((num_train_envs,)),
                init_obs, init_env_state, num_train_envs, config["num_steps"],
            )
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = compute_max_returns(dones, rewards)
            scores = compute_score(config, dones, values, max_returns, advantages)
            success = rollout_success_from_rewards(rewards)
            sampler, _ = level_sampler.insert_batch(
                sampler, child_levels, scores,
                {"max_return": max_returns, "success_ema": success, "success_obs_count": jnp.ones_like(success)},
            )

            # Track uphill statistics
            batch_improved = jnp.sum(scores > parent_scores).astype(jnp.int32)
            batch_compared = jnp.array(num_train_envs, dtype=jnp.int32)
            uphill_frac = batch_improved.astype(jnp.float32) / jnp.maximum(
                batch_compared.astype(jnp.float32), 1.0
            )

            (rng, train_state), losses = update_actor_critic_rnn(
                rng, train_state,
                ActorCritic.initialize_carry((num_train_envs,)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                num_train_envs, config["num_steps"], config["num_minibatches"],
                config["epoch_ppo"], config["clip_eps"], config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": child_levels.wall_map.sum() / num_train_envs,
                "coevo_used": use_coevo.astype(jnp.int32),
            }
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_mutation_updates=train_state.num_mutation_updates + 1,
                num_uphill=train_state.num_uphill + batch_improved,
                num_compared=train_state.num_compared + batch_compared,
                uphill_last_fraction=uphill_frac,
                mutation_last_level_batch=child_levels,
            )
            return (rng, train_state), metrics

        rng, train_state = carry
        rng, rng_replay = jax.random.split(rng)

        # ACCEL branching: DR → replay → mutate → DR → ...
        s = train_state.update_state
        branch = (1 - s) * level_sampler.sample_replay_decision(train_state.sampler, rng_replay) + 2 * s

        return jax.lax.switch(
            branch,
            [on_new_levels, on_replay_levels, on_mutate_levels],
            rng, train_state,
        )

    # ------------------------------------------------------------------
    # Eval function
    # ------------------------------------------------------------------
    def eval(rng: chex.PRNGKey, train_state: TrainState):
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, num_levels), levels, env_params
        )
        states, rewards, episode_lengths = evaluate_rnn(
            rng, eval_env, env_params, train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs, init_env_state, env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths

    @jax.jit
    def train_and_eval_step(runner_state, _):
        (rng, train_state), metrics = jax.lax.scan(
            train_step, runner_state, None, config["eval_freq"]
        )

        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(
            jax.random.split(rng_eval, config["eval_num_attempts"]), train_state
        )

        eval_solve_rates = jnp.where(cum_rewards > 0, 1.0, 0.0).mean(axis=0)
        eval_returns = cum_rewards.mean(axis=0)

        states, episode_lengths = jax.tree_util.tree_map(
            lambda x: x[0], (states, episode_lengths)
        )
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params)
        frames = images.transpose(0, 1, 4, 2, 3)

        metrics["update_count"] = (
            train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates
        )
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"] = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        metrics["dr_levels"] = jax.vmap(env_renderer.render_level, (0, None))(
            train_state.dr_last_level_batch, env_params
        )
        metrics["replay_levels"] = jax.vmap(env_renderer.render_level, (0, None))(
            train_state.replay_last_level_batch, env_params
        )
        metrics["mutation_levels"] = jax.vmap(env_renderer.render_level, (0, None))(
            train_state.mutation_last_level_batch, env_params
        )
        highest_scoring_level = level_sampler.get_levels(
            train_state.sampler, train_state.sampler["scores"].argmax()
        )
        highest_weighted_level = level_sampler.get_levels(
            train_state.sampler, level_sampler.level_weights(train_state.sampler).argmax()
        )
        metrics["highest_scoring_level"] = env_renderer.render_level(highest_scoring_level, env_params)
        metrics["highest_weighted_level"] = env_renderer.render_level(highest_weighted_level, env_params)

        return (rng, train_state), metrics

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log_eval(stats, train_state_info, coevo_log: dict):
        print(f"Logging update: {stats['update_count']}")
        env_steps = stats["update_count"] * config["num_train_envs"] * config["num_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / max(stats["time_delta"], 1e-6),
        }

        solve_rates = stats["eval_solve_rates"]
        returns = stats["eval_returns"]
        log_dict.update({f"solve_rate/{name}": sr for name, sr in zip(config["eval_levels"], solve_rates)})
        log_dict["solve_rate/mean"] = solve_rates.mean()
        log_dict.update({f"return/{name}": r for name, r in zip(config["eval_levels"], returns)})
        log_dict["return/mean"] = returns.mean()
        log_dict["eval_ep_lengths/mean"] = stats["eval_ep_lengths"].mean()

        log_dict.update(train_state_info["log"])
        log_dict.update(coevo_log)

        log_dict["images/highest_scoring_level"] = wandb.Image(
            np.array(stats["highest_scoring_level"]), caption="Highest scoring level"
        )
        log_dict["images/highest_weighted_level"] = wandb.Image(
            np.array(stats["highest_weighted_level"]), caption="Highest weighted level"
        )
        for s in ["dr", "replay", "mutation"]:
            if train_state_info["info"][f"num_{s}_updates"] > 0:
                log_dict[f"images/{s}_levels"] = [
                    wandb.Image(np.array(img)) for img in stats[f"{s}_levels"]
                ]
        for i, level_name in enumerate(config["eval_levels"]):
            frames, ep_len = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
            frames = np.array(frames[: int(ep_len)])
            log_dict[f"animations/{level_name}"] = wandb.Video(frames, fps=4, format="gif")

        wandb.log(log_dict)

    # ------------------------------------------------------------------
    # Initialize and run
    # ------------------------------------------------------------------
    rng_global, rng_init, rng_train = jax.random.split(rng_global, 3)
    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)

    num_eval_steps = config["num_updates"] // config["eval_freq"]

    # Keep mutable references to the level model state (updated outside JIT)
    lm_params = level_model_params_np
    lm_opt_state = level_model_opt_state
    rng_retrain = jax.random.PRNGKey(config["seed"] + 1000)

    for eval_step in range(num_eval_steps):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics["time_delta"] = curr_time - start_time

        train_state_now = runner_state[1]
        total_updates = int(
            train_state_now.num_dr_updates
            + train_state_now.num_replay_updates
            + train_state_now.num_mutation_updates
        )
        buffer_size = int(train_state_now.sampler["size"])

        # -----------------------------------------------
        # Co-evolutionary retraining (outside JIT)
        # -----------------------------------------------
        coevo_log: dict = {}
        past_burn_in = total_updates >= int(config["coevo_burn_in"])
        should_retrain = (
            past_burn_in
            and config["use_coevo"]
            and (eval_step % int(config["coevo_retrain_every"]) == 0)
            and buffer_size >= int(config["coevo_min_buffer_size"])
        )

        if should_retrain:
            print(f"[CoEvo] Retraining level model at update {total_updates} "
                  f"(buffer_size={buffer_size})...")
            retrain_start = time.time()

            # Extract buffer grids as numpy
            buffer_grids = extract_buffer_grids(train_state_now.sampler, height, width)

            # Compute static structural targets (BFS-based, on-the-fly)
            static_targets = np.array(
                jax.jit(compute_structural_targets_from_grids)(jnp.array(buffer_grids)),
                dtype=np.float32,
            )
            p_ema_buf = np.array(
                train_state_now.sampler["levels_extra"]["success_ema"][:buffer_size],
                dtype=np.float32,
            )
            obs_count_buf = np.array(
                train_state_now.sampler["levels_extra"]["success_obs_count"][:buffer_size],
                dtype=np.float32,
            )

            # Mix buffer with reservoir to prevent latent collapse
            dataset = build_dataset_with_reservoir(
                buffer_grids, static_targets, p_ema_buf, obs_count_buf,
                reservoir,
                reservoir_fraction=float(config.get("coevo_reservoir_fraction", 0.3)),
            )

            # Retrain the level model
            rng_retrain, rng_rt = jax.random.split(rng_retrain)
            lm_params, lm_opt_state, retrain_metrics = retrain_level_model(
                lm_params, lm_opt_state, dataset, rng_rt, lm_cfg,
                n_steps=int(config["coevo_retrain_steps"]),
            )
            retrain_elapsed = time.time() - retrain_start
            print(f"[CoEvo] Retrain done in {retrain_elapsed:.1f}s. "
                  f"loss={retrain_metrics.get('total', float('nan')):.4f}")

            # Compute PCA over buffer latents for pca-scaled mutation
            latents_np = encode_batch_np(lm_params, buffer_grids, latent_dim, height, width)
            eigvecs, eigvals = compute_pca_from_latents(latents_np)

            # Compute buffer latent diversity (mean pairwise distance, sampled)
            if len(latents_np) > 1:
                n_sample = min(len(latents_np), 512)
                idx = np.random.choice(len(latents_np), size=n_sample, replace=False)
                sample_z = latents_np[idx]
                # Mean pairwise L2 distance (upper triangle only)
                diffs = sample_z[:, None, :] - sample_z[None, :, :]
                dists = np.sqrt(np.sum(diffs ** 2, axis=-1) + 1e-8)
                mask = np.triu(np.ones((n_sample, n_sample), dtype=bool), k=1)
                buffer_diversity = float(dists[mask].mean())
            else:
                buffer_diversity = 0.0

            # Push updated params + PCA back into TrainState
            train_state_now = runner_state[1].replace(
                level_model_params=core.freeze(lm_params),
                level_model_initialized=jnp.array(True),
                level_model_pca_eigvecs=jnp.array(eigvecs, dtype=jnp.float32),
                level_model_pca_eigvals=jnp.array(eigvals, dtype=jnp.float32),
            )
            runner_state = (runner_state[0], train_state_now)

            coevo_log = {
                "coevo/retrain_loss": retrain_metrics.get("total", 0.0),
                "coevo/retrain_recon": retrain_metrics.get("recon", 0.0),
                "coevo/retrain_kl": retrain_metrics.get("kl", 0.0),
                "coevo/retrain_static": retrain_metrics.get("static", 0.0),
                "coevo/retrain_time_s": retrain_elapsed,
                "coevo/buffer_latent_diversity": buffer_diversity,
                "coevo/buffer_size": buffer_size,
                "coevo/dataset_size": len(dataset["grids"]),
            }

        log_eval(
            metrics,
            train_state_to_log_dict(runner_state[1], level_sampler),
            coevo_log,
        )

        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()

    return runner_state[1]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="COEVO_ACCEL_TEST")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="train")

    # Checkpointing
    parser.add_argument("--checkpoint_save_interval", type=int, default=2)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)

    # Eval
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument("--eval_levels", nargs="+", default=[
        "SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped",
        "Labyrinth2", "StandardMaze", "StandardMaze2", "StandardMaze3",
        "NarrowBridge", "ForkDeception", "PerimeterRun", "SpiralPocket",
        "SymmetricCross", "ZigZagTunnel", "RoomKeyhole", "DualLoopChoice",
        "CentralChoke", "LongDetour", "DeadendFan", "OpenFieldBarriers",
        "ParallelCorridors", "CornerTrapEscape", "SnakeSpine",
    ])

    group = parser.add_argument_group("Training params")

    # PPO
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--max_grad_norm", type=float, default=0.5)
    mut = group.add_mutually_exclusive_group()
    mut.add_argument("--num_updates", type=int, default=30000)
    mut.add_argument("--num_env_steps", type=int, default=None)
    group.add_argument("--num_steps", type=int, default=256)
    group.add_argument("--num_train_envs", type=int, default=32)
    group.add_argument("--num_minibatches", type=int, default=1)
    group.add_argument("--gamma", type=float, default=0.995)
    group.add_argument("--epoch_ppo", type=int, default=5)
    group.add_argument("--clip_eps", type=float, default=0.2)
    group.add_argument("--gae_lambda", type=float, default=0.98)
    group.add_argument("--entropy_coeff", type=float, default=1e-3)
    group.add_argument("--critic_coeff", type=float, default=0.5)

    # PLR / ACCEL
    group.add_argument("--score_function", type=str, default="MaxMC", choices=["MaxMC", "pvl"])
    group.add_argument("--exploratory_grad_updates", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--level_buffer_capacity", type=int, default=4000)
    group.add_argument("--replay_prob", type=float, default=0.8)
    group.add_argument("--staleness_coeff", type=float, default=0.3)
    group.add_argument("--temperature", type=float, default=0.3)
    group.add_argument("--topk_k", type=int, default=4)
    group.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    group.add_argument("--prioritization", type=str, default="rank", choices=["rank", "topk"])
    group.add_argument("--buffer_duplicate_check", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--num_edits", type=int, default=5)
    group.add_argument("--success_ema_alpha", type=float, default=0.1)

    # Env
    group.add_argument("--agent_view_size", type=int, default=5)
    group.add_argument("--n_walls", type=int, default=25)

    # Co-evolutionary ACCEL
    group.add_argument("--use_coevo", action=argparse.BooleanOptionalAction, default=True,
                       help="Enable co-evolutionary level model retraining.")
    group.add_argument("--coevo_burn_in", type=int, default=500,
                       help="ACCEL updates before first level model retrain.")
    group.add_argument("--coevo_retrain_every", type=int, default=1,
                       help="Retrain every N eval cycles (each = eval_freq updates).")
    group.add_argument("--coevo_retrain_steps", type=int, default=1000,
                       help="Gradient steps per level model retrain.")
    group.add_argument("--coevo_min_buffer_size", type=int, default=50,
                       help="Min buffer entries before first retrain.")
    group.add_argument("--coevo_mutation_prob", type=float, default=0.8,
                       help="Prob of latent mutation vs random mutation after burn-in.")
    group.add_argument("--coevo_sigma", type=float, default=0.5,
                       help="Gaussian noise sigma in latent space.")
    group.add_argument("--coevo_mutation_mode", type=str, default="gaussian",
                       choices=["gaussian", "pca"],
                       help="Perturbation mode: gaussian or pca-scaled.")
    group.add_argument("--coevo_decode_temperature", type=float, default=0.25,
                       help="Gumbel-max temperature for VAE decoder.")
    group.add_argument("--coevo_reservoir_size", type=int, default=100,
                       help="Number of random mazes in the retraining reservoir.")
    group.add_argument("--coevo_reservoir_fraction", type=float, default=0.3,
                       help="Fraction of each retrain batch from the reservoir.")

    # Level model (online VAE) hyperparameters
    group.add_argument("--level_model_latent_dim", type=int, default=32,
                       help="Latent dimension of the online MazeTaskAwareVAE.")
    group.add_argument("--level_model_lr", type=float, default=3e-4)
    group.add_argument("--level_model_batch_size", type=int, default=64)
    group.add_argument("--level_model_beta", type=float, default=0.05,
                       help="KL weight (beta-VAE).")
    group.add_argument("--level_model_lambda_static", type=float, default=1.0)
    group.add_argument("--level_model_lambda_metric", type=float, default=0.2)
    group.add_argument("--level_model_lambda_valid", type=float, default=0.5)

    config = vars(parser.parse_args())

    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (
            config["num_train_envs"] * config["num_steps"]
        )

    if config["run_name"] is None:
        config["run_name"] = f"coevo_accel_seed{config['seed']}"

    if config["mode"] == "eval":
        os.environ["WANDB_MODE"] = "disabled"

    main(config, project=config["project"])
