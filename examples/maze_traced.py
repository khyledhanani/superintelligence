"""TRACED for maze ACCEL, with an optional checkpoint-420 PCA editor."""

from __future__ import annotations

import json
import os
import sys
import time
from enum import IntEnum
from typing import Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState as BaseTrainState
from gymnax.environments import spaces

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from es.cluttr_encoder import extract_encoder_params, encode_levels_to_latents
from es.env_bridge import cluttr_sequence_to_level, level_to_cluttr_sequence
from es.online_level_model import compute_pca_from_latents
from es.traced_transition_model import ImageTransitionPredictionModel
from es.vae_decoder import (
    decode_latent_to_env,
    extract_decoder_params,
    load_vae_params,
    repair_cluttr_sequence,
)
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator, make_level_mutator_minimax
from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.level_sampler import LevelSampler
from jaxued.linen import ResetRNN
from jaxued.utils import compute_max_mean_returns_epcount, compute_max_returns
from jaxued.wrappers import AutoReplayWrapper


class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1


class TrainState(BaseTrainState):
    sampler: dict = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    num_uphill: int
    num_compared: int
    uphill_last_fraction: chex.Array
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_inds: chex.Array = struct.field(pytree_node=True)
    replay_last_task_difficulty: chex.Array = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    traced_prev_replay_inds: chex.Array = struct.field(pytree_node=True)
    traced_prev_replay_count: int
    traced_prev_replay_valid: chex.Array = struct.field(pytree_node=True)
    traced_seed_scores: chex.Array = struct.field(pytree_node=True)
    traced_regret_scores: chex.Array = struct.field(pytree_node=True)
    traced_current_regret_score: chex.Array = struct.field(pytree_node=True)
    traced_previous_regret_score: chex.Array = struct.field(pytree_node=True)
    traced_regret_diff_changed: chex.Array = struct.field(pytree_node=True)
    traced_regret_seen: chex.Array = struct.field(pytree_node=True)
    traced_partial_seed_scores: chex.Array = struct.field(pytree_node=True)
    traced_partial_seed_max_scores: chex.Array = struct.field(pytree_node=True)
    traced_partial_seed_steps: chex.Array = struct.field(pytree_node=True)
    transition_params: dict = struct.field(pytree_node=True)
    transition_opt_state: optax.OptState = struct.field(pytree_node=True)
    transition_last_loss: chex.Array
    transition_last_atpl_mean: chex.Array
    traced_pca_eigvecs: chex.Array = struct.field(pytree_node=True)
    traced_pca_eigvals: chex.Array = struct.field(pytree_node=True)


def compute_score(config, dones, values, max_returns, advantages):
    if config["score_function"] != "pvl":
        raise ValueError("TRACED requires --score_function pvl.")
    from jaxued.utils import positive_value_loss

    return positive_value_loss(dones, advantages)


def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values


def rollout_success_from_rewards(rewards: chex.Array) -> chex.Array:
    return (jnp.max(rewards, axis=0) > 0.0).astype(jnp.float32)


def update_success_ema(
    old_ema: chex.Array,
    old_count: chex.Array,
    success: chex.Array,
    alpha: float,
) -> tuple[chex.Array, chex.Array]:
    new_ema = (1.0 - alpha) * old_ema + alpha * success
    new_count = old_count + 1.0
    return new_ema, new_count


class ActorCritic(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(
            5,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="scalar_embed",
        )(dir_embed)

        embedding = jnp.append(img_embed, dir_embed, axis=-1)
        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))(
            (embedding, dones), initial_carry=hidden
        )

        actor_mean = nn.Dense(
            32,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="actor0",
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor1",
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            32,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="critic0",
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic1",
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, 256)
        )


def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: tuple,
    num_envs: int,
    num_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool = True,
):
    def update_epoch(carry, _):
        def update_minibatch(train_state, batch_info):
            init_hstate, obs, actions, dones, log_probs, values, targets, advantages = batch_info

            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(params, (obs, dones), init_hstate)
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (
                    -jnp.minimum(
                        ratio * adv,
                        jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv,
                    )
                ).mean()

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum(
                    (values_pred - targets) ** 2,
                    (values_pred_clipped - targets) ** 2,
                ).mean()
                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy
                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0).reshape(n_minibatch, -1, *x.shape[1:]),
                init_hstate,
            ),
            *jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)


def sample_trajectories_rnn_with_next_obs(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
):
    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x_: x_[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value = value.squeeze(0)
        action = action.squeeze(0)
        log_prob = log_prob.squeeze(0)

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, next_obs, info)

    (rng, train_state, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_envs, dtype=bool),
        ),
        None,
        length=max_episode_length,
    )

    x = jax.tree_util.tree_map(lambda x_: x_[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)

    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj


def sample_replay_trajectories_rnn_with_next_obs(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    level_sampler: LevelSampler,
    sampler: dict,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    init_levels: Level,
    init_level_inds: chex.Array,
    num_envs: int,
    max_episode_length: int,
):
    def sample_step(carry, _):
        rng, sampler, train_state, hstate, obs, env_state, levels, level_inds, last_done = carry
        rng, rng_action, rng_step, rng_resample, rng_reset = jax.random.split(rng, 5)

        x = jax.tree_util.tree_map(lambda x_: x_[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value = value.squeeze(0)
        action = action.squeeze(0)
        log_prob = log_prob.squeeze(0)

        next_obs, next_env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        step_level_inds = level_inds
        rng_resample_split = jax.random.split(rng_resample, num_envs)
        rng_reset_split = jax.random.split(rng_reset, num_envs)

        def maybe_resample_one(carry_i, env_idx):
            sampler_i, obs_i, env_state_i, levels_i, level_inds_i = carry_i
            done_i = done[env_idx]

            def on_done(data):
                sampler_d, obs_d, env_state_d, levels_d, level_inds_d = data
                sampler_d, (new_level_idx, new_level) = level_sampler.sample_replay_level(
                    sampler_d, rng_resample_split[env_idx]
                )
                new_obs, new_env_state = env.reset_to_level(rng_reset_split[env_idx], new_level, env_params)
                obs_d = jax.tree_util.tree_map(lambda arr, val: arr.at[env_idx].set(val), obs_d, new_obs)
                env_state_d = jax.tree_util.tree_map(
                    lambda arr, val: arr.at[env_idx].set(val), env_state_d, new_env_state
                )
                levels_d = jax.tree_util.tree_map(lambda arr, val: arr.at[env_idx].set(val), levels_d, new_level)
                level_inds_d = level_inds_d.at[env_idx].set(new_level_idx)
                return sampler_d, obs_d, env_state_d, levels_d, level_inds_d

            return (
                jax.lax.cond(
                    done_i,
                    on_done,
                    lambda data: data,
                    (sampler_i, obs_i, env_state_i, levels_i, level_inds_i),
                ),
                None,
            )

        (sampler, next_obs, next_env_state, levels, level_inds), _ = jax.lax.scan(
            maybe_resample_one,
            (sampler, next_obs, next_env_state, levels, level_inds),
            jnp.arange(num_envs),
            length=num_envs,
        )

        carry = (rng, sampler, train_state, hstate, next_obs, next_env_state, levels, level_inds, done)
        return carry, (obs, action, reward, done, log_prob, value, next_obs, info, step_level_inds)

    (
        rng,
        sampler,
        train_state,
        hstate,
        last_obs,
        last_env_state,
        last_levels,
        last_level_inds,
        last_done,
    ), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            sampler,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            init_levels,
            init_level_inds,
            jnp.zeros(num_envs, dtype=bool),
        ),
        None,
        length=max_episode_length,
    )

    x = jax.tree_util.tree_map(lambda x_: x_[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)

    return (
        rng,
        sampler,
        train_state,
        hstate,
        last_obs,
        last_env_state,
        last_levels,
        last_level_inds,
        last_value.squeeze(0),
    ), traj


def evaluate_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
):
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]

    def step(carry, _):
        rng, train_state, hstate, obs, env_state, done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)
        x = jax.tree_util.tree_map(lambda x_: x_[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.mode().squeeze(0)
        next_obs, env_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), env_state, action, env_params)
        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (env_state, reward, done)

    (_, _, _, _, _, _), (states, rewards, dones) = jax.lax.scan(
        step,
        (rng, train_state, init_hstate, init_obs, init_env_state, jnp.zeros((num_levels,), dtype=bool)),
        None,
        length=max_episode_length,
    )

    first_done = jnp.argmax(dones, axis=0)
    has_done = jnp.any(dones, axis=0)
    episode_lengths = jnp.where(has_done, first_done + 1, max_episode_length)
    return states, rewards, episode_lengths


def setup_checkpointing(config: dict, train_state: TrainState) -> ocp.CheckpointManager:
    plain_config = dict(config)
    overall_save_dir = os.path.join(
        os.getcwd(),
        "checkpoints",
        f"{plain_config['run_name']}",
        str(plain_config["seed"]),
    )
    os.makedirs(overall_save_dir, exist_ok=True)
    with open(os.path.join(overall_save_dir, "config.json"), "w") as f:
        json.dump(plain_config, f, indent=2)

    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, "models"),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=config["max_number_of_checkpoints"],
            save_interval_steps=1,
            create=True,
        ),
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    return checkpoint_manager


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
        "mutation/cumulative_uphill_fraction": train_state.num_uphill / jnp.maximum(train_state.num_compared, 1),
        "traced/transition_last_loss": train_state.transition_last_loss,
        "traced/transition_last_atpl_mean": train_state.transition_last_atpl_mean,
    }
    return {
        "log": log,
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        },
    }


def main(config=None, project: str = "MAZE_TRACED"):
    run = wandb.init(config=config, project=project, group=config.get("run_name", "traced"), tags=["TRACED"])
    config = wandb.config

    if config["score_function"] != "pvl":
        raise ValueError("TRACED requires --score_function pvl.")
    if config["traced_use_pca_mutation"] and config["n_walls"] > 50:
        raise ValueError("checkpoint_420000 CLUTTR VAE only supports mazes with <= 50 walls.")

    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    for prefix in ["solve_rate", "level_sampler", "return", "eval_ep_lengths", "mutation", "traced"]:
        wandb.define_metric(f"{prefix}/*", step_metric="num_updates")

    env = Maze(
        max_height=13,
        max_width=13,
        agent_view_size=config["agent_view_size"],
        normalize_obs=True,
    )
    eval_env = env
    sample_random_level = make_level_generator(env.max_height, env.max_width, config["n_walls"])
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

    action_space = env.action_space(env_params)
    if not isinstance(action_space, spaces.Discrete):
        raise ValueError("maze_traced.py currently expects a discrete action space.")
    action_dim = int(action_space.n)
    transition_action_dim = 1

    rng = jax.random.PRNGKey(config["seed"])
    rng, rng_dummy, rng_tm_init = jax.random.split(rng, 3)
    dummy_level = sample_random_level(rng_dummy)
    dummy_obs, _ = env.reset_to_level(rng_dummy, dummy_level, env_params)
    image_shape = tuple(dummy_obs.image.shape)

    transition_model = ImageTransitionPredictionModel(
        action_dim=transition_action_dim,
        hidden_dim=int(config["traced_transition_hidden_dim"]),
    )
    dummy_images = jnp.zeros((1, 1, *image_shape), dtype=jnp.float32)
    dummy_actions = jnp.zeros((1, 1, transition_action_dim), dtype=jnp.float32)
    transition_params = transition_model.init(rng_tm_init, dummy_images, dummy_actions)["params"]
    transition_tx = optax.chain(
        optax.clip_by_global_norm(float(config["max_grad_norm"])),
        optax.adamw(
            learning_rate=float(config["traced_transition_lr"]),
            weight_decay=float(config["traced_transition_weight_decay"]),
        ),
    )
    transition_opt_state = transition_tx.init(transition_params)

    @jax.jit
    def update_transition_model(
        params,
        opt_state,
        obs_images,
        actions_scalar,
        next_images,
        valid_mask,
    ):
        def loss_fn(p):
            pred_next = transition_model.apply({"params": p}, obs_images, actions_scalar)
            loss_raw = jnp.abs(pred_next - next_images)
            loss_per_pair = loss_raw.mean(axis=(-1, -2, -3))
            loss = (loss_per_pair * valid_mask).sum() / jnp.maximum(valid_mask.sum(), 1.0)
            return loss, loss_per_pair

        (loss, loss_per_pair), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = transition_tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, loss_per_pair

    encoder_params = None
    decoder_params = None
    traced_pca_latent_dim = 1
    if config["traced_use_pca_mutation"]:
        traced_ckpt = config["traced_vae_checkpoint"]
        if traced_ckpt is None:
            traced_ckpt = os.path.join(ROOT_DIR, "vae", "model", "checkpoint_420000.pkl")
        elif not os.path.isabs(traced_ckpt):
            traced_ckpt = os.path.abspath(os.path.join(os.getcwd(), traced_ckpt))
        full_vae_params = load_vae_params(traced_ckpt)
        encoder_params = extract_encoder_params(full_vae_params)
        decoder_params = extract_decoder_params(full_vae_params)
        traced_pca_latent_dim = int(encoder_params["mean_layer"]["kernel"].shape[-1])

    @jax.jit
    def create_train_state(rng_init: chex.PRNGKey) -> TrainState:
        def linear_schedule(count):
            frac = 1.0 - (count // (config["num_minibatches"] * config["epoch_ppo"])) / config["num_updates"]
            return config["lr"] * frac

        obs, _ = env.reset_to_level(rng_init, sample_random_level(rng_init), env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...],
                256,
                axis=0,
            ),
            obs,
        )
        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        network = ActorCritic(action_dim)
        network_params = network.init(
            rng_init,
            init_x,
            ActorCritic.initialize_carry((config["num_train_envs"],)),
        )
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
                "traced_pvl": jnp.array(0.0, dtype=jnp.float32),
                "traced_atpl": jnp.array(0.0, dtype=jnp.float32),
                "traced_task_difficulty": jnp.array(0.0, dtype=jnp.float32),
                "traced_prev_task_difficulty": jnp.array(0.0, dtype=jnp.float32),
                "traced_colearnability": jnp.array(0.0, dtype=jnp.float32),
                "traced_priority": jnp.array(0.0, dtype=jnp.float32),
            },
        )
        pholder_level_batch = jax.tree_util.tree_map(
            lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0),
            pholder_level,
        )
        pholder_inds = jnp.zeros((config["num_train_envs"],), dtype=jnp.int32)
        pholder_prev_inds = jnp.zeros((config["level_buffer_capacity"],), dtype=jnp.int32)
        pholder_seed_scores = jnp.zeros((config["level_buffer_capacity"],), dtype=jnp.float32)
        pholder_regret_flags = jnp.zeros((config["level_buffer_capacity"],), dtype=jnp.int32)
        pholder_regret_seen = jnp.zeros((config["level_buffer_capacity"],), dtype=bool)
        pholder_partial_scores = jnp.zeros(
            (config["num_train_envs"], config["level_buffer_capacity"]), dtype=jnp.float32
        )
        pholder_partial_max = jnp.full(
            (config["num_train_envs"], config["level_buffer_capacity"]),
            -jnp.inf,
            dtype=jnp.float32,
        )
        pholder_partial_steps = jnp.zeros(
            (config["num_train_envs"], config["level_buffer_capacity"]), dtype=jnp.int32
        )

        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=UpdateState.DR,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            num_uphill=0,
            num_compared=0,
            uphill_last_fraction=jnp.array(0.0, dtype=jnp.float32),
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            replay_last_level_inds=pholder_inds,
            replay_last_task_difficulty=jnp.zeros((config["num_train_envs"],), dtype=jnp.float32),
            mutation_last_level_batch=pholder_level_batch,
            traced_prev_replay_inds=pholder_prev_inds,
            traced_prev_replay_count=0,
            traced_prev_replay_valid=jnp.array(False),
            traced_seed_scores=pholder_seed_scores,
            traced_regret_scores=pholder_seed_scores,
            traced_current_regret_score=pholder_seed_scores,
            traced_previous_regret_score=pholder_seed_scores,
            traced_regret_diff_changed=pholder_regret_flags,
            traced_regret_seen=pholder_regret_seen,
            traced_partial_seed_scores=pholder_partial_scores,
            traced_partial_seed_max_scores=pholder_partial_max,
            traced_partial_seed_steps=pholder_partial_steps,
            transition_params=transition_params,
            transition_opt_state=transition_opt_state,
            transition_last_loss=jnp.array(0.0, dtype=jnp.float32),
            transition_last_atpl_mean=jnp.array(0.0, dtype=jnp.float32),
            traced_pca_eigvecs=jnp.eye(traced_pca_latent_dim, dtype=jnp.float32),
            traced_pca_eigvals=jnp.ones((traced_pca_latent_dim,), dtype=jnp.float32),
        )

    num_train_envs = int(config["num_train_envs"])
    traced_num_mutation_parents = max(1, int(config["traced_num_mutation_parents"]))
    traced_transition_prob_weight = float(config["traced_transition_prob_weight"])
    traced_colearnability_weight = float(config["traced_colearnability_weight"])
    traced_score_alpha = float(config["traced_score_alpha"])
    traced_max_score_coef = float(config["traced_max_score_coef"])

    def build_traced_extras(max_returns, success, pvl, atpl, task_difficulty, success_obs_count):
        return {
            "max_return": max_returns,
            "success_ema": success,
            "success_obs_count": success_obs_count,
            "traced_pvl": pvl,
            "traced_atpl": atpl,
            "traced_task_difficulty": task_difficulty,
            "traced_prev_task_difficulty": task_difficulty,
            "traced_colearnability": jnp.zeros_like(task_difficulty),
            "traced_priority": task_difficulty,
        }

    def prepare_transition_pairs(obs: Observation, actions: chex.Array, dones: chex.Array):
        obs_bt = jnp.swapaxes(obs.image, 0, 1).astype(jnp.float32)  # (B, T, H, W, C)
        actions_bt = jnp.swapaxes(actions, 0, 1).astype(jnp.float32)  # (B, T)
        dones_bt = jnp.swapaxes(dones, 0, 1)  # (B, T)
        pair_obs = obs_bt[:, :-1]
        pair_next_obs = obs_bt[:, 1:]
        pair_actions = actions_bt[:, :-1, None]
        pair_valid = (~dones_bt[:, :-1]).astype(jnp.float32)
        return pair_obs, pair_actions, pair_next_obs, pair_valid

    def mask_to_compact_indices(mask: chex.Array) -> tuple[chex.Array, chex.Array]:
        idx = jnp.arange(mask.shape[0], dtype=jnp.int32)
        sentinel = jnp.array(mask.shape[0], dtype=jnp.int32)
        sortable = jnp.where(mask, idx, sentinel)
        sorted_idx = jnp.sort(sortable)
        count = mask.astype(jnp.int32).sum()
        compact = jnp.where(
            jnp.arange(mask.shape[0], dtype=jnp.int32) < count,
            sorted_idx,
            jnp.zeros_like(sorted_idx),
        )
        return compact, count

    def sync_sampler_scores_with_traced(sampler: dict, seed_scores: chex.Array) -> dict:
        valid = jnp.arange(level_sampler.capacity, dtype=jnp.int32) < sampler["size"]
        synced_scores = jnp.where(valid, seed_scores, -jnp.inf)
        level_extras = sampler["levels_extra"]
        level_extras = {
            **level_extras,
            "traced_task_difficulty": jnp.where(valid, seed_scores, level_extras["traced_task_difficulty"]),
            "traced_priority": jnp.where(valid, synced_scores, level_extras["traced_priority"]),
        }
        return {**sampler, "scores": synced_scores, "levels_extra": level_extras}

    def initialize_inserted_seed_state(
        train_state: TrainState, inserted_inds: chex.Array, inserted_scores: chex.Array
    ) -> TrainState:
        def body(i, state):
            idx = inserted_inds[i]
            score = inserted_scores[i]

            def do_update(st):
                seed_scores = st.traced_seed_scores.at[idx].set(score)
                regret_scores = st.traced_regret_scores.at[idx].set(score)
                current_regret = st.traced_current_regret_score.at[idx].set(score)
                previous_regret = st.traced_previous_regret_score.at[idx].set(0.0)
                regret_diff_changed = st.traced_regret_diff_changed.at[idx].set(1)
                regret_seen = st.traced_regret_seen.at[idx].set(False)
                partial_scores = st.traced_partial_seed_scores.at[:, idx].set(0.0)
                partial_max = st.traced_partial_seed_max_scores.at[:, idx].set(-jnp.inf)
                partial_steps = st.traced_partial_seed_steps.at[:, idx].set(0)
                return st.replace(
                    traced_seed_scores=seed_scores,
                    traced_regret_scores=regret_scores,
                    traced_current_regret_score=current_regret,
                    traced_previous_regret_score=previous_regret,
                    traced_regret_diff_changed=regret_diff_changed,
                    traced_regret_seen=regret_seen,
                    traced_partial_seed_scores=partial_scores,
                    traced_partial_seed_max_scores=partial_max,
                    traced_partial_seed_steps=partial_steps,
                )

            return jax.lax.cond(idx >= 0, do_update, lambda st: st, state)

        return jax.lax.fori_loop(0, inserted_inds.shape[0], body, train_state)

    def partial_update_seed_score(
        state: tuple,
        actor_index: int,
        seed_idx: chex.Array,
        score: chex.Array,
        max_score: chex.Array,
        num_steps: chex.Array,
        done: chex.Array,
        running_mean: bool,
    ):
        (
            seed_scores,
            regret_scores,
            current_regret,
            previous_regret,
            regret_diff_changed,
            regret_seen,
            partial_scores,
            partial_max_scores,
            partial_steps,
        ) = state

        partial_score = partial_scores[actor_index, seed_idx]
        partial_max_score = partial_max_scores[actor_index, seed_idx]
        partial_num_steps = partial_steps[actor_index, seed_idx]
        running_num_steps = partial_num_steps + num_steps
        merged_score = partial_score + (score - partial_score) * num_steps.astype(jnp.float32)
        if running_mean:
            merged_score = jnp.where(
                running_num_steps > 0,
                merged_score / jnp.maximum(running_num_steps.astype(jnp.float32), 1.0),
                merged_score,
            )
        merged_max_score = jnp.maximum(partial_max_score, max_score)

        def done_update(args):
            (
                seed_scores_u,
                regret_scores_u,
                current_regret_u,
                previous_regret_u,
                regret_diff_changed_u,
                regret_seen_u,
                partial_scores_u,
                partial_max_scores_u,
                partial_steps_u,
            ) = args
            partial_scores_u = partial_scores_u.at[actor_index, seed_idx].set(0.0)
            partial_max_scores_u = partial_max_scores_u.at[actor_index, seed_idx].set(-jnp.inf)
            partial_steps_u = partial_steps_u.at[actor_index, seed_idx].set(0)

            old_score = regret_scores_u[seed_idx]
            total_score = traced_max_score_coef * merged_max_score + (1.0 - traced_max_score_coef) * merged_score
            smoothed_score = (1.0 - traced_score_alpha) * old_score + traced_score_alpha * total_score

            regret_diff_changed_u = regret_diff_changed_u.at[seed_idx].set(1)
            current_regret_u = current_regret_u.at[seed_idx].set(smoothed_score)
            regret_scores_u = regret_scores_u.at[seed_idx].set(smoothed_score)
            seed_scores_u = seed_scores_u.at[seed_idx].set(smoothed_score)

            return (
                seed_scores_u,
                regret_scores_u,
                current_regret_u,
                previous_regret_u,
                regret_diff_changed_u,
                regret_seen_u,
                partial_scores_u,
                partial_max_scores_u,
                partial_steps_u,
            )

        def running_update(args):
            (
                seed_scores_u,
                regret_scores_u,
                current_regret_u,
                previous_regret_u,
                regret_diff_changed_u,
                regret_seen_u,
                partial_scores_u,
                partial_max_scores_u,
                partial_steps_u,
            ) = args
            partial_scores_u = partial_scores_u.at[actor_index, seed_idx].set(merged_score)
            partial_max_scores_u = partial_max_scores_u.at[actor_index, seed_idx].set(merged_max_score)
            partial_steps_u = partial_steps_u.at[actor_index, seed_idx].set(running_num_steps)
            return (
                seed_scores_u,
                regret_scores_u,
                current_regret_u,
                previous_regret_u,
                regret_diff_changed_u,
                regret_seen_u,
                partial_scores_u,
                partial_max_scores_u,
                partial_steps_u,
            )

        return jax.lax.cond(done, done_update, running_update, state)

    def update_seed_scores_from_rollout(
        train_state: TrainState,
        level_inds_steps: chex.Array,
        dones: chex.Array,
        pvl_steps: chex.Array,
        pair_losses: chex.Array,
        pair_valid: chex.Array,
    ) -> TrainState:
        level_inds_bt = jnp.swapaxes(level_inds_steps, 0, 1)
        dones_bt = jnp.swapaxes(dones, 0, 1)
        pvl_bt = jnp.swapaxes(pvl_steps, 0, 1)
        rollout_len = level_inds_steps.shape[0]

        state = (
            train_state.traced_seed_scores,
            train_state.traced_regret_scores,
            train_state.traced_current_regret_score,
            train_state.traced_previous_regret_score,
            train_state.traced_regret_diff_changed,
            train_state.traced_regret_seen,
            train_state.traced_partial_seed_scores,
            train_state.traced_partial_seed_max_scores,
            train_state.traced_partial_seed_steps,
        )

        def actor_loop(actor_idx, state_in):
            actor_levels = level_inds_bt[actor_idx]
            actor_dones = dones_bt[actor_idx]
            actor_pvl = pvl_bt[actor_idx]
            actor_pair_losses = pair_losses[actor_idx]
            actor_pair_valid = pair_valid[actor_idx].astype(bool)

            init = (
                actor_levels[0],
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(-jnp.inf, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(-jnp.inf, dtype=jnp.float32),
                jnp.array(0, dtype=jnp.int32),
                state_in,
            )

            def step_loop(t, carry):
                (
                    seg_seed_idx,
                    seg_len,
                    seg_pvl_sum,
                    seg_pvl_max,
                    seg_pair_sum,
                    seg_pair_max,
                    seg_pair_count,
                    state_mid,
                ) = carry

                def add_pair(c):
                    (
                        seg_seed_idx_i,
                        seg_len_i,
                        seg_pvl_sum_i,
                        seg_pvl_max_i,
                        seg_pair_sum_i,
                        seg_pair_max_i,
                        seg_pair_count_i,
                        state_i,
                    ) = c
                    pair_ok = actor_pair_valid[t - 1]
                    pair_val = actor_pair_losses[t - 1]
                    seg_pair_sum_i = seg_pair_sum_i + jnp.where(pair_ok, pair_val, 0.0)
                    seg_pair_max_i = jnp.where(pair_ok, jnp.maximum(seg_pair_max_i, pair_val), seg_pair_max_i)
                    seg_pair_count_i = seg_pair_count_i + pair_ok.astype(jnp.int32)
                    return (
                        seg_seed_idx_i,
                        seg_len_i,
                        seg_pvl_sum_i,
                        seg_pvl_max_i,
                        seg_pair_sum_i,
                        seg_pair_max_i,
                        seg_pair_count_i,
                        state_i,
                    )

                (
                    seg_seed_idx,
                    seg_len,
                    seg_pvl_sum,
                    seg_pvl_max,
                    seg_pair_sum,
                    seg_pair_max,
                    seg_pair_count,
                    state_mid,
                ) = jax.lax.cond(t > 0, add_pair, lambda c: c, carry)

                pvl_t = actor_pvl[t]
                seg_pvl_sum = seg_pvl_sum + pvl_t
                seg_pvl_max = jnp.maximum(seg_pvl_max, pvl_t)
                seg_len = seg_len + 1

                done_t = actor_dones[t]

                def on_done(c):
                    (
                        seg_seed_idx_i,
                        seg_len_i,
                        seg_pvl_sum_i,
                        seg_pvl_max_i,
                        seg_pair_sum_i,
                        seg_pair_max_i,
                        seg_pair_count_i,
                        state_i,
                    ) = c

                    score_mean = seg_pvl_sum_i / jnp.maximum(seg_len_i.astype(jnp.float32), 1.0)
                    score_max = seg_pvl_max_i
                    prob_mean = jnp.where(
                        seg_pair_count_i > 0,
                        seg_pair_sum_i / jnp.maximum(seg_pair_count_i.astype(jnp.float32), 1.0),
                        0.0,
                    )
                    prob_max = jnp.where(seg_pair_count_i > 0, seg_pair_max_i, 0.0)
                    total_score = score_mean + traced_transition_prob_weight * prob_mean
                    total_max = score_max + traced_transition_prob_weight * prob_max

                    state_i = partial_update_seed_score(
                        state_i,
                        actor_idx,
                        seg_seed_idx_i,
                        total_score,
                        total_max,
                        seg_len_i,
                        jnp.array(True),
                        running_mean=False,
                    )
                    next_seed = jax.lax.select(t + 1 < rollout_len, actor_levels[t + 1], seg_seed_idx_i)
                    return (
                        next_seed,
                        jnp.array(0, dtype=jnp.int32),
                        jnp.array(0.0, dtype=jnp.float32),
                        jnp.array(-jnp.inf, dtype=jnp.float32),
                        jnp.array(0.0, dtype=jnp.float32),
                        jnp.array(-jnp.inf, dtype=jnp.float32),
                        jnp.array(0, dtype=jnp.int32),
                        state_i,
                    )

                return jax.lax.cond(
                    done_t,
                    on_done,
                    lambda c: c,
                    (
                        seg_seed_idx,
                        seg_len,
                        seg_pvl_sum,
                        seg_pvl_max,
                        seg_pair_sum,
                        seg_pair_max,
                        seg_pair_count,
                        state_mid,
                    ),
                )

            (
                seg_seed_idx,
                seg_len,
                seg_pvl_sum,
                seg_pvl_max,
                seg_pair_sum,
                seg_pair_max,
                seg_pair_count,
                state_out,
            ) = jax.lax.fori_loop(0, rollout_len, step_loop, init)

            def flush_partial(st):
                score_mean = seg_pvl_sum / jnp.maximum(seg_len.astype(jnp.float32), 1.0)
                score_max = seg_pvl_max
                prob_mean = jnp.where(
                    seg_pair_count > 0,
                    seg_pair_sum / jnp.maximum(seg_pair_count.astype(jnp.float32), 1.0),
                    0.0,
                )
                prob_max = jnp.where(seg_pair_count > 0, seg_pair_max, 0.0)
                total_score = score_mean + traced_transition_prob_weight * prob_mean
                total_max = score_max + traced_transition_prob_weight * prob_max
                return partial_update_seed_score(
                    st,
                    actor_idx,
                    seg_seed_idx,
                    total_score,
                    total_max,
                    seg_len,
                    jnp.array(False),
                    running_mean=False,
                )

            return jax.lax.cond(seg_len > 0, flush_partial, lambda st: st, state_out)

        state = jax.lax.fori_loop(0, num_train_envs, actor_loop, state)
        (
            seed_scores,
            regret_scores,
            current_regret,
            previous_regret,
            regret_diff_changed,
            regret_seen,
            partial_scores,
            partial_max_scores,
            partial_steps,
        ) = state
        return train_state.replace(
            traced_seed_scores=seed_scores,
            traced_regret_scores=regret_scores,
            traced_current_regret_score=current_regret,
            traced_previous_regret_score=previous_regret,
            traced_regret_diff_changed=regret_diff_changed,
            traced_regret_seen=regret_seen,
            traced_partial_seed_scores=partial_scores,
            traced_partial_seed_max_scores=partial_max_scores,
            traced_partial_seed_steps=partial_steps,
        )

    def update_regret_diff(train_state: TrainState):
        changed_mask = train_state.traced_regret_diff_changed == 1
        changed_count = changed_mask.astype(jnp.int32).sum()
        first_seen = (~train_state.traced_regret_seen) & (train_state.traced_previous_regret_score == 0.0)
        diffs = jnp.where(
            changed_mask,
            jnp.where(
                first_seen,
                0.0,
                train_state.traced_previous_regret_score - train_state.traced_current_regret_score,
            ),
            0.0,
        )
        mean_diff = jnp.where(
            changed_count > 0,
            diffs.sum() / jnp.maximum(changed_count.astype(jnp.float32), 1.0),
            0.0,
        )
        max_diff = jnp.where(changed_count > 0, jnp.max(jnp.where(changed_mask, diffs, -jnp.inf)), 0.0)
        min_diff = jnp.where(changed_count > 0, jnp.min(jnp.where(changed_mask, diffs, jnp.inf)), 0.0)
        std_diff = jnp.where(
            changed_count > 0,
            jnp.sqrt(
                (
                    jnp.where(changed_mask, diffs - mean_diff, 0.0) ** 2
                ).sum()
                / jnp.maximum(changed_count.astype(jnp.float32), 1.0)
            ),
            0.0,
        )

        prev_inds = train_state.traced_prev_replay_inds
        prev_count = train_state.traced_prev_replay_count
        prev_mask = jnp.arange(prev_inds.shape[0], dtype=jnp.int32) < prev_count
        bonus = traced_colearnability_weight * mean_diff
        bonus_vals = jnp.where(prev_mask, bonus, 0.0)
        seed_scores = train_state.traced_seed_scores.at[prev_inds].add(bonus_vals)

        previous_regret = jnp.where(
            changed_mask,
            train_state.traced_current_regret_score,
            train_state.traced_previous_regret_score,
        )
        regret_diff_changed = jnp.where(changed_mask, 0, train_state.traced_regret_diff_changed)
        regret_seen = jnp.where(changed_mask, True, train_state.traced_regret_seen)
        new_prev_inds, new_prev_count = mask_to_compact_indices(changed_mask)

        train_state = train_state.replace(
            traced_seed_scores=seed_scores,
            traced_previous_regret_score=previous_regret,
            traced_regret_diff_changed=regret_diff_changed,
            traced_regret_seen=regret_seen,
            traced_prev_replay_inds=new_prev_inds,
            traced_prev_replay_count=new_prev_count,
            traced_prev_replay_valid=new_prev_count > 0,
        )
        return train_state, mean_diff, max_diff, min_diff, std_diff, bonus

    def traced_after_update(train_state: TrainState) -> TrainState:
        state = (
            train_state.traced_seed_scores,
            train_state.traced_regret_scores,
            train_state.traced_current_regret_score,
            train_state.traced_previous_regret_score,
            train_state.traced_regret_diff_changed,
            train_state.traced_regret_seen,
            train_state.traced_partial_seed_scores,
            train_state.traced_partial_seed_max_scores,
            train_state.traced_partial_seed_steps,
        )

        num_capacity = train_state.traced_seed_scores.shape[0]

        def actor_loop(actor_idx, state_in):
            def seed_loop(seed_idx, state_mid):
                partial_score = state_mid[6][actor_idx, seed_idx]
                return jax.lax.cond(
                    partial_score != 0.0,
                    lambda st: partial_update_seed_score(
                        st,
                        actor_idx,
                        jnp.array(seed_idx, dtype=jnp.int32),
                        jnp.array(0.0, dtype=jnp.float32),
                        jnp.array(-jnp.inf, dtype=jnp.float32),
                        jnp.array(0, dtype=jnp.int32),
                        jnp.array(True),
                        running_mean=True,
                    ),
                    lambda st: st,
                    state_mid,
                )

            return jax.lax.fori_loop(0, num_capacity, seed_loop, state_in)

        state = jax.lax.fori_loop(0, num_train_envs, actor_loop, state)
        (
            seed_scores,
            regret_scores,
            current_regret,
            previous_regret,
            regret_diff_changed,
            regret_seen,
            partial_scores,
            partial_max_scores,
            partial_steps,
        ) = state
        return train_state.replace(
            traced_seed_scores=seed_scores,
            traced_regret_scores=regret_scores,
            traced_current_regret_score=current_regret,
            traced_previous_regret_score=previous_regret,
            traced_regret_diff_changed=regret_diff_changed,
            traced_regret_seen=regret_seen,
            traced_partial_seed_scores=jnp.zeros_like(partial_scores),
            traced_partial_seed_max_scores=partial_max_scores,
            traced_partial_seed_steps=jnp.zeros_like(partial_steps),
        )

    def select_mutation_parent_indices(task_difficulty: chex.Array) -> chex.Array:
        num_parents = min(traced_num_mutation_parents, num_train_envs)
        easy = jnp.argsort(task_difficulty)[:num_parents]
        repeats = (num_train_envs + num_parents - 1) // num_parents
        return jnp.tile(easy, repeats)[:num_train_envs]

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        def on_new_levels(rng_step: chex.PRNGKey, train_state: TrainState):
            sampler = train_state.sampler
            rng_step, rng_levels, rng_reset = jax.random.split(rng_step, 3)
            new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, num_train_envs))
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, num_train_envs), new_levels, env_params
            )
            (
                (rng_step, train_state, _, _, _, last_value),
                (obs, actions, rewards, dones, log_probs, values, next_obs, _),
            ) = sample_trajectories_rnn_with_next_obs(
                rng_step,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((num_train_envs,)),
                init_obs,
                init_env_state,
                num_train_envs,
                config["num_steps"],
            )
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            pvl_steps = jnp.maximum(targets - values, 0.0)
            pvl = jnp.mean(pvl_steps, axis=0)
            max_returns = compute_max_returns(dones, rewards)
            pair_obs, pair_actions, pair_next_obs, pair_valid = prepare_transition_pairs(obs, actions, dones)
            transition_params_new, transition_opt_state_new, transition_loss, pair_losses = update_transition_model(
                train_state.transition_params,
                train_state.transition_opt_state,
                pair_obs,
                pair_actions,
                pair_next_obs,
                pair_valid,
            )
            pair_den = jnp.maximum(pair_valid.sum(axis=1), 1.0)
            atpl_mean = (pair_losses * pair_valid).sum(axis=1) / pair_den
            task_difficulty = pvl + traced_transition_prob_weight * atpl_mean
            success = rollout_success_from_rewards(rewards)
            sampler, inserted_inds = level_sampler.insert_batch(
                sampler,
                new_levels,
                task_difficulty,
                build_traced_extras(
                    max_returns,
                    success,
                    pvl,
                    atpl_mean,
                    task_difficulty,
                    jnp.ones_like(success),
                ),
            )

            (rng_step, train_state), losses = update_actor_critic_rnn(
                rng_step,
                train_state,
                ActorCritic.initialize_carry((num_train_envs,)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                num_train_envs,
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )

            train_state = train_state.replace(
                transition_params=transition_params_new,
                transition_opt_state=transition_opt_state_new,
                transition_last_loss=transition_loss,
                transition_last_atpl_mean=atpl_mean.mean(),
            )
            train_state = initialize_inserted_seed_state(train_state, inserted_inds, task_difficulty)

            def run_regret_pass(state):
                return update_regret_diff(state)

            def skip_regret_pass(state):
                zero = jnp.array(0.0, dtype=jnp.float32)
                return state, zero, zero, zero, zero, zero

            (
                train_state,
                mean_regret_diff,
                _max_regret_diff,
                _min_regret_diff,
                _std_regret_diff,
                colearnability_bonus,
            ) = jax.lax.cond(
                config["exploratory_grad_updates"],
                run_regret_pass,
                skip_regret_pass,
                train_state,
            )
            train_state = traced_after_update(train_state)
            sampler = sync_sampler_scores_with_traced(sampler, train_state.traced_seed_scores)
            inserted_mask = inserted_inds >= 0
            inserted_score_mean = jnp.where(
                inserted_mask.any(),
                jnp.where(inserted_mask, train_state.traced_seed_scores[inserted_inds], 0.0).sum()
                / jnp.maximum(inserted_mask.astype(jnp.float32).sum(), 1.0),
                0.0,
            )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": new_levels.wall_map.sum() / num_train_envs,
                "traced_transition_loss": transition_loss,
                "traced_pvl_mean": pvl.mean(),
                "traced_atpl_mean": atpl_mean.mean(),
                "traced_task_difficulty_mean": inserted_score_mean,
                "traced_mean_regret_diff": mean_regret_diff,
                "traced_colearnability_bonus": colearnability_bonus,
                "traced_pca_used": jnp.array(0, dtype=jnp.int32),
                "mutation_parent_difficulty_mean": jnp.array(0.0, dtype=jnp.float32),
                "mutation_uphill_fraction": jnp.array(0.0, dtype=jnp.float32),
            }

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=new_levels,
            )
            return (rng_step, train_state), metrics

        def on_replay_levels(rng_step: chex.PRNGKey, train_state: TrainState):
            sampler = train_state.sampler
            rng_step, rng_levels, rng_reset = jax.random.split(rng_step, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, num_train_envs)
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, num_train_envs), levels, env_params
            )
            (
                (rng_step, sampler, train_state, _, _, _, _, _, last_value),
                (obs, actions, rewards, dones, log_probs, values, _next_obs, _, replay_level_inds_steps),
            ) = sample_replay_trajectories_rnn_with_next_obs(
                rng_step,
                env,
                env_params,
                level_sampler,
                sampler,
                train_state,
                ActorCritic.initialize_carry((num_train_envs,)),
                init_obs,
                init_env_state,
                levels,
                level_inds,
                num_train_envs,
                config["num_steps"],
            )
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            pvl_steps = jnp.maximum(targets - values, 0.0)
            pvl = jnp.mean(pvl_steps, axis=0)
            pair_obs, pair_actions, pair_next_obs, pair_valid = prepare_transition_pairs(obs, actions, dones)
            transition_params_new, transition_opt_state_new, transition_loss, pair_losses = update_transition_model(
                train_state.transition_params,
                train_state.transition_opt_state,
                pair_obs,
                pair_actions,
                pair_next_obs,
                pair_valid,
            )
            pair_den = jnp.maximum(pair_valid.sum(axis=1), 1.0)
            atpl_mean = (pair_losses * pair_valid).sum(axis=1) / pair_den

            train_state = update_seed_scores_from_rollout(
                train_state,
                replay_level_inds_steps,
                dones,
                pvl_steps,
                pair_losses,
                pair_valid,
            )

            level_extras = level_sampler.get_levels_extra(sampler, level_inds)
            max_returns = jnp.maximum(level_extras["max_return"], compute_max_returns(dones, rewards))
            task_difficulty = train_state.traced_seed_scores[level_inds]
            success = rollout_success_from_rewards(rewards)
            new_success_ema, new_success_obs_count = update_success_ema(
                level_extras["success_ema"],
                level_extras["success_obs_count"],
                success,
                float(config["success_ema_alpha"]),
            )
            sampler = level_sampler.update_batch(
                sampler,
                level_inds,
                task_difficulty,
                build_traced_extras(
                    max_returns,
                    new_success_ema,
                    pvl,
                    atpl_mean,
                    task_difficulty,
                    new_success_obs_count,
                ),
            )
            mean_returns, _, _ = compute_max_mean_returns_epcount(dones, rewards)
            batched_value_loss = jnp.clip(jnp.mean(jnp.abs(targets - values), axis=0), -1.0, 1.0)
            mutation_parent_difficulty = mean_returns - batched_value_loss

            (rng_step, train_state), losses = update_actor_critic_rnn(
                rng_step,
                train_state,
                ActorCritic.initialize_carry((num_train_envs,)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                num_train_envs,
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=True,
            )

            train_state = train_state.replace(
                transition_params=transition_params_new,
                transition_opt_state=transition_opt_state_new,
                transition_last_loss=transition_loss,
                transition_last_atpl_mean=atpl_mean.mean(),
            )
            (
                train_state,
                mean_regret_diff,
                _max_regret_diff,
                _min_regret_diff,
                _std_regret_diff,
                colearnability_bonus,
            ) = update_regret_diff(train_state)
            train_state = traced_after_update(train_state)
            sampler = sync_sampler_scores_with_traced(sampler, train_state.traced_seed_scores)
            sampled_scores = train_state.traced_seed_scores[level_inds]

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": levels.wall_map.sum() / num_train_envs,
                "traced_transition_loss": transition_loss,
                "traced_pvl_mean": pvl.mean(),
                "traced_atpl_mean": atpl_mean.mean(),
                "traced_task_difficulty_mean": sampled_scores.mean(),
                "traced_mean_regret_diff": mean_regret_diff,
                "traced_colearnability_bonus": colearnability_bonus,
                "traced_pca_used": jnp.array(0, dtype=jnp.int32),
                "mutation_parent_difficulty_mean": jnp.array(0.0, dtype=jnp.float32),
                "mutation_uphill_fraction": jnp.array(0.0, dtype=jnp.float32),
            }

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
                replay_last_level_inds=level_inds,
                replay_last_task_difficulty=mutation_parent_difficulty,
            )
            return (rng_step, train_state), metrics

        def on_mutate_levels(rng_step: chex.PRNGKey, train_state: TrainState):
            sampler = train_state.sampler
            parent_levels = train_state.replay_last_level_batch
            parent_difficulty = train_state.replay_last_task_difficulty
            rng_step, rng_mutate, rng_decode, rng_levels, rng_reset = jax.random.split(rng_step, 5)

            local_parent_idx = select_mutation_parent_indices(parent_difficulty)
            selected_parent_levels = jax.tree_util.tree_map(lambda x: x[local_parent_idx], parent_levels)
            selected_parent_difficulty = parent_difficulty[local_parent_idx]

            if config["traced_use_pca_mutation"]:
                parent_seqs = jax.vmap(level_to_cluttr_sequence)(
                    selected_parent_levels.wall_map,
                    selected_parent_levels.goal_pos,
                    selected_parent_levels.agent_pos,
                )
                parent_latents = encode_levels_to_latents(encoder_params, parent_seqs)
                noise = jax.random.normal(rng_mutate, parent_latents.shape)
                scaled = (
                    noise
                    * jnp.sqrt(jnp.maximum(train_state.traced_pca_eigvals, 1e-8))
                    * float(config["traced_pca_sigma"])
                )
                child_latents = parent_latents + scaled @ train_state.traced_pca_eigvecs
                child_sequences = decode_latent_to_env(
                    decoder_params,
                    child_latents,
                    rng_key=rng_decode,
                    temperature=float(config["traced_decode_temperature"]),
                )
                child_sequences = jax.vmap(repair_cluttr_sequence)(child_sequences)
                child_levels = jax.vmap(cluttr_sequence_to_level)(
                    child_sequences,
                    jax.random.split(rng_levels, num_train_envs),
                )
                pca_used = jnp.array(1, dtype=jnp.int32)
            else:
                child_levels = jax.vmap(mutate_level, (0, 0, None))(
                    jax.random.split(rng_mutate, num_train_envs),
                    selected_parent_levels,
                    config["num_edits"],
                )
                pca_used = jnp.array(0, dtype=jnp.int32)

            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, num_train_envs), child_levels, env_params
            )
            (
                (rng_step, train_state, _, _, _, last_value),
                (obs, actions, rewards, dones, log_probs, values, next_obs, _),
            ) = sample_trajectories_rnn_with_next_obs(
                rng_step,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((num_train_envs,)),
                init_obs,
                init_env_state,
                num_train_envs,
                config["num_steps"],
            )
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            pvl_steps = jnp.maximum(targets - values, 0.0)
            pvl = jnp.mean(pvl_steps, axis=0)
            max_returns = compute_max_returns(dones, rewards)
            pair_obs, pair_actions, pair_next_obs, pair_valid = prepare_transition_pairs(obs, actions, dones)
            transition_params_new, transition_opt_state_new, transition_loss, pair_losses = update_transition_model(
                train_state.transition_params,
                train_state.transition_opt_state,
                pair_obs,
                pair_actions,
                pair_next_obs,
                pair_valid,
            )
            pair_den = jnp.maximum(pair_valid.sum(axis=1), 1.0)
            atpl_mean = (pair_losses * pair_valid).sum(axis=1) / pair_den
            task_difficulty = pvl + traced_transition_prob_weight * atpl_mean
            success = rollout_success_from_rewards(rewards)
            sampler, inserted_inds = level_sampler.insert_batch(
                sampler,
                child_levels,
                task_difficulty,
                build_traced_extras(
                    max_returns,
                    success,
                    pvl,
                    atpl_mean,
                    task_difficulty,
                    jnp.ones_like(success),
                ),
            )

            batch_improved = jnp.sum(task_difficulty > selected_parent_difficulty).astype(jnp.int32)
            batch_compared = jnp.array(num_train_envs, dtype=jnp.int32)
            uphill_frac = batch_improved.astype(jnp.float32) / jnp.maximum(batch_compared.astype(jnp.float32), 1.0)

            (rng_step, train_state), losses = update_actor_critic_rnn(
                rng_step,
                train_state,
                ActorCritic.initialize_carry((num_train_envs,)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                num_train_envs,
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=False,
            )

            train_state = train_state.replace(
                transition_params=transition_params_new,
                transition_opt_state=transition_opt_state_new,
                transition_last_loss=transition_loss,
                transition_last_atpl_mean=atpl_mean.mean(),
            )
            train_state = initialize_inserted_seed_state(train_state, inserted_inds, task_difficulty)
            train_state = traced_after_update(train_state)
            sampler = sync_sampler_scores_with_traced(sampler, train_state.traced_seed_scores)
            inserted_mask = inserted_inds >= 0
            inserted_score_mean = jnp.where(
                inserted_mask.any(),
                jnp.where(inserted_mask, train_state.traced_seed_scores[inserted_inds], 0.0).sum()
                / jnp.maximum(inserted_mask.astype(jnp.float32).sum(), 1.0),
                0.0,
            )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": child_levels.wall_map.sum() / num_train_envs,
                "traced_transition_loss": transition_loss,
                "traced_pvl_mean": pvl.mean(),
                "traced_atpl_mean": atpl_mean.mean(),
                "traced_task_difficulty_mean": inserted_score_mean,
                "traced_mean_regret_diff": jnp.array(0.0, dtype=jnp.float32),
                "traced_colearnability_bonus": jnp.array(0.0, dtype=jnp.float32),
                "traced_pca_used": pca_used,
                "mutation_parent_difficulty_mean": selected_parent_difficulty.mean(),
                "mutation_uphill_fraction": uphill_frac,
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
            return (rng_step, train_state), metrics

        rng_step, train_state = carry
        rng_step, rng_replay = jax.random.split(rng_step)
        should_replay = level_sampler.sample_replay_decision(train_state.sampler, rng_replay)

        def run_new_levels(c):
            return on_new_levels(*c)

        def run_replay_and_mutate(c):
            (rng_i, state_i), replay_metrics = on_replay_levels(*c)
            (rng_i, state_i), mutate_metrics = on_mutate_levels(rng_i, state_i)
            merged_metrics = {
                **replay_metrics,
                "mutation_parent_difficulty_mean": mutate_metrics["mutation_parent_difficulty_mean"],
                "mutation_uphill_fraction": mutate_metrics["mutation_uphill_fraction"],
                "traced_pca_used": mutate_metrics["traced_pca_used"],
            }
            return (rng_i, state_i), merged_metrics

        return jax.lax.cond(
            should_replay,
            run_replay_and_mutate,
            run_new_levels,
            (rng_step, train_state),
        )

    def eval_once(rng_eval: chex.PRNGKey, train_state: TrainState):
        rng_eval, rng_reset = jax.random.split(rng_eval)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, num_levels), levels, env_params
        )
        states, rewards, episode_lengths = evaluate_rnn(
            rng_eval,
            eval_env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths

    @jax.jit
    def train_and_eval_step(runner_state, _):
        (rng_eval, train_state), train_metrics = jax.lax.scan(
            train_step, runner_state, None, config["eval_freq"]
        )
        train_metrics = jax.tree_util.tree_map(lambda x: x.mean(), train_metrics)

        rng_eval, rng_eval_many = jax.random.split(rng_eval)
        states, cum_rewards, episode_lengths = jax.vmap(eval_once, (0, None))(
            jax.random.split(rng_eval_many, config["eval_num_attempts"]), train_state
        )

        eval_solve_rates = jnp.where(cum_rewards > 0, 1.0, 0.0).mean(axis=0)
        eval_returns = cum_rewards.mean(axis=0)

        states, episode_lengths = jax.tree_util.tree_map(lambda x: x[0], (states, episode_lengths))
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params)
        frames = images.transpose(0, 1, 4, 2, 3)

        metrics = dict(train_metrics)
        metrics["update_count"] = train_state.num_dr_updates + train_state.num_replay_updates
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"] = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        metrics["dr_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.dr_last_level_batch, env_params)
        metrics["replay_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.replay_last_level_batch, env_params)
        metrics["mutation_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.mutation_last_level_batch, env_params)

        highest_scoring_level = level_sampler.get_levels(train_state.sampler, train_state.sampler["scores"].argmax())
        highest_weighted_level = level_sampler.get_levels(
            train_state.sampler, level_sampler.level_weights(train_state.sampler).argmax()
        )
        metrics["highest_scoring_level"] = env_renderer.render_level(highest_scoring_level, env_params)
        metrics["highest_weighted_level"] = env_renderer.render_level(highest_weighted_level, env_params)
        return (rng_eval, train_state), metrics

    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")
        env_steps = stats["update_count"] * config["num_train_envs"] * config["num_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / max(stats["time_delta"], 1e-6),
            "traced/transition_loss": stats["traced_transition_loss"],
            "traced/pvl_mean": stats["traced_pvl_mean"],
            "traced/atpl_mean": stats["traced_atpl_mean"],
            "traced/task_difficulty_mean": stats["traced_task_difficulty_mean"],
            "traced/mean_regret_diff": stats["traced_mean_regret_diff"],
            "traced/colearnability_bonus": stats["traced_colearnability_bonus"],
            "traced/pca_used": stats["traced_pca_used"],
            "mutation/parent_difficulty_mean": stats["mutation_parent_difficulty_mean"],
            "mutation/uphill_fraction": stats["mutation_uphill_fraction"],
        }

        solve_rates = stats["eval_solve_rates"]
        returns = stats["eval_returns"]
        log_dict.update({f"solve_rate/{name}": sr for name, sr in zip(config["eval_levels"], solve_rates)})
        log_dict["solve_rate/mean"] = solve_rates.mean()
        log_dict.update({f"return/{name}": r for name, r in zip(config["eval_levels"], returns)})
        log_dict["return/mean"] = returns.mean()
        log_dict["eval_ep_lengths/mean"] = stats["eval_ep_lengths"].mean()
        log_dict.update(train_state_info["log"])

        log_dict["images/highest_scoring_level"] = wandb.Image(
            np.array(stats["highest_scoring_level"]),
            caption="Highest scoring level",
        )
        log_dict["images/highest_weighted_level"] = wandb.Image(
            np.array(stats["highest_weighted_level"]),
            caption="Highest weighted level",
        )
        for name in ["dr", "replay", "mutation"]:
            if train_state_info["info"][f"num_{name}_updates"] > 0:
                log_dict[f"images/{name}_levels"] = [
                    wandb.Image(np.array(img)) for img in stats[f"{name}_levels"]
                ]
        for i, level_name in enumerate(config["eval_levels"]):
            frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
            frames = np.array(frames[: int(episode_length)])
            log_dict[f"animations/{level_name}"] = wandb.Video(frames, fps=4, format="gif")
        wandb.log(log_dict)

    rng, rng_init, rng_train = jax.random.split(rng, 3)
    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    checkpoint_manager = None
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state)

    num_eval_steps = config["num_updates"] // config["eval_freq"]
    for eval_step in range(num_eval_steps):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        metrics["time_delta"] = time.time() - start_time

        if (
            config["traced_use_pca_mutation"]
            and eval_step % int(config["traced_pca_update_every"]) == 0
        ):
            buffer_size = int(runner_state[1].sampler["size"])
            if buffer_size >= traced_pca_latent_dim + 1:
                level_state = runner_state[1].sampler["levels"]
                wall_maps = jnp.array(np.array(level_state.wall_map[:buffer_size]))
                goal_pos = jnp.array(np.array(level_state.goal_pos[:buffer_size]))
                agent_pos = jnp.array(np.array(level_state.agent_pos[:buffer_size]))
                seqs = jax.jit(jax.vmap(level_to_cluttr_sequence))(wall_maps, goal_pos, agent_pos)
                latent_chunks = []
                for i in range(0, buffer_size, 256):
                    latent_chunks.append(np.array(encode_levels_to_latents(encoder_params, seqs[i : i + 256])))
                latents = np.concatenate(latent_chunks, axis=0)
                eigvecs, eigvals = compute_pca_from_latents(latents)
                runner_state = (
                    runner_state[0],
                    runner_state[1].replace(
                        traced_pca_eigvecs=jnp.array(eigvecs, dtype=jnp.float32),
                        traced_pca_eigvals=jnp.array(eigvals, dtype=jnp.float32),
                    ),
                )

        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
        if checkpoint_manager is not None:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()

    return runner_state[1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="MAZE_TRACED_TEST")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--checkpoint_save_interval", type=int, default=2)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument(
        "--eval_levels",
        nargs="+",
        default=[
            "SixteenRooms",
            "SixteenRooms2",
            "Labyrinth",
            "LabyrinthFlipped",
            "Labyrinth2",
            "StandardMaze",
            "StandardMaze2",
            "StandardMaze3",
            "NarrowBridge",
            "ForkDeception",
            "PerimeterRun",
            "SpiralPocket",
            "SymmetricCross",
            "ZigZagTunnel",
            "RoomKeyhole",
            "DualLoopChoice",
            "CentralChoke",
            "LongDetour",
            "DeadendFan",
            "OpenFieldBarriers",
            "ParallelCorridors",
            "CornerTrapEscape",
            "SnakeSpine",
        ],
    )

    group = parser.add_argument_group("Training params")
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
    group.add_argument("--score_function", type=str, default="pvl", choices=["pvl"])
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
    group.add_argument("--agent_view_size", type=int, default=5)
    group.add_argument("--n_walls", type=int, default=25)
    group.add_argument("--traced_colearnability_weight", type=float, default=1.0)
    group.add_argument("--traced_transition_prob_weight", type=float, default=1.0)
    group.add_argument("--traced_score_alpha", type=float, default=1.0)
    group.add_argument("--traced_max_score_coef", type=float, default=0.0)
    group.add_argument("--traced_transition_hidden_dim", type=int, default=128)
    group.add_argument("--traced_transition_lr", type=float, default=1e-4)
    group.add_argument("--traced_transition_weight_decay", type=float, default=1e-5)
    group.add_argument("--traced_num_mutation_parents", type=int, default=4)
    group.add_argument("--traced_use_pca_mutation", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--traced_vae_checkpoint", type=str, default=None)
    group.add_argument("--traced_pca_sigma", type=float, default=0.5)
    group.add_argument("--traced_decode_temperature", type=float, default=0.25)
    group.add_argument("--traced_pca_update_every", type=int, default=1)

    config = vars(parser.parse_args())
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (
            config["num_train_envs"] * config["num_steps"]
        )
    if config["run_name"] is None:
        suffix = "pca" if config["traced_use_pca_mutation"] else "edit"
        config["run_name"] = f"maze_traced_{suffix}_seed{config['seed']}"
    if config["mode"] == "eval":
        raise ValueError("maze_traced.py currently supports training mode only.")
    main(config, project=config["project"])
