import json
import time
from typing import Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import core, struct
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax
import distrax
import os
import sys
import orbax.checkpoint as ocp
import wandb
from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator, make_level_mutator_minimax
from jaxued.level_sampler import LevelSampler
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from jaxued.wrappers import AutoReplayWrapper
import chex
from enum import IntEnum

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from es.vae_decoder import load_vae_params, extract_decoder_params, decode_latent_to_env, repair_cluttr_sequence
from es.cluttr_encoder import extract_encoder_params, encode_levels_to_latents
from es.env_bridge import level_to_cluttr_sequence, cluttr_sequence_to_level
from es.maze_ae import (
    load_maze_ae_params, extract_maze_encoder_params, extract_maze_decoder_params,
    maze_level_to_grid, encode_maze_levels, decode_maze_latents, predict_task_targets,
    compute_structural_targets,
)
from es.cnn_maze_vae import (
    load_cnn_vae_params,
    encode_cnn_vae_levels,
    decode_cnn_vae_latents,
    latent_dim_from_params as cnn_vae_latent_dim,
)
from es import plwm_scoring
from es.online_level_model import (
    compute_pca_from_latents,
    encode_batch_np,
    extract_buffer_grids,
)
from es.map_elites_mutation_service import (
    MapElitesArchive,
    OBS_BINS,
    DIST_BINS,
    BFS_PATH_BINS,
    DENSE_OBS_BINS,
    num_cells,
    make_latent_projections,
    init_map_elites_archive,
    map_elites_mutate_levels,
    map_elites_insert_batch,
    map_elites_stats,
)

class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    me_archive: MapElitesArchive = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    num_plwm_compared: int
    num_plwm_improved: int
    num_plwm_inserted: int
    num_plwm_inserted_improved: int
    plwm_last_uphill_fraction: chex.Array
    plwm_last_insert_fraction: chex.Array
    plwm_last_accepted_uphill_fraction: chex.Array
    plwm_delta_score_sum: chex.Array
    plwm_accepted_delta_score_sum: chex.Array
    plwm_last_mean_delta_score: chex.Array
    plwm_last_mean_accepted_delta_score: chex.Array
    plwm_last_scale_global_fraction: chex.Array
    plwm_last_scale_meso_fraction: chex.Array
    plwm_last_scale_local_fraction: chex.Array
    plwm_last_schedule_progress: chex.Array
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_inds: chex.Array = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    # Online frontier estimator (run-local, task-specific).
    frontier_params: core.FrozenDict[str, chex.Array] = struct.field(pytree_node=True)
    frontier_updates: int
    frontier_last_loss: chex.Array
    frontier_last_p_mae: chex.Array
    frontier_last_std: chex.Array
    # PCA directions over buffer latents (for --plwm_use_pca_mutation)
    plwm_pca_eigvecs: chex.Array = struct.field(pytree_node=True)   # (D, D)
    plwm_pca_eigvals: chex.Array = struct.field(pytree_node=True)   # (D,)
    plwm_pca_mean: chex.Array = struct.field(pytree_node=True)      # (D,)

# region PPO helper functions
def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float): 
        lambd (float): 
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """
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

def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
) -> Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]:
    """This samples trajectories from the environment using the agent specified by the `train_state`.

    Args:

        rng (chex.PRNGKey): Singleton 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): Singleton
        init_hstate (chex.ArrayTree): This is the init RNN hidden state, has to have shape (NUM_ENVS, ...)
        init_obs (Observation): The initial observation, shape (NUM_ENVS, ...)
        init_env_state (EnvState): The initial env state (NUM_ENVS, ...)
        num_envs (int): The number of envs that are vmapped over.
        max_episode_length (int): The maximum episode length, i.e., the number of steps to do the rollouts for.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]: (rng, train_state, hstate, last_obs, last_env_state, last_value), traj, where traj is (obs, action, reward, done, log_prob, value, info). The first element in the tuple consists of arrays that have shapes (NUM_ENVS, ...) (except `rng` and and `train_state` which are singleton). The second element in the tuple is of shape (NUM_STEPS, NUM_ENVS, ...), and it contains the trajectory.
    """
    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info)

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

    x = jax.tree_util.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)

    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj

def evaluate_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """This runs the RNN on the environment, given an initial state and observation, and returns (states, rewards, episode_lengths)

    Args:
        rng (chex.PRNGKey): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): Shape (num_levels, )
        init_obs (Observation): Shape (num_levels, )
        init_env_state (EnvState): Shape (num_levels, )
        max_episode_length (int): 

    Returns:
        Tuple[chex.Array, chex.Array, chex.Array]: (States, rewards, episode lengths) ((NUM_STEPS, NUM_LEVELS), (NUM_STEPS, NUM_LEVELS), (NUM_LEVELS,)
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]
    
    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)
        
        next_mask = mask & ~done
        episode_length += mask

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (state, reward)
    
    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths

def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool=True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Args:
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): 
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int): 
        n_steps (int): 
        n_minibatch (int): 
        n_epochs (int): 
        clip_eps (float): 
        entropy_coeff (float): 
        critic_coeff (float): 
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]: It returns a new rng, the updated train_state, and the losses. The losses have structure (loss, (l_vf, l_clip, entropy))
    """
    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages
    
    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, actions, last_dones, log_probs, values, targets, advantages = minibatch
            
            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(params, (obs, last_dones), init_hstate)
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean()

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

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
                lambda x: jnp.take(x, permutation, axis=0)
                .reshape(n_minibatch, -1, *x.shape[1:]),
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

class ActorCritic(nn.Module):
    """This is an actor critic class that uses an LSTM
    """
    action_dim: Sequence[int]
    
    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs
        
        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)
        
        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)
        
        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))
# endregion

# region checkpointing
def setup_checkpointing(config: dict, train_state: TrainState, env: UnderspecifiedEnv, env_params: EnvParams) -> ocp.CheckpointManager:
    """This takes in the train state and config, and returns an orbax checkpoint manager.
        It also saves the config in `checkpoints/run_name/seed/config.json`

    Args:
        config (dict): 
        train_state (TrainState): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 

    Returns:
        ocp.CheckpointManager: 
    """
    overall_save_dir = os.path.join(os.getcwd(), "checkpoints", f"{config['run_name']}", str(config['seed']))
    os.makedirs(overall_save_dir, exist_ok=True)
    
    # save the config
    with open(os.path.join(overall_save_dir, 'config.json'), 'w+') as f:
        f.write(json.dumps(config.as_dict(), indent=True))
    
    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, 'models'),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config['checkpoint_save_interval'],
            max_to_keep=config['max_number_of_checkpoints'],
        )
    )
    return checkpoint_manager
#endregion

def train_state_to_log_dict(
    train_state: TrainState, level_sampler: LevelSampler, use_map_elites: bool = False,
    use_plwm: bool = False,
) -> dict:
    """To prevent the entire (large) train_state to be copied to the CPU when doing logging, this function returns all of the important information in a dictionary format.

        Anything in the `log` key will be logged to wandb.
    
    Args:
        train_state (TrainState): 
        level_sampler (LevelSampler): 

    Returns:
        dict: 
    """
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)
    log = {
        "level_sampler/size": sampler["size"],
        "level_sampler/episode_count": sampler["episode_count"],
        "level_sampler/max_score": sampler["scores"].max(),
        "level_sampler/weighted_score": (sampler["scores"] * level_sampler.level_weights(sampler)).sum(),
        "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
    }

    if use_map_elites:
        me = map_elites_stats(
            train_state.me_archive,
            current_step=train_state.num_mutation_updates,
        )
        log.update(
            {
                "map_elites/occupied_cells": me["occupied_cells"],
                "map_elites/coverage": me["coverage"],
                "map_elites/best_fitness": me["best_fitness"],
                "map_elites/mean_fitness": me["mean_fitness"],
            }
        )
        if "mean_staleness" in me:
            log["map_elites/mean_staleness"] = me["mean_staleness"]

    # Generic mutation uphill stats (valid for PLWM and base ACCEL minimax mutation).
    log["mutation/num_mutations"] = train_state.num_mutation_updates
    log["mutation/last_uphill_fraction"] = train_state.plwm_last_uphill_fraction
    log["mutation/cumulative_uphill_fraction"] = (
        train_state.num_plwm_improved / jnp.maximum(train_state.num_plwm_compared, 1)
    )
    log["mutation/num_compared"] = train_state.num_plwm_compared
    log["mutation/num_improved"] = train_state.num_plwm_improved
    log["mutation/last_insert_fraction"] = train_state.plwm_last_insert_fraction
    log["mutation/cumulative_insert_fraction"] = (
        train_state.num_plwm_inserted / jnp.maximum(train_state.num_plwm_compared, 1)
    )
    log["mutation/num_inserted"] = train_state.num_plwm_inserted
    log["mutation/last_accepted_uphill_fraction"] = train_state.plwm_last_accepted_uphill_fraction
    log["mutation/cumulative_accepted_uphill_fraction"] = (
        train_state.num_plwm_inserted_improved / jnp.maximum(train_state.num_plwm_inserted, 1)
    )
    log["mutation/num_inserted_improved"] = train_state.num_plwm_inserted_improved
    log["mutation/last_mean_delta_score"] = train_state.plwm_last_mean_delta_score
    log["mutation/cumulative_mean_delta_score"] = (
        train_state.plwm_delta_score_sum / jnp.maximum(train_state.num_plwm_compared, 1)
    )
    log["mutation/last_mean_accepted_delta_score"] = train_state.plwm_last_mean_accepted_delta_score
    log["mutation/cumulative_mean_accepted_delta_score"] = (
        train_state.plwm_accepted_delta_score_sum / jnp.maximum(train_state.num_plwm_inserted, 1)
    )
    log["mutation/scale_fraction_global"] = train_state.plwm_last_scale_global_fraction
    log["mutation/scale_fraction_meso"] = train_state.plwm_last_scale_meso_fraction
    log["mutation/scale_fraction_local"] = train_state.plwm_last_scale_local_fraction
    log["mutation/schedule_progress"] = train_state.plwm_last_schedule_progress

    if use_plwm:
        # Backward-compatible aliases for existing dashboards.
        log["plwm/num_mutations"] = log["mutation/num_mutations"]
        log["plwm/last_uphill_fraction"] = log["mutation/last_uphill_fraction"]
        log["plwm/cumulative_uphill_fraction"] = log["mutation/cumulative_uphill_fraction"]
        log["plwm/num_compared"] = log["mutation/num_compared"]
        log["plwm/num_improved"] = log["mutation/num_improved"]
        log["plwm/last_insert_fraction"] = log["mutation/last_insert_fraction"]
        log["plwm/cumulative_insert_fraction"] = log["mutation/cumulative_insert_fraction"]
        log["plwm/num_inserted"] = log["mutation/num_inserted"]
        log["plwm/last_accepted_uphill_fraction"] = log["mutation/last_accepted_uphill_fraction"]
        log["plwm/cumulative_accepted_uphill_fraction"] = log["mutation/cumulative_accepted_uphill_fraction"]
        log["plwm/num_inserted_improved"] = log["mutation/num_inserted_improved"]
        log["plwm/last_mean_delta_score"] = log["mutation/last_mean_delta_score"]
        log["plwm/cumulative_mean_delta_score"] = log["mutation/cumulative_mean_delta_score"]
        log["plwm/last_mean_accepted_delta_score"] = log["mutation/last_mean_accepted_delta_score"]
        log["plwm/cumulative_mean_accepted_delta_score"] = log["mutation/cumulative_mean_accepted_delta_score"]
        log["plwm/scale_fraction_global"] = log["mutation/scale_fraction_global"]
        log["plwm/scale_fraction_meso"] = log["mutation/scale_fraction_meso"]
        log["plwm/scale_fraction_local"] = log["mutation/scale_fraction_local"]
        log["plwm/schedule_progress"] = log["mutation/schedule_progress"]
        log["plwm/frontier_updates"] = train_state.frontier_updates
        log["plwm/frontier_last_loss"] = train_state.frontier_last_loss
        log["plwm/frontier_last_p_mae"] = train_state.frontier_last_p_mae
        log["plwm/frontier_last_std"] = train_state.frontier_last_std

    return {
        "log": log,
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        },
    }

def compute_score(config, dones, values, max_returns, advantages):
    if config['score_function'] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config['score_function'] == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")


def rollout_success_from_rewards(rewards: chex.Array) -> chex.Array:
    """Binary success per environment from rollout rewards, shape (num_envs,)."""
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


def build_frontier_features(
    latents: chex.Array,
    static_reg: chex.Array,
    valid_prob: chex.Array,
) -> chex.Array:
    """Construct estimator features from frozen VAE latents + static/valid outputs."""
    z = jnp.clip(latents, -4.0, 4.0)
    s = jnp.clip(static_reg, 0.0, 1.0)
    v = jnp.clip(valid_prob, 0.0, 1.0)[:, None]
    return jnp.concatenate([z, s, v], axis=-1)


def compute_structural_frontier_stats(levels: Level) -> dict[str, chex.Array]:
    """Build CLUTTR-path frontier features from decoded level geometry."""
    targets = compute_structural_targets(
        levels.wall_map,
        levels.goal_pos,
        levels.agent_pos,
    )
    valid_prob = jnp.clip(targets[:, 0], 0.0, 1.0)
    static_reg = jnp.clip(targets[:, 1:], 0.0, 1.0)
    return {
        "static_reg": static_reg,
        "valid_prob": valid_prob,
        "invalid_prob": 1.0 - valid_prob,
        "wall_density_pred": static_reg[:, 0],
        "bfs_norm_pred": static_reg[:, 1],
    }


def init_frontier_params(
    rng: chex.PRNGKey,
    feature_dim: int,
    hidden_dim: int,
    ensemble_size: int,
) -> core.FrozenDict[str, chex.Array]:
    """Initialize a small frontier MLP ensemble."""
    rng_w1, rng_w2 = jax.random.split(rng)
    scale1 = jnp.sqrt(2.0 / float(max(feature_dim, 1)))
    scale2 = jnp.sqrt(2.0 / float(max(hidden_dim, 1)))
    return core.freeze(
        {
            "w1": (
                scale1
                * jax.random.normal(
                    rng_w1,
                    (ensemble_size, feature_dim, hidden_dim),
                    dtype=jnp.float32,
                )
            ),
            "b1": jnp.zeros((ensemble_size, hidden_dim), dtype=jnp.float32),
            "w2": (
                scale2
                * jax.random.normal(
                    rng_w2,
                    (ensemble_size, hidden_dim),
                    dtype=jnp.float32,
                )
            ),
            "b2": jnp.zeros((ensemble_size,), dtype=jnp.float32),
        }
    )


def frontier_predict_members(
    features: chex.Array,
    params: core.FrozenDict[str, chex.Array],
) -> chex.Array:
    """Predict success probabilities for each ensemble member, shape (batch, ensemble)."""
    hidden = jnp.einsum("bd,edh->beh", features, params["w1"]) + params["b1"][None, :, :]
    hidden = jax.nn.relu(hidden)
    logits = jnp.einsum("beh,eh->be", hidden, params["w2"]) + params["b2"][None, :]
    return jax.nn.sigmoid(logits)


def frontier_predict_stats(
    features: chex.Array,
    params: core.FrozenDict[str, chex.Array],
) -> tuple[chex.Array, chex.Array]:
    preds = frontier_predict_members(features, params)
    return preds.mean(axis=-1), preds.std(axis=-1)


def frontier_candidate_novelty(
    candidate_latents: chex.Array,
    parent_latents: chex.Array,
    knn_k: int,
) -> chex.Array:
    """Explicit novelty bonus: move away from the current replay batch, not just the parent."""
    batch_size, num_candidates, latent_dim = candidate_latents.shape
    delta = jnp.linalg.norm(candidate_latents - parent_latents[:, None, :], axis=-1)
    if batch_size <= 1:
        return delta / jnp.sqrt(float(latent_dim))

    flat = candidate_latents.reshape((batch_size * num_candidates, latent_dim))
    dists = jnp.linalg.norm(flat[:, None, :] - parent_latents[None, :, :], axis=-1)
    parent_idx = jnp.repeat(jnp.arange(batch_size), num_candidates)
    dists = dists.at[jnp.arange(batch_size * num_candidates), parent_idx].set(jnp.inf)
    knn = jnp.sort(dists, axis=-1)[:, :knn_k].mean(axis=-1).reshape((batch_size, num_candidates))
    scale = jnp.sqrt(float(latent_dim))
    return 0.5 * (delta / scale) + 0.5 * (knn / scale)


def frontier_candidate_novelty_from_coords(
    candidate_coords: chex.Array,
    parent_coords: chex.Array,
    knn_k: int,
) -> chex.Array:
    """Novelty bonus in a user-specified coordinate system, typically whitened PCA."""
    batch_size, num_candidates, coord_dim = candidate_coords.shape
    delta = jnp.linalg.norm(candidate_coords - parent_coords[:, None, :], axis=-1)
    if batch_size <= 1:
        return delta / jnp.sqrt(float(coord_dim))

    flat = candidate_coords.reshape((batch_size * num_candidates, coord_dim))
    dists = jnp.linalg.norm(flat[:, None, :] - parent_coords[None, :, :], axis=-1)
    parent_idx = jnp.repeat(jnp.arange(batch_size), num_candidates)
    dists = dists.at[jnp.arange(batch_size * num_candidates), parent_idx].set(jnp.inf)
    knn = jnp.sort(dists, axis=-1)[:, :knn_k].mean(axis=-1).reshape((batch_size, num_candidates))
    scale = jnp.sqrt(float(coord_dim))
    return 0.5 * (delta / scale) + 0.5 * (knn / scale)


def project_to_whitened_pca(
    latents: chex.Array,
    pca_mean: chex.Array,
    pca_eigvecs: chex.Array,
    pca_eigvals: chex.Array,
) -> chex.Array:
    centered = latents - pca_mean[None, :]
    coords = centered @ pca_eigvecs.T
    return coords / jnp.sqrt(jnp.maximum(pca_eigvals[None, :], 1e-8))


def build_signed_pc_bank(
    latent_dim: int,
    num_candidates: int,
    pc_ids: Sequence[int],
    primary_pc_scale: float = 1.0,
) -> jnp.ndarray:
    bank = np.zeros((num_candidates, latent_dim), dtype=np.float32)
    if num_candidates <= 0 or latent_dim <= 0 or not pc_ids:
        return jnp.asarray(bank)

    valid_pc_ids = [int(pc) for pc in pc_ids if 0 <= int(pc) < latent_dim]
    if not valid_pc_ids:
        valid_pc_ids = [0]

    for i in range(num_candidates):
        pc = valid_pc_ids[i % len(valid_pc_ids)]
        sign = 1.0 if ((i // len(valid_pc_ids)) % 2 == 0) else -1.0
        scale = primary_pc_scale if pc == 0 else 1.0
        bank[i, pc] = sign * scale

    return jnp.asarray(bank)


def build_global_pca_offset_bank(
    latent_dim: int,
    num_candidates: int,
    global_top_k: int,
    pc1_scale: float,
    pc1_fraction: float,
) -> jnp.ndarray:
    """Create coarse PCA probes: capped PC1 plus a few top local PCs."""
    bank = np.zeros((num_candidates, latent_dim), dtype=np.float32)
    if num_candidates <= 0 or latent_dim <= 0:
        return jnp.asarray(bank)

    if latent_dim == 1:
        return build_signed_pc_bank(latent_dim, num_candidates, [0], primary_pc_scale=pc1_scale)

    other_ids = list(range(1, min(global_top_k, latent_dim)))
    pc1_count = int(round(pc1_fraction * num_candidates))
    if pc1_fraction > 0.0:
        pc1_count = max(2, pc1_count)
    pc1_count = min(num_candidates, pc1_count)
    if not other_ids:
        pc1_count = num_candidates
    elif pc1_count == num_candidates:
        pc1_count = max(1, num_candidates - 1)
    if pc1_count % 2 == 1 and pc1_count < num_candidates:
        pc1_count += 1

    if pc1_count > 0:
        bank[:pc1_count] = np.asarray(
            build_signed_pc_bank(latent_dim, pc1_count, [0], primary_pc_scale=pc1_scale)
        )
    if pc1_count < num_candidates:
        bank[pc1_count:] = np.asarray(
            build_signed_pc_bank(latent_dim, num_candidates - pc1_count, other_ids, primary_pc_scale=pc1_scale)
        )
    return jnp.asarray(bank)


def training_progress_fraction(train_state: TrainState, total_updates: int) -> chex.Array:
    total = (
        train_state.num_dr_updates
        + train_state.num_replay_updates
        + train_state.num_mutation_updates
    )
    return jnp.clip(
        total / jnp.maximum(jnp.asarray(total_updates, dtype=jnp.float32), 1.0),
        0.0,
        1.0,
    )


def annealed_pca_scale_mix(
    progress: chex.Array,
    global_start: float,
    global_end: float,
    local_start: float,
    local_end: float,
) -> chex.Array:
    global_w = (1.0 - progress) * float(global_start) + progress * float(global_end)
    local_w = (1.0 - progress) * float(local_start) + progress * float(local_end)
    meso_w = jnp.maximum(1e-3, 1.0 - global_w - local_w)
    weights = jnp.stack([global_w, meso_w, local_w], axis=0)
    return weights / jnp.maximum(weights.sum(), 1e-6)


def condition_pca_scale_mix(
    base_weights: chex.Array,
    parent_success_ema: chex.Array,
    low: float,
    high: float,
    strength: float,
) -> chex.Array:
    band_width = max(float(high) - float(low), 1e-3)
    outside_dist = jnp.where(
        parent_success_ema < float(low),
        float(low) - parent_success_ema,
        jnp.where(parent_success_ema > float(high), parent_success_ema - float(high), 0.0),
    )
    frontier_proximity = jnp.clip(1.0 - outside_dist / band_width, 0.0, 1.0)
    weights = jnp.broadcast_to(base_weights[None, :], (parent_success_ema.shape[0], 3))
    weights = weights.at[:, 0].set(weights[:, 0] * (1.0 + float(strength) * (1.0 - frontier_proximity)))
    weights = weights.at[:, 2].set(weights[:, 2] * (1.0 + float(strength) * frontier_proximity))
    return weights / jnp.maximum(weights.sum(axis=1, keepdims=True), 1e-6)


def expand_multiscale_pca_candidate_offsets(
    global_bank: chex.Array,
    meso_bank: chex.Array,
    local_bank: chex.Array,
    mix_weights: chex.Array,
) -> tuple[chex.Array, chex.Array]:
    """Map per-parent scale weights to a deterministic candidate bank allocation."""
    num_candidates = global_bank.shape[0]
    positions = (jnp.arange(num_candidates, dtype=jnp.float32) + 0.5) / jnp.maximum(float(num_candidates), 1.0)
    global_cut = mix_weights[:, 0:1]
    meso_cut = mix_weights[:, 0:1] + mix_weights[:, 1:2]
    is_global = positions[None, :] < global_cut
    is_meso = (positions[None, :] >= global_cut) & (positions[None, :] < meso_cut)
    is_local = ~(is_global | is_meso)
    offsets = jnp.where(
        is_global[..., None],
        global_bank[None, :, :],
        jnp.where(is_meso[..., None], meso_bank[None, :, :], local_bank[None, :, :]),
    )
    fractions = jnp.stack(
        [
            is_global.astype(jnp.float32).mean(axis=1),
            is_meso.astype(jnp.float32).mean(axis=1),
            is_local.astype(jnp.float32).mean(axis=1),
        ],
        axis=-1,
    )
    return offsets, fractions


def build_structured_pca_candidates(
    parent_latents: chex.Array,
    parent_success_ema: chex.Array,
    train_state: TrainState,
    config: dict,
    global_bank: chex.Array,
    meso_bank: chex.Array,
    local_bank: chex.Array,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Create multi-scale PCA candidates with annealed and parent-conditioned allocation."""
    progress = training_progress_fraction(train_state, int(config["num_updates"]))
    base_mix = annealed_pca_scale_mix(
        progress,
        config["plwm_pca_global_weight_start"],
        config["plwm_pca_global_weight_end"],
        config["plwm_pca_local_weight_start"],
        config["plwm_pca_local_weight_end"],
    )
    mix_weights = condition_pca_scale_mix(
        base_mix,
        parent_success_ema,
        config["plwm_target_success_low"],
        config["plwm_target_success_high"],
        config["plwm_pca_success_conditioning"],
    )
    candidate_offsets_white_unit, scale_fractions = expand_multiscale_pca_candidate_offsets(
        global_bank,
        meso_bank,
        local_bank,
        mix_weights,
    )
    parent_pca_coords = project_to_whitened_pca(
        parent_latents,
        train_state.plwm_pca_mean,
        train_state.plwm_pca_eigvecs,
        train_state.plwm_pca_eigvals,
    )
    candidate_offsets_white = float(config["plwm_sigma"]) * candidate_offsets_white_unit
    scaled = candidate_offsets_white * jnp.sqrt(jnp.maximum(train_state.plwm_pca_eigvals[None, None, :], 1e-8))
    candidate_latents = parent_latents[:, None, :] + scaled @ train_state.plwm_pca_eigvecs
    novelty_candidate_coords = parent_pca_coords[:, None, :] + candidate_offsets_white
    schedule_stats = jnp.array(
        [
            scale_fractions[:, 0].mean(),
            scale_fractions[:, 1].mean(),
            scale_fractions[:, 2].mean(),
            progress,
        ],
        dtype=jnp.float32,
    )
    return candidate_latents, parent_pca_coords, novelty_candidate_coords, mix_weights, schedule_stats


def frontier_update_step(
    params: core.FrozenDict[str, chex.Array],
    features: chex.Array,
    targets: chex.Array,
    weights: chex.Array,
    lr: float,
    l2: float,
) -> tuple[core.FrozenDict[str, chex.Array], chex.Array, chex.Array, chex.Array]:
    """Single weighted SGD update for an ensemble MLP frontier estimator."""
    denom = jnp.maximum(weights.sum(), 1e-6)

    def loss_fn(curr_params):
        preds = frontier_predict_members(features, curr_params)
        logits = jnp.log(jnp.clip(preds, 1e-6, 1.0 - 1e-6)) - jnp.log1p(-jnp.clip(preds, 1e-6, 1.0 - 1e-6))
        bce = optax.sigmoid_binary_cross_entropy(logits, targets[:, None])
        data_loss = (weights[:, None] * bce).sum() / jnp.maximum(
            denom * preds.shape[-1], 1e-6
        )
        reg = 0.5 * (
            jnp.sum(curr_params["w1"] ** 2) + jnp.sum(curr_params["w2"] ** 2)
        ) / float(preds.shape[-1])
        loss = data_loss + l2 * reg
        mean_pred = preds.mean(axis=-1)
        p_mae = (weights * jnp.abs(mean_pred - targets)).sum() / denom
        pred_std = (weights * preds.std(axis=-1)).sum() / denom
        return loss, (p_mae, pred_std)

    (loss, (p_mae, pred_std)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return core.freeze(new_params), loss, p_mae, pred_std


def structural_difficulty_surrogate(
    wall_map: chex.Array,
    goal_pos: chex.Array,
    agent_pos: chex.Array,
    *,
    weight_bfs: float,
    weight_slack: float,
    weight_dead_ends: float,
    weight_walls: float,
    weight_branches: float,
    require_solvable: bool,
) -> chex.Array:
    """Structural difficulty proxy used for surrogate-guided PLWM candidate ranking."""
    return plwm_scoring.structural_difficulty_surrogate(
        wall_map,
        goal_pos,
        agent_pos,
        weight_bfs=weight_bfs,
        weight_slack=weight_slack,
        weight_dead_ends=weight_dead_ends,
        weight_walls=weight_walls,
        weight_branches=weight_branches,
        require_solvable=require_solvable,
    )

def main(config=None, project="JAXUED_TEST"):
    tags = []
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")
    if config["use_map_elites_mutation"]:
        tags.append("MAP_ELITES_MUT")
    if config["use_plwm_mutation"]:
        tags.append("PLWM_MUT")
    run = wandb.init(config=config, project=project, group=config["run_name"], tags=tags)
    config = wandb.config
    
    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")
    wandb.define_metric("mutation/*", step_metric="num_updates")
    wandb.define_metric("plwm/*", step_metric="num_updates")

    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")
        
        # generic stats
        env_steps = stats["update_count"] * config["num_train_envs"] * config["num_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats['time_delta'],
        }
        
        # evaluation performance
        solve_rates = stats['eval_solve_rates']
        returns     = stats["eval_returns"]
        log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config["eval_levels"], solve_rates)})
        log_dict.update({"solve_rate/mean": solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config["eval_levels"], returns)})
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()})
        
        # level sampler
        log_dict.update(train_state_info["log"])

        # images
        log_dict.update({"images/highest_scoring_level": wandb.Image(np.array(stats["highest_scoring_level"]), caption="Highest scoring level")})
        log_dict.update({"images/highest_weighted_level": wandb.Image(np.array(stats["highest_weighted_level"]), caption="Highest weighted level")})

        for s in ['dr', 'replay', 'mutation']:
            if train_state_info['info'][f'num_{s}_updates'] > 0:
                log_dict.update({f"images/{s}_levels": [wandb.Image(np.array(image)) for image in stats[f"{s}_levels"]]})

        # animations
        for i, level_name in enumerate(config["eval_levels"]):
            frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
            frames = np.array(frames[:episode_length])
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4, format="gif")})
        
        wandb.log(log_dict)

    decoder_params = None
    encoder_params = None
    maze_encoder_params = None
    maze_decoder_params = None
    maze_full_params = None
    cnn_vae_params = None
    frontier_feature_dim = 1
    me_latent_dim = int(config["me_latent_dim"])
    me_descriptor_mode = str(config["me_descriptor_mode"]).lower()
    if me_descriptor_mode not in ("behavior", "latent", "hybrid", "bfs"):
        raise ValueError(
            f"Invalid --me_descriptor_mode={config['me_descriptor_mode']}. "
            "Choose from: behavior, latent, hybrid, bfs."
        )

    me_latent_bin_count = int(config["me_latent_bin_count"])
    if me_latent_bin_count < 2:
        raise ValueError("--me_latent_bin_count must be >= 2.")
    me_latent_bins = jnp.linspace(
        float(config["me_latent_bin_min"]),
        float(config["me_latent_bin_max"]),
        me_latent_bin_count + 1,
        dtype=jnp.float32,
    )
    me_latent_projections = make_latent_projections(
        me_latent_dim, seed=int(config["me_latent_descriptor_seed"])
    )

    if me_descriptor_mode == "behavior":
        me_axis1_bins = OBS_BINS.astype(jnp.float32)
        me_axis2_bins = DIST_BINS.astype(jnp.float32)
        me_insert_latent_projections = None
    elif me_descriptor_mode == "latent":
        me_axis1_bins = me_latent_bins
        me_axis2_bins = me_latent_bins
        me_insert_latent_projections = me_latent_projections
    elif me_descriptor_mode == "hybrid":
        # Hybrid: behaviorally meaningful distance axis + latent geometry axis.
        me_axis1_bins = DIST_BINS.astype(jnp.float32)
        me_axis2_bins = me_latent_bins
        me_insert_latent_projections = me_latent_projections
    else:
        # "bfs": BFS path-length × obstacle-count — fully structural, no latent projection.
        # 8 × 8 = 64 cells; every cell corresponds to a distinct difficulty region.
        me_axis1_bins = BFS_PATH_BINS.astype(jnp.float32)
        me_axis2_bins = DENSE_OBS_BINS.astype(jnp.float32)
        me_insert_latent_projections = None
    me_archive_cells = num_cells(me_axis1_bins, me_axis2_bins)
    me_staleness_decay_rate = float(config["me_staleness_decay_rate"])

    if config["use_map_elites_mutation"] and config["mode"] == "train":
        if not config["use_accel"]:
            raise ValueError("--use_map_elites_mutation requires --use_accel")

        me_checkpoint_path = config["me_vae_checkpoint"]
        if me_checkpoint_path is None:
            me_checkpoint_path = os.path.join(ROOT_DIR, "vae", "model", "checkpoint_420000.pkl")
        elif not os.path.isabs(me_checkpoint_path):
            me_checkpoint_path = os.path.abspath(os.path.join(os.getcwd(), me_checkpoint_path))

        print(f"Loading MAP-Elites decoder from {me_checkpoint_path}...")
        full_vae_params = load_vae_params(me_checkpoint_path)
        decoder_params = extract_decoder_params(full_vae_params)
        print(
            f"MAP-Elites mutation enabled: latent_dim={me_latent_dim}, "
            f"sigma={config['me_mutation_sigma']}, temp={config['me_decode_temperature']}, "
            f"descriptor_mode={me_descriptor_mode}, cells={me_archive_cells}"
        )

    if config["use_plwm_mutation"] and config["mode"] == "train":
        if not config["use_accel"]:
            raise ValueError("--use_plwm_mutation requires --use_accel")
        if config["use_map_elites_mutation"]:
            raise ValueError("--use_plwm_mutation and --use_map_elites_mutation are mutually exclusive")
        if config["plwm_online_frontier_guided"] and config["plwm_task_aware_guided"]:
            raise ValueError("--plwm_online_frontier_guided and --plwm_task_aware_guided are mutually exclusive.")
        if config["plwm_use_maze_ae"] and config["plwm_use_cnn_vae"]:
            raise ValueError("--plwm_use_maze_ae and --plwm_use_cnn_vae are mutually exclusive.")
        if config["plwm_surrogate_guided"] and config["plwm_task_aware_guided"] and (
            (not config["plwm_use_maze_ae"]) or config["plwm_use_cnn_vae"]
        ):
            raise ValueError("--plwm_task_aware_guided currently requires --plwm_use_maze_ae and does not support --plwm_use_cnn_vae.")
        if int(config["plwm_frontier_hidden_dim"]) < 1:
            raise ValueError("--plwm_frontier_hidden_dim must be >= 1.")
        if int(config["plwm_frontier_ensemble_size"]) < 1:
            raise ValueError("--plwm_frontier_ensemble_size must be >= 1.")
        if int(config["plwm_frontier_novelty_knn"]) < 1:
            raise ValueError("--plwm_frontier_novelty_knn must be >= 1.")

        if config["plwm_use_maze_ae"]:
            # Grid-based CNN AE path — load only the MazeAE checkpoint
            mae_path = config["plwm_mae_checkpoint"]
            if mae_path is None:
                mae_path = os.path.join(ROOT_DIR, "vae", "model_maze_ae", "checkpoint_final.pkl")
            elif not os.path.isabs(mae_path):
                mae_path = os.path.abspath(os.path.join(os.getcwd(), mae_path))
            print(f"Loading Maze AE from {mae_path}...")
            full_mae_params = load_maze_ae_params(mae_path)
            if config["plwm_surrogate_guided"] and config["plwm_task_aware_guided"]:
                required = ("StaticHead_0", "CurriculumHead_0", "ValidHead_0")
                missing = [k for k in required if k not in full_mae_params]
                if missing:
                    raise ValueError(
                        "Task-aware PLWM requires a task-aware MazeAE checkpoint with heads. "
                        f"Missing keys: {missing}"
                    )
            maze_full_params = full_mae_params
            maze_encoder_params = extract_maze_encoder_params(full_mae_params)
            maze_decoder_params = extract_maze_decoder_params(full_mae_params)
            if "mean_layer" in maze_encoder_params:
                frontier_feature_dim = int(maze_encoder_params["mean_layer"]["kernel"].shape[-1]) + 6 + 1
            else:
                frontier_feature_dim = int(me_latent_dim) + 6 + 1
            print(
                f"PLWM mutation enabled (Maze AE, grid-based): "
                f"sigma={config['plwm_sigma']}, temp={config['plwm_decode_temperature']}"
            )
        elif config["plwm_use_cnn_vae"]:
            cnn_path = config["plwm_cnn_vae_checkpoint"]
            if cnn_path is None:
                cnn_path = os.path.join(ROOT_DIR, "vae", "checkpoints", "cnn_vae", "run11_1M")
            elif not os.path.isabs(cnn_path):
                cnn_path = os.path.abspath(os.path.join(os.getcwd(), cnn_path))
            print(f"Loading CNN maze VAE from {cnn_path}...")
            cnn_vae_params = load_cnn_vae_params(cnn_path)
            frontier_feature_dim = int(cnn_vae_latent_dim(cnn_vae_params)) + 6 + 1
            print(
                "PLWM mutation enabled (CNN VAE, grid-based): "
                f"sigma={config['plwm_sigma']}, temp=deterministic-walls"
            )
        else:
            # CLUTTR sequence VAE path
            plwm_checkpoint_path = config["me_vae_checkpoint"]
            if plwm_checkpoint_path is None:
                plwm_checkpoint_path = os.path.join(ROOT_DIR, "vae", "model", "checkpoint_420000.pkl")
            elif not os.path.isabs(plwm_checkpoint_path):
                plwm_checkpoint_path = os.path.abspath(os.path.join(os.getcwd(), plwm_checkpoint_path))
            print(f"Loading PLWM CLUTTR encoder+decoder from {plwm_checkpoint_path}...")
            full_vae_params = load_vae_params(plwm_checkpoint_path)
            encoder_params = extract_encoder_params(full_vae_params)
            decoder_params = extract_decoder_params(full_vae_params)
            frontier_feature_dim = int(encoder_params["mean_layer"]["kernel"].shape[-1]) + 6 + 1
            print(
                f"PLWM mutation enabled (CLUTTR VAE): latent_dim={me_latent_dim}, "
                f"sigma={config['plwm_sigma']}, temp={config['plwm_decode_temperature']}"
            )

        if config["plwm_surrogate_guided"]:
            if int(config["plwm_surrogate_num_candidates"]) < 1:
                raise ValueError("--plwm_surrogate_num_candidates must be >= 1.")
            mode = (
                "online-frontier"
                if config["plwm_online_frontier_guided"]
                else ("task-aware" if config["plwm_task_aware_guided"] else "structural")
            )
            print(
                "PLWM surrogate guidance enabled: "
                f"mode={mode}, "
                f"candidates={config['plwm_surrogate_num_candidates']}, "
                f"weights(bfs={config['plwm_surrogate_weight_bfs']}, "
                f"slack={config['plwm_surrogate_weight_slack']}, "
                f"dead={config['plwm_surrogate_weight_dead_ends']}, "
                f"walls={config['plwm_surrogate_weight_walls']}, "
                f"branch={config['plwm_surrogate_weight_branches']}), "
                f"require_solvable={config['plwm_surrogate_require_solvable']}"
            )
            if config["plwm_online_frontier_guided"]:
                print(
                    "Frontier ensemble: "
                    f"hidden={config['plwm_frontier_hidden_dim']}, "
                    f"ensemble={config['plwm_frontier_ensemble_size']}, "
                    f"uncertainty_w={config['plwm_frontier_uncertainty_weight']}, "
                    f"novelty_w={config['plwm_frontier_novelty_weight']}, "
                    f"novelty_knn={config['plwm_frontier_novelty_knn']}"
                )
        elif config["plwm_online_frontier_guided"] or config["plwm_task_aware_guided"]:
            raise ValueError(
                "--plwm_online_frontier_guided and --plwm_task_aware_guided require "
                "--plwm_surrogate_guided."
            )

    # Latent dim for PLWM PCA (derived from loaded checkpoint; 1 when unused)
    plwm_latent_dim = 1
    if config["plwm_use_pca_mutation"] and config["use_plwm_mutation"]:
        if config["plwm_use_maze_ae"] and maze_encoder_params is not None:
            if "mean_layer" in maze_encoder_params:
                # Variational encoder: use mean_layer output dim
                plwm_latent_dim = int(maze_encoder_params["mean_layer"]["kernel"].shape[-1])
            elif "Dense_1" in maze_encoder_params:
                # Deterministic legacy encoder: use last Dense output dim
                plwm_latent_dim = int(maze_encoder_params["Dense_1"]["kernel"].shape[-1])
        elif config["plwm_use_cnn_vae"] and cnn_vae_params is not None:
            plwm_latent_dim = int(cnn_vae_latent_dim(cnn_vae_params))
        elif not config["plwm_use_maze_ae"] and encoder_params is not None:
            # CLUTTR VAE encoder also has mean_layer key
            if "mean_layer" in encoder_params:
                plwm_latent_dim = int(encoder_params["mean_layer"]["kernel"].shape[-1])

    local_pc_ids = list(range(1, min(int(config["plwm_pca_local_top_k"]), plwm_latent_dim)))
    if not local_pc_ids:
        local_pc_ids = [0]
    plwm_pca_global_bank = float(config["plwm_pca_global_scale"]) * build_global_pca_offset_bank(
        latent_dim=plwm_latent_dim,
        num_candidates=int(config["plwm_surrogate_num_candidates"]),
        global_top_k=int(config["plwm_pca_global_top_k"]),
        pc1_scale=float(config["plwm_pca_pc1_scale"]),
        pc1_fraction=float(config["plwm_pca_pc1_candidate_fraction"]),
    )
    plwm_pca_meso_bank = float(config["plwm_pca_meso_scale"]) * build_signed_pc_bank(
        latent_dim=plwm_latent_dim,
        num_candidates=int(config["plwm_surrogate_num_candidates"]),
        pc_ids=local_pc_ids,
    )
    plwm_pca_local_bank = float(config["plwm_pca_local_scale"]) * build_signed_pc_bank(
        latent_dim=plwm_latent_dim,
        num_candidates=int(config["plwm_surrogate_num_candidates"]),
        pc_ids=local_pc_ids,
    )

    # Setup the environment
    env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    eval_env = env
    sample_random_level = make_level_generator(env.max_height, env.max_width, config["n_walls"])
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    mutate_level = make_level_mutator_minimax(100)

    # And the level sampler    
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config['topk_k']},
        duplicate_check=config['buffer_duplicate_check'],
    )

    me_update_period = max(int(config["me_update_period"]), 1)
    me_min_obstacles = int(config["me_min_obstacles"])
    me_min_distance = int(config["me_min_distance"])
    
    @jax.jit
    def create_train_state(rng) -> TrainState:
        # Creates the train state
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["num_minibatches"] * config["epoch_ppo"]))
                / config["num_updates"]
            )
            return config["lr"] * frac
        rng_model, rng_frontier, rng_level = jax.random.split(rng, 3)
        obs, _ = env.reset_to_level(rng_level, sample_random_level(rng_level), env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
            obs,
        )
        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        network = ActorCritic(env.action_space(env_params).n)
        network_params = network.init(rng_model, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
            # optax.adam(learning_rate=config["lr"], eps=1e-5),
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
        me_archive = init_map_elites_archive(me_latent_dim, cells=me_archive_cells)
        pholder_level_batch = jax.tree_util.tree_map(lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), pholder_level)
        pholder_level_inds = jnp.zeros((config["num_train_envs"],), dtype=jnp.int32)
        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            me_archive=me_archive,
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            num_plwm_compared=0,
            num_plwm_improved=0,
            num_plwm_inserted=0,
            num_plwm_inserted_improved=0,
            plwm_last_uphill_fraction=jnp.array(0.0, dtype=jnp.float32),
            plwm_last_insert_fraction=jnp.array(0.0, dtype=jnp.float32),
            plwm_last_accepted_uphill_fraction=jnp.array(0.0, dtype=jnp.float32),
            plwm_delta_score_sum=jnp.array(0.0, dtype=jnp.float32),
            plwm_accepted_delta_score_sum=jnp.array(0.0, dtype=jnp.float32),
            plwm_last_mean_delta_score=jnp.array(0.0, dtype=jnp.float32),
            plwm_last_mean_accepted_delta_score=jnp.array(0.0, dtype=jnp.float32),
            plwm_last_scale_global_fraction=jnp.array(0.0, dtype=jnp.float32),
            plwm_last_scale_meso_fraction=jnp.array(0.0, dtype=jnp.float32),
            plwm_last_scale_local_fraction=jnp.array(0.0, dtype=jnp.float32),
            plwm_last_schedule_progress=jnp.array(0.0, dtype=jnp.float32),
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            replay_last_level_inds=pholder_level_inds,
            mutation_last_level_batch=pholder_level_batch,
            frontier_params=init_frontier_params(
                rng_frontier,
                frontier_feature_dim,
                int(config["plwm_frontier_hidden_dim"]),
                int(config["plwm_frontier_ensemble_size"]),
            ),
            frontier_updates=0,
            frontier_last_loss=jnp.array(0.0, dtype=jnp.float32),
            frontier_last_p_mae=jnp.array(0.0, dtype=jnp.float32),
            frontier_last_std=jnp.array(0.0, dtype=jnp.float32),
            plwm_pca_eigvecs=jnp.eye(plwm_latent_dim, dtype=jnp.float32),
            plwm_pca_eigvals=jnp.ones(plwm_latent_dim, dtype=jnp.float32),
            plwm_pca_mean=jnp.zeros(plwm_latent_dim, dtype=jnp.float32),
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """
            This is the main training loop. It basically calls either `on_new_levels`, `on_replay_levels`, or `on_mutate_levels` at every step.
        """
        def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                Samples new (randomly-generated) levels and evaluates the policy on these. It also then adds the levels to the level buffer if they have high-enough scores.
                The agent is updated on these trajectories iff `config["exploratory_grad_updates"]` is True.
            """
            sampler = train_state.sampler
            
            # Reset
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_train_envs"]))
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), new_levels, env_params)
            # Rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = compute_max_returns(dones, rewards)
            scores = compute_score(config, dones, values, max_returns, advantages)
            success = rollout_success_from_rewards(rewards)
            sampler, _ = level_sampler.insert_batch(
                sampler,
                new_levels,
                scores,
                {
                    "max_return": max_returns,
                    "success_ema": success,
                    "success_obs_count": jnp.ones_like(success),
                },
            )
            
            # Update: train_state only modified if exploratory_grad_updates is on
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )
            
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": new_levels.wall_map.sum() / config["num_train_envs"],
                "me_insertions": jnp.array(0, dtype=jnp.int32),
                "plwm_batch_uphill_fraction": jnp.array(0.0, dtype=jnp.float32),
                "plwm_batch_insert_fraction": jnp.array(0.0, dtype=jnp.float32),
                "plwm_batch_accepted_uphill_fraction": jnp.array(0.0, dtype=jnp.float32),
                "plwm_batch_mean_delta_score": jnp.array(0.0, dtype=jnp.float32),
            }
            
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=new_levels,
            )
            return (rng, train_state), metrics
        
        def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                This samples levels from the level buffer, and updates the policy on them.
            """
            sampler = train_state.sampler
            
            # Collect trajectories on replay levels
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, config["num_train_envs"])
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params)
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
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

            frontier_params = train_state.frontier_params
            frontier_updates_inc = jnp.array(0, dtype=jnp.int32)
            frontier_last_loss = train_state.frontier_last_loss
            frontier_last_p_mae = train_state.frontier_last_p_mae
            frontier_last_std = train_state.frontier_last_std
            if (
                config["use_plwm_mutation"]
                and config["plwm_surrogate_guided"]
                and config["plwm_online_frontier_guided"]
            ):
                if config["plwm_use_maze_ae"] or config["plwm_use_cnn_vae"]:
                    replay_grids = jax.vmap(maze_level_to_grid)(
                        levels.wall_map,
                        levels.goal_pos,
                        levels.agent_pos,
                    )
                    if config["plwm_use_maze_ae"]:
                        replay_latents = encode_maze_levels(maze_encoder_params, replay_grids)
                        replay_preds = predict_task_targets(maze_full_params, replay_grids)
                    else:
                        replay_latents = encode_cnn_vae_levels(cnn_vae_params, replay_grids)
                        replay_preds = compute_structural_frontier_stats(levels)
                    replay_features = build_frontier_features(
                        replay_latents,
                        replay_preds["static_reg"],
                        replay_preds["valid_prob"],
                    )
                else:
                    replay_seqs = jax.vmap(level_to_cluttr_sequence)(
                        levels.wall_map,
                        levels.goal_pos,
                        levels.agent_pos,
                    )
                    replay_latents = encode_levels_to_latents(encoder_params, replay_seqs)
                    replay_preds = compute_structural_frontier_stats(levels)
                    replay_features = build_frontier_features(
                        replay_latents,
                        replay_preds["static_reg"],
                        replay_preds["valid_prob"],
                    )
                frontier_targets = new_success_ema
                frontier_weights = jnp.minimum(
                    1.0,
                    new_success_obs_count / float(config["plwm_frontier_conf_ref"]),
                )
                frontier_params, frontier_last_loss, frontier_last_p_mae, frontier_last_std = frontier_update_step(
                    frontier_params,
                    replay_features,
                    frontier_targets,
                    frontier_weights,
                    lr=float(config["plwm_frontier_lr"]),
                    l2=float(config["plwm_frontier_l2"]),
                )
                frontier_updates_inc = jnp.array(1, dtype=jnp.int32)

            sampler = level_sampler.update_batch(
                sampler,
                level_inds,
                scores,
                {
                    "max_return": max_returns,
                    "success_ema": new_success_ema,
                    "success_obs_count": new_success_obs_count,
                },
            )
            
            # Update the policy using trajectories collected from replay levels
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=True,
            )
                            
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],
                "me_insertions": jnp.array(0, dtype=jnp.int32),
                "plwm_batch_uphill_fraction": jnp.array(0.0, dtype=jnp.float32),
                "plwm_batch_insert_fraction": jnp.array(0.0, dtype=jnp.float32),
                "plwm_batch_accepted_uphill_fraction": jnp.array(0.0, dtype=jnp.float32),
                "plwm_batch_mean_delta_score": jnp.array(0.0, dtype=jnp.float32),
            }
            
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
                replay_last_level_inds=level_inds,
                frontier_params=frontier_params,
                frontier_updates=train_state.frontier_updates + frontier_updates_inc,
                frontier_last_loss=frontier_last_loss,
                frontier_last_p_mae=frontier_last_p_mae,
                frontier_last_std=frontier_last_std,
            )
            return (rng, train_state), metrics
        
        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                This mutates the previous batch of replay levels and potentially adds them to the level buffer.
                This also updates the policy iff `config["exploratory_grad_updates"]` is True.
            """
            sampler = train_state.sampler
            me_archive = train_state.me_archive
            rng, rng_mutate, rng_reset = jax.random.split(rng, 3)
            plwm_schedule_stats = jnp.zeros((4,), dtype=jnp.float32)
            
            # mutate
            if config["use_map_elites_mutation"]:
                child_levels, child_latents, child_sequences = map_elites_mutate_levels(
                    rng_mutate,
                    me_archive,
                    decoder_params=decoder_params,
                    batch_size=config["num_train_envs"],
                    latent_sigma=config["me_mutation_sigma"],
                    decode_temperature=config["me_decode_temperature"],
                    uniform_fraction=config["me_uniform_parent_fraction"],
                    softmax_temperature=config["me_fitness_softmax_temp"],
                    current_step=train_state.num_mutation_updates,
                    staleness_decay_rate=me_staleness_decay_rate,
                )
            elif config["use_plwm_mutation"]:
                # PLR-Weighted Latent Mutation:
                #   Parents = replay_last_level_batch (already PLR score-weighted).
                #   Encode -> perturb -> decode -> evaluate -> PLR insert.
                parent_levels = train_state.replay_last_level_batch
                parent_level_inds = train_state.replay_last_level_inds
                parent_scores = train_state.sampler["scores"][parent_level_inds]
                parent_extras = level_sampler.get_levels_extra(train_state.sampler, parent_level_inds)
                parent_success_ema = parent_extras["success_ema"]
                rng_mutate, rng_encode, rng_decode, rng_levels_key = jax.random.split(rng_mutate, 4)
                batch_size = config["num_train_envs"]
                num_candidates = int(config["plwm_surrogate_num_candidates"])

                if config["plwm_use_maze_ae"] or config["plwm_use_cnn_vae"]:
                    # ---- Grid-based latent model path (MazeAE or CNN-VAE) ----
                    # Convert Level -> (H, W, 3) grid
                    parent_grids = jax.vmap(maze_level_to_grid)(
                        parent_levels.wall_map,
                        parent_levels.goal_pos,
                        parent_levels.agent_pos,
                    )  # (batch, H, W, 3)

                    # Encode
                    if config["plwm_use_maze_ae"]:
                        parent_latents = encode_maze_levels(maze_encoder_params, parent_grids)
                    else:
                        parent_latents = encode_cnn_vae_levels(cnn_vae_params, parent_grids)

                    if config["plwm_surrogate_guided"]:
                        # Sample multiple perturbation candidates per parent and keep the
                        # highest-scoring candidate under the selected ranking objective.
                        candidate_noise = None
                        novelty_parent_coords = None
                        novelty_candidate_coords = None
                        if config["plwm_use_pca_mutation"]:
                            if config["plwm_pca_structured_candidates"]:
                                (
                                    candidate_latents,
                                    parent_pca_coords,
                                    novelty_candidate_coords,
                                    _mix_weights,
                                    plwm_schedule_stats,
                                ) = build_structured_pca_candidates(
                                    parent_latents,
                                    parent_success_ema,
                                    train_state,
                                    config,
                                    plwm_pca_global_bank,
                                    plwm_pca_meso_bank,
                                    plwm_pca_local_bank,
                                )
                                novelty_parent_coords = parent_pca_coords
                            else:
                                candidate_noise = jax.random.normal(
                                    rng_encode,
                                    (batch_size, num_candidates, parent_latents.shape[-1]),
                                )
                                scaled = candidate_noise * jnp.sqrt(
                                    jnp.maximum(train_state.plwm_pca_eigvals[None, None, :], 1e-8)
                                ) * float(config["plwm_sigma"])
                                candidate_latents = parent_latents[:, None, :] + scaled @ train_state.plwm_pca_eigvecs
                                if config["plwm_pca_whitened_novelty"]:
                                    novelty_parent_coords = project_to_whitened_pca(
                                        parent_latents,
                                        train_state.plwm_pca_mean,
                                        train_state.plwm_pca_eigvecs,
                                        train_state.plwm_pca_eigvals,
                                    )
                                    novelty_candidate_coords = project_to_whitened_pca(
                                        candidate_latents.reshape((batch_size * num_candidates, -1)),
                                        train_state.plwm_pca_mean,
                                        train_state.plwm_pca_eigvecs,
                                        train_state.plwm_pca_eigvals,
                                    ).reshape((batch_size, num_candidates, -1))
                        else:
                            candidate_noise = jax.random.normal(
                                rng_encode,
                                (batch_size, num_candidates, parent_latents.shape[-1]),
                            )
                            candidate_latents = (
                                parent_latents[:, None, :] + float(config["plwm_sigma"]) * candidate_noise
                            )
                        flat_latents = candidate_latents.reshape((batch_size * num_candidates, -1))

                        if config["plwm_use_maze_ae"]:
                            candidate_levels_flat = decode_maze_latents(
                                maze_decoder_params,
                                flat_latents,
                                jax.random.split(rng_decode, batch_size * num_candidates),
                                wall_threshold=0.5,
                                temperature=float(config["plwm_decode_temperature"]),
                            )
                        else:
                            candidate_levels_flat = decode_cnn_vae_latents(
                                cnn_vae_params,
                                flat_latents,
                                rng_decode,
                            )

                        if config["plwm_online_frontier_guided"]:
                            if config["plwm_use_maze_ae"]:
                                parent_preds = predict_task_targets(maze_full_params, parent_grids)
                                candidate_grids_flat = jax.vmap(maze_level_to_grid)(
                                    candidate_levels_flat.wall_map,
                                    candidate_levels_flat.goal_pos,
                                    candidate_levels_flat.agent_pos,
                                )
                                cand_preds = predict_task_targets(maze_full_params, candidate_grids_flat)
                            else:
                                parent_preds = compute_structural_frontier_stats(parent_levels)
                                cand_preds = compute_structural_frontier_stats(candidate_levels_flat)

                            cand_features = build_frontier_features(
                                flat_latents,
                                cand_preds["static_reg"],
                                cand_preds["valid_prob"],
                            )
                            p_frontier, p_frontier_std = frontier_predict_stats(
                                cand_features,
                                train_state.frontier_params,
                            )
                            p_frontier = p_frontier.reshape((batch_size, num_candidates))
                            p_frontier_std = p_frontier_std.reshape((batch_size, num_candidates))
                            l_frontier = p_frontier * (1.0 - p_frontier)
                            if config["plwm_use_pca_mutation"] and config["plwm_pca_whitened_novelty"]:
                                novelty_bonus = frontier_candidate_novelty_from_coords(
                                    novelty_candidate_coords,
                                    novelty_parent_coords,
                                    max(1, int(config["plwm_frontier_novelty_knn"])),
                                )
                            else:
                                novelty_bonus = frontier_candidate_novelty(
                                    candidate_latents,
                                    parent_latents,
                                    max(1, int(config["plwm_frontier_novelty_knn"])),
                                )

                            task_scores = plwm_scoring.task_aware_objective(
                                p_pred=p_frontier,
                                learnability_pred=l_frontier,
                                invalid_prob=cand_preds["invalid_prob"].reshape((batch_size, num_candidates)),
                                bfs_norm_pred=cand_preds["bfs_norm_pred"].reshape((batch_size, num_candidates)),
                                wall_density_pred=cand_preds["wall_density_pred"].reshape((batch_size, num_candidates)),
                                parent_bfs_norm=parent_preds["bfs_norm_pred"][:, None],
                                parent_wall_density=parent_preds["wall_density_pred"][:, None],
                                a=float(config["plwm_task_weight_a"]),
                                b=float(config["plwm_task_weight_b"]),
                                c=float(config["plwm_task_weight_c"]),
                                d=float(config["plwm_task_weight_d"]),
                                e=float(config["plwm_task_weight_e"]),
                                low=float(config["plwm_target_success_low"]),
                                high=float(config["plwm_target_success_high"]),
                                delta_bfs_norm=float(config["plwm_task_delta_bfs_steps"]) / 169.0,
                            )
                            surrogate_scores = (
                                task_scores
                                + float(config["plwm_frontier_uncertainty_weight"]) * p_frontier_std
                                + float(config["plwm_frontier_novelty_weight"]) * novelty_bonus
                            )
                        elif config["plwm_task_aware_guided"]:
                            parent_preds = predict_task_targets(maze_full_params, parent_grids)
                            candidate_grids_flat = jax.vmap(maze_level_to_grid)(
                                candidate_levels_flat.wall_map,
                                candidate_levels_flat.goal_pos,
                                candidate_levels_flat.agent_pos,
                            )
                            cand_preds = predict_task_targets(maze_full_params, candidate_grids_flat)

                            task_scores = plwm_scoring.task_aware_objective(
                                p_pred=cand_preds["p_pred"].reshape((batch_size, num_candidates)),
                                learnability_pred=cand_preds["l_pred"].reshape((batch_size, num_candidates)),
                                invalid_prob=cand_preds["invalid_prob"].reshape((batch_size, num_candidates)),
                                bfs_norm_pred=cand_preds["bfs_norm_pred"].reshape((batch_size, num_candidates)),
                                wall_density_pred=cand_preds["wall_density_pred"].reshape((batch_size, num_candidates)),
                                parent_bfs_norm=parent_preds["bfs_norm_pred"][:, None],
                                parent_wall_density=parent_preds["wall_density_pred"][:, None],
                                a=float(config["plwm_task_weight_a"]),
                                b=float(config["plwm_task_weight_b"]),
                                c=float(config["plwm_task_weight_c"]),
                                d=float(config["plwm_task_weight_d"]),
                                e=float(config["plwm_task_weight_e"]),
                                low=float(config["plwm_target_success_low"]),
                                high=float(config["plwm_target_success_high"]),
                                delta_bfs_norm=float(config["plwm_task_delta_bfs_steps"]) / 169.0,
                            )
                            surrogate_scores = task_scores
                        else:
                            surrogate_scores_flat = structural_difficulty_surrogate(
                                candidate_levels_flat.wall_map,
                                candidate_levels_flat.goal_pos,
                                candidate_levels_flat.agent_pos,
                                weight_bfs=float(config["plwm_surrogate_weight_bfs"]),
                                weight_slack=float(config["plwm_surrogate_weight_slack"]),
                                weight_dead_ends=float(config["plwm_surrogate_weight_dead_ends"]),
                                weight_walls=float(config["plwm_surrogate_weight_walls"]),
                                weight_branches=float(config["plwm_surrogate_weight_branches"]),
                                require_solvable=bool(config["plwm_surrogate_require_solvable"]),
                            )
                            surrogate_scores = surrogate_scores_flat.reshape((batch_size, num_candidates))

                        best_idx = jnp.argmax(surrogate_scores, axis=1)
                        row_idx = jnp.arange(batch_size)

                        child_latents = candidate_latents[row_idx, best_idx]
                        candidate_levels = jax.tree_util.tree_map(
                            lambda x: x.reshape((batch_size, num_candidates, *x.shape[1:])),
                            candidate_levels_flat,
                        )
                        child_levels = jax.tree_util.tree_map(
                            lambda x: x[row_idx, best_idx],
                            candidate_levels,
                        )
                    else:
                        noise = jax.random.normal(rng_encode, parent_latents.shape)
                        if config["plwm_use_pca_mutation"]:
                            scaled = noise * jnp.sqrt(jnp.maximum(train_state.plwm_pca_eigvals, 1e-8)) * float(config["plwm_sigma"])
                            child_latents = parent_latents + scaled @ train_state.plwm_pca_eigvecs
                        else:
                            child_latents = parent_latents + float(config["plwm_sigma"]) * noise

                        # Decode -> Level
                        if config["plwm_use_maze_ae"]:
                            child_levels = decode_maze_latents(
                                maze_decoder_params,
                                child_latents,
                                jax.random.split(rng_decode, batch_size),
                                wall_threshold=0.5,
                                temperature=float(config["plwm_decode_temperature"]),
                            )
                        else:
                            child_levels = decode_cnn_vae_latents(
                                cnn_vae_params,
                                child_latents,
                                rng_decode,
                            )
                else:
                    # ---- CLUTTR sequence VAE path (legacy, ≤50 walls) ----
                    parent_seqs = jax.vmap(level_to_cluttr_sequence)(
                        parent_levels.wall_map,
                        parent_levels.goal_pos,
                        parent_levels.agent_pos,
                    )  # (batch, 52)

                    parent_latents = encode_levels_to_latents(encoder_params, parent_seqs)

                    if config["plwm_surrogate_guided"]:
                        candidate_noise = None
                        novelty_parent_coords = None
                        novelty_candidate_coords = None
                        if config["plwm_use_pca_mutation"]:
                            if config["plwm_pca_structured_candidates"]:
                                (
                                    candidate_latents,
                                    parent_pca_coords,
                                    novelty_candidate_coords,
                                    _mix_weights,
                                    plwm_schedule_stats,
                                ) = build_structured_pca_candidates(
                                    parent_latents,
                                    parent_success_ema,
                                    train_state,
                                    config,
                                    plwm_pca_global_bank,
                                    plwm_pca_meso_bank,
                                    plwm_pca_local_bank,
                                )
                                novelty_parent_coords = parent_pca_coords
                            else:
                                candidate_noise = jax.random.normal(
                                    rng_encode,
                                    (batch_size, num_candidates, parent_latents.shape[-1]),
                                )
                                scaled = candidate_noise * jnp.sqrt(
                                    jnp.maximum(train_state.plwm_pca_eigvals[None, None, :], 1e-8)
                                ) * float(config["plwm_sigma"])
                                candidate_latents = parent_latents[:, None, :] + scaled @ train_state.plwm_pca_eigvecs
                                if config["plwm_pca_whitened_novelty"]:
                                    novelty_parent_coords = project_to_whitened_pca(
                                        parent_latents,
                                        train_state.plwm_pca_mean,
                                        train_state.plwm_pca_eigvecs,
                                        train_state.plwm_pca_eigvals,
                                    )
                                    novelty_candidate_coords = project_to_whitened_pca(
                                        candidate_latents.reshape((batch_size * num_candidates, -1)),
                                        train_state.plwm_pca_mean,
                                        train_state.plwm_pca_eigvecs,
                                        train_state.plwm_pca_eigvals,
                                    ).reshape((batch_size, num_candidates, -1))
                        else:
                            candidate_noise = jax.random.normal(
                                rng_encode,
                                (batch_size, num_candidates, parent_latents.shape[-1]),
                            )
                            candidate_latents = (
                                parent_latents[:, None, :] + float(config["plwm_sigma"]) * candidate_noise
                            )
                        flat_latents = candidate_latents.reshape((batch_size * num_candidates, -1))

                        child_sequences_flat = decode_latent_to_env(
                            decoder_params,
                            flat_latents,
                            rng_key=rng_decode,
                            temperature=float(config["plwm_decode_temperature"]),
                        )
                        child_sequences_flat = jax.vmap(repair_cluttr_sequence)(child_sequences_flat)
                        candidate_levels_flat = jax.vmap(cluttr_sequence_to_level)(
                            child_sequences_flat,
                            jax.random.split(rng_levels_key, batch_size * num_candidates),
                        )
                        if config["plwm_online_frontier_guided"]:
                            parent_struct = compute_structural_frontier_stats(parent_levels)
                            cand_struct = compute_structural_frontier_stats(candidate_levels_flat)
                            cand_features = build_frontier_features(
                                flat_latents,
                                cand_struct["static_reg"],
                                cand_struct["valid_prob"],
                            )
                            p_frontier, p_frontier_std = frontier_predict_stats(
                                cand_features,
                                train_state.frontier_params,
                            )
                            p_frontier = p_frontier.reshape((batch_size, num_candidates))
                            p_frontier_std = p_frontier_std.reshape((batch_size, num_candidates))
                            l_frontier = p_frontier * (1.0 - p_frontier)
                            if config["plwm_use_pca_mutation"] and config["plwm_pca_whitened_novelty"]:
                                novelty_bonus = frontier_candidate_novelty_from_coords(
                                    novelty_candidate_coords,
                                    novelty_parent_coords,
                                    max(1, int(config["plwm_frontier_novelty_knn"])),
                                )
                            else:
                                novelty_bonus = frontier_candidate_novelty(
                                    candidate_latents,
                                    parent_latents,
                                    max(1, int(config["plwm_frontier_novelty_knn"])),
                                )
                            surrogate_scores = plwm_scoring.task_aware_objective(
                                p_pred=p_frontier,
                                learnability_pred=l_frontier,
                                invalid_prob=cand_struct["invalid_prob"].reshape((batch_size, num_candidates)),
                                bfs_norm_pred=cand_struct["bfs_norm_pred"].reshape((batch_size, num_candidates)),
                                wall_density_pred=cand_struct["wall_density_pred"].reshape((batch_size, num_candidates)),
                                parent_bfs_norm=parent_struct["bfs_norm_pred"][:, None],
                                parent_wall_density=parent_struct["wall_density_pred"][:, None],
                                a=float(config["plwm_task_weight_a"]),
                                b=float(config["plwm_task_weight_b"]),
                                c=float(config["plwm_task_weight_c"]),
                                d=float(config["plwm_task_weight_d"]),
                                e=float(config["plwm_task_weight_e"]),
                                low=float(config["plwm_target_success_low"]),
                                high=float(config["plwm_target_success_high"]),
                                delta_bfs_norm=float(config["plwm_task_delta_bfs_steps"]) / 169.0,
                            )
                            surrogate_scores = (
                                surrogate_scores
                                + float(config["plwm_frontier_uncertainty_weight"]) * p_frontier_std
                                + float(config["plwm_frontier_novelty_weight"]) * novelty_bonus
                            )
                        else:
                            surrogate_scores_flat = structural_difficulty_surrogate(
                                candidate_levels_flat.wall_map,
                                candidate_levels_flat.goal_pos,
                                candidate_levels_flat.agent_pos,
                                weight_bfs=float(config["plwm_surrogate_weight_bfs"]),
                                weight_slack=float(config["plwm_surrogate_weight_slack"]),
                                weight_dead_ends=float(config["plwm_surrogate_weight_dead_ends"]),
                                weight_walls=float(config["plwm_surrogate_weight_walls"]),
                                weight_branches=float(config["plwm_surrogate_weight_branches"]),
                                require_solvable=bool(config["plwm_surrogate_require_solvable"]),
                            )
                            surrogate_scores = surrogate_scores_flat.reshape((batch_size, num_candidates))
                        best_idx = jnp.argmax(surrogate_scores, axis=1)
                        row_idx = jnp.arange(batch_size)

                        child_latents = candidate_latents[row_idx, best_idx]
                        child_sequences = child_sequences_flat.reshape((batch_size, num_candidates, -1))[row_idx, best_idx]
                        candidate_levels = jax.tree_util.tree_map(
                            lambda x: x.reshape((batch_size, num_candidates, *x.shape[1:])),
                            candidate_levels_flat,
                        )
                        child_levels = jax.tree_util.tree_map(
                            lambda x: x[row_idx, best_idx],
                            candidate_levels,
                        )
                    else:
                        noise = jax.random.normal(rng_encode, parent_latents.shape)
                        if config["plwm_use_pca_mutation"]:
                            scaled = noise * jnp.sqrt(jnp.maximum(train_state.plwm_pca_eigvals, 1e-8)) * float(config["plwm_sigma"])
                            child_latents = parent_latents + scaled @ train_state.plwm_pca_eigvecs
                        else:
                            child_latents = parent_latents + float(config["plwm_sigma"]) * noise

                        child_sequences = decode_latent_to_env(
                            decoder_params, child_latents,
                            rng_key=rng_decode,
                            temperature=float(config["plwm_decode_temperature"]),
                        )
                        child_sequences = jax.vmap(repair_cluttr_sequence)(child_sequences)
                        child_levels = jax.vmap(cluttr_sequence_to_level)(
                            child_sequences,
                            jax.random.split(rng_levels_key, batch_size),
                        )
            else:
                parent_levels = train_state.replay_last_level_batch
                parent_level_inds = train_state.replay_last_level_inds
                parent_scores = train_state.sampler["scores"][parent_level_inds]
                child_levels = jax.vmap(mutate_level, (0, 0, None))(
                    jax.random.split(rng_mutate, config["num_train_envs"]),
                    parent_levels,
                    config["num_edits"],
                )
                child_latents = None
                child_sequences = None
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), child_levels, env_params)

            # rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = compute_max_returns(dones, rewards)
            scores = compute_score(config, dones, values, max_returns, advantages)
            success = rollout_success_from_rewards(rewards)
            sampler, inserted_level_inds = level_sampler.insert_batch(
                sampler,
                child_levels,
                scores,
                {
                    "max_return": max_returns,
                    "success_ema": success,
                    "success_obs_count": jnp.ones_like(success),
                },
            )

            # Track how often selected PLWM mutations improve over their replay parent score.
            plwm_batch_uphill_fraction = jnp.array(0.0, dtype=jnp.float32)
            plwm_batch_compared = jnp.array(0, dtype=jnp.int32)
            plwm_batch_improved = jnp.array(0, dtype=jnp.int32)
            plwm_batch_inserted = jnp.array(0, dtype=jnp.int32)
            plwm_batch_inserted_improved = jnp.array(0, dtype=jnp.int32)
            plwm_batch_insert_fraction = jnp.array(0.0, dtype=jnp.float32)
            plwm_batch_accepted_uphill_fraction = jnp.array(0.0, dtype=jnp.float32)
            plwm_batch_mean_delta_score = jnp.array(0.0, dtype=jnp.float32)
            plwm_batch_mean_accepted_delta_score = jnp.array(0.0, dtype=jnp.float32)
            plwm_batch_delta_score_sum = jnp.array(0.0, dtype=jnp.float32)
            plwm_batch_accepted_delta_score_sum = jnp.array(0.0, dtype=jnp.float32)
            if not config["use_map_elites_mutation"]:
                inserted_mask = inserted_level_inds >= 0
                delta_scores = scores - parent_scores
                plwm_batch_compared = jnp.array(config["num_train_envs"], dtype=jnp.int32)
                plwm_batch_improved = jnp.sum(scores > parent_scores).astype(jnp.int32)
                plwm_batch_inserted = inserted_mask.astype(jnp.int32).sum()
                plwm_batch_inserted_improved = jnp.sum(inserted_mask & (scores > parent_scores)).astype(jnp.int32)
                plwm_batch_uphill_fraction = plwm_batch_improved.astype(jnp.float32) / jnp.maximum(
                    plwm_batch_compared.astype(jnp.float32), 1.0
                )
                plwm_batch_insert_fraction = plwm_batch_inserted.astype(jnp.float32) / jnp.maximum(
                    plwm_batch_compared.astype(jnp.float32), 1.0
                )
                plwm_batch_accepted_uphill_fraction = plwm_batch_inserted_improved.astype(jnp.float32) / jnp.maximum(
                    plwm_batch_inserted.astype(jnp.float32), 1.0
                )
                plwm_batch_delta_score_sum = delta_scores.sum(dtype=jnp.float32)
                plwm_batch_mean_delta_score = plwm_batch_delta_score_sum / jnp.maximum(
                    plwm_batch_compared.astype(jnp.float32), 1.0
                )
                plwm_batch_accepted_delta_score_sum = jnp.where(inserted_mask, delta_scores, 0.0).sum(dtype=jnp.float32)
                plwm_batch_mean_accepted_delta_score = plwm_batch_accepted_delta_score_sum / jnp.maximum(
                    plwm_batch_inserted.astype(jnp.float32), 1.0
                )

            if config["use_map_elites_mutation"]:
                should_update_archive = (train_state.num_mutation_updates % me_update_period) == 0

                def _update_archive(archive):
                    return map_elites_insert_batch(
                        archive,
                        child_latents,
                        child_sequences,
                        scores,
                        descriptor_mode=me_descriptor_mode,
                        axis1_bins=me_axis1_bins,
                        axis2_bins=me_axis2_bins,
                        latent_projections=me_insert_latent_projections,
                        min_obstacles=me_min_obstacles,
                        min_distance=me_min_distance,
                        current_step=train_state.num_mutation_updates,
                    )

                me_archive, me_insertions = jax.lax.cond(
                    should_update_archive,
                    _update_archive,
                    lambda archive: (archive, jnp.array(0, dtype=jnp.int32)),
                    me_archive,
                )
            else:
                # PLWM and minimax mutation do not use the ME archive
                me_insertions = jnp.array(0, dtype=jnp.int32)
            
            # Update: train_state only modified if exploratory_grad_updates is on
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )
            
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": child_levels.wall_map.sum() / config["num_train_envs"],
                "me_insertions": me_insertions,
                "plwm_batch_uphill_fraction": plwm_batch_uphill_fraction,
                "plwm_batch_insert_fraction": plwm_batch_insert_fraction,
                "plwm_batch_accepted_uphill_fraction": plwm_batch_accepted_uphill_fraction,
                "plwm_batch_mean_delta_score": plwm_batch_mean_delta_score,
            }
            
            train_state = train_state.replace(
                sampler=sampler,
                me_archive=me_archive,
                update_state=UpdateState.DR,
                num_mutation_updates=train_state.num_mutation_updates + 1,
                num_plwm_compared=train_state.num_plwm_compared + plwm_batch_compared,
                num_plwm_improved=train_state.num_plwm_improved + plwm_batch_improved,
                num_plwm_inserted=train_state.num_plwm_inserted + plwm_batch_inserted,
                num_plwm_inserted_improved=train_state.num_plwm_inserted_improved + plwm_batch_inserted_improved,
                plwm_last_uphill_fraction=plwm_batch_uphill_fraction,
                plwm_last_insert_fraction=plwm_batch_insert_fraction,
                plwm_last_accepted_uphill_fraction=plwm_batch_accepted_uphill_fraction,
                plwm_delta_score_sum=train_state.plwm_delta_score_sum + plwm_batch_delta_score_sum,
                plwm_accepted_delta_score_sum=train_state.plwm_accepted_delta_score_sum + plwm_batch_accepted_delta_score_sum,
                plwm_last_mean_delta_score=plwm_batch_mean_delta_score,
                plwm_last_mean_accepted_delta_score=plwm_batch_mean_accepted_delta_score,
                plwm_last_scale_global_fraction=plwm_schedule_stats[0],
                plwm_last_scale_meso_fraction=plwm_schedule_stats[1],
                plwm_last_scale_local_fraction=plwm_schedule_stats[2],
                plwm_last_schedule_progress=plwm_schedule_stats[3],
                mutation_last_level_batch=child_levels,
            )
            return (rng, train_state), metrics
    
        rng, train_state = carry
        rng, rng_replay = jax.random.split(rng)
        
        # The train step makes a decision on which branch to take, either on_new, on_replay or on_mutate.
        # on_mutate is only called if the replay branch has been taken before (as it uses `train_state.update_state`).
        if config["use_accel"]:
            s = train_state.update_state
            branch = (1 - s) * level_sampler.sample_replay_decision(train_state.sampler, rng_replay) + 2 * s
        else:
            branch = level_sampler.sample_replay_decision(train_state.sampler, rng_replay).astype(int)
        
        return jax.lax.switch(
            branch,
            [
                on_new_levels,
                on_replay_levels,
                on_mutate_levels,
            ],
            rng, train_state
        )
    
    def eval(rng: chex.PRNGKey, train_state: TrainState):
        """
        This evaluates the current policy on the set of evaluation levels specified by config["eval_levels"].
        It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
        """
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
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
        return states, cum_rewards, episode_lengths # (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
    
    @jax.jit
    def train_and_eval_step(runner_state, _):
        """
            This function runs the train_step for a certain number of iterations, and then evaluates the policy.
            It returns the updated train state, and a dictionary of metrics.
        """
        # Train
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])

        # Eval
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, config["eval_num_attempts"]), train_state)
        
        # Collect Metrics
        eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0) # (num_eval_levels,)
        eval_returns = cum_rewards.mean(axis=0) # (num_eval_levels,)
        
        # just grab the first run
        states, episode_lengths = jax.tree_util.tree_map(lambda x: x[0], (states, episode_lengths)) # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params) # (num_steps, num_eval_levels, ...)
        frames = images.transpose(0, 1, 4, 2, 3) # WandB expects color channel before image dimensions when dealing with animations for some reason
        
        metrics["update_count"] = train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"]  = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        metrics["dr_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.dr_last_level_batch, env_params)
        metrics["replay_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.replay_last_level_batch, env_params)
        metrics["mutation_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.mutation_last_level_batch, env_params)
        
        highest_scoring_level = level_sampler.get_levels(train_state.sampler, train_state.sampler["scores"].argmax())
        highest_weighted_level = level_sampler.get_levels(train_state.sampler, level_sampler.level_weights(train_state.sampler).argmax())
        
        metrics["highest_scoring_level"] = env_renderer.render_level(highest_scoring_level, env_params)
        metrics["highest_weighted_level"] = env_renderer.render_level(highest_weighted_level, env_params)
        
        return (rng, train_state), metrics
    
    def eval_checkpoint(og_config):
        """
            This function is what is used to evaluate a saved checkpoint *after* training. It first loads the checkpoint and then runs evaluation.
            It saves the states, cum_rewards and episode_lengths to a .npz file in the `results/run_name/seed` directory.
        """
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))
        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, 'config.json')) as f: config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_directory, 'models'), item_handlers=ocp.StandardCheckpointHandler())

            train_state_og: TrainState = create_train_state(rng_init)
            step = checkpoint_manager.latest_step() if og_config['checkpoint_to_eval'] == -1 else og_config['checkpoint_to_eval']

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint['params']
            train_state = train_state_og.replace(params=params)
            return train_state, config
        
        train_state, config = load(rng_init, og_config['checkpoint_directory'])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, og_config["eval_num_attempts"]), train_state)
        save_loc = og_config['checkpoint_directory'].replace('checkpoints', 'results')
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(os.path.join(save_loc, 'results.npz'), states=np.asarray(states), cum_rewards=np.asarray(cum_rewards), episode_lengths=np.asarray(episode_lengths), levels=config['eval_levels'])
        return states, cum_rewards, episode_lengths

    if config['mode'] == 'eval':
        return eval_checkpoint(config) # evaluate and exit early

    # Set up the train states
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)
    
    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)
    
    # And run the train_eval_sep function for the specified number of updates
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time

        # Recompute PCA directions over encoded buffer latents
        if (
            config["plwm_use_pca_mutation"]
            and config["use_plwm_mutation"]
            and eval_step % config["plwm_pca_update_every"] == 0
        ):
            _buf_size = int(runner_state[1].sampler["size"])
            if _buf_size >= plwm_latent_dim + 1:
                _chunk_size = 256
                _latent_chunks = []
                if config["plwm_use_maze_ae"] or config["plwm_use_cnn_vae"]:
                    _grids = extract_buffer_grids(runner_state[1].sampler, eval_env.max_height, eval_env.max_width)
                    for _i in range(0, len(_grids), _chunk_size):
                        _chunk = jnp.array(_grids[_i : _i + _chunk_size])
                        if config["plwm_use_maze_ae"]:
                            _latent_chunks.append(np.array(encode_maze_levels(maze_encoder_params, _chunk)))
                        else:
                            _latent_chunks.append(np.array(encode_cnn_vae_levels(cnn_vae_params, _chunk)))
                else:
                    # CLUTTR VAE path: extract sequences from buffer and encode
                    _lvls = runner_state[1].sampler["levels"]
                    _wall_maps = jnp.array(np.array(_lvls.wall_map[:_buf_size]))
                    _goal_pos  = jnp.array(np.array(_lvls.goal_pos[:_buf_size]))
                    _agent_pos = jnp.array(np.array(_lvls.agent_pos[:_buf_size]))
                    _seqs = jax.jit(jax.vmap(level_to_cluttr_sequence))(_wall_maps, _goal_pos, _agent_pos)
                    for _i in range(0, _buf_size, _chunk_size):
                        _chunk_seqs = _seqs[_i : _i + _chunk_size]
                        _latent_chunks.append(np.array(encode_levels_to_latents(encoder_params, _chunk_seqs)))
                _latents = np.concatenate(_latent_chunks, axis=0)
                _eigvecs, _eigvals = compute_pca_from_latents(_latents)
                _mean = _latents.mean(axis=0).astype(np.float32)
                runner_state = (
                    runner_state[0],
                    runner_state[1].replace(
                        plwm_pca_eigvecs=jnp.array(_eigvecs),
                        plwm_pca_eigvals=jnp.array(_eigvals),
                        plwm_pca_mean=jnp.array(_mean),
                    ),
                )

        log_eval(
            metrics,
            train_state_to_log_dict(
                runner_state[1],
                level_sampler,
                use_map_elites=config["use_map_elites_mutation"],
                use_plwm=config["use_plwm_mutation"],
            ),
        )
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()
    return runner_state[1]

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="JAXUED_TEST")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    # === Train vs Eval ===
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)
    # === CHECKPOINTING ===
    parser.add_argument("--checkpoint_save_interval", type=int, default=2)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)
    # === EVAL ===
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument("--eval_levels", nargs='+', default=[
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
    ])
    group = parser.add_argument_group('Training params')
    # === PPO === 
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--max_grad_norm", type=float, default=0.5)
    mut_group = group.add_mutually_exclusive_group()
    mut_group.add_argument("--num_updates", type=int, default=30000)
    mut_group.add_argument("--num_env_steps", type=int, default=None)
    group.add_argument("--num_steps", type=int, default=256)
    group.add_argument("--num_train_envs", type=int, default=32)
    group.add_argument("--num_minibatches", type=int, default=1)
    group.add_argument("--gamma", type=float, default=0.995)
    group.add_argument("--epoch_ppo", type=int, default=5)
    group.add_argument("--clip_eps", type=float, default=0.2)
    group.add_argument("--gae_lambda", type=float, default=0.98)
    group.add_argument("--entropy_coeff", type=float, default=1e-3)
    group.add_argument("--critic_coeff", type=float, default=0.5)
    # === PLR ===
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
    # === ACCEL ===
    group.add_argument("--use_accel", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--num_edits", type=int, default=5)
    group.add_argument("--use_map_elites_mutation", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--me_vae_checkpoint", type=str, default=None)
    group.add_argument("--me_latent_dim", type=int, default=64)
    group.add_argument("--me_mutation_sigma", type=float, default=0.5)
    group.add_argument("--me_decode_temperature", type=float, default=0.25)
    group.add_argument("--me_uniform_parent_fraction", type=float, default=0.5)
    group.add_argument("--me_fitness_softmax_temp", type=float, default=0.5)
    group.add_argument("--me_descriptor_mode", type=str, default="bfs", choices=["behavior", "latent", "hybrid", "bfs"])
    group.add_argument("--me_latent_descriptor_seed", type=int, default=0)
    group.add_argument("--me_latent_bin_min", type=float, default=-4.0)
    group.add_argument("--me_latent_bin_max", type=float, default=4.0)
    group.add_argument("--me_latent_bin_count", type=int, default=8)
    group.add_argument("--me_staleness_decay_rate", type=float, default=2e-5,
                       help="Staleness decay rate for archive fitness (0 = disabled). "
                            "Half-life = ln(2)/rate. Default 2e-5 → half-life ~35k mutation steps.")
    group.add_argument("--me_update_period", type=int, default=1)
    group.add_argument("--me_min_obstacles", type=int, default=5)
    group.add_argument("--me_min_distance", type=int, default=3)
    # === PLWM (PLR-Weighted Latent Mutation) ===
    group.add_argument("--use_plwm_mutation", action=argparse.BooleanOptionalAction, default=False,
                       help="Enable PLR-Weighted Latent Mutation: encode replay parents to VAE "
                            "latents, perturb, decode, and insert back into PLR. Requires "
                            "--use_accel. Mutually exclusive with --use_map_elites_mutation.")
    group.add_argument("--plwm_sigma", type=float, default=0.5,
                       help="Std-dev of Gaussian noise added in latent space for PLWM. "
                            "Lower values = smaller mutations. Default 0.5.")
    group.add_argument("--plwm_decode_temperature", type=float, default=0.25,
                       help="Gumbel-max sampling temperature for PLWM decoder. "
                            "Lower = more deterministic. Default 0.25.")
    group.add_argument("--success_ema_alpha", type=float, default=0.1,
                       help="EMA smoothing factor for per-level success labels in replay extras.")
    group.add_argument("--plwm_surrogate_guided", action=argparse.BooleanOptionalAction, default=False,
                       help="If set, sample multiple latent perturbation candidates per parent "
                            "and choose the best candidate using a structural difficulty surrogate.")
    group.add_argument("--plwm_task_aware_guided", action=argparse.BooleanOptionalAction, default=False,
                       help="When surrogate-guided PLWM is enabled, use task-aware beta-VAE head "
                            "predictions for candidate ranking instead of the structural surrogate.")
    group.add_argument("--plwm_online_frontier_guided", action=argparse.BooleanOptionalAction, default=False,
                       help="When surrogate-guided PLWM is enabled, rank candidates with an online "
                            "run-local frontier estimator trained on replay success labels.")
    group.add_argument("--plwm_frontier_lr", type=float, default=1e-2,
                       help="Learning rate for online frontier estimator SGD updates.")
    group.add_argument("--plwm_frontier_l2", type=float, default=1e-4,
                       help="L2 regularization for online frontier estimator weights.")
    group.add_argument("--plwm_frontier_hidden_dim", type=int, default=64,
                       help="Hidden width of each frontier ensemble member.")
    group.add_argument("--plwm_frontier_ensemble_size", type=int, default=4,
                       help="Number of ensemble members in the frontier scorer.")
    group.add_argument("--plwm_frontier_conf_ref", type=float, default=20.0,
                       help="Confidence reference count in w=min(1,count/ref) for online frontier updates.")
    group.add_argument("--plwm_frontier_uncertainty_weight", type=float, default=0.25,
                       help="Bonus weight on frontier ensemble std during candidate ranking.")
    group.add_argument("--plwm_frontier_novelty_weight", type=float, default=0.15,
                       help="Bonus weight on explicit latent novelty during candidate ranking.")
    group.add_argument("--plwm_frontier_novelty_knn", type=int, default=3,
                       help="Number of nearest replay parents used for novelty scoring.")
    group.add_argument("--plwm_target_success_low", type=float, default=0.3,
                       help="Lower bound of the target success-probability band.")
    group.add_argument("--plwm_target_success_high", type=float, default=0.7,
                       help="Upper bound of the target success-probability band.")
    group.add_argument("--plwm_task_delta_bfs_steps", type=float, default=2.0,
                       help="Desired BFS-length increase relative to the parent in steps.")
    group.add_argument("--plwm_task_weight_a", type=float, default=1.0,
                       help="Weight on predicted learnability in task-aware PLWM objective.")
    group.add_argument("--plwm_task_weight_b", type=float, default=2.0,
                       help="Weight on invalidity penalty in task-aware PLWM objective.")
    group.add_argument("--plwm_task_weight_c", type=float, default=0.5,
                       help="Weight on success-band penalty in task-aware PLWM objective.")
    group.add_argument("--plwm_task_weight_d", type=float, default=0.25,
                       help="Weight on BFS-target deviation penalty in task-aware PLWM objective.")
    group.add_argument("--plwm_task_weight_e", type=float, default=0.1,
                       help="Weight on wall-density drift penalty in task-aware PLWM objective.")
    group.add_argument("--plwm_surrogate_num_candidates", type=int, default=8,
                       help="Number of latent perturbation candidates per parent when surrogate-guided "
                            "PLWM is enabled. Default 8.")
    group.add_argument("--plwm_surrogate_weight_bfs", type=float, default=1.0,
                       help="Weight for BFS path length in structural surrogate score.")
    group.add_argument("--plwm_surrogate_weight_slack", type=float, default=1.0,
                       help="Weight for path slack (BFS - Manhattan) in structural surrogate score.")
    group.add_argument("--plwm_surrogate_weight_dead_ends", type=float, default=0.2,
                       help="Weight for dead-end count in structural surrogate score.")
    group.add_argument("--plwm_surrogate_weight_walls", type=float, default=0.05,
                       help="Weight for wall count in structural surrogate score.")
    group.add_argument("--plwm_surrogate_weight_branches", type=float, default=0.0,
                       help="Weight for branch-point count in structural surrogate score.")
    group.add_argument("--plwm_surrogate_require_solvable", action=argparse.BooleanOptionalAction, default=True,
                       help="If set, surrogate score for unsolvable candidate levels is forced to -inf "
                            "so only solvable candidates are selected.")
    group.add_argument("--plwm_use_maze_ae", action=argparse.BooleanOptionalAction, default=False,
                       help="Use the grid-based task-aware Maze beta-VAE instead of the CLUTTR sequence "
                            "VAE for PLWM encoding/decoding. Requires --use_plwm_mutation. "
                            "Supports any wall count (no 50-wall truncation).")
    group.add_argument("--plwm_use_cnn_vae", action=argparse.BooleanOptionalAction, default=False,
                       help="Use the CNN maze VAE checkpoint for PLWM encoding/decoding. "
                            "Intended for structural or online-frontier guided PLWM; "
                            "task-aware-guided PLWM still requires --plwm_use_maze_ae.")
    group.add_argument("--plwm_mae_checkpoint", type=str, default=None,
                       help="Path to the MazeAE checkpoint pickle. Defaults to "
                            "vae/model_maze_ae/checkpoint_final.pkl relative to project root.")
    group.add_argument("--plwm_cnn_vae_checkpoint", type=str, default=None,
                       help="Path to the CNN maze VAE Orbax checkpoint directory. Defaults to "
                            "vae/checkpoints/cnn_vae/run11_1M relative to project root.")
    group.add_argument("--plwm_use_pca_mutation", action=argparse.BooleanOptionalAction, default=False,
                       help="Scale PLWM latent perturbations along PCA directions of the replay buffer "
                            "latents instead of isotropic Gaussian. Requires --use_plwm_mutation "
                            "and supports --plwm_use_maze_ae or --plwm_use_cnn_vae.")
    group.add_argument("--plwm_pca_update_every", type=int, default=1,
                       help="Recompute buffer-latent PCA every N eval cycles. Default 1 (every eval).")
    group.add_argument("--plwm_pca_structured_candidates", action=argparse.BooleanOptionalAction, default=True,
                       help="When surrogate-guided PCA mutation is enabled, use a structured multi-scale "
                            "candidate bank with global/meso/local PCA probes.")
    group.add_argument("--plwm_pca_whitened_novelty", action=argparse.BooleanOptionalAction, default=True,
                       help="When using PCA mutation with online frontier guidance, compute novelty in "
                            "whitened PCA coordinates instead of raw latent L2.")
    group.add_argument("--plwm_pca_local_top_k", type=int, default=5,
                       help="Number of top PCs reserved for local structured candidates. "
                            "PC1 is handled separately.")
    group.add_argument("--plwm_pca_global_top_k", type=int, default=3,
                       help="Number of top PCs reserved for coarse global structured candidates.")
    group.add_argument("--plwm_pca_pc1_scale", type=float, default=0.25,
                       help="Relative scale for explicit PC1 probes inside the structured PCA bank, "
                            "expressed as a multiplier on --plwm_sigma.")
    group.add_argument("--plwm_pca_pc1_candidate_fraction", type=float, default=0.25,
                       help="Fraction of surrogate-guided PCA candidates allocated to explicit PC1 probes.")
    group.add_argument("--plwm_pca_global_scale", type=float, default=1.2,
                       help="Relative scale multiplier for global structured PCA candidates.")
    group.add_argument("--plwm_pca_meso_scale", type=float, default=0.7,
                       help="Relative scale multiplier for meso structured PCA candidates.")
    group.add_argument("--plwm_pca_local_scale", type=float, default=0.3,
                       help="Relative scale multiplier for local structured PCA candidates.")
    group.add_argument("--plwm_pca_global_weight_start", type=float, default=0.35,
                       help="Early-training fraction of structured candidates allocated to the global scale.")
    group.add_argument("--plwm_pca_global_weight_end", type=float, default=0.10,
                       help="Late-training fraction of structured candidates allocated to the global scale.")
    group.add_argument("--plwm_pca_local_weight_start", type=float, default=0.20,
                       help="Early-training fraction of structured candidates allocated to the local scale.")
    group.add_argument("--plwm_pca_local_weight_end", type=float, default=0.60,
                       help="Late-training fraction of structured candidates allocated to the local scale.")
    group.add_argument("--plwm_pca_success_conditioning", type=float, default=0.75,
                       help="How strongly parent success_ema shifts the structured PCA mix toward "
                            "global moves away from the target success band and local moves near it.")
    # === ENV CONFIG ===
    group.add_argument("--agent_view_size", type=int, default=5)
    # === DR CONFIG ===
    group.add_argument("--n_walls", type=int, default=25)
    
    config = vars(parser.parse_args())
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])
    config["group_name"] = ''.join([str(config[key]) for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])])
    
    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'
    
    # wandb.login()
    main(config, project=config["project"])
