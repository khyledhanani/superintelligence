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
from es.env_bridge import level_to_cluttr_sequence, cluttr_sequence_to_level, bfs_path_length
from es.maze_ae import (
    load_maze_ae_params, extract_maze_encoder_params, extract_maze_decoder_params,
    maze_level_to_grid, encode_maze_levels, decode_maze_latents,
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
    plwm_last_uphill_fraction: chex.Array
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_inds: chex.Array = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)

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

    if use_plwm:
        log["plwm/num_mutations"] = train_state.num_mutation_updates
        log["plwm/last_uphill_fraction"] = train_state.plwm_last_uphill_fraction
        log["plwm/cumulative_uphill_fraction"] = (
            train_state.num_plwm_improved / jnp.maximum(train_state.num_plwm_compared, 1)
        )
        log["plwm/num_compared"] = train_state.num_plwm_compared
        log["plwm/num_improved"] = train_state.num_plwm_improved

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
    """Structural difficulty proxy used for surrogate-guided PLWM candidate ranking.

    Inputs are batched:
      wall_map   : (B, H, W) bool
      goal_pos   : (B, 2) [x, y]
      agent_pos  : (B, 2) [x, y]
    Returns:
      surrogate score per candidate, shape (B,)
    """
    h = wall_map.shape[1]
    w = wall_map.shape[2]
    inf = h * w

    bfs = jax.vmap(bfs_path_length)(wall_map, agent_pos, goal_pos).astype(jnp.float32)
    manhattan = (
        jnp.abs(goal_pos[:, 0].astype(jnp.int32) - agent_pos[:, 0].astype(jnp.int32))
        + jnp.abs(goal_pos[:, 1].astype(jnp.int32) - agent_pos[:, 1].astype(jnp.int32))
    ).astype(jnp.float32)
    slack = bfs - manhattan

    walls = wall_map.astype(jnp.bool_)
    free = ~walls

    up = jnp.roll(free, -1, axis=1).at[:, -1, :].set(False)
    down = jnp.roll(free, 1, axis=1).at[:, 0, :].set(False)
    left = jnp.roll(free, -1, axis=2).at[:, :, -1].set(False)
    right = jnp.roll(free, 1, axis=2).at[:, :, 0].set(False)
    degree = up.astype(jnp.int32) + down.astype(jnp.int32) + left.astype(jnp.int32) + right.astype(jnp.int32)

    dead_ends = jnp.sum(free & (degree == 1), axis=(1, 2)).astype(jnp.float32)
    branch_points = jnp.sum(free & (degree >= 3), axis=(1, 2)).astype(jnp.float32)
    wall_count = jnp.sum(walls, axis=(1, 2)).astype(jnp.float32)

    score = (
        weight_bfs * bfs
        + weight_slack * slack
        + weight_dead_ends * dead_ends
        + weight_walls * wall_count
        + weight_branches * branch_points
    )
    if require_solvable:
        score = jnp.where(bfs < inf, score, -1e9)
    return score

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
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4)})
        
        wandb.log(log_dict)

    decoder_params = None
    encoder_params = None
    maze_encoder_params = None
    maze_decoder_params = None
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

        if config["plwm_use_maze_ae"]:
            # Grid-based CNN AE path — load only the MazeAE checkpoint
            mae_path = config["plwm_mae_checkpoint"]
            if mae_path is None:
                mae_path = os.path.join(ROOT_DIR, "vae", "model_maze_ae", "checkpoint_final.pkl")
            elif not os.path.isabs(mae_path):
                mae_path = os.path.abspath(os.path.join(os.getcwd(), mae_path))
            print(f"Loading Maze AE from {mae_path}...")
            full_mae_params = load_maze_ae_params(mae_path)
            maze_encoder_params = extract_maze_encoder_params(full_mae_params)
            maze_decoder_params = extract_maze_decoder_params(full_mae_params)
            print(
                f"PLWM mutation enabled (Maze AE, grid-based): "
                f"sigma={config['plwm_sigma']}, temp={config['plwm_decode_temperature']}"
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
            print(
                f"PLWM mutation enabled (CLUTTR VAE): latent_dim={me_latent_dim}, "
                f"sigma={config['plwm_sigma']}, temp={config['plwm_decode_temperature']}"
            )

        if config["plwm_surrogate_guided"]:
            if int(config["plwm_surrogate_num_candidates"]) < 1:
                raise ValueError("--plwm_surrogate_num_candidates must be >= 1.")
            print(
                "PLWM surrogate guidance enabled: "
                f"candidates={config['plwm_surrogate_num_candidates']}, "
                f"weights(bfs={config['plwm_surrogate_weight_bfs']}, "
                f"slack={config['plwm_surrogate_weight_slack']}, "
                f"dead={config['plwm_surrogate_weight_dead_ends']}, "
                f"walls={config['plwm_surrogate_weight_walls']}, "
                f"branch={config['plwm_surrogate_weight_branches']}), "
                f"require_solvable={config['plwm_surrogate_require_solvable']}"
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
        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
            obs,
        )
        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        network = ActorCritic(env.action_space(env_params).n)
        network_params = network.init(rng, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
            # optax.adam(learning_rate=config["lr"], eps=1e-5),
        )
        pholder_level = sample_random_level(jax.random.PRNGKey(0))
        sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
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
            plwm_last_uphill_fraction=jnp.array(0.0, dtype=jnp.float32),
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            replay_last_level_inds=pholder_level_inds,
            mutation_last_level_batch=pholder_level_batch,
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
            sampler, _ = level_sampler.insert_batch(sampler, new_levels, scores, {"max_return": max_returns})
            
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
            max_returns = jnp.maximum(level_sampler.get_levels_extra(sampler, level_inds)["max_return"], compute_max_returns(dones, rewards))
            scores = compute_score(config, dones, values, max_returns, advantages)
            sampler = level_sampler.update_batch(sampler, level_inds, scores, {"max_return": max_returns})
            
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
            """
                This mutates the previous batch of replay levels and potentially adds them to the level buffer.
                This also updates the policy iff `config["exploratory_grad_updates"]` is True.
            """
            sampler = train_state.sampler
            me_archive = train_state.me_archive
            rng, rng_mutate, rng_reset = jax.random.split(rng, 3)
            
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
                rng_mutate, rng_encode, rng_decode, rng_levels_key = jax.random.split(rng_mutate, 4)
                batch_size = config["num_train_envs"]
                num_candidates = int(config["plwm_surrogate_num_candidates"])

                if config["plwm_use_maze_ae"]:
                    # ---- Grid-based CNN AE path (full wall coverage) ----
                    # Convert Level -> (H, W, 3) grid
                    parent_grids = jax.vmap(maze_level_to_grid)(
                        parent_levels.wall_map,
                        parent_levels.goal_pos,
                        parent_levels.agent_pos,
                    )  # (batch, H, W, 3)

                    # Encode
                    parent_latents = encode_maze_levels(maze_encoder_params, parent_grids)

                    if config["plwm_surrogate_guided"]:
                        # Sample multiple perturbation candidates per parent and keep the
                        # structurally hardest one under a configurable surrogate.
                        candidate_noise = jax.random.normal(
                            rng_encode,
                            (batch_size, num_candidates, parent_latents.shape[-1]),
                        )
                        candidate_latents = (
                            parent_latents[:, None, :] + float(config["plwm_sigma"]) * candidate_noise
                        )
                        flat_latents = candidate_latents.reshape((batch_size * num_candidates, -1))

                        candidate_levels_flat = decode_maze_latents(
                            maze_decoder_params,
                            flat_latents,
                            jax.random.split(rng_decode, batch_size * num_candidates),
                            wall_threshold=0.5,
                            temperature=float(config["plwm_decode_temperature"]),
                        )
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
                        # Perturb once per parent (legacy PLWM behavior).
                        noise = jax.random.normal(rng_encode, parent_latents.shape)
                        child_latents = parent_latents + float(config["plwm_sigma"]) * noise

                        # Decode -> Level (with Gumbel-max sampling + repair)
                        child_levels = decode_maze_latents(
                            maze_decoder_params,
                            child_latents,
                            jax.random.split(rng_decode, batch_size),
                            wall_threshold=0.5,
                            temperature=float(config["plwm_decode_temperature"]),
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
                parent_scores = jnp.zeros((config["num_train_envs"],), dtype=jnp.float32)
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
            sampler, _ = level_sampler.insert_batch(sampler, child_levels, scores, {"max_return": max_returns})

            # Track how often selected PLWM mutations improve over their replay parent score.
            plwm_batch_uphill_fraction = jnp.array(0.0, dtype=jnp.float32)
            plwm_batch_compared = jnp.array(0, dtype=jnp.int32)
            plwm_batch_improved = jnp.array(0, dtype=jnp.int32)
            if config["use_plwm_mutation"]:
                plwm_batch_compared = jnp.array(config["num_train_envs"], dtype=jnp.int32)
                plwm_batch_improved = jnp.sum(scores > parent_scores).astype(jnp.int32)
                plwm_batch_uphill_fraction = plwm_batch_improved.astype(jnp.float32) / jnp.maximum(
                    plwm_batch_compared.astype(jnp.float32), 1.0
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
            }
            
            train_state = train_state.replace(
                sampler=sampler,
                me_archive=me_archive,
                update_state=UpdateState.DR,
                num_mutation_updates=train_state.num_mutation_updates + 1,
                num_plwm_compared=train_state.num_plwm_compared + plwm_batch_compared,
                num_plwm_improved=train_state.num_plwm_improved + plwm_batch_improved,
                plwm_last_uphill_fraction=plwm_batch_uphill_fraction,
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
    group.add_argument("--plwm_surrogate_guided", action=argparse.BooleanOptionalAction, default=False,
                       help="If set, sample multiple latent perturbation candidates per parent "
                            "and choose the best candidate using a structural difficulty surrogate.")
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
                       help="Use the grid-based CNN AE (SIGReg) instead of the CLUTTR sequence "
                            "VAE for PLWM encoding/decoding. Requires --use_plwm_mutation. "
                            "Supports any wall count (no 50-wall truncation).")
    group.add_argument("--plwm_mae_checkpoint", type=str, default=None,
                       help="Path to the MazeAE checkpoint pickle. Defaults to "
                            "vae/model_maze_ae/checkpoint_final.pkl relative to project root.")
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
