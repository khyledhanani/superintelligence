"""
RND-Enhanced PLR/ACCEL Implementation for Maze Environment

This file extends maze_plr.py with Random Network Distillation (RND) for
novelty-based level scoring. RND measures how "novel" a level is by computing
the prediction error of a learned predictor network against a fixed random
target network.

Key additions:
- RNDTargetNetwork: Fixed random network that produces target embeddings
- RNDPredictorNetwork: Trainable network that learns to predict target embeddings
- Level encoding: Converts Level structs to flat vectors for RND input
- Combined scoring: base_score + rnd_weight * normalized_rnd_bonus
- Running normalization: Tracks mean/var of RND bonuses for stable scaling

Usage:
    python examples/maze_plr_rnd.py --use_accel --use_rnd --rnd_weight 0.1
"""

import json
import time
from typing import Sequence, Tuple, NamedTuple
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


class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1


# =============================================================================
# RND Networks
# =============================================================================

class RNDTargetNetwork(nn.Module):
    """Fixed random network that produces target embeddings.
    
    This network is initialized randomly and never updated. It provides
    the "ground truth" embeddings that the predictor tries to match.
    """
    hidden_dim: int = 256
    output_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return x


class RNDPredictorNetwork(nn.Module):
    """Trainable network that learns to predict target embeddings.
    
    This network is trained to minimize the prediction error against the
    target network. High prediction error indicates a novel input that
    the predictor hasn't seen much of.
    """
    hidden_dim: int = 256
    output_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return x


@struct.dataclass
class RNDState:
    """State for the RND module, stored separately for clarity."""
    target_params: chex.ArrayTree
    predictor_params: chex.ArrayTree
    predictor_opt_state: optax.OptState
    # Running statistics for normalization
    running_mean: chex.Array  # scalar
    running_var: chex.Array   # scalar
    update_count: int         # for running stats update


def encode_level(level: Level, max_height: int, max_width: int) -> chex.Array:
    """Encode a Level struct into a flat vector for RND input.
    
    The encoding includes:
    - Flattened wall map (binary)
    - Normalized agent position
    - Normalized goal position  
    - Normalized agent direction
    
    Args:
        level: The Level to encode
        max_height: Maximum maze height (for normalization)
        max_width: Maximum maze width (for normalization)
    
    Returns:
        Flat array of shape (max_height * max_width + 5,)
    """
    # Flatten wall map
    wall_flat = level.wall_map.flatten().astype(jnp.float32)
    
    # Normalize positions to [0, 1]
    agent_pos_norm = level.agent_pos.astype(jnp.float32) / jnp.array([max_width, max_height], dtype=jnp.float32)
    goal_pos_norm = level.goal_pos.astype(jnp.float32) / jnp.array([max_width, max_height], dtype=jnp.float32)
    
    # Normalize direction to [0, 1]
    agent_dir_norm = jnp.array([level.agent_dir / 4.0], dtype=jnp.float32)
    
    return jnp.concatenate([wall_flat, agent_pos_norm, goal_pos_norm, agent_dir_norm])


def encode_levels_batch(levels: Level, max_height: int, max_width: int) -> chex.Array:
    """Vectorized level encoding for a batch of levels."""
    return jax.vmap(encode_level, in_axes=(0, None, None))(levels, max_height, max_width)


def create_rnd_state(
    rng: chex.PRNGKey,
    input_dim: int,
    hidden_dim: int = 256,
    output_dim: int = 64,
    learning_rate: float = 1e-4,
) -> Tuple[RNDState, RNDTargetNetwork, RNDPredictorNetwork]:
    """Initialize RND networks and state.
    
    Args:
        rng: Random key for initialization
        input_dim: Size of level encoding (max_height * max_width + 5)
        hidden_dim: Hidden layer size for RND networks
        output_dim: Embedding dimension
        learning_rate: Learning rate for predictor optimizer
    
    Returns:
        Tuple of (RNDState, target_network, predictor_network)
    """
    rng_target, rng_predictor = jax.random.split(rng)
    
    target_net = RNDTargetNetwork(hidden_dim=hidden_dim, output_dim=output_dim)
    predictor_net = RNDPredictorNetwork(hidden_dim=hidden_dim, output_dim=output_dim)
    
    # Initialize with dummy input
    dummy_input = jnp.zeros((1, input_dim))
    target_params = target_net.init(rng_target, dummy_input)
    predictor_params = predictor_net.init(rng_predictor, dummy_input)
    
    # Optimizer for predictor only (target is frozen)
    predictor_optimizer = optax.adam(learning_rate)
    predictor_opt_state = predictor_optimizer.init(predictor_params)
    
    rnd_state = RNDState(
        target_params=target_params,
        predictor_params=predictor_params,
        predictor_opt_state=predictor_opt_state,
        running_mean=jnp.array(0.0),
        running_var=jnp.array(1.0),
        update_count=0,
    )
    
    return rnd_state, target_net, predictor_net


def compute_rnd_bonus(
    level_encodings: chex.Array,
    rnd_state: RNDState,
    target_net: RNDTargetNetwork,
    predictor_net: RNDPredictorNetwork,
) -> chex.Array:
    """Compute RND bonus (prediction error) for a batch of level encodings.
    
    Args:
        level_encodings: Batch of encoded levels, shape (batch_size, input_dim)
        rnd_state: Current RND state with network params
        target_net: Target network (for apply)
        predictor_net: Predictor network (for apply)
    
    Returns:
        RND bonuses, shape (batch_size,)
    """
    target_emb = target_net.apply(rnd_state.target_params, level_encodings)
    pred_emb = predictor_net.apply(rnd_state.predictor_params, level_encodings)
    
    # Mean squared error per sample
    rnd_bonus = ((target_emb - pred_emb) ** 2).mean(axis=-1)
    return rnd_bonus


def normalize_rnd_bonus(
    rnd_bonus: chex.Array,
    rnd_state: RNDState,
) -> chex.Array:
    """Normalize RND bonus using running statistics.
    
    Args:
        rnd_bonus: Raw RND bonuses, shape (batch_size,)
        rnd_state: RND state containing running mean/var
    
    Returns:
        Normalized bonuses, shape (batch_size,)
    """
    normalized = (rnd_bonus - rnd_state.running_mean) / (jnp.sqrt(rnd_state.running_var) + 1e-8)
    # Clip to prevent extreme values
    return jnp.clip(normalized, -5.0, 5.0)


def update_rnd_running_stats(
    rnd_state: RNDState,
    rnd_bonus: chex.Array,
    momentum: float = 0.99,
) -> RNDState:
    """Update running mean and variance of RND bonuses.
    
    Uses exponential moving average for stable normalization.
    
    Args:
        rnd_state: Current RND state
        rnd_bonus: New batch of RND bonuses
        momentum: EMA momentum (higher = slower adaptation)
    
    Returns:
        Updated RND state
    """
    batch_mean = rnd_bonus.mean()
    batch_var = rnd_bonus.var()
    
    # Use lower momentum for initial updates to adapt faster
    effective_momentum = jnp.minimum(momentum, 1.0 - 1.0 / (rnd_state.update_count + 1))
    
    new_mean = effective_momentum * rnd_state.running_mean + (1 - effective_momentum) * batch_mean
    new_var = effective_momentum * rnd_state.running_var + (1 - effective_momentum) * batch_var
    
    return rnd_state.replace(
        running_mean=new_mean,
        running_var=jnp.maximum(new_var, 1e-8),  # Prevent zero variance
        update_count=rnd_state.update_count + 1,
    )


def update_rnd_predictor(
    rnd_state: RNDState,
    level_encodings: chex.Array,
    target_net: RNDTargetNetwork,
    predictor_net: RNDPredictorNetwork,
    predictor_optimizer: optax.GradientTransformation,
) -> Tuple[RNDState, chex.Array]:
    """Update the RND predictor to reduce prediction error.
    
    This trains the predictor to better predict the target network's outputs,
    which decreases the RND bonus for frequently seen level types.
    
    Args:
        rnd_state: Current RND state
        level_encodings: Batch of encoded levels to train on
        target_net: Target network (frozen)
        predictor_net: Predictor network (to be updated)
        predictor_optimizer: Optimizer for predictor
    
    Returns:
        Tuple of (updated RND state, loss value)
    """
    def rnd_loss_fn(predictor_params):
        target_emb = target_net.apply(rnd_state.target_params, level_encodings)
        pred_emb = predictor_net.apply(predictor_params, level_encodings)
        return ((target_emb - pred_emb) ** 2).mean()
    
    loss, grads = jax.value_and_grad(rnd_loss_fn)(rnd_state.predictor_params)
    updates, new_opt_state = predictor_optimizer.update(
        grads, rnd_state.predictor_opt_state, rnd_state.predictor_params
    )
    new_predictor_params = optax.apply_updates(rnd_state.predictor_params, updates)
    
    new_rnd_state = rnd_state.replace(
        predictor_params=new_predictor_params,
        predictor_opt_state=new_opt_state,
    )
    
    return new_rnd_state, loss


# =============================================================================
# Modified TrainState with RND
# =============================================================================

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # RND state
    rnd_state: RNDState = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)


# =============================================================================
# PPO Helper Functions (unchanged from maze_plr.py)
# =============================================================================

def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Compute GAE advantages and targets."""
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
    """Sample trajectories using the policy."""
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
    """Evaluate the policy on given initial states."""
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
    """PPO update for actor-critic with RNN."""
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
    """Actor-Critic network with LSTM."""
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


# =============================================================================
# Checkpointing
# =============================================================================

def setup_checkpointing(config: dict, train_state: TrainState, env: UnderspecifiedEnv, env_params: EnvParams) -> ocp.CheckpointManager:
    """Set up checkpointing with orbax."""
    overall_save_dir = os.path.join(os.getcwd(), "checkpoints", f"{config['run_name']}", str(config['seed']))
    os.makedirs(overall_save_dir, exist_ok=True)
    
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


# =============================================================================
# Logging Utilities
# =============================================================================

def train_state_to_log_dict(train_state: TrainState, level_sampler: LevelSampler, config: dict) -> dict:
    """Extract logging information from train state."""
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)
    
    log_dict = {
        "log": {
            "level_sampler/size": sampler["size"],
            "level_sampler/episode_count": sampler["episode_count"],
            "level_sampler/max_score": sampler["scores"].max(),
            "level_sampler/weighted_score": (sampler["scores"] * level_sampler.level_weights(sampler)).sum(),
            "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
        },
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        }
    }
    
    # Add RND stats if enabled
    if config["use_rnd"]:
        log_dict["log"]["rnd/running_mean"] = train_state.rnd_state.running_mean
        log_dict["log"]["rnd/running_var"] = train_state.rnd_state.running_var
        log_dict["log"]["rnd/update_count"] = train_state.rnd_state.update_count
    
    return log_dict


# =============================================================================
# Score Computation with RND
# =============================================================================

def compute_base_score(config, dones, values, max_returns, advantages):
    """Compute the base regret-based score."""
    if config['score_function'] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config['score_function'] == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")


def compute_score_with_rnd(
    config,
    dones,
    values,
    max_returns,
    advantages,
    levels: Level,
    rnd_state: RNDState,
    target_net: RNDTargetNetwork,
    predictor_net: RNDPredictorNetwork,
) -> Tuple[chex.Array, chex.Array]:
    """Compute combined score with RND novelty bonus.
    
    Args:
        config: Configuration dict
        dones, values, max_returns, advantages: Standard scoring inputs
        levels: Batch of levels being scored
        rnd_state: Current RND state
        target_net, predictor_net: RND networks
    
    Returns:
        Tuple of (combined_scores, raw_rnd_bonuses)
    """
    # Base regret score
    base_score = compute_base_score(config, dones, values, max_returns, advantages)
    
    if not config["use_rnd"]:
        return base_score, jnp.zeros_like(base_score)
    
    # Encode levels
    level_encodings = encode_levels_batch(levels, config["max_height"], config["max_width"])
    
    # Compute RND bonus
    rnd_bonus = compute_rnd_bonus(level_encodings, rnd_state, target_net, predictor_net)
    
    # Normalize RND bonus
    normalized_rnd = normalize_rnd_bonus(rnd_bonus, rnd_state)
    
    # Combine scores
    combined_score = base_score + config["rnd_weight"] * normalized_rnd
    
    return combined_score, rnd_bonus


# =============================================================================
# Main Training Loop
# =============================================================================

def main(config=None, project="JAXUED_TEST"):
    tags = []
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")
    if config["use_rnd"]:
        tags.append("RND")
    
    run = wandb.init(config=config, project=project, group=config["run_name"], tags=tags)
    config = wandb.config
    
    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")
    if config["use_rnd"]:
        wandb.define_metric("rnd/*", step_metric="num_updates")

    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")
        
        env_steps = stats["update_count"] * config["num_train_envs"] * config["num_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats['time_delta'],
        }
        
        solve_rates = stats['eval_solve_rates']
        returns = stats["eval_returns"]
        log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config["eval_levels"], solve_rates)})
        log_dict.update({"solve_rate/mean": solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config["eval_levels"], returns)})
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()})
        
        log_dict.update(train_state_info["log"])

        log_dict.update({"images/highest_scoring_level": wandb.Image(np.array(stats["highest_scoring_level"]), caption="Highest scoring level")})
        log_dict.update({"images/highest_weighted_level": wandb.Image(np.array(stats["highest_weighted_level"]), caption="Highest weighted level")})

        for s in ['dr', 'replay', 'mutation']:
            if train_state_info['info'][f'num_{s}_updates'] > 0:
                log_dict.update({f"images/{s}_levels": [wandb.Image(np.array(image)) for image in stats[f"{s}_levels"]]})

        for i, level_name in enumerate(config["eval_levels"]):
            frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
            frames = np.array(frames[:episode_length])
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4)})
        
        # Log RND-specific metrics
        if config["use_rnd"] and "rnd_loss" in stats:
            log_dict.update({"rnd/loss": float(stats["rnd_loss"])})
        
        wandb.log(log_dict)
    
    # Setup environment
    env = Maze(max_height=config["max_height"], max_width=config["max_width"], 
               agent_view_size=config["agent_view_size"], normalize_obs=True)
    eval_env = env
    sample_random_level = make_level_generator(env.max_height, env.max_width, config["n_walls"])
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    mutate_level = make_level_mutator_minimax(100)

    # Level sampler
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config['topk_k']},
        duplicate_check=config['buffer_duplicate_check'],
    )
    
    # RND networks (created once, used in jitted functions)
    rnd_input_dim = config["max_height"] * config["max_width"] + 5
    _, target_net, predictor_net = create_rnd_state(
        jax.random.PRNGKey(0), rnd_input_dim,
        hidden_dim=config["rnd_hidden_dim"],
        output_dim=config["rnd_output_dim"],
        learning_rate=config["rnd_lr"],
    )
    rnd_predictor_optimizer = optax.adam(config["rnd_lr"])
    
    @jax.jit
    def create_train_state(rng) -> TrainState:
        """Create initial training state including RND."""
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["num_minibatches"] * config["epoch_ppo"]))
                / config["num_updates"]
            )
            return config["lr"] * frac
        
        rng, rng_rnd = jax.random.split(rng)
        
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
        )
        
        pholder_level = sample_random_level(jax.random.PRNGKey(0))
        sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
        pholder_level_batch = jax.tree_util.tree_map(
            lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), pholder_level
        )
        
        # Initialize RND state
        rnd_state, _, _ = create_rnd_state(
            rng_rnd, rnd_input_dim,
            hidden_dim=config["rnd_hidden_dim"],
            output_dim=config["rnd_output_dim"],
            learning_rate=config["rnd_lr"],
        )
        
        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            rnd_state=rnd_state,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            mutation_last_level_batch=pholder_level_batch,
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """Main training step with RND integration."""
        
        def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
            """Sample and evaluate new random levels."""
            sampler = train_state.sampler
            rnd_state = train_state.rnd_state
            
            # Reset
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_train_envs"]))
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, config["num_train_envs"]), new_levels, env_params
            )
            
            # Rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng, env, env_params, train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs, init_env_state,
                config["num_train_envs"], config["num_steps"],
            )
            
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = compute_max_returns(dones, rewards)
            
            # Compute scores with RND
            scores, rnd_bonuses = compute_score_with_rnd(
                config, dones, values, max_returns, advantages,
                new_levels, rnd_state, target_net, predictor_net,
            )
            
            # Update RND running stats and predictor
            if config["use_rnd"]:
                rnd_state = update_rnd_running_stats(rnd_state, rnd_bonuses)
                level_encodings = encode_levels_batch(new_levels, config["max_height"], config["max_width"])
                rnd_state, rnd_loss = update_rnd_predictor(
                    rnd_state, level_encodings, target_net, predictor_net, rnd_predictor_optimizer
                )
            else:
                rnd_loss = 0.0
            
            sampler, _ = level_sampler.insert_batch(sampler, new_levels, scores, {"max_return": max_returns})
            
            # Update policy (only if exploratory_grad_updates)
            (rng, train_state), losses = update_actor_critic_rnn(
                rng, train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"], config["num_steps"],
                config["num_minibatches"], config["epoch_ppo"],
                config["clip_eps"], config["entropy_coeff"], config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )
            
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": new_levels.wall_map.sum() / config["num_train_envs"],
                "rnd_loss": rnd_loss,
            }
            
            train_state = train_state.replace(
                sampler=sampler,
                rnd_state=rnd_state,
                update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=new_levels,
            )
            return (rng, train_state), metrics
        
        def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):
            """Sample and train on replay levels."""
            sampler = train_state.sampler
            rnd_state = train_state.rnd_state
            
            # Sample replay levels
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(
                sampler, rng_levels, config["num_train_envs"]
            )
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params
            )
            
            # Rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng, env, env_params, train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs, init_env_state,
                config["num_train_envs"], config["num_steps"],
            )
            
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = jnp.maximum(
                level_sampler.get_levels_extra(sampler, level_inds)["max_return"],
                compute_max_returns(dones, rewards)
            )
            
            # Compute scores with RND
            scores, rnd_bonuses = compute_score_with_rnd(
                config, dones, values, max_returns, advantages,
                levels, rnd_state, target_net, predictor_net,
            )
            
            # Update RND (even on replay to keep predictor learning)
            if config["use_rnd"]:
                rnd_state = update_rnd_running_stats(rnd_state, rnd_bonuses)
                level_encodings = encode_levels_batch(levels, config["max_height"], config["max_width"])
                rnd_state, rnd_loss = update_rnd_predictor(
                    rnd_state, level_encodings, target_net, predictor_net, rnd_predictor_optimizer
                )
            else:
                rnd_loss = 0.0
            
            sampler = level_sampler.update_batch(sampler, level_inds, scores, {"max_return": max_returns})
            
            # Update policy
            (rng, train_state), losses = update_actor_critic_rnn(
                rng, train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"], config["num_steps"],
                config["num_minibatches"], config["epoch_ppo"],
                config["clip_eps"], config["entropy_coeff"], config["critic_coeff"],
                update_grad=True,
            )
            
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],
                "rnd_loss": rnd_loss,
            }
            
            train_state = train_state.replace(
                sampler=sampler,
                rnd_state=rnd_state,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
            )
            return (rng, train_state), metrics
        
        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            """Mutate replay levels (ACCEL)."""
            sampler = train_state.sampler
            rnd_state = train_state.rnd_state
            rng, rng_mutate, rng_reset = jax.random.split(rng, 3)
            
            # Mutate
            parent_levels = train_state.replay_last_level_batch
            child_levels = jax.vmap(mutate_level, (0, 0, None))(
                jax.random.split(rng_mutate, config["num_train_envs"]), parent_levels, config["num_edits"]
            )
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, config["num_train_envs"]), child_levels, env_params
            )

            # Rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng, env, env_params, train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs, init_env_state,
                config["num_train_envs"], config["num_steps"],
            )
            
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = compute_max_returns(dones, rewards)
            
            # Compute scores with RND - this is where RND really shines for ACCEL
            # Novel mutations get higher scores even if regret is similar
            scores, rnd_bonuses = compute_score_with_rnd(
                config, dones, values, max_returns, advantages,
                child_levels, rnd_state, target_net, predictor_net,
            )
            
            # Update RND
            if config["use_rnd"]:
                rnd_state = update_rnd_running_stats(rnd_state, rnd_bonuses)
                level_encodings = encode_levels_batch(child_levels, config["max_height"], config["max_width"])
                rnd_state, rnd_loss = update_rnd_predictor(
                    rnd_state, level_encodings, target_net, predictor_net, rnd_predictor_optimizer
                )
            else:
                rnd_loss = 0.0
            
            sampler, _ = level_sampler.insert_batch(sampler, child_levels, scores, {"max_return": max_returns})
            
            # Update policy (only if exploratory_grad_updates)
            (rng, train_state), losses = update_actor_critic_rnn(
                rng, train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"], config["num_steps"],
                config["num_minibatches"], config["epoch_ppo"],
                config["clip_eps"], config["entropy_coeff"], config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )
            
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": child_levels.wall_map.sum() / config["num_train_envs"],
                "rnd_loss": rnd_loss,
            }
            
            train_state = train_state.replace(
                sampler=sampler,
                rnd_state=rnd_state,
                update_state=UpdateState.DR,
                num_mutation_updates=train_state.num_mutation_updates + 1,
                mutation_last_level_batch=child_levels,
            )
            return (rng, train_state), metrics
    
        rng, train_state = carry
        rng, rng_replay = jax.random.split(rng)
        
        # Branch selection logic
        if config["use_accel"]:
            s = train_state.update_state
            branch = (1 - s) * level_sampler.sample_replay_decision(train_state.sampler, rng_replay) + 2 * s
        else:
            branch = level_sampler.sample_replay_decision(train_state.sampler, rng_replay).astype(int)
        
        return jax.lax.switch(
            branch,
            [on_new_levels, on_replay_levels, on_mutate_levels],
            rng, train_state
        )
    
    def eval(rng: chex.PRNGKey, train_state: TrainState):
        """Evaluate on held-out levels."""
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, num_levels), levels, env_params
        )
        states, rewards, episode_lengths = evaluate_rnn(
            rng, eval_env, env_params, train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs, init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths
    
    @jax.jit
    def train_and_eval_step(runner_state, _):
        """Combined train and eval step."""
        # Train
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])

        # Eval
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(
            jax.random.split(rng_eval, config["eval_num_attempts"]), train_state
        )
        
        # Collect metrics
        eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0)
        eval_returns = cum_rewards.mean(axis=0)
        
        states, episode_lengths = jax.tree_util.tree_map(lambda x: x[0], (states, episode_lengths))
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params)
        frames = images.transpose(0, 1, 4, 2, 3)
        
        metrics["update_count"] = train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"] = episode_lengths
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
        """Evaluate a saved checkpoint."""
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))
        
        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, 'config.json')) as f:
                config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(
                os.path.join(os.getcwd(), checkpoint_directory, 'models'),
                item_handlers=ocp.StandardCheckpointHandler()
            )

            train_state_og: TrainState = create_train_state(rng_init)
            step = checkpoint_manager.latest_step() if og_config['checkpoint_to_eval'] == -1 else og_config['checkpoint_to_eval']

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint['params']
            train_state = train_state_og.replace(params=params)
            return train_state, config
        
        train_state, config = load(rng_init, og_config['checkpoint_directory'])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(
            jax.random.split(rng_eval, og_config["eval_num_attempts"]), train_state
        )
        save_loc = og_config['checkpoint_directory'].replace('checkpoints', 'results')
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_loc, 'results.npz'),
            states=np.asarray(states),
            cum_rewards=np.asarray(cum_rewards),
            episode_lengths=np.asarray(episode_lengths),
            levels=config['eval_levels']
        )
        return states, cum_rewards, episode_lengths

    if config['mode'] == 'eval':
        return eval_checkpoint(config)

    # Initialize training
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)
    
    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)
    
    # Run training
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler, config))
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()
    
    return runner_state[1]


if __name__ == "__main__":
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
    
    # === RND (NEW) ===
    group.add_argument("--use_rnd", action=argparse.BooleanOptionalAction, default=False,
                       help="Enable RND-based novelty bonus for level scoring")
    group.add_argument("--rnd_weight", type=float, default=0.1,
                       help="Weight of RND bonus in combined score (base_score + rnd_weight * rnd_bonus)")
    group.add_argument("--rnd_lr", type=float, default=1e-4,
                       help="Learning rate for RND predictor network")
    group.add_argument("--rnd_hidden_dim", type=int, default=256,
                       help="Hidden dimension for RND networks")
    group.add_argument("--rnd_output_dim", type=int, default=64,
                       help="Output embedding dimension for RND networks")
    
    # === ENV CONFIG ===
    group.add_argument("--agent_view_size", type=int, default=5)
    group.add_argument("--max_height", type=int, default=13)
    group.add_argument("--max_width", type=int, default=13)
    
    # === DR CONFIG ===
    group.add_argument("--n_walls", type=int, default=25)
    
    config = vars(parser.parse_args())
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])
    config["group_name"] = ''.join([str(config[key]) for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])])
    
    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'
    
    main(config, project=config["project"])
