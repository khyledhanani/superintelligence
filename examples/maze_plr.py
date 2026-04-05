import json
import logging
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
import yaml
import pickle
import sys
from enum import IntEnum

# Metrics imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# VAE + CMA-ES imports (conditional on --use_cmaes flag)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vae'))
from vae_model import CluttrVAE
from vae_level_utils import decode_latent_to_levels, level_to_tokens, tokens_to_level
from cmaes_manager import CMAESManager

from llm.injection_config import LLMInjectionConfig
from llm.injector import LLMInjectionManager
from llm.level_cache import LevelCache

class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    es_state: chex.ArrayTree = struct.field(pytree_node=True)
    # === Proximity gate for LLM-lineage mutations (JIT-visible) ===
    seed_embeddings: chex.Array = struct.field(pytree_node=True)      # (max_ancestors, 257)
    proximity_thresholds: chex.Array = struct.field(pytree_node=True)  # (max_ancestors,)
    n_seeds: int = struct.field(pytree_node=True)                      # number of registered seeds
    fixed_dist_n_llm: int = struct.field(pytree_node=True)             # LLM replay slots for fixed distribution
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_inds: chex.Array = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)

# region PPO helper functions
def compute_insertion_embeddings(
    hstates: chex.Array,
    actions: chex.Array,
    dones: chex.Array,
) -> chex.Array:
    """Compute per-level mean [hstate(256), action(1)] embedding, masked to first episode.

    Args:
        hstates: (NUM_STEPS, NUM_ENVS, 256) — LSTM output hidden states
        actions: (NUM_STEPS, NUM_ENVS) — discrete actions
        dones: (NUM_STEPS, NUM_ENVS) — episode termination flags

    Returns:
        (NUM_ENVS, 257) — mean state-action embedding per level
    """
    # Mask: 1 for all timesteps up to and including the first done, 0 after
    pre_done = jnp.concatenate([jnp.zeros((1,) + dones.shape[1:]), dones[:-1]], axis=0)
    episode_mask = (jnp.cumsum(pre_done, axis=0) == 0).astype(jnp.float32)  # (T, num_envs)
    ep_lengths = jnp.maximum(episode_mask.sum(axis=0), 1.0)  # (num_envs,) avoid div by zero

    mean_h = (hstates * episode_mask[..., None]).sum(axis=0) / ep_lengths[..., None]  # (num_envs, 256)
    mean_a = (actions.astype(jnp.float32) * episode_mask).sum(axis=0) / ep_lengths  # (num_envs,)
    return jnp.concatenate([mean_h, mean_a[:, None]], axis=-1)  # (num_envs, 257)


def _sample_masked_level(level_sampler_obj, sampler, rng, mask):
    """Sample one replay level from slots where mask is True.

    Like sample_replay_level but zeros out weights for masked-out slots.
    """
    weights = level_sampler_obj.level_weights(sampler)
    weights = jnp.where(mask, weights, 0.0)
    weights = weights / jnp.maximum(weights.sum(), 1e-8)
    idx = jax.random.choice(rng, level_sampler_obj.capacity, p=weights)
    new_episode_count = sampler["episode_count"] + 1
    sampler = {
        **sampler,
        "timestamps": sampler["timestamps"].at[idx].set(new_episode_count),
        "episode_count": new_episode_count,
    }
    return sampler, (idx, jax.tree_util.tree_map(lambda x: x[idx], sampler["levels"]))


def compute_td_errors(
    gamma: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> chex.Array:
    """Compute per-step TD errors: δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t).

    Args:
        gamma: Discount factor
        last_value: Shape (NUM_ENVS,) — value estimate at final step
        values: Shape (NUM_STEPS, NUM_ENVS)
        rewards: Shape (NUM_STEPS, NUM_ENVS)
        dones: Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        TD errors of shape (NUM_STEPS, NUM_ENVS)
    """
    # next_values[t] = values[t+1] for t < T-1, last_value for t = T-1
    next_values = jnp.concatenate([values[1:], last_value[None, :]], axis=0)
    td_errors = rewards + gamma * next_values * (1 - dones) - values
    return td_errors

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
) -> Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict, chex.Array]]:
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
        (rng, train_state, hstate, last_obs, last_env_state, last_value), traj
        where traj is (obs, action, reward, done, log_prob, value, info, agent_pos, hstate_h).
        hstate_h is the LSTM output hidden state of shape (NUM_STEPS, NUM_ENVS, 256).
    """
    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        # Capture agent position before the step (matches obs)
        agent_pos = env_state.env_state.agent_pos  # (num_envs, 2) — env_state is AutoReplayState

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        # Extract LSTM output hidden state (carry_h, not carry_c)
        hstate_h = hstate[1]  # (num_envs, 256)

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info, agent_pos, hstate_h)

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
def _upload_to_gcs(local_path, gcs_bucket, gcs_path):
    """Upload a local file to GCS. Uses google.cloud.storage if available, else gcloud CLI."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
    except (ImportError, Exception) as e:
        print(f"[GCS] Python client failed ({e}), falling back to gcloud CLI")
        import subprocess
        dest = f"gs://{gcs_bucket}/{gcs_path}"
        subprocess.run(["gcloud", "storage", "cp", local_path, dest], check=True)
    print(f"[GCS] Uploaded {local_path} -> gs://{gcs_bucket}/{gcs_path}")


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
    if config.get("gcs_bucket"):
        overall_save_dir = f"gs://{config['gcs_bucket']}/{config['gcs_prefix']}/checkpoints/{config['run_name']}/{config['seed']}"
        # Save config to GCS
        config_json = json.dumps(dict(config), indent=2)
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(config["gcs_bucket"])
            blob = bucket.blob(f"{config['gcs_prefix']}/checkpoints/{config['run_name']}/{config['seed']}/config.json")
            blob.upload_from_string(config_json)
        except (ImportError, Exception):
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(config_json)
                tmp_path = f.name
            subprocess.run(["gcloud", "storage", "cp", tmp_path, f"{overall_save_dir}/config.json"], check=True)
            os.remove(tmp_path)
        print(f"[GCS] Config saved to {overall_save_dir}/config.json")
    else:
        if config.get("output_dir"):
            overall_save_dir = os.path.join(config["output_dir"], "checkpoints")
        else:
            overall_save_dir = os.path.join(os.getcwd(), "checkpoints", f"{config['run_name']}", str(config['seed']))
        os.makedirs(overall_save_dir, exist_ok=True)
        with open(os.path.join(overall_save_dir, 'config.json'), 'w+') as f:
            f.write(json.dumps(dict(config), indent=2))

    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, 'models'),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config['checkpoint_save_interval'],
            max_to_keep=config['max_number_of_checkpoints'],
        )
    )
    return checkpoint_manager
#endregion

def train_state_to_log_dict(train_state: TrainState, level_sampler: LevelSampler) -> dict:
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

    scores = sampler["scores"]
    mean_score = (scores * idx).sum() / s
    return {
        "log":{
            "level_sampler/size": sampler["size"],
            "level_sampler/episode_count": sampler["episode_count"],
            "level_sampler/max_score": scores.max(),
            "level_sampler/weighted_score": (scores * level_sampler.level_weights(sampler)).sum(),
            "level_sampler/mean_score": mean_score,
            "level_sampler/score_std": jnp.sqrt(((jnp.where(idx, scores, 0) - mean_score) ** 2 * idx).sum() / s),
        },
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        }
    }

def compute_score(config, dones, values, max_returns, advantages):
    if config['score_function'] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config['score_function'] == "pvl":
        return positive_value_loss(dones, advantages)
    elif config['score_function'] == "sfl":
        # SFL doesn't use regret-based scores; return zeros as placeholder
        # (actual SFL scores computed separately via multi-rollout eval)
        return jnp.zeros(dones.shape[1])
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")

def main(config=None, project="JAXUED_TEST"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("wandb").setLevel(logging.WARNING)
    tags = []
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")
    if config.get("use_cmaes"):
        tags.append("CMA-ES")
    if config.get("use_llm"):
        tags.append("llm")
    if config.get("score_function") == "sfl":
        tags.append("SFL")
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")
    print("=" * 60)

    wb_group = config.get("wandb_group") or config["run_name"]
    wb_entity = config.get("wandb_entity") or None
    run = wandb.init(config=config, project=project, entity=wb_entity, group=wb_group, tags=tags)
    config = wandb.config
    
    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")
    wandb.define_metric("gen/*", step_metric="num_updates")
    if config["use_cmaes"]:
        wandb.define_metric("cmaes/*", step_metric="num_updates")
    wandb.define_metric("buffer/*", step_metric="num_updates")
    wandb.define_metric("prox_gate/*", step_metric="num_updates")
    wandb.define_metric("interp_compare/*", step_metric="num_updates")
    if config.get("use_llm"):
        wandb.define_metric("llm/*", step_metric="num_updates")

    # --- CMA-ES + VAE setup ---
    vae_decode_fn = None
    vae_encode_fn = None
    cmaes_mgr = None
    _need_vae = config["use_cmaes"] or config.get("interp_compare_interval", 0) > 0
    if _need_vae:
        assert config["vae_checkpoint_path"] is not None, "--vae_checkpoint_path required when --use_cmaes or --interp_compare_interval"
        assert config["vae_config_path"] is not None, "--vae_config_path required when --use_cmaes or --interp_compare_interval"

        # Load VAE config
        with open(config["vae_config_path"]) as f:
            vae_cfg = yaml.safe_load(f)

        # Instantiate model with config dimensions
        vae = CluttrVAE(
            vocab_size=vae_cfg["vocab_size"],
            embed_dim=vae_cfg["embed_dim"],
            latent_dim=vae_cfg["latent_dim"],
            seq_len=vae_cfg["seq_len"],
            enc_lstm_dim=vae_cfg.get("enc_lstm_dim", 300),
            dec_lstm_dim=vae_cfg.get("dec_lstm_dim", 400),
        )

        # Load checkpoint
        with open(config["vae_checkpoint_path"], "rb") as f:
            vae_ckpt = pickle.load(f)
        vae_params = vae_ckpt["params"] if isinstance(vae_ckpt, dict) and "params" in vae_ckpt else vae_ckpt

        # Build pure encode/decode functions
        def vae_encode_fn(tokens_batch):
            mean, _ = vae.apply({"params": vae_params}, tokens_batch, train=False, method=vae.encode)
            return mean

        def vae_decode_fn(z):
            return vae.apply({"params": vae_params}, z, method=vae.decode)

        # VAE grid parameters (defaults match 13x13; 21x21 config overrides them)
        _vae_grid_size = vae_cfg.get("grid_size", 13)
        _vae_vocab_size = vae_cfg["vocab_size"]
        _vae_max_walls = vae_cfg.get("max_walls", 50)

        from functools import partial
        level_to_tokens_vae = partial(level_to_tokens, grid_size=_vae_grid_size, max_walls=_vae_max_walls)
        decode_latent_to_levels_vae = partial(decode_latent_to_levels, grid_size=_vae_grid_size, vocab_size=_vae_vocab_size)

        print(f"[VAE] Loaded from {config['vae_checkpoint_path']}, latent_dim={vae_cfg['latent_dim']}, grid_size={_vae_grid_size}")

    if config["use_cmaes"]:
        # Initialize CMA-ES manager
        cmaes_mgr = CMAESManager(
            popsize=config["num_train_envs"],
            latent_dim=vae_cfg["latent_dim"],
            sigma_init=config["cmaes_sigma_init"],
        )
        print(f"[CMA-ES] popsize={config['num_train_envs']}")

    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")
        
        # generic stats
        _offset = config.get("wandb_update_offset", 0)
        effective_update = stats["update_count"] + _offset
        env_steps = effective_update * config["num_train_envs"] * config["num_steps"]
        log_dict = {
            "num_updates": effective_update,
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

        # Validity rate and insertion rate logging (averaged over eval_freq steps, excluding replay steps where it's 0)
        if "gen/valid_structure_pct" in stats:
            valid_pct = np.array(stats["gen/valid_structure_pct"])
            gen_mask = valid_pct > 0  # DR and mutation steps have non-zero validity
            if gen_mask.any():
                log_dict["gen/valid_structure_pct"] = float(valid_pct[gen_mask].mean())


        # CMA-ES metrics (averaged over the eval_freq training steps)
        if config.get("use_cmaes") and "cmaes/valid_structure_pct" in stats:
            # stats from scan have shape (eval_freq,); take mean of DR steps only (non-zero entries)
            valid_pct = np.array(stats["cmaes/valid_structure_pct"])
            dr_mask = valid_pct > 0  # only DR steps have non-zero valid_structure_pct
            if dr_mask.any():
                log_dict["cmaes/valid_structure_pct"] = float(valid_pct[dr_mask].mean())
                log_dict["cmaes/mean_fitness"] = float(np.array(stats["cmaes/mean_fitness"])[dr_mask].mean())
                log_dict["cmaes/mean_episode_length"] = float(np.array(stats["cmaes/mean_episode_length"])[dr_mask].mean())
                log_dict["cmaes/sigma"] = float(np.array(stats["cmaes/sigma"])[dr_mask].mean())
                log_dict["cmaes/pop_spread"] = float(np.array(stats["cmaes/pop_spread"])[dr_mask].mean())
                log_dict["cmaes/mean_z_norm"] = float(np.array(stats["cmaes/mean_z_norm"])[dr_mask].mean())

        # Buffer diversity metrics (unconditional — logged for all run types)
        if _buffer_embeddings is not None:
            size = int(runner_state[1].sampler["size"])
            emb_buf = _buffer_embeddings[:size]
            # Filter to non-zero embeddings (slots that have been written at least once)
            norms = np.sqrt(np.sum(emb_buf ** 2, axis=1))
            valid_mask = norms > 1e-6
            n_valid = int(valid_mask.sum())

            if n_valid >= 2:
                valid_emb = emb_buf[valid_mask]

                # Pairwise L2 distance (subsampled for speed)
                from scipy.spatial.distance import pdist
                n_sub = min(300, n_valid)
                sub_idx = np.random.choice(n_valid, n_sub, replace=False)
                pairwise_dists = pdist(valid_emb[sub_idx])
                log_dict["buffer/emb_pairwise_l2_mean"] = float(pairwise_dists.mean())
                log_dict["buffer/emb_pairwise_l2_std"] = float(pairwise_dists.std())

                # Log-det covariance (log-volume of embedding space covered)
                centered = valid_emb - valid_emb.mean(axis=0)
                cov = (centered.T @ centered) / (n_valid - 1)
                eigvals = np.linalg.eigvalsh(cov)
                log_det = float(np.sum(np.log(eigvals[eigvals > 1e-10])))
                log_dict["buffer/emb_log_det_cov"] = log_det

        # LLM provenance embedding metrics (only when LLM injection is active)
        if _buffer_embeddings is not None and _buffer_ancestor_ids is not None:
            size = int(runner_state[1].sampler["size"])
            emb_buf = _buffer_embeddings[:size]
            anc_buf = _buffer_ancestor_ids[:size]
            llm_mask = anc_buf >= 0
            org_mask = anc_buf < 0
            n_llm = int(llm_mask.sum())
            n_org = int(org_mask.sum())

            log_dict["provenance/n_llm_in_buffer"] = n_llm
            log_dict["provenance/llm_buffer_pct"] = n_llm / max(size, 1)

            if n_org > 0 and n_llm > 0:
                org_centroid = emb_buf[org_mask].mean(axis=0)
                llm_centroid = emb_buf[llm_mask].mean(axis=0)
                llm_to_org = np.linalg.norm(emb_buf[llm_mask] - org_centroid, axis=1)
                log_dict["provenance/llm_organic_centroid_dist"] = float(llm_to_org.mean())
                log_dict["provenance/inter_cluster_dist"] = float(np.linalg.norm(llm_centroid - org_centroid))

                seed_embs_np = np.asarray(runner_state[1].seed_embeddings)
                llm_anc = anc_buf[llm_mask]
                safe_anc = np.clip(llm_anc, 0, seed_embs_np.shape[0] - 1)
                llm_to_seed = np.linalg.norm(emb_buf[llm_mask] - seed_embs_np[safe_anc], axis=1)
                log_dict["provenance/llm_seed_dist_mean"] = float(llm_to_seed.mean())

                if n_llm > 1:
                    n_sub = min(200, n_llm)
                    sub_idx = np.random.choice(n_llm, n_sub, replace=False)
                    from scipy.spatial.distance import pdist
                    intra_dists = pdist(emb_buf[llm_mask][sub_idx])
                    log_dict["provenance/intra_llm_spread"] = float(intra_dists.mean())

        # Proximity gate stats (aggregated over eval_freq steps, LLM injection only)
        if config.get("use_llm") and "prox_gate/n_blocked" in stats:
            blocked = np.array(stats["prox_gate/n_blocked"])
            llm_parents = np.array(stats["prox_gate/n_llm_parents"])
            total_blocked = int(blocked.sum())
            total_llm = int(llm_parents.sum())
            log_dict["prox_gate/total_blocked"] = total_blocked
            log_dict["prox_gate/total_llm_parents"] = total_llm
            log_dict["prox_gate/block_rate"] = total_blocked / max(total_llm, 1)
            log_dict["prox_gate/mean_dist"] = float(np.array(stats["prox_gate/mean_dist"]).mean())

        # Provenance-aware replay rate tracking
        if _buffer_origins is not None and "replay_level_inds" in stats:
            all_inds = np.array(stats["replay_level_inds"])  # (eval_freq, num_train_envs)
            valid_mask = all_inds >= 0  # -1 = DR or mutation step (no replay)
            valid_inds = all_inds[valid_mask].flatten()

            # Buffer-level provenance counts
            size = int(runner_state[1].sampler["size"])
            origins_slice = _buffer_origins[:size]
            n_buf_original = int((origins_slice == 1).sum())   # direct LLM seeds
            n_buf_descendants = int((origins_slice == 2).sum()) # mutation descendants
            n_buf_lineage = n_buf_original + n_buf_descendants
            log_dict["provenance/buffer_original_injected"] = n_buf_original
            log_dict["provenance/buffer_descendants"] = n_buf_descendants
            log_dict["provenance/buffer_lineage_total"] = n_buf_lineage
            log_dict["provenance/buffer_lineage_pct"] = n_buf_lineage / max(size, 1)

            # Per-ancestor counts: how many descendants each LLM seed has in buffer
            if _buffer_ancestor_ids is not None:
                ancestor_slice = _buffer_ancestor_ids[:size]
                lineage_ancestors = ancestor_slice[origins_slice > 0]
                if len(lineage_ancestors) > 0 and (lineage_ancestors >= 0).any():
                    unique_a, counts_a = np.unique(
                        lineage_ancestors[lineage_ancestors >= 0], return_counts=True)
                    log_dict["provenance/n_active_ancestors"] = len(unique_a)
                    for a_id, cnt in zip(unique_a, counts_a):
                        log_dict[f"provenance/ancestor_{a_id}_count"] = int(cnt)
                else:
                    log_dict["provenance/n_active_ancestors"] = 0

            if len(valid_inds) > 0:
                # Count replays by origin type
                replayed_origins = _buffer_origins[valid_inds]
                n_total = len(valid_inds)
                n_organic = int((replayed_origins == 0).sum())
                n_injected = int((replayed_origins > 0).sum())
                log_dict["provenance/replay_total"] = n_total
                log_dict["provenance/replay_organic"] = n_organic
                log_dict["provenance/replay_injected"] = n_injected
                log_dict["provenance/replay_injected_pct"] = n_injected / max(n_total, 1)
                # How many unique injected slots were replayed
                unique_injected = len(set(valid_inds[replayed_origins > 0]))
                log_dict["provenance/replay_unique_injected"] = unique_injected

        wandb.log(log_dict)

    # Setup the environment
    env = Maze(max_height=config["maze_height"], max_width=config["maze_width"], agent_view_size=config["agent_view_size"], normalize_obs=True)
    eval_env = env
    sample_random_level = make_level_generator(env.max_height, env.max_width, config["n_walls"])
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    mutate_level = make_level_mutator_minimax(100)

    # --- SFL: multi-rollout learnability scoring ---
    def compute_sfl_scores(rng, train_state, levels, max_returns):
        """Estimate learnability = p * (1-p) via multi-rollout evaluation."""
        train_success = (max_returns > 0).astype(jnp.float32)

        def sfl_eval_step(carry, rng_eval):
            rng_r, rng_e = jax.random.split(rng_eval)
            init_obs_e, init_env_state_e = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
                jax.random.split(rng_r, config["num_train_envs"]), levels, env_params)
            _, rewards_e, _ = evaluate_rnn(
                rng_e, eval_env, env_params, train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs_e, init_env_state_e,
                env_params.max_steps_in_episode)
            success = (rewards_e.sum(axis=0) > 0).astype(jnp.float32)
            return carry, success

        eval_rngs = jax.random.split(rng, config["num_sfl_rollouts"] - 1)
        _, eval_successes = jax.lax.scan(sfl_eval_step, jnp.int32(0), eval_rngs)

        all_successes = jnp.concatenate([train_success[None], eval_successes], axis=0)
        p = all_successes.mean(axis=0)
        return p * (1 - p)

    def compute_level_scores(rng, train_state, levels, dones, values, max_returns, advantages):
        """Unified score dispatch: SFL uses multi-rollout eval, others use regret."""
        if config["score_function"] == "sfl":
            return compute_sfl_scores(rng, train_state, levels, max_returns)
        else:
            return compute_score(config, dones, values, max_returns, advantages)

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
    
    # Initialize CMA-ES state OUTSIDE jit to avoid tracing issues with evosax
    if cmaes_mgr is not None:
        es_state_init = cmaes_mgr.initialize(jax.random.PRNGKey(42))
        # Verify shapes before entering any jit context
        print(f"[CMA-ES] Initialized es_state: mean.shape={es_state_init.mean.shape}, "
              f"p_std.shape={es_state_init.p_std.shape}, C.shape={es_state_init.C.shape}")
        assert es_state_init.mean.shape == (cmaes_mgr.latent_dim,), (
            f"CMA-ES state.mean has shape {es_state_init.mean.shape}, "
            f"expected ({cmaes_mgr.latent_dim},). "
            f"This likely means evosax inferred the wrong num_dims."
        )
    else:
        es_state_init = None

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
        sampler = level_sampler.initialize(pholder_level, {
            "max_return": -jnp.inf,
            "ancestor_id": jnp.int32(-1),
        })
        pholder_level_batch = jax.tree_util.tree_map(lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), pholder_level)
        max_anc = config.get("max_llm_ancestors", 256)

        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            es_state=es_state_init,
            seed_embeddings=jnp.zeros((max_anc, 257), dtype=jnp.float32),
            proximity_thresholds=jnp.full(max_anc, jnp.inf, dtype=jnp.float32),
            n_seeds=jnp.int32(0),
            fixed_dist_n_llm=jnp.int32(0),
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            replay_last_level_inds=jnp.full(config["num_train_envs"], -1, dtype=jnp.int32),
            mutation_last_level_batch=pholder_level_batch,
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """
            This is the main training loop. It basically calls either `on_new_levels`, `on_replay_levels`, or `on_mutate_levels` at every step.
        """
        def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                Generates new levels and evaluates the policy on them.
                When use_cmaes=True: uses CMA-ES to search the VAE latent space.
                When use_cmaes=False: generates random levels (original behavior).
                Levels are added to the PLR buffer based on scores.
                The agent is updated iff `config["exploratory_grad_updates"]` is True.
            """
            sampler = train_state.sampler
            es_state = train_state.es_state

            # Generate levels
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            if config["use_cmaes"]:
                # CMA-ES: ask for candidate latent vectors, decode to levels
                rng, rng_ask, rng_decode = jax.random.split(rng, 3)
                z_population, es_state = cmaes_mgr.ask(rng_ask, es_state)
                new_levels = decode_latent_to_levels_vae(vae_decode_fn, z_population, rng_decode)
            else:
                new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_train_envs"]))

            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), new_levels, env_params)
            # Rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info, agent_positions, hstates),
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
            rng, rng_score = jax.random.split(rng)
            scores = compute_level_scores(rng_score, train_state, new_levels, dones, values, max_returns, advantages)

            # CMA-ES: tell fitness and insert into buffer
            if config["use_cmaes"]:
                # CMA-ES minimizes; negate scores so high-regret = low fitness
                rng, rng_tell = jax.random.split(rng)
                es_state = cmaes_mgr.tell(rng_tell, z_population, -scores, es_state)

                # Periodic reset to prevent stagnation
                should_reset = (train_state.num_dr_updates % config["cmaes_reset_interval"]) == 0
                rng, rng_reset_es = jax.random.split(rng)
                fresh_es_state = cmaes_mgr.initialize(rng_reset_es)
                es_state = jax.tree_util.tree_map(
                    lambda fresh, old: jnp.where(should_reset, fresh, old),
                    fresh_es_state, es_state
                )

            sampler, dr_inserted_inds = level_sampler.insert_batch(sampler, new_levels, scores, {
                "max_return": max_returns,
                "ancestor_id": jnp.full(config["num_train_envs"], -1, dtype=jnp.int32),
            })

            # Stale embedding: mean [hstate(256), action(1)] over first episode
            dr_embeddings = compute_insertion_embeddings(hstates, actions, dones)

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

            # Validity check for generated levels (CMA-ES or random)
            is_valid = jax.vmap(lambda l: l.is_well_formatted())(new_levels)

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": new_levels.wall_map.sum() / config["num_train_envs"],
                "gen/valid_structure_pct": is_valid.mean() * 100,
                "replay_level_inds": jnp.full(config["num_train_envs"], -1, dtype=jnp.int32),
                "mutation_parent_inds": jnp.full(config["num_train_envs"], -1, dtype=jnp.int32),
                "mutation_child_inds": jnp.full(config["num_train_envs"], -1, dtype=jnp.int32),
                "insertion_inds": dr_inserted_inds,  # (num_train_envs,) -1 if not inserted
                "insertion_embeddings": dr_embeddings,  # (num_train_envs, 257)
                "prox_gate/n_blocked": jnp.int32(0),
                "prox_gate/n_llm_parents": jnp.int32(0),
                "prox_gate/mean_dist": jnp.float32(0.0),
            }

            # CMA-ES monitoring metrics
            if config["use_cmaes"]:
                metrics["cmaes/valid_structure_pct"] = is_valid.mean() * 100
                metrics["cmaes/mean_fitness"] = scores.mean()
                metrics["cmaes/mean_episode_length"] = dones.sum(axis=0).mean()
                # Step size (sigma) — tracks exploration vs convergence
                metrics["cmaes/sigma"] = es_state.std
                # Spread of population in latent space (std of z-vectors across candidates)
                metrics["cmaes/pop_spread"] = z_population.std()
                # Mean norm of latent vectors (how far from origin)
                metrics["cmaes/mean_z_norm"] = jnp.linalg.norm(z_population, axis=-1).mean()

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                es_state=es_state,
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
            if config.get("fixed_distribution"):
                # Fixed distribution: sample n_llm from LLM-lineage, rest from organic.
                # n_llm comes from train_state (dynamic, updated by injector).
                # Use full num_train_envs scans with masked sampling, then select.
                n_envs = config["num_train_envs"]
                ancestor_ids_buf = sampler["levels_extra"]["ancestor_id"]
                is_llm_mask = ancestor_ids_buf >= 0
                is_org_mask = ancestor_ids_buf < 0
                has_llm = jnp.any(is_llm_mask)

                def _sample_fixed_dist(sampler_rng):
                    sampler, rng_levels = sampler_rng
                    n_llm = train_state.fixed_dist_n_llm
                    rng_llm, rng_org = jax.random.split(rng_levels)

                    # Sample n_envs from each pool, then slice
                    def _sample_llm(sampler, rng_k):
                        return _sample_masked_level(level_sampler, sampler, rng_k, is_llm_mask)
                    def _sample_org(sampler, rng_k):
                        return _sample_masked_level(level_sampler, sampler, rng_k, is_org_mask)

                    sampler, (llm_inds, llm_levels) = jax.lax.scan(
                        _sample_llm, sampler, jax.random.split(rng_llm, n_envs), length=n_envs)
                    sampler, (org_inds, org_levels) = jax.lax.scan(
                        _sample_org, sampler, jax.random.split(rng_org, n_envs), length=n_envs)

                    # Build index arrays: first n_llm from LLM pool, rest from organic pool
                    idx = jnp.arange(n_envs)
                    use_llm = idx < n_llm
                    org_idx = jnp.clip(idx - n_llm, 0, n_envs - 1)
                    level_inds = jnp.where(use_llm, llm_inds, org_inds[org_idx])
                    def _select_level_leaf(l, o):
                        mask = use_llm.reshape((-1,) + (1,) * (l.ndim - 1))
                        return jnp.where(mask, l, o[org_idx])
                    levels = jax.tree_util.tree_map(_select_level_leaf, llm_levels, org_levels)
                    return sampler, (level_inds, levels)

                def _sample_standard(sampler_rng):
                    sampler, rng_levels = sampler_rng
                    return level_sampler.sample_replay_levels(sampler, rng_levels, n_envs)

                sampler, (level_inds, levels) = jax.lax.cond(
                    has_llm & (train_state.fixed_dist_n_llm > 0),
                    _sample_fixed_dist,
                    _sample_standard,
                    (sampler, rng_levels))
            else:
                sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, config["num_train_envs"])
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params)
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info, agent_positions, hstates),
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
            rng, rng_score = jax.random.split(rng)
            scores = compute_level_scores(rng_score, train_state, levels, dones, values, max_returns, advantages)
            existing_ancestor_ids = sampler["levels_extra"]["ancestor_id"][level_inds]
            sampler = level_sampler.update_batch(sampler, level_inds, scores, {
                "max_return": max_returns,
                "ancestor_id": existing_ancestor_ids,
            })
            
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
                "gen/valid_structure_pct": jnp.float32(0.0),  # no new levels generated
                "replay_level_inds": level_inds,  # (num_train_envs,) for provenance tracking
                "mutation_parent_inds": jnp.full(config["num_train_envs"], -1, dtype=jnp.int32),
                "mutation_child_inds": jnp.full(config["num_train_envs"], -1, dtype=jnp.int32),
                "insertion_inds": level_inds,  # replay slots to update embeddings for
                "insertion_embeddings": compute_insertion_embeddings(hstates, actions, dones),
                "prox_gate/n_blocked": jnp.int32(0),
                "prox_gate/n_llm_parents": jnp.int32(0),
                "prox_gate/mean_dist": jnp.float32(0.0),
            }
            if config["use_cmaes"]:
                metrics["cmaes/valid_structure_pct"] = jnp.float32(0.0)
                metrics["cmaes/mean_fitness"] = jnp.float32(0.0)
                metrics["cmaes/mean_episode_length"] = jnp.float32(0.0)
                metrics["cmaes/sigma"] = jnp.float32(0.0)
                metrics["cmaes/pop_spread"] = jnp.float32(0.0)
                metrics["cmaes/mean_z_norm"] = jnp.float32(0.0)

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                es_state=train_state.es_state,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
                replay_last_level_inds=level_inds,
            )
            return (rng, train_state), metrics

        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                This mutates the previous batch of replay levels and potentially adds them to the level buffer.
                This also updates the policy iff `config["exploratory_grad_updates"]` is True.

                For LLM-lineage parents with proximity gate enabled, generates K candidate
                mutations per parent (--llm_mutation_retries), scores them all, and picks the
                best one that passes the gate. Organic parents get a single mutation as before.
            """
            sampler = train_state.sampler
            parent_levels = train_state.replay_last_level_batch
            parent_ancestor_ids = sampler["levels_extra"]["ancestor_id"][train_state.replay_last_level_inds]
            is_llm_parent = parent_ancestor_ids >= 0
            n_llm_parents = is_llm_parent.sum()
            n_envs = config["num_train_envs"]
            K = config.get("llm_mutation_retries", 1)
            use_retries = config.get("proximity_gate_ratio", 0) > 0 and K > 1

            if use_retries:
                # Generate K mutations per parent for ALL slots (fixed shape for JIT).
                # For organic parents we'll just use candidate 0.
                rng, rng_mutate, rng_reset = jax.random.split(rng, 3)

                # Tile parents K times: (K * n_envs,)
                tiled_parents = jax.tree_util.tree_map(
                    lambda x: jnp.tile(x, (K,) + (1,) * (x.ndim - 1)), parent_levels)
                tiled_rngs = jax.random.split(rng_mutate, K * n_envs)
                all_children = jax.vmap(mutate_level, (0, 0, None))(
                    tiled_rngs, tiled_parents, config["num_edits"])

                # Rollout on all K * n_envs candidates
                init_obs_all, init_state_all = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                    jax.random.split(rng_reset, K * n_envs), all_children, env_params)
                (
                    (rng, train_state, _, _, _, last_value_all),
                    (obs_all, actions_all, rewards_all, dones_all, log_probs_all,
                     values_all, info_all, positions_all, hstates_all),
                ) = sample_trajectories_rnn(
                    rng, env, env_params, train_state,
                    ActorCritic.initialize_carry((K * n_envs,)),
                    init_obs_all, init_state_all, K * n_envs, config["num_steps"],
                )

                # Proxy score for candidate selection: use max_return as a quick
                # learnability signal (solved=1, unsolved=0). Full SFL scoring
                # happens after selecting the best candidate per slot.
                max_returns_all = compute_max_returns(dones_all, rewards_all)
                proxy_scores_all = (max_returns_all > 0).astype(jnp.float32)
                embs_all = compute_insertion_embeddings(hstates_all, actions_all, dones_all)

                # Reshape to (K, n_envs, ...)
                proxy_scores_K = proxy_scores_all.reshape(K, n_envs)
                embs_K = embs_all.reshape(K, n_envs, 257)
                max_returns_K = max_returns_all.reshape(K, n_envs)

                # Proximity gate on all K candidates
                max_anc = config["max_llm_ancestors"]
                safe_idx = jnp.clip(parent_ancestor_ids, 0, max_anc - 1)
                ancestor_embs = train_state.seed_embeddings[safe_idx]       # (n_envs, 257)
                thresholds = train_state.proximity_thresholds[safe_idx]     # (n_envs,)
                dists_K = jnp.linalg.norm(
                    embs_K - ancestor_embs[None, :, :], axis=-1)            # (K, n_envs)
                gate_pass_K = ~is_llm_parent[None, :] | (dists_K < thresholds[None, :])  # (K, n_envs)

                # For each env slot, pick the best passing candidate (highest proxy score among passing)
                gated_scores_K = jnp.where(gate_pass_K, proxy_scores_K, -jnp.inf)  # (K, n_envs)
                best_k = jnp.argmax(gated_scores_K, axis=0)                   # (n_envs,)

                # For organic parents, just use candidate 0
                best_k = jnp.where(is_llm_parent, best_k, 0)

                # Gather best candidate per env
                env_idx = jnp.arange(n_envs)
                flat_idx = best_k * n_envs + env_idx  # index into (K * n_envs,)
                child_levels = jax.tree_util.tree_map(lambda x: x[flat_idx], all_children)
                max_returns = max_returns_all[flat_idx]
                mutation_embeddings = embs_all[flat_idx]

                # Proper SFL scoring on the selected n_envs levels
                rng, rng_score = jax.random.split(rng)
                scores = compute_level_scores(
                    rng_score, train_state, child_levels,
                    dones_all[:, :n_envs], values_all[:, :n_envs],  # dummy, only used for non-SFL
                    max_returns,
                    compute_gae(config["gamma"], config["gae_lambda"],
                                last_value_all[:n_envs], values_all[:, :n_envs],
                                rewards_all[:, :n_envs], dones_all[:, :n_envs])[1])

                # Final gate check on selected candidates (organic always pass)
                dists_to_seed = dists_K[best_k, env_idx]
                final_gate_pass = ~is_llm_parent | (dists_to_seed < thresholds)
                scores = jnp.where(final_gate_pass, scores, -jnp.inf)

                prox_n_blocked = jnp.sum(~final_gate_pass & is_llm_parent)
                prox_mean_dist = jnp.where(
                    n_llm_parents > 0,
                    jnp.where(is_llm_parent, dists_to_seed, 0.0).sum() / jnp.maximum(n_llm_parents, 1),
                    0.0)

                # Use candidate 0's trajectory data for PPO update (all envs trained on first candidate)
                obs = jax.tree_util.tree_map(lambda x: x[:, :n_envs], obs_all)
                actions = actions_all[:, :n_envs]
                rewards = rewards_all[:, :n_envs]
                dones = dones_all[:, :n_envs]
                log_probs = log_probs_all[:, :n_envs]
                values = values_all[:, :n_envs]
                hstates = hstates_all[:, :n_envs]
                advantages, targets = compute_gae(
                    config["gamma"], config["gae_lambda"],
                    last_value_all[:n_envs], values, rewards, dones)

            else:
                # Standard single-mutation path
                rng, rng_mutate, rng_reset = jax.random.split(rng, 3)
                child_levels = jax.vmap(mutate_level, (0, 0, None))(
                    jax.random.split(rng_mutate, n_envs), parent_levels, config["num_edits"])
                init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                    jax.random.split(rng_reset, n_envs), child_levels, env_params)
                (
                    (rng, train_state, hstate, last_obs, last_env_state, last_value),
                    (obs, actions, rewards, dones, log_probs, values, info, agent_positions, hstates),
                ) = sample_trajectories_rnn(
                    rng, env, env_params, train_state,
                    ActorCritic.initialize_carry((n_envs,)),
                    init_obs, init_env_state, n_envs, config["num_steps"],
                )
                advantages, targets = compute_gae(
                    config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
                max_returns = compute_max_returns(dones, rewards)
                rng, rng_score = jax.random.split(rng)
                scores = compute_level_scores(
                    rng_score, train_state, child_levels, dones, values, max_returns, advantages)
                mutation_embeddings = compute_insertion_embeddings(hstates, actions, dones)

                # Proximity gate (single candidate)
                prox_n_blocked = jnp.int32(0)
                prox_mean_dist = jnp.float32(0.0)
                if config.get("proximity_gate_ratio", 0) > 0:
                    max_anc = config["max_llm_ancestors"]
                    safe_idx = jnp.clip(parent_ancestor_ids, 0, max_anc - 1)
                    ancestor_embs = train_state.seed_embeddings[safe_idx]
                    thresholds = train_state.proximity_thresholds[safe_idx]
                    dists_to_seed = jnp.linalg.norm(mutation_embeddings - ancestor_embs, axis=-1)
                    gate_pass = ~is_llm_parent | (dists_to_seed < thresholds)
                    scores = jnp.where(gate_pass, scores, -jnp.inf)
                    prox_n_blocked = jnp.sum(~gate_pass & is_llm_parent)
                    prox_mean_dist = jnp.where(
                        n_llm_parents > 0,
                        jnp.where(is_llm_parent, dists_to_seed, 0.0).sum() / jnp.maximum(n_llm_parents, 1),
                        0.0)

            sampler, mutation_inserted_inds = level_sampler.insert_batch(sampler, child_levels, scores, {
                "max_return": max_returns,
                "ancestor_id": parent_ancestor_ids,
            })

            # Update: train_state only modified if exploratory_grad_updates is on
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((n_envs,)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                n_envs,
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )

            # Validity check for mutated levels
            is_valid_mut = jax.vmap(lambda l: l.is_well_formatted())(child_levels)

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": child_levels.wall_map.sum() / n_envs,
                "gen/valid_structure_pct": is_valid_mut.mean() * 100,
                "replay_level_inds": jnp.full(n_envs, -1, dtype=jnp.int32),
                # Lineage tracking: parent buffer indices and where their children landed
                "mutation_parent_inds": train_state.replay_last_level_inds,  # (n_envs,)
                "mutation_child_inds": mutation_inserted_inds,  # (n_envs,) -1 if not inserted
                "insertion_inds": mutation_inserted_inds,
                "insertion_embeddings": mutation_embeddings,  # (n_envs, 257)
                "prox_gate/n_blocked": prox_n_blocked,
                "prox_gate/n_llm_parents": n_llm_parents,
                "prox_gate/mean_dist": prox_mean_dist,
            }
            if config["use_cmaes"]:
                metrics["cmaes/valid_structure_pct"] = jnp.float32(0.0)
                metrics["cmaes/mean_fitness"] = jnp.float32(0.0)
                metrics["cmaes/mean_episode_length"] = jnp.float32(0.0)
                metrics["cmaes/sigma"] = jnp.float32(0.0)
                metrics["cmaes/pop_spread"] = jnp.float32(0.0)
                metrics["cmaes/mean_z_norm"] = jnp.float32(0.0)

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                es_state=train_state.es_state,
                num_mutation_updates=train_state.num_mutation_updates + 1,
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

    # --- Resume agent params from checkpoint ---
    if config.get("resume_checkpoint_dir"):
        resume_dir = config["resume_checkpoint_dir"]
        print(f"[Resume] Loading agent params from {resume_dir}...")
        from cross_evaluate import load_agent as _load_agent
        _resume_ts, _, _, _ = _load_agent(resume_dir)
        if _resume_ts is not None:
            train_state = train_state.replace(params=_resume_ts.params)
            print(f"[Resume] Restored agent params from {resume_dir}")
        else:
            print(f"[Resume] WARNING: Failed to load params from {resume_dir}, starting fresh")

    # --- Provenance tracking (numpy, outside JAX) ---
    # origins: 0=organic, 1=LLM seed, 2=LLM mutation descendant
    # origin_ids: uint64 hash per level for matching across dumps
    # ancestor_ids: int32 per slot — which LLM seed (0-indexed) this slot descends from
    #   -1 = organic (no LLM ancestor), 0 = 1st LLM seed, 1 = 2nd, etc.
    _buffer_origins = None      # np.ndarray (capacity,) int32
    _buffer_origin_ids = None   # np.ndarray (capacity,) uint64
    _buffer_ancestor_ids = None # np.ndarray (capacity,) int32  — LLM seed index (-1=organic)
    _prev_buffer_hashes = None  # np.ndarray (capacity,) uint64 — snapshot before each eval step
    # Stale embeddings: mean [hstate(256), action(1)] at insertion time
    # Allocated unconditionally so buffer dumps always contain embeddings.
    capacity = train_state.sampler["scores"].shape[0]
    _buffer_embeddings = np.zeros((capacity, 257), dtype=np.float32)

    # --- Preload buffer from merged .npz ---
    if config.get("preload_buffer_npz"):
        import hashlib as _hl
        preload_path = config["preload_buffer_npz"]
        print(f"[Preload] Loading buffer from {preload_path}...")
        preload = np.load(preload_path, allow_pickle=True)
        pre_tokens = preload["tokens"]
        pre_scores = preload["scores"]
        pre_size = int(preload["size"])

        # Reconstruct Level objects from tokens
        pre_levels_list = [tokens_to_level(jnp.array(pre_tokens[i])) for i in range(pre_size)]
        pre_levels_batched = Level.stack(pre_levels_list)

        # Overwrite sampler fields
        sampler = train_state.sampler
        capacity = sampler["scores"].shape[0]
        new_levels = jax.tree_util.tree_map(
            lambda orig, loaded: orig.at[:pre_size].set(loaded),
            sampler["levels"], pre_levels_batched,
        )
        new_scores = sampler["scores"].at[:pre_size].set(jnp.array(pre_scores[:pre_size], dtype=jnp.float32))
        if "timestamps" in preload:
            new_timestamps = sampler["timestamps"].at[:pre_size].set(
                jnp.array(preload["timestamps"][:pre_size], dtype=jnp.int32))
        else:
            new_timestamps = sampler["timestamps"]

        new_sampler = {**sampler, "levels": new_levels, "scores": new_scores,
                       "timestamps": new_timestamps, "size": pre_size}
        train_state = train_state.replace(sampler=new_sampler)

        # Load provenance arrays
        _buffer_origins = np.zeros(capacity, dtype=np.int32)
        _buffer_origin_ids = np.zeros(capacity, dtype=np.uint64)
        _buffer_ancestor_ids = np.full(capacity, -1, dtype=np.int32)
        if "origins" in preload:
            n = min(len(preload["origins"]), capacity)
            _buffer_origins[:n] = preload["origins"][:n]
        if "origin_ids" in preload:
            n = min(len(preload["origin_ids"]), capacity)
            _buffer_origin_ids[:n] = preload["origin_ids"][:n]
        if "ancestor_ids" in preload:
            n = min(len(preload["ancestor_ids"]), capacity)
            _buffer_ancestor_ids[:n] = preload["ancestor_ids"][:n]
        else:
            # Backward compat: assign unique ancestor IDs to each injected level
            # based on its unique origin_id hash
            injected_mask = _buffer_origins[:pre_size] > 0
            if injected_mask.any():
                unique_hashes = np.unique(_buffer_origin_ids[:pre_size][injected_mask])
                hash_to_id = {h: i for i, h in enumerate(unique_hashes)}
                for slot in range(pre_size):
                    if _buffer_origins[slot] > 0:
                        _buffer_ancestor_ids[slot] = hash_to_id[_buffer_origin_ids[slot]]
                print(f"[Preload] Inferred {len(unique_hashes)} unique LLM ancestor IDs from origin_ids")
        if "embeddings" in preload:
            n = min(len(preload["embeddings"]), capacity)
            _buffer_embeddings[:n] = preload["embeddings"][:n]

        n_injected = int((_buffer_origins[:pre_size] > 0).sum())
        unique_ancestors = len(set(_buffer_ancestor_ids[:pre_size][_buffer_origins[:pre_size] > 0].tolist()))
        print(f"[Preload] Loaded {pre_size} levels ({n_injected} injected from "
              f"{unique_ancestors} LLM ancestors, {pre_size - n_injected} organic)")

        # Write ancestor_ids into levels_extra (JIT-visible)
        new_ancestor_id_arr = train_state.sampler["levels_extra"]["ancestor_id"].at[:pre_size].set(
            jnp.array(_buffer_ancestor_ids[:pre_size], dtype=jnp.int32))
        new_levels_extra = {**train_state.sampler["levels_extra"], "ancestor_id": new_ancestor_id_arr}
        new_sampler = {**train_state.sampler, "levels_extra": new_levels_extra}
        train_state = train_state.replace(sampler=new_sampler)

        # Compute seed embeddings and proximity thresholds from preloaded buffer
        if config.get("proximity_gate_ratio", 0) > 0 or config.get("fixed_distribution"):
            unique_anc_ids = sorted(set(
                int(a) for a in _buffer_ancestor_ids[:pre_size] if a >= 0))
            n_seeds = len(unique_anc_ids)
            max_anc = config.get("max_llm_ancestors", 256)

            if n_seeds > 0:
                print(f"[Preload] Setting up seed embeddings for {n_seeds} ancestors...")
                seed_embs = np.zeros((max_anc, 257), dtype=np.float32)
                prox_thresholds = np.full(max_anc, np.inf, dtype=np.float32)

                if "embeddings" in preload:
                    # Use stored embeddings
                    org_mask = _buffer_ancestor_ids[:pre_size] == -1
                    organic_centroid = _buffer_embeddings[:pre_size][org_mask].mean(axis=0) if org_mask.any() else np.zeros(257)

                    ratio = config.get("proximity_gate_ratio", 0.5)
                    for anc_id in unique_anc_ids:
                        if anc_id >= max_anc:
                            print(f"  WARNING: ancestor_id {anc_id} >= max_llm_ancestors {max_anc}, skipping")
                            continue
                        seed_slots = np.where(
                            (_buffer_ancestor_ids[:pre_size] == anc_id) &
                            (_buffer_origins[:pre_size] == 1)
                        )[0]
                        if len(seed_slots) == 0:
                            seed_slots = np.where(_buffer_ancestor_ids[:pre_size] == anc_id)[0]
                        if len(seed_slots) > 0:
                            seed_emb = _buffer_embeddings[seed_slots].mean(axis=0)
                            dist_to_organic = float(np.linalg.norm(seed_emb - organic_centroid))
                            seed_embs[anc_id] = seed_emb
                            prox_thresholds[anc_id] = ratio * dist_to_organic
                            print(f"  Ancestor {anc_id}: dist={dist_to_organic:.4f}, thr={prox_thresholds[anc_id]:.4f}")
                else:
                    # No stored embeddings — compute by rolling out agent on seed levels
                    print(f"[Preload] No embeddings in buffer, computing via agent rollouts...")
                    # Collect seed level indices (one representative per ancestor)
                    seed_level_list = []
                    seed_anc_list = []
                    for anc_id in unique_anc_ids:
                        if anc_id >= max_anc:
                            continue
                        slots = np.where(
                            (_buffer_ancestor_ids[:pre_size] == anc_id) &
                            (_buffer_origins[:pre_size] == 1)
                        )[0]
                        if len(slots) == 0:
                            slots = np.where(_buffer_ancestor_ids[:pre_size] == anc_id)[0]
                        if len(slots) > 0:
                            seed_level_list.append(pre_levels_list[slots[0]])
                            seed_anc_list.append(anc_id)

                    if seed_level_list:
                        n_batch = len(seed_level_list)
                        seed_levels_batched = Level.stack(seed_level_list)
                        rng_emb = jax.random.PRNGKey(42)
                        rng_emb, rng_reset, rng_rollout = jax.random.split(rng_emb, 3)
                        init_obs_s, init_state_s = jax.vmap(
                            env.reset_to_level, in_axes=(0, 0, None)
                        )(jax.random.split(rng_reset, n_batch), seed_levels_batched, env_params)
                        (_, _, _, _, _, _), traj_s = sample_trajectories_rnn(
                            rng_rollout, env, env_params, train_state,
                            ActorCritic.initialize_carry((n_batch,)),
                            init_obs_s, init_state_s, n_batch,
                            config["num_steps"],
                        )
                        _, actions_s, _, dones_s, _, _, _, _, hstates_s = traj_s
                        seed_embeddings_computed = np.asarray(
                            compute_insertion_embeddings(hstates_s, actions_s, dones_s))

                        # Also compute organic centroid by rolling out on a sample of organic levels
                        org_slots = np.where(_buffer_ancestor_ids[:pre_size] == -1)[0]
                        n_org_sample = min(64, len(org_slots))
                        if n_org_sample > 0:
                            org_sample_idx = org_slots[np.random.RandomState(42).choice(len(org_slots), n_org_sample, replace=False)]
                            org_levels_list = [pre_levels_list[i] for i in org_sample_idx]
                            org_levels_batched = Level.stack(org_levels_list)
                            rng_emb, rng_org_reset, rng_org_roll = jax.random.split(rng_emb, 3)
                            init_obs_o, init_state_o = jax.vmap(
                                env.reset_to_level, in_axes=(0, 0, None)
                            )(jax.random.split(rng_org_reset, n_org_sample), org_levels_batched, env_params)
                            (_, _, _, _, _, _), traj_o = sample_trajectories_rnn(
                                rng_org_roll, env, env_params, train_state,
                                ActorCritic.initialize_carry((n_org_sample,)),
                                init_obs_o, init_state_o, n_org_sample,
                                config["num_steps"],
                            )
                            _, actions_o, _, dones_o, _, _, _, _, hstates_o = traj_o
                            org_embeddings = np.asarray(
                                compute_insertion_embeddings(hstates_o, actions_o, dones_o))
                            organic_centroid = org_embeddings.mean(axis=0)
                        else:
                            organic_centroid = np.zeros(257, dtype=np.float32)

                        ratio = config.get("proximity_gate_ratio", 0.5)
                        for i, anc_id in enumerate(seed_anc_list):
                            seed_emb = seed_embeddings_computed[i]
                            dist_to_organic = float(np.linalg.norm(seed_emb - organic_centroid))
                            seed_embs[anc_id] = seed_emb
                            prox_thresholds[anc_id] = ratio * dist_to_organic
                            print(f"  Ancestor {anc_id}: dist={dist_to_organic:.4f}, thr={prox_thresholds[anc_id]:.4f}")

                train_state = train_state.replace(
                    seed_embeddings=jnp.array(seed_embs),
                    proximity_thresholds=jnp.array(prox_thresholds),
                    n_seeds=jnp.int32(n_seeds),
                )
                print(f"[Preload] Registered {n_seeds} seed embeddings for proximity gate")

            # Fixed distribution: compute target LLM count for replay sampling
            if config.get("fixed_distribution") and n_injected > 0:
                injection_pct = n_injected / pre_size
                n_llm = max(1, round(injection_pct * config["num_train_envs"]))
                train_state = train_state.replace(
                    fixed_dist_n_llm=jnp.int32(n_llm))
                print(f"[Preload] Fixed distribution: {injection_pct:.1%} injection -> "
                      f"{n_llm}/{config['num_train_envs']} LLM replay slots")

    # Save seed levels for periodic embedding recomputation
    _prox_gate_seed_levels = None  # List[(anc_id, Level)] or None
    _prox_gate_seed_batched = None  # Batched Level or None
    _prox_gate_seed_anc_ids = None  # list of ancestor IDs matching batch order

    if config.get("proximity_gate_ratio", 0) > 0 and config.get("preload_buffer_npz"):
        # Reconstruct seed level list from preloaded data
        _prox_gate_seed_anc_ids = []
        _seed_level_list = []
        max_anc = config.get("max_llm_ancestors", 256)
        for anc_id in sorted(set(int(a) for a in _buffer_ancestor_ids[:int(train_state.sampler["size"])] if a >= 0)):
            if anc_id >= max_anc:
                continue
            slots = np.where(
                (_buffer_ancestor_ids[:int(train_state.sampler["size"])] == anc_id) &
                (_buffer_origins[:int(train_state.sampler["size"])] == 1)
            )[0]
            if len(slots) == 0:
                slots = np.where(_buffer_ancestor_ids[:int(train_state.sampler["size"])] == anc_id)[0]
            if len(slots) > 0:
                _seed_level_list.append(pre_levels_list[slots[0]])
                _prox_gate_seed_anc_ids.append(anc_id)
        if _seed_level_list:
            _prox_gate_seed_batched = Level.stack(_seed_level_list)
            print(f"[ProxGate] Saved {len(_seed_level_list)} seed levels for periodic recomputation")

    def _recompute_seed_embeddings(runner_state):
        """Recompute seed embeddings and proximity thresholds using the current agent.

        Called at each eval step to keep embeddings fresh as the agent evolves.
        """
        if _prox_gate_seed_batched is None:
            return runner_state

        rng, train_state = runner_state
        n_batch = len(_prox_gate_seed_anc_ids)
        max_anc = config.get("max_llm_ancestors", 256)
        ratio = config.get("proximity_gate_ratio", 0.5)

        # Roll out current agent on seed levels (read-only, don't propagate modified train_state)
        rng, rng_reset, rng_rollout = jax.random.split(rng, 3)
        init_obs_s, init_state_s = jax.vmap(
            env.reset_to_level, in_axes=(0, 0, None)
        )(jax.random.split(rng_reset, n_batch), _prox_gate_seed_batched, env_params)
        _, traj_s = sample_trajectories_rnn(
            rng_rollout, env, env_params, train_state,
            ActorCritic.initialize_carry((n_batch,)),
            init_obs_s, init_state_s, n_batch, config["num_steps"],
        )
        # Unpack only what we need (discard modified train_state from rollout)
        _, actions_s, _, dones_s, _, _, _, _, hstates_s = traj_s
        seed_embs_new = np.asarray(compute_insertion_embeddings(hstates_s, actions_s, dones_s))

        # Organic centroid from current buffer embeddings
        size = int(train_state.sampler["size"])
        if _buffer_embeddings is not None:
            org_mask = _buffer_ancestor_ids[:size] == -1
            if org_mask.any():
                organic_centroid = _buffer_embeddings[:size][org_mask].mean(axis=0)
            else:
                organic_centroid = np.zeros(257, dtype=np.float32)
        else:
            organic_centroid = np.zeros(257, dtype=np.float32)

        # Update thresholds
        seed_embs_arr = np.array(train_state.seed_embeddings, copy=True)
        prox_thr_arr = np.array(train_state.proximity_thresholds, copy=True)
        for i, anc_id in enumerate(_prox_gate_seed_anc_ids):
            seed_embs_arr[anc_id] = seed_embs_new[i]
            dist = float(np.linalg.norm(seed_embs_new[i] - organic_centroid))
            prox_thr_arr[anc_id] = ratio * dist

        train_state = train_state.replace(
            seed_embeddings=jnp.array(seed_embs_arr),
            proximity_thresholds=jnp.array(prox_thr_arr),
        )
        return (rng, train_state)

    runner_state = (rng_train, train_state)

    def dump_buffer(train_state, update_num):
        """Save PLR buffer as .npy (VAE token format) + .npz (full metadata).

        If provenance tracking is active, saves origins array with exact
        lineage: 0=organic, 1=original LLM seed, 2=mutation descendant.
        Ancestor IDs track which specific LLM seed each descendant traces to.
        """
        nonlocal _buffer_origins, _buffer_origin_ids, _buffer_ancestor_ids

        sampler = train_state.sampler
        size = int(sampler["size"])
        if size == 0:
            return

        buffer_levels = jax.tree_util.tree_map(lambda x: x[:size], sampler["levels"])
        _l2t = level_to_tokens_vae if _need_vae else level_to_tokens
        tokens = jax.vmap(_l2t)(buffer_levels)
        tokens_np = np.asarray(tokens)

        dump_data = {
            "tokens": tokens_np,
            "scores": np.asarray(sampler["scores"][:size]),
            "timestamps": np.asarray(sampler["timestamps"][:size]),
            "size": size,
            "update_num": update_num,
        }

        # --- Provenance: use live-tracked _buffer_origins ---
        if _buffer_origins is not None:
            current_origins = _buffer_origins[:size].copy()
            current_origin_ids = _buffer_origin_ids[:size].copy() if _buffer_origin_ids is not None else np.zeros(size, dtype=np.uint64)

            n_original = int((current_origins == 1).sum())
            n_descendants = int((current_origins == 2).sum())
            n_lineage = n_original + n_descendants

            dump_data["origins"] = current_origins
            dump_data["origin_ids"] = current_origin_ids
            dump_data["n_injected_original"] = n_original
            dump_data["n_injected_descendants"] = n_descendants
            dump_data["n_injected_lineage"] = n_lineage

            # Ancestor IDs: which LLM seed each descendant traces back to
            if _buffer_ancestor_ids is not None:
                current_ancestor_ids = _buffer_ancestor_ids[:size].copy()
                dump_data["ancestor_ids"] = current_ancestor_ids
                # Per-ancestor breakdown
                lineage_ancestors = current_ancestor_ids[current_origins > 0]
                if len(lineage_ancestors) > 0:
                    unique, counts = np.unique(lineage_ancestors[lineage_ancestors >= 0], return_counts=True)
                    ancestor_str = ", ".join(f"LLM#{a}:{c}" for a, c in zip(unique, counts))
                else:
                    ancestor_str = "none"
            else:
                ancestor_str = "n/a"

            print(f"  [Provenance] {n_original} original + {n_descendants} descendants = "
                  f"{n_lineage} lineage levels in buffer | ancestors: {ancestor_str}")

        # Stale embeddings: 257D [hstate(256), action(1)] at insertion time
        if _buffer_embeddings is not None:
            dump_data["embeddings"] = _buffer_embeddings[:size].copy()

        if config.get("output_dir"):
            dump_dir = os.path.join(config["output_dir"], "buffer_dumps")
        else:
            dump_dir = os.path.join("/tmp", "buffer_dumps", f"{config['run_name']}", str(config["seed"]))
        os.makedirs(dump_dir, exist_ok=True)
        tag = f"_{update_num}" if update_num > 0 else "_final"
        tokens_path = os.path.join(dump_dir, f"buffer_tokens{tag}.npy")
        dump_path = os.path.join(dump_dir, f"buffer_dump{tag}.npz")
        np.save(tokens_path, np.asarray(tokens))
        np.savez_compressed(dump_path, **dump_data)
        print(f"[Buffer dump @ {update_num}] {size} levels -> {dump_path}")

        if config.get("gcs_bucket"):
            gcs_base = f"{config['gcs_prefix']}/buffer_dumps/{config['run_name']}/{config['seed']}"
            _upload_to_gcs(tokens_path, config["gcs_bucket"], f"{gcs_base}/buffer_tokens{tag}.npy")
            _upload_to_gcs(dump_path, config["gcs_bucket"], f"{gcs_base}/buffer_dump{tag}.npz")

    def _compute_buffer_hashes(train_state):
        """Compute MD5 hashes of all buffer level tokens (numpy, outside JAX)."""
        import hashlib as _hl
        sampler = train_state.sampler
        size = int(sampler["size"])
        buffer_levels = jax.tree_util.tree_map(lambda x: x[:size], sampler["levels"])
        _l2t = level_to_tokens_vae if _need_vae else level_to_tokens
        tokens = np.asarray(jax.vmap(_l2t)(buffer_levels))
        hashes = np.zeros(sampler["scores"].shape[0], dtype=np.uint64)
        for i in range(size):
            hashes[i] = int(_hl.md5(tokens[i].tobytes()).hexdigest()[:16], 16)
        return hashes

    def _update_lineage(train_state, metrics):
        """Exact lineage tracking for mutation descendants of injected levels.

        Uses mutation_parent_inds and mutation_child_inds from the JIT scan
        to trace exact parent→child relationships. For each mutation step:
        - mutation_parent_inds[i] = buffer index of parent level
        - mutation_child_inds[i] = buffer index where child was inserted (-1 if not)

        If parent had origin > 0 (injected/descendant), mark child as origin=2.
        Also clears origin for any slot whose level was replaced by a non-descendant.
        """
        nonlocal _buffer_origins, _buffer_origin_ids, _buffer_ancestor_ids, _prev_buffer_hashes
        if _buffer_origins is None or _prev_buffer_hashes is None:
            return

        new_hashes = _compute_buffer_hashes(train_state)
        size = int(train_state.sampler["size"])

        # Save pre-update origins and ancestors (from BEFORE this window)
        old_origins = _buffer_origins.copy()
        old_ancestors = _buffer_ancestor_ids.copy()

        # Detect changed slots (level was evicted and replaced)
        changed_mask = (new_hashes[:size] != _prev_buffer_hashes[:size])
        changed_slots = set(np.where(changed_mask)[0].tolist())

        # First: apply exact lineage from mutation parent→child pairs.
        # We process these BEFORE clearing changed slots, because a changed
        # slot might be a mutation child that should inherit lineage.
        # metrics arrays have shape (eval_freq, num_train_envs)
        parent_inds = np.array(metrics["mutation_parent_inds"])
        child_inds = np.array(metrics["mutation_child_inds"])

        # Track which changed slots were assigned lineage via mutation
        lineage_assigned = set()

        n_new_descendants = 0
        n_mutations_from_lineage = 0
        n_total_mutations = 0

        for step in range(parent_inds.shape[0]):
            for env in range(parent_inds.shape[1]):
                p_idx = int(parent_inds[step, env])
                c_idx = int(child_inds[step, env])
                if p_idx < 0 or c_idx < 0:
                    continue  # not a mutation step, or child wasn't inserted
                n_total_mutations += 1
                if old_origins[p_idx] > 0:
                    n_mutations_from_lineage += 1
                    n_new_descendants += 1
                    old_origins[c_idx] = 2
                    old_ancestors[c_idx] = old_ancestors[p_idx]  # inherit ancestor ID
                    lineage_assigned.add(c_idx)
                else:
                    # Parent was organic; if child landed in a slot that
                    # previously held lineage, that lineage is now gone
                    old_origins[c_idx] = 0
                    old_ancestors[c_idx] = -1

        # Now set final _buffer_origins and _buffer_ancestor_ids:
        # - Changed slots with lineage assigned → origin=2, ancestor inherited
        # - Changed slots that were mutation targets (child inserted) but not lineage → origin=0
        # - Changed slots that were NOT mutation targets → keep existing origin
        #   (their hash changed due to score/timestamp updates, not level replacement)
        mutation_target_slots = set(int(child_inds[s, e])
                                     for s in range(parent_inds.shape[0])
                                     for e in range(parent_inds.shape[1])
                                     if int(child_inds[s, e]) >= 0)
        for slot in changed_slots:
            if slot in lineage_assigned:
                _buffer_origins[slot] = 2
                _buffer_ancestor_ids[slot] = old_ancestors[slot]
            elif slot in mutation_target_slots:
                # Slot received a new level (organic mutation) — clear lineage
                _buffer_origins[slot] = 0
                _buffer_ancestor_ids[slot] = -1
            # else: hash changed but slot wasn't a mutation target — keep existing origin

        # Update hashes
        _buffer_origin_ids[:size] = new_hashes[:size]
        _prev_buffer_hashes = new_hashes

        n_original = int((_buffer_origins[:size] == 1).sum())
        n_descendants = int((_buffer_origins[:size] == 2).sum())
        n_total_lineage = n_original + n_descendants

        # Per-ancestor breakdown
        lineage_mask = _buffer_origins[:size] > 0
        lineage_ancestors = _buffer_ancestor_ids[:size][lineage_mask]
        if len(lineage_ancestors) > 0 and (lineage_ancestors >= 0).any():
            unique_a, counts_a = np.unique(lineage_ancestors[lineage_ancestors >= 0], return_counts=True)
            ancestor_breakdown = ", ".join(f"#{a}:{c}" for a, c in zip(unique_a, counts_a))
        else:
            ancestor_breakdown = "none"

        print(f"  [Lineage] {n_original} original + {n_descendants} descendants = "
              f"{n_total_lineage} lineage ({len(changed_slots)} slots changed, "
              f"{n_new_descendants} new descendants from "
              f"{n_mutations_from_lineage}/{n_total_mutations} lineage mutations)")
        print(f"  [Lineage] Per-ancestor: {ancestor_breakdown}")

    def _update_embeddings(metrics):
        """Store stale 257D embeddings for levels inserted during this eval window."""
        nonlocal _buffer_embeddings
        if _buffer_embeddings is None:
            return

        ins_inds = np.array(metrics["insertion_inds"])      # (eval_freq, num_train_envs)
        ins_embs = np.array(metrics["insertion_embeddings"]) # (eval_freq, num_train_envs, 257)

        n_stored = 0
        for step in range(ins_inds.shape[0]):
            for env in range(ins_inds.shape[1]):
                slot = int(ins_inds[step, env])
                if slot >= 0:
                    _buffer_embeddings[slot] = ins_embs[step, env]
                    n_stored += 1

    # LLM injection setup
    llm_config = LLMInjectionConfig.from_config_dict(config)
    llm_injector = None
    if llm_config.enabled:
        agent_evaluator = None
        if llm_config.gate_enabled:
            from llm.agent_evaluator import AgentEvaluator
            # Construct with live train_state params — will be refreshed each injection event
            rng_init, train_state_init = runner_state
            agent_evaluator = AgentEvaluator(
                apply_fn=train_state_init.apply_fn,
                params=train_state_init.params,
                env_params=env_params,
            )
        # Build level cache dir: results/<run_name>/llm_levels/<seed>
        # Provides per-run audit trail for accepted LLM levels (EXPT-02)
        llm_cache_dir = os.path.join("results", config["run_name"], "llm_levels", str(config["seed"]))
        level_cache = LevelCache(llm_cache_dir)
        print(f"[LLM] Level cache: {llm_cache_dir}")

        # Initialize provenance tracking arrays if not already set (online-only runs)
        if _buffer_origins is None:
            capacity = train_state.sampler["scores"].shape[0]
            _buffer_origins = np.zeros(capacity, dtype=np.int32)
            _buffer_origin_ids = np.zeros(capacity, dtype=np.uint64)
            _buffer_ancestor_ids = np.full(capacity, -1, dtype=np.int32)
            print(f"[LLM] Initialized provenance tracking arrays (capacity={capacity})")

        llm_injector = LLMInjectionManager(
            config=llm_config,
            level_sampler=level_sampler,
            eval_freq=config["eval_freq"],
            agent_evaluator=agent_evaluator,
            level_cache=level_cache,
            training_config=config,
        )
        # Pass numpy provenance arrays so injector can update them
        llm_injector._buffer_origins = _buffer_origins
        llm_injector._buffer_ancestor_ids = _buffer_ancestor_ids

        # Sync next_ancestor_id with preloaded ancestors to avoid ID collisions
        if _buffer_ancestor_ids is not None:
            max_existing = int(_buffer_ancestor_ids.max())
            if max_existing >= 0:
                llm_injector.next_ancestor_id = max_existing + 1
                print(f"[LLM] Synced next_ancestor_id to {llm_injector.next_ancestor_id} "
                      f"(from preloaded buffer)")
        gate_status = "gate=ON" if llm_config.gate_enabled else "gate=OFF"
        print(f"[LLM] Injection enabled: interval={llm_config.injection_interval}, "
              f"n_raw={llm_config.n_raw}, start_step={llm_config.inject_start_step}, {gate_status}")
        # Print injection schedule
        eval_freq = config["eval_freq"]
        num_updates = config["num_updates"]
        inject_steps = []
        for es in range(num_updates // eval_freq):
            cs = (es + 1) * eval_freq
            if cs >= llm_config.inject_start_step and cs % llm_config.injection_interval == 0:
                inject_steps.append(cs)
        print(f"[LLM] Injection schedule ({len(inject_steps)} events): {inject_steps}")

    # --- Shadow comparison: interpolation vs ACCEL mutations (logging only) ---
    def _slerp_batch(z1, z2, alpha):
        """Spherical linear interpolation preserving norm through direction interpolation."""
        norm1 = jnp.linalg.norm(z1, axis=-1, keepdims=True)
        norm2 = jnp.linalg.norm(z2, axis=-1, keepdims=True)
        z1_n = z1 / jnp.maximum(norm1, 1e-8)
        z2_n = z2 / jnp.maximum(norm2, 1e-8)
        cos_omega = jnp.clip(jnp.sum(z1_n * z2_n, axis=-1, keepdims=True), -1.0, 1.0)
        omega = jnp.arccos(cos_omega)
        sin_omega = jnp.sin(omega)
        safe = sin_omega > 1e-6
        s1 = jnp.where(safe, jnp.sin((1 - alpha) * omega) / sin_omega, 1 - alpha)
        s2 = jnp.where(safe, jnp.sin(alpha * omega) / sin_omega, alpha)
        z_dir = s1 * z1_n + s2 * z2_n
        norm_interp = (1 - alpha) * norm1 + alpha * norm2
        z_dir_norm = jnp.linalg.norm(z_dir, axis=-1, keepdims=True)
        return z_dir * (norm_interp / jnp.maximum(z_dir_norm, 1e-8))

    def _pair_random(n_items, n_pairs, rng_seed):
        """Random permutation pairing."""
        rng = np.random.RandomState(rng_seed)
        idx = rng.permutation(n_items)
        return idx[:n_pairs], idx[n_pairs:2 * n_pairs]

    def _pair_max_dissimilar(embeddings, n_pairs):
        """Greedy max-dissimilarity pairing."""
        from sklearn.metrics import pairwise_distances
        K = len(embeddings)
        dist_matrix = pairwise_distances(embeddings, metric="euclidean")
        np.fill_diagonal(dist_matrix, -np.inf)
        available = set(range(K))
        paired_a, paired_b = [], []
        for _ in range(n_pairs):
            mask = np.full_like(dist_matrix, -np.inf)
            avail = sorted(available)
            for i in avail:
                for j in avail:
                    if i != j:
                        mask[i, j] = dist_matrix[i, j]
            flat_idx = np.argmax(mask)
            i, j = np.unravel_index(flat_idx, mask.shape)
            paired_a.append(i)
            paired_b.append(j)
            available.discard(i)
            available.discard(j)
        return np.array(paired_a), np.array(paired_b)

    _interp_history = {"xs": [], "sfl": {}, "sr": {}, "ratio": {}}

    def compare_interp_vs_accel(train_state, eval_step):
        """Shadow comparison: score 4 interpolation methods vs ACCEL children.

        Methods: random+linear, random+slerp, max_dissim+linear, max_dissim+slerp.
        Runs outside JIT at interp_compare_interval. Does NOT modify train_state
        or the buffer — purely for logging to wandb.
        """
        interval = config.get("interp_compare_interval", 0)
        if interval <= 0 or vae_decode_fn is None:
            return
        updates_so_far = (eval_step + 1) * config["eval_freq"]
        if updates_so_far % interval != 0:
            return

        sampler = train_state.sampler
        buf_size = int(np.asarray(sampler["size"]))
        top_k = min(config.get("interp_compare_top_k", 64), buf_size)
        if top_k < 4:
            return

        alpha = config.get("interp_compare_alpha", 0.2)
        num_attempts = config.get("interp_compare_num_attempts", 10)
        n_children = top_k // 2  # 2 parents per child

        # Extract top-K levels by score
        scores_np = np.asarray(sampler["scores"][:buf_size])
        top_k_idx = np.argsort(-scores_np)[:top_k]
        levels_pytree = sampler["levels"]
        top_levels_list = [jax.tree_util.tree_map(lambda x: x[i], levels_pytree) for i in top_k_idx]
        top_levels = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *top_levels_list)

        # Compute agent behavioural embeddings for top-K
        rng_emb = jax.random.PRNGKey(eval_step + 5000)
        rng_emb, rng_reset_emb, rng_eval_emb = jax.random.split(rng_emb, 3)
        init_obs_emb, init_state_emb = jax.vmap(env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset_emb, top_k), top_levels, env_params)
        (_, _, _, _, _, _), traj_emb = sample_trajectories_rnn(
            rng_eval_emb, env, env_params, train_state,
            ActorCritic.initialize_carry((top_k,)),
            init_obs_emb, init_state_emb, top_k, config["num_steps"])
        _, actions_emb, _, dones_emb, _, _, _, _, hstates_emb = traj_emb
        top_embeddings = np.asarray(compute_insertion_embeddings(hstates_emb, actions_emb, dones_emb))

        # Encode top-K to VAE latent space
        top_tokens = np.asarray(jax.vmap(level_to_tokens_vae)(top_levels))
        means = vae_encode_fn(jnp.array(top_tokens))

        # Build pairings
        idx_maxd_a, idx_maxd_b = _pair_max_dissimilar(top_embeddings, n_children)
        idx_rand_a, idx_rand_b = _pair_random(top_k, n_children, rng_seed=eval_step)

        # Generate children for all 4 methods
        methods = {
            "rand_linear":    alpha * means[idx_rand_a] + (1 - alpha) * means[idx_rand_b],
            "rand_slerp":     _slerp_batch(means[idx_rand_a], means[idx_rand_b], alpha),
            "maxd_linear":    alpha * means[idx_maxd_a] + (1 - alpha) * means[idx_maxd_b],
            "maxd_slerp":     _slerp_batch(means[idx_maxd_a], means[idx_maxd_b], alpha),
        }

        # Decode all children and score them
        def score_levels(levels, rng_base):
            n = jax.tree_util.tree_flatten(levels)[0][0].shape[0]
            max_steps = env_params.max_steps_in_episode
            all_solved = []
            for attempt in range(num_attempts):
                rng = jax.random.PRNGKey(attempt + 1000 + rng_base)
                rng, rng_r, rng_e = jax.random.split(rng, 3)
                init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
                    jax.random.split(rng_r, n), levels, env_params)
                _, rewards, _ = evaluate_rnn(
                    rng_e, eval_env, env_params, train_state,
                    ActorCritic.initialize_carry((n,)),
                    init_obs, init_env_state, max_steps)
                all_solved.append(np.asarray((rewards.sum(axis=0) > 0).astype(jnp.float32)))
            solved = np.stack(all_solved, axis=0)
            sr = solved.mean(axis=0)
            sfl = sr * (1 - sr)
            return sfl, sr

        # Score all methods + ACCEL
        method_sfl, method_sr = {}, {}
        rng_offset = 0
        for method_name, z_child in methods.items():
            rng_decode = jax.random.PRNGKey(eval_step + 6000 + rng_offset)
            levels = decode_latent_to_levels_vae(vae_decode_fn, z_child, rng_decode)
            sfl, sr = score_levels(levels, eval_step * 100 + rng_offset)
            method_sfl[method_name] = float(sfl.mean())
            method_sr[method_name] = float(sr.mean())
            rng_offset += 10

        accel_parents = train_state.replay_last_level_batch
        n_accel = config["num_train_envs"]
        rng_accel = jax.random.PRNGKey(eval_step + 7000)
        accel_levels = jax.vmap(mutate_level, (0, 0, None))(
            jax.random.split(rng_accel, n_accel), accel_parents, config["num_edits"])
        accel_sfl, accel_sr = score_levels(accel_levels, eval_step * 100 + 50)
        method_sfl["accel"] = float(accel_sfl.mean())
        method_sr["accel"] = float(accel_sr.mean())

        # Ratios vs ACCEL
        accel_sfl_val = max(method_sfl["accel"], 1e-8)
        method_ratio = {m: method_sfl[m] / accel_sfl_val for m in methods}

        # Accumulate history for line_series charts
        all_methods = list(methods.keys()) + ["accel"]
        _interp_history["xs"].append(updates_so_far)
        for m in all_methods:
            _interp_history["sfl"].setdefault(m, []).append(method_sfl[m])
            _interp_history["sr"].setdefault(m, []).append(method_sr[m])
        for m in methods:
            _interp_history["ratio"].setdefault(m, []).append(method_ratio[m])

        # Log combined line_series charts (one plot per metric type)
        xs_list = _interp_history["xs"]
        wandb.log({
            "num_updates": updates_so_far,
            "interp_compare/sfl_all": wandb.plot.line_series(
                xs=xs_list,
                ys=[_interp_history["sfl"][m] for m in all_methods],
                keys=all_methods,
                title="SFL by Method",
                xname="num_updates",
            ),
            "interp_compare/solve_rate_all": wandb.plot.line_series(
                xs=xs_list,
                ys=[_interp_history["sr"][m] for m in all_methods],
                keys=all_methods,
                title="Solve Rate by Method",
                xname="num_updates",
            ),
            "interp_compare/sfl_ratio_all": wandb.plot.line_series(
                xs=xs_list,
                ys=[_interp_history["ratio"][m] for m in methods],
                keys=list(methods.keys()),
                title="SFL Ratio vs ACCEL (>1 = better)",
                xname="num_updates",
            ),
        })

        best_method = max(methods.keys(), key=lambda m: method_sfl[m])
        print(f"[InterpCompare @ {updates_so_far}] "
              f"Best: {best_method} SFL={method_sfl[best_method]:.4f} | "
              f"ACCEL SFL={method_sfl['accel']:.4f} | "
              f"rand_lin={method_sfl['rand_linear']:.4f} "
              f"rand_slp={method_sfl['rand_slerp']:.4f} "
              f"maxd_lin={method_sfl['maxd_linear']:.4f} "
              f"maxd_slp={method_sfl['maxd_slerp']:.4f}")

    # And run the train_eval_sep function for the specified number of updates
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)

    # Initialize lineage snapshot if provenance tracking is active
    if _buffer_origins is not None:
        _prev_buffer_hashes = _compute_buffer_hashes(runner_state[1])

    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time

        # Update lineage tracking (must run before log_eval so provenance counts are current)
        _update_lineage(runner_state[1], metrics)
        _update_embeddings(metrics)

        # Recompute seed embeddings with current agent (keeps proximity gate fresh)
        if config.get("proximity_gate_ratio", 0) > 0:
            runner_state = _recompute_seed_embeddings(runner_state)

        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))

        # Shadow comparison: interpolation vs ACCEL mutations
        compare_interp_vs_accel(runner_state[1], eval_step)

        # LLM injection hook
        if llm_injector is not None:
            llm_injector._buffer_embeddings = _buffer_embeddings
            runner_state = llm_injector.maybe_inject(runner_state, eval_step)

        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()

        # Buffer dump: periodic interval and/or specific steps
        updates_so_far = (eval_step + 1) * config["eval_freq"]
        should_dump = False
        if config["buffer_dump_interval"] > 0 and updates_so_far % config["buffer_dump_interval"] == 0:
            should_dump = True
        if config.get("_buffer_dump_steps") and updates_so_far in config["_buffer_dump_steps"]:
            should_dump = True
        if should_dump:
            dump_buffer(runner_state[1], updates_so_far)

    # === End-of-run buffer dump ===
    final_train_state = runner_state[1]
    sampler = final_train_state.sampler
    size = int(sampler["size"])
    print(f"[Buffer dump] Saving {size} levels (final)...")
    dump_buffer(final_train_state, 0)  # tag = "_final"

    buffer_levels = jax.tree_util.tree_map(lambda x: x[:size], sampler["levels"])
    buffer_scores = np.asarray(sampler["scores"][:size])
    _l2t = level_to_tokens_vae if _need_vae else level_to_tokens
    tokens = jax.vmap(_l2t)(buffer_levels)

    # === Post-training: evaluate agent on buffer levels ===
    if config.get("skip_post_eval"):
        print("[Post-training] Skipped (--skip_post_eval). Use evaluate_buffer.py on the checkpoint later.")
        wandb.finish()
        return

    print(f"\n[Post-training] Evaluating agent on {size} buffer levels...")
    eval_env_post = Maze(max_height=config["maze_height"], max_width=config["maze_width"], agent_view_size=config["agent_view_size"], normalize_obs=True)
    max_steps = env_params.max_steps_in_episode
    num_eval_attempts = 5

    all_solve_rates = []
    for attempt in range(num_eval_attempts):
        rng_attempt = jax.random.PRNGKey(attempt + 2000)
        rng_attempt, rng_reset, rng_eval = jax.random.split(rng_attempt, 3)
        init_obs, init_env_state = jax.vmap(eval_env_post.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, size), buffer_levels, env_params
        )
        states, rewards, episode_lengths = evaluate_rnn(
            rng_eval, eval_env_post, env_params, final_train_state,
            ActorCritic.initialize_carry((size,)),
            init_obs, init_env_state, max_steps,
        )
        mask = jnp.arange(max_steps)[:, None] < episode_lengths[None, :]
        cum_rewards = (rewards * mask).sum(axis=0)
        all_solve_rates.append((cum_rewards > 0).astype(float))

    solve_rates = np.asarray(jnp.stack(all_solve_rates).mean(axis=0))
    # Get paths from last attempt
    agent_paths = np.asarray(states.agent_pos)  # (max_steps, size, 2)
    ep_lengths = np.asarray(episode_lengths)

    print(f"  Mean solve rate: {solve_rates.mean():.1%}")
    print(f"  Unsolved (0%): {(solve_rates == 0).sum()} | Fully solved (100%): {(solve_rates == 1.0).sum()}")

    # Save evaluation results
    if config.get("output_dir"):
        dump_dir = os.path.join(config["output_dir"], "buffer_dumps")
    else:
        dump_dir = os.path.join("/tmp", "buffer_dumps", f"{config['run_name']}", str(config["seed"]))
    os.makedirs(dump_dir, exist_ok=True)
    gcs_base = f"{config['gcs_prefix']}/buffer_dumps/{config['run_name']}/{config['seed']}"
    eval_path = os.path.join(dump_dir, "buffer_eval.npz")
    np.savez_compressed(eval_path, solve_rates=solve_rates, paths=agent_paths,
                        episode_lengths=ep_lengths, buffer_scores=buffer_scores, tokens=np.asarray(tokens))
    print(f"[Buffer eval] Saved: {eval_path}")
    if config.get("gcs_bucket"):
        _upload_to_gcs(eval_path, config["gcs_bucket"], f"{gcs_base}/buffer_eval.npz")

    # Log summary to wandb
    wandb.summary["buffer/mean_solve_rate"] = float(solve_rates.mean())
    wandb.summary["buffer/unsolved_count"] = int((solve_rates == 0).sum())
    wandb.summary["buffer/fully_solved_count"] = int((solve_rates == 1.0).sum())
    wandb.summary["buffer/mean_score"] = float(buffer_scores.mean())

    # === Post-training: render hardest levels with agent paths ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = np.argsort(solve_rates)  # hardest first
        n_show = min(16, size)
        ncols = min(4, n_show)
        nrows = (n_show + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[None, :]

        for idx in range(n_show):
            level_idx = order[idx]
            ax = axes[idx // ncols, idx % ncols]
            level = jax.tree_util.tree_map(lambda x: x[level_idx], buffer_levels)
            img = np.asarray(env_renderer.render_level(level, env_params))
            ax.imshow(img)

            path = agent_paths[:, level_idx, :]
            ep_len = int(ep_lengths[level_idx])
            path = path[:ep_len]
            tile_size = 8  # matches MazeRenderer tile_size
            px = (path[:, 0].astype(float) + 0.5) * tile_size
            py = (path[:, 1].astype(float) + 0.5) * tile_size
            ax.plot(px, py, 'r-', linewidth=1, alpha=0.7)
            if len(px) > 0:
                ax.plot(px[0], py[0], 'go', markersize=4)
                ax.plot(px[-1], py[-1], 'rs', markersize=4)

            ax.set_title(f"Solve:{solve_rates[level_idx]:.0%} Score:{buffer_scores[level_idx]:.2f}", fontsize=8)
            ax.axis("off")

        for idx in range(n_show, nrows * ncols):
            axes[idx // ncols, idx % ncols].axis("off")

        plt.suptitle(f"Hardest {n_show} Buffer Levels", fontsize=12)
        plt.tight_layout()
        plot_path = os.path.join(dump_dir, "hardest_levels.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved: {plot_path}")
        if config.get("gcs_bucket"):
            _upload_to_gcs(plot_path, config["gcs_bucket"], f"{gcs_base}/hardest_levels.png")
        wandb.log({"buffer/hardest_levels": wandb.Image(plot_path)})
    except Exception as e:
        print(f"[Plot] Skipped rendering: {e}")

    # === Post-training: PCA of buffer snapshots in VAE latent space ===
    if vae_decode_fn is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA

            print("\n[Post-training] PCA analysis of buffer snapshots in VAE latent space...")

            # Build VAE encode function
            def vae_encode_fn(tokens_batch):
                mean, _ = vae.apply({"params": vae_params}, tokens_batch, train=False, method=vae.encode)
                return mean

            # Collect all periodic buffer dumps + final
            if config.get("output_dir"):
                dump_dir_pca = os.path.join(config["output_dir"], "buffer_dumps")
            else:
                dump_dir_pca = os.path.join("/tmp", "buffer_dumps", f"{config['run_name']}", str(config["seed"]))
            snapshot_labels = []
            snapshot_latents = []
            snapshot_scores = []

            # Find all dump files in order
            dump_files = sorted([
                f for f in os.listdir(dump_dir_pca)
                if f.startswith("buffer_dump_") and f.endswith(".npz")
            ])

            for dump_file in dump_files:
                data = np.load(os.path.join(dump_dir_pca, dump_file))
                toks = jnp.array(data["tokens"])
                sc = data["scores"]
                tag = dump_file.replace("buffer_dump_", "").replace(".npz", "")

                # Encode through VAE in batches
                latents = []
                for i in range(0, len(toks), 512):
                    batch = toks[i:i + 512]
                    latents.append(np.asarray(vae_encode_fn(batch)))
                latents = np.concatenate(latents, axis=0)

                snapshot_labels.append(tag)
                snapshot_latents.append(latents)
                snapshot_scores.append(sc)
                print(f"  Encoded {tag}: {len(latents)} levels")

            if len(snapshot_latents) >= 1:
                # Fit PCA on all snapshots combined
                all_latents = np.concatenate(snapshot_latents, axis=0)
                pca = PCA(n_components=2)
                pca.fit(all_latents)

                # Plot: one color per snapshot timestep
                fig, axes = plt.subplots(1, 2, figsize=(18, 7))
                cmap = plt.cm.viridis
                n_snaps = len(snapshot_labels)
                colors = [cmap(i / max(n_snaps - 1, 1)) for i in range(n_snaps)]

                for i, (label, latents, sc) in enumerate(zip(snapshot_labels, snapshot_latents, snapshot_scores)):
                    proj = pca.transform(latents)
                    axes[0].scatter(proj[:, 0], proj[:, 1], c=[colors[i]], alpha=0.3, s=6, label=label)

                axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
                axes[0].set_title("Buffer Evolution in VAE Latent Space")
                axes[0].legend(markerscale=3, fontsize=8)

                # Right plot: final buffer colored by score
                final_proj = pca.transform(snapshot_latents[-1])
                final_sc = snapshot_scores[-1]
                valid = np.isfinite(final_sc) & (final_sc > -1e6)
                sc_plot = axes[1].scatter(final_proj[valid, 0], final_proj[valid, 1],
                                          c=final_sc[valid], cmap="plasma", alpha=0.4, s=8)
                plt.colorbar(sc_plot, ax=axes[1], label="Score (regret)")
                axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
                axes[1].set_title("Final Buffer — Colored by Score")

                plt.tight_layout()
                pca_path = os.path.join(dump_dir_pca, "buffer_pca_evolution.png")
                plt.savefig(pca_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"[PCA] Saved: {pca_path}")

                if config.get("gcs_bucket"):
                    gcs_base = f"{config['gcs_prefix']}/buffer_dumps/{config['run_name']}/{config['seed']}"
                    _upload_to_gcs(pca_path, config["gcs_bucket"], f"{gcs_base}/buffer_pca_evolution.png")
                wandb.log({"buffer/pca_evolution": wandb.Image(pca_path)})
        except Exception as e:
            print(f"[PCA] Skipped latent analysis: {e}")

    return final_train_state

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="JAXUED_TEST")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="Wandb run group (defaults to run_name if not set)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb entity/team (default: use wandb default)")
    parser.add_argument("--wandb_update_offset", type=int, default=0,
                        help="Offset added to num_updates in wandb logs (e.g. 10000 for warmstart)")
    parser.add_argument("--seed", type=int, default=0)
    # === Train vs Eval ===
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)
    # === OUTPUT ===
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Co-locate checkpoints + buffer dumps under one directory. "
                             "Creates output_dir/checkpoints/ and output_dir/buffer_dumps/")
    # === CHECKPOINTING ===
    parser.add_argument("--checkpoint_save_interval", type=int, default=1)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=120)
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
    group.add_argument("--score_function", type=str, default="MaxMC",
                       choices=["MaxMC", "pvl", "sfl"],
                       help="Level scoring: MaxMC (regret), pvl (positive value loss), sfl (learnability p*(1-p))")
    group.add_argument("--num_sfl_rollouts", type=int, default=10,
                       help="Number of evaluation rollouts for SFL learnability estimation")
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
    # === PROXIMITY GATE (LLM-lineage ACCEL mutations) ===
    group.add_argument("--proximity_gate_ratio", type=float, default=0.0,
                       help="Proximity gate for LLM-lineage ACCEL mutations: "
                            "child must have dist(child_emb, seed_emb) < ratio * dist(seed_emb, organic_centroid). "
                            "0 = disabled.")
    group.add_argument("--max_llm_ancestors", type=int, default=256,
                       help="Max distinct LLM ancestor seeds to track (static JAX array size)")
    group.add_argument("--fixed_distribution", action=argparse.BooleanOptionalAction, default=False,
                       help="Fix LLM/organic replay ratio to match initial injection percentage")
    group.add_argument("--llm_mutation_retries", type=int, default=5,
                       help="Number of mutation candidates per LLM-lineage parent when proximity gate is active. "
                            "Best passing candidate is selected. Only used when --proximity_gate_ratio > 0.")
    group.add_argument("--llm_buffer_state", type=str, default=None,
                       choices=["stale", "fresh"],
                       help="'stale' uses incrementally updated embeddings, "
                            "'fresh' recomputes all buffer embeddings at each injection event "
                            "(default: from config.yaml)")
    # === ENV CONFIG ===
    group.add_argument("--maze_height", type=int, default=13)
    group.add_argument("--maze_width", type=int, default=13)
    group.add_argument("--agent_view_size", type=int, default=5)
    # === DR CONFIG ===
    group.add_argument("--n_walls", type=int, default=25)
    # === CMA-ES + VAE CONFIG ===
    group.add_argument("--use_cmaes", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--vae_checkpoint_path", type=str, default=None,
                       help="Path to VAE .pkl checkpoint file")
    group.add_argument("--vae_config_path", type=str, default=None,
                       help="Path to VAE config.yaml (run directory)")
    group.add_argument("--cmaes_sigma_init", type=float, default=1.0)
    group.add_argument("--cmaes_reset_interval", type=int, default=500,
                       help="Reset CMA-ES every N DR updates to prevent stagnation")
    # === GCS CONFIG ===
    group.add_argument("--gcs_bucket", type=str, default=None,
                       help="GCS bucket name for saving checkpoints/artifacts (e.g. 'ucl-ued-project-bucket')")
    group.add_argument("--gcs_prefix", type=str, default="accel",
                       help="Prefix path within GCS bucket")
    group.add_argument("--buffer_dump_interval", type=int, default=0,
                       help="Dump PLR buffer every N updates. 0 to disable periodic dumps.")
    group.add_argument("--buffer_dump_steps", type=str, default=None,
                       help="Comma-separated list of specific update steps to dump buffer (e.g. '2500,5000,10000,30000')")
    group.add_argument("--preload_buffer_npz", type=str, default=None,
                       help="Path to a merged_buffer .npz to preload into the PLR sampler at init. "
                            "Supports provenance tracking via 'origins' and 'origin_ids' arrays.")
    group.add_argument("--resume_checkpoint_dir", type=str, default=None,
                       help="Path to a checkpoint directory to restore agent params from "
                            "(for resuming training, not eval). Loads params only, not sampler.")
    group.add_argument("--skip_post_eval", action="store_true", default=False,
                       help="Skip post-training buffer evaluation, rendering, and PCA (run evaluate_buffer.py separately)")
    # === INTERP vs ACCEL SHADOW COMPARISON ===
    group.add_argument("--interp_compare_interval", type=int, default=0,
                       help="Log interp vs ACCEL mutation comparison every N updates (0 to disable). "
                            "Requires --use_cmaes (VAE). Does NOT affect training.")
    group.add_argument("--interp_compare_top_k", type=int, default=64,
                       help="Top-K buffer levels to use for interpolation comparison")
    group.add_argument("--interp_compare_alpha", type=float, default=0.2,
                       help="Interpolation alpha for shadow comparison")
    group.add_argument("--interp_compare_num_attempts", type=int, default=10,
                       help="Number of rollout attempts for scoring in shadow comparison")

    # === LLM INJECTION CONFIG ===
    llm_group = parser.add_argument_group('LLM Injection')
    llm_group.add_argument("--use_llm", action=argparse.BooleanOptionalAction, default=False,
                           help="Enable LLM-based maze injection into PLR buffer")
    llm_group.add_argument("--llm_provider", type=str, default="",
                           help="LLM provider (required when --use_llm): 'openrouter', 'openai', 'ollama'")
    llm_group.add_argument("--llm_model", type=str, default="",
                           help="Model name for the LLM provider")
    llm_group.add_argument("--llm_config", type=str, default="llm/config.yaml",
                           help="Path to LLM config YAML file")
    llm_group.add_argument("--llm_inject_interval", type=int, default=3000,
                           help="Number of eval steps between LLM injection events")
    llm_group.add_argument("--llm_inject_start_step", type=int, default=5000,
                           help="Training step at which LLM injection begins (no injection before this step)")
    llm_group.add_argument("--llm_batch_size", type=int, default=25,
                           help="Number of raw mazes requested from LLM per injection round")
    llm_group.add_argument("--llm_target_buffer_pct", type=float, default=0.0,
                           help="Target LLM buffer fill %% (0=disabled, 0.05=5%%). "
                                "Repeats injection rounds until target met or max rounds hit.")
    llm_group.add_argument("--llm_n_references", type=int, default=None,
                           help="Number of buffer mazes shown to LLM as reference context (default: from config.yaml)")
    llm_group.add_argument("--llm_ref_strategy", type=str, default=None,
                           choices=["hardest", "random", "diverse", "kmedoid", "hybrid-kmedoid"],
                           help="Strategy for selecting reference mazes from PLR buffer (default: from config.yaml)")
    llm_group.add_argument("--llm_hybrid_difficulty_percentile", type=float, default=None,
                           help="Percentile threshold for hybrid-kmedoid difficulty filter "
                                "(default: from config.yaml)")
    llm_group.add_argument("--llm_amplification", action=argparse.BooleanOptionalAction, default=True,
                           help="Enable mutation amplification of LLM seed mazes")
    llm_group.add_argument("--llm_mutations_per_seed", type=int, default=30,
                           help="Number of wall-flip mutations per LLM seed maze")
    llm_group.add_argument("--llm_gate", action=argparse.BooleanOptionalAction, default=None,
                           help="Enable decision gate (default: from config.yaml)")
    llm_group.add_argument("--llm_difficulty_threshold", type=float, default=None,
                           help="Minimum difficulty score for gate acceptance (default: from config.yaml)")
    llm_group.add_argument("--llm_difficulty_gate_mode", type=str, default=None,
                           choices=["fixed", "buffer_min", "buffer_mean", "reference_mean", "competitive"],
                           help="How LLM difficulty threshold is set (default: from config.yaml)")
    llm_group.add_argument("--llm_min_diversity", type=float, default=None,
                           help="Minimum diversity score for gate acceptance (default: from config.yaml)")
    llm_group.add_argument("--llm_diversity_gate_mode", type=str, default=None,
                           choices=["fixed", "buffer_median", "disabled"],
                           help="How LLM diversity threshold is set (default: from config.yaml)")
    llm_group.add_argument("--llm_diversity_metric", type=str, default=None,
                           choices=["td_error_emd", "experience_divergence", "position_dtw", "embedding_l2", "cenie"],
                           help="Diversity metric for decision gate (default: from config.yaml)")
    llm_group.add_argument("--llm_max_diversity_retries", type=int, default=None,
                           help="Max LLM retries when gate rejects a maze (default: from config.yaml)")
    llm_group.add_argument("--llm_n_rollouts", type=int, default=None,
                           help="Number of agent rollouts per candidate for gate evaluation (default: from config.yaml)")

    config = vars(parser.parse_args())
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])
    config["group_name"] = config.get("wandb_group") or config.get("run_name") or ""
    # Parse --buffer_dump_steps into a set of ints
    if config.get("buffer_dump_steps"):
        config["_buffer_dump_steps"] = set(int(s.strip()) for s in config["buffer_dump_steps"].split(","))
    else:
        config["_buffer_dump_steps"] = set()
    
    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'
    
    # wandb.login()
    main(config, project=config["project"])
