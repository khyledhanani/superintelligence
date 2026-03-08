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
from metrics.trajectory_metrics import compute_pairwise_metrics
from metrics.trajectory_cache import TrajectoryCache

# VAE + CMA-ES imports (conditional on --use_cmaes flag)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vae'))
from vae_model import CluttrVAE
from vae_level_utils import decode_latent_to_levels, level_to_tokens
from cmaes_manager import CMAESManager

class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    es_state: chex.ArrayTree = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
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
        Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict, chex.Array]]: (rng, train_state, hstate, last_obs, last_env_state, last_value), traj, where traj is (obs, action, reward, done, log_prob, value, info, agent_pos). agent_pos has shape (NUM_STEPS, NUM_ENVS, 2). The first element in the tuple consists of arrays that have shapes (NUM_ENVS, ...) (except `rng` and and `train_state` which are singleton). The second element in the tuple is of shape (NUM_STEPS, NUM_ENVS, ...), and it contains the trajectory.
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

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info, agent_pos)

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
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")

def main(config=None, project="JAXUED_TEST"):
    tags = []
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")
    if config.get("use_cmaes"):
        tags.append("CMA-ES")
    run = wandb.init(config=config, project=project, group=config["run_name"], tags=tags)
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
    wandb.define_metric("diversity/*", step_metric="num_updates")

    # --- CMA-ES + VAE setup ---
    vae_decode_fn = None
    cmaes_mgr = None
    if config["use_cmaes"]:
        assert config["vae_checkpoint_path"] is not None, "--vae_checkpoint_path required when --use_cmaes"
        assert config["vae_config_path"] is not None, "--vae_config_path required when --use_cmaes"

        # Load VAE config
        with open(config["vae_config_path"]) as f:
            vae_cfg = yaml.safe_load(f)

        # Instantiate model with config dimensions
        vae = CluttrVAE(
            vocab_size=vae_cfg["vocab_size"],
            embed_dim=vae_cfg["embed_dim"],
            latent_dim=vae_cfg["latent_dim"],
            seq_len=vae_cfg["seq_len"],
        )

        # Load checkpoint
        with open(config["vae_checkpoint_path"], "rb") as f:
            vae_ckpt = pickle.load(f)
        vae_params = vae_ckpt["params"] if isinstance(vae_ckpt, dict) and "params" in vae_ckpt else vae_ckpt

        # Build pure decode function: z (latent_dim,) -> logits (seq_len, vocab_size)
        def vae_decode_fn(z):
            return vae.apply({"params": vae_params}, z, method=vae.decode)

        # Initialize CMA-ES manager
        cmaes_mgr = CMAESManager(
            popsize=config["num_train_envs"],
            latent_dim=vae_cfg["latent_dim"],
            sigma_init=config["cmaes_sigma_init"],
        )
        print(f"[CMA-ES] VAE loaded from {config['vae_checkpoint_path']}")
        print(f"[CMA-ES] latent_dim={vae_cfg['latent_dim']}, popsize={config['num_train_envs']}")

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

        wandb.log(log_dict)
    
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
        sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
        pholder_level_batch = jax.tree_util.tree_map(lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), pholder_level)

        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            es_state=es_state_init,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
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
                new_levels = decode_latent_to_levels(vae_decode_fn, z_population, rng_decode)
            else:
                new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_train_envs"]))

            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), new_levels, env_params)
            # Rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info, agent_positions),
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

            # Validity check for generated levels (CMA-ES or random)
            is_valid = jax.vmap(lambda l: l.is_well_formatted())(new_levels)

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": new_levels.wall_map.sum() / config["num_train_envs"],
                "gen/valid_structure_pct": is_valid.mean() * 100,
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
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, config["num_train_envs"])
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params)
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info, agent_positions),
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
                "gen/valid_structure_pct": jnp.float32(0.0),  # no new levels generated
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
            )
            return (rng, train_state), metrics
        
        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                This mutates the previous batch of replay levels and potentially adds them to the level buffer.
                This also updates the policy iff `config["exploratory_grad_updates"]` is True.
            """
            sampler = train_state.sampler
            rng, rng_mutate, rng_reset = jax.random.split(rng, 3)
            
            # mutate
            parent_levels = train_state.replay_last_level_batch
            child_levels = jax.vmap(mutate_level, (0, 0, None))(jax.random.split(rng_mutate, config["num_train_envs"]), parent_levels, config["num_edits"])
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), child_levels, env_params)

            # rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info, agent_positions),
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

            # Validity check for mutated levels
            is_valid_mut = jax.vmap(lambda l: l.is_well_formatted())(child_levels)

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": child_levels.wall_map.sum() / config["num_train_envs"],
                "gen/valid_structure_pct": is_valid_mut.mean() * 100,
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
    runner_state = (rng_train, train_state)
    
    def dump_buffer(train_state, update_num):
        """Save PLR buffer as .npy (VAE token format) + .npz (full metadata). Uploads to GCS."""
        sampler = train_state.sampler
        size = int(sampler["size"])
        if size == 0:
            return

        buffer_levels = jax.tree_util.tree_map(lambda x: x[:size], sampler["levels"])
        tokens = jax.vmap(level_to_tokens)(buffer_levels)

        dump_data = {
            "tokens": np.asarray(tokens),
            "scores": np.asarray(sampler["scores"][:size]),
            "timestamps": np.asarray(sampler["timestamps"][:size]),
            "size": size,
            "update_num": update_num,
        }

        dump_dir = os.path.join("/tmp", "buffer_dumps", f"{config['run_name']}", str(config["seed"]))
        os.makedirs(dump_dir, exist_ok=True)
        tag = f"_{update_num}k" if update_num > 0 else "_final"
        tokens_path = os.path.join(dump_dir, f"buffer_tokens{tag}.npy")
        dump_path = os.path.join(dump_dir, f"buffer_dump{tag}.npz")
        np.save(tokens_path, np.asarray(tokens))
        np.savez_compressed(dump_path, **dump_data)
        print(f"[Buffer dump @ {update_num}k] {size} levels -> {tokens_path}")

        if config.get("gcs_bucket"):
            gcs_base = f"{config['gcs_prefix']}/buffer_dumps/{config['run_name']}/{config['seed']}"
            _upload_to_gcs(tokens_path, config["gcs_bucket"], f"{gcs_base}/buffer_tokens{tag}.npy")
            _upload_to_gcs(dump_path, config["gcs_bucket"], f"{gcs_base}/buffer_dump{tag}.npz")

    # === Trajectory cache + diversity metrics ===
    traj_cache = TrajectoryCache()
    diversity_log_interval = config.get("diversity_log_interval", 100)  # every N eval_steps
    diversity_sample_size = config.get("diversity_sample_size", 20)  # num levels to subsample

    @jax.jit
    def collect_buffer_trajectories(rng, train_state, level_indices):
        """Run agent on buffer levels and return trajectory data for caching."""
        sampler = train_state.sampler
        levels = level_sampler.get_levels(sampler, level_indices)
        num_levels = level_indices.shape[0]
        rng, rng_reset = jax.random.split(rng)
        init_obs, init_env_state = jax.vmap(env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, num_levels), levels, env_params
        )
        (
            (rng, _, _, _, _, _),
            (obs, actions, rewards, dones, log_probs, values, info, agent_positions),
        ) = sample_trajectories_rnn(
            rng, env, env_params, train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs, init_env_state, num_levels,
            env_params.max_steps_in_episode,
        )
        return obs.image, actions, rewards, dones, values, agent_positions

    def update_cache_and_compute_metrics(rng, train_state):
        """Collect trajectories on buffer subsample, update cache, compute pairwise metrics."""
        sampler = train_state.sampler
        size = int(sampler["size"])
        if size < diversity_sample_size:
            return {}

        # Sample fixed number of indices from buffer (fixed shape for JIT stability)
        n_sample = diversity_sample_size
        rng_np = np.random.default_rng(int(jax.random.randint(rng, (), 0, 2**31)))
        sample_indices = jnp.array(rng_np.choice(size, size=n_sample, replace=False))

        # Collect trajectories (JIT'd)
        rng, rng_collect = jax.random.split(rng)
        obs, actions, rewards, dones, values, agent_positions = collect_buffer_trajectories(
            rng_collect, train_state, sample_indices
        )

        # Move to numpy for cache + metrics
        obs_np = np.asarray(obs)  # (T, N, *obs_shape)
        actions_np = np.asarray(actions)  # (T, N)
        dones_np = np.asarray(dones)  # (T, N)
        values_np = np.asarray(values)  # (T, N)
        positions_np = np.asarray(agent_positions)  # (T, N, 2)
        indices_np = np.asarray(sample_indices)

        # Update cache
        traj_cache.update_batch(indices_np, obs_np, positions_np, values_np, dones_np, actions_np)

        # Sync cache with buffer (evict stale entries)
        active_indices = set(range(size))
        traj_cache.sync_with_buffer(active_indices)

        # Build trajectory list for pairwise metrics
        trajectories = []
        for i in range(n_sample):
            trajectories.append({
                "observations": obs_np[:, i],
                "positions": positions_np[:, i],
                "values": values_np[:, i],
                "dones": dones_np[:, i],
            })

        # Compute pairwise metrics
        pairwise = compute_pairwise_metrics(trajectories)

        # Build wandb log dict
        diversity_log = {}
        for metric_name, values_arr in pairwise.items():
            if len(values_arr) > 0:
                diversity_log[f"diversity/{metric_name}/mean"] = float(values_arr.mean())
                diversity_log[f"diversity/{metric_name}/std"] = float(values_arr.std())
                diversity_log[f"diversity/{metric_name}/min"] = float(values_arr.min())
                diversity_log[f"diversity/{metric_name}/max"] = float(values_arr.max())
        diversity_log["diversity/cache_size"] = traj_cache.size

        return diversity_log

    # And run the train_eval_sep function for the specified number of updates
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()

        # Periodic buffer dump at configured intervals
        updates_so_far = (eval_step + 1) * config["eval_freq"]
        if config["buffer_dump_interval"] > 0 and updates_so_far % config["buffer_dump_interval"] == 0:
            dump_buffer(runner_state[1], updates_so_far // 1000)

        # Periodic diversity metrics computation
        if diversity_log_interval > 0 and (eval_step + 1) % diversity_log_interval == 0:
            rng_diversity = jax.random.PRNGKey(eval_step + 9999)
            diversity_metrics = update_cache_and_compute_metrics(rng_diversity, runner_state[1])
            if diversity_metrics:
                diversity_metrics["num_updates"] = int(updates_so_far)
                wandb.log(diversity_metrics)
                obs_dtw = diversity_metrics.get('diversity/obs_dtw_distances/mean', float('nan'))
                pos_dtw = diversity_metrics.get('diversity/pos_dtw_distances/mean', float('nan'))
                val_corr = diversity_metrics.get('diversity/value_correlations/mean', float('nan'))
                jaccard = diversity_metrics.get('diversity/jaccard_indices/mean', float('nan'))
                print(f"  [Diversity] obs_dtw={obs_dtw:.4f}, pos_dtw={pos_dtw:.4f}, "
                      f"val_corr={val_corr:.4f}, jaccard={jaccard:.4f}")

    # === End-of-run buffer dump ===
    final_train_state = runner_state[1]
    sampler = final_train_state.sampler
    size = int(sampler["size"])
    print(f"[Buffer dump] Saving {size} levels (final)...")
    dump_buffer(final_train_state, 0)  # tag = "_final"

    buffer_levels = jax.tree_util.tree_map(lambda x: x[:size], sampler["levels"])
    buffer_scores = np.asarray(sampler["scores"][:size])
    tokens = jax.vmap(level_to_tokens)(buffer_levels)

    # === Post-training: evaluate agent on buffer levels ===
    if config.get("skip_post_eval"):
        print("[Post-training] Skipped (--skip_post_eval). Use evaluate_buffer.py on the checkpoint later.")
        wandb.finish()
        return

    print(f"\n[Post-training] Evaluating agent on {size} buffer levels...")
    eval_env_post = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)
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
    # === ENV CONFIG ===
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
    group.add_argument("--buffer_dump_interval", type=int, default=10000,
                       help="Dump PLR buffer (VAE token format) every N updates. 0 to disable periodic dumps.")
    group.add_argument("--skip_post_eval", action="store_true", default=False,
                       help="Skip post-training buffer evaluation, rendering, and PCA (run evaluate_buffer.py separately)")
    # === DIVERSITY METRICS CONFIG ===
    group.add_argument("--diversity_log_interval", type=int, default=100,
                       help="Compute diversity metrics every N eval steps (0 to disable)")
    group.add_argument("--diversity_sample_size", type=int, default=20,
                       help="Number of buffer levels to subsample for pairwise diversity metrics")

    config = vars(parser.parse_args())
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])
    config["group_name"] = ''.join([str(config[key]) for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])])
    
    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'
    
    # wandb.login()
    main(config, project=config["project"])
