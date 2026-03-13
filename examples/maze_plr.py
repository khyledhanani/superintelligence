import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"

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
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss, accumulate_rollout_stats
from jaxued.wrappers import AutoReplayWrapper
import chex
import yaml
import pickle
import sys
from enum import IntEnum

# VAE + CMA-ES imports (conditional on --use_cmaes flag)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vae'))
from vae_model import CluttrVAE
from vae_level_utils import decode_latent_to_levels, level_to_tokens, tokens_to_level
from cmaes_manager import CMAESManager
from cenie_scorer import CENIEScorer

class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1


def fit_pca(latent_means, n_components, fitness_scores=None):
    """Fit PCA on latent means and return (pca_mean, pca_components).

    If fitness_scores is provided, uses fitness-aware PCA:
      - PC1 = direction of maximum fitness change (linear regression)
      - PC2...k = max-variance directions orthogonal to fitness direction
    Otherwise, standard PCA (max-variance directions).

    Args:
        latent_means: (N, D) array of encoded latent means.
        n_components: number of principal components to keep.
        fitness_scores: optional (N,) array of fitness/score values.

    Returns:
        pca_mean: (D,) mean of the latent vectors.
        pca_components: (n_components, D) top principal component directions.
    """
    pca_mean = jnp.mean(latent_means, axis=0)
    centered = latent_means - pca_mean

    if fitness_scores is not None and n_components >= 1:
        # Fitness-aware: PC1 = fitness gradient direction
        fitness_centered = fitness_scores - jnp.mean(fitness_scores)
        # Direction of max fitness change: w = Z^T @ f (unnormalized gradient)
        w = centered.T @ fitness_centered  # (D,)
        w_norm = jnp.linalg.norm(w)
        w = w / (w_norm + 1e-8)

        # Correlation between fitness and projection onto this direction
        proj = centered @ w  # (N,)
        corr = float(jnp.corrcoef(proj, fitness_centered)[0, 1])
        print(f"[PCA] PC0 = fitness gradient direction (corr with fitness: {corr:.3f})")

        components = [w]

        if n_components > 1:
            # Project out the fitness direction from the data
            residual = centered - (centered @ w[:, None]) @ w[None, :]
            # Standard PCA on residuals for remaining components
            _, S, Vt = jnp.linalg.svd(residual, full_matrices=False)
            for i in range(n_components - 1):
                components.append(Vt[i])
            explained = S[: n_components - 1] ** 2
            total = jnp.sum(S ** 2)
            print(f"[PCA] PC1-{n_components-1}: {float(jnp.sum(explained) / total) * 100:.1f}% "
                  f"of residual variance")

        components = jnp.stack(components)
    else:
        # Standard PCA
        _, S, Vt = jnp.linalg.svd(centered, full_matrices=False)
        components = Vt[:n_components]
        explained = S[:n_components] ** 2
        total = jnp.sum(S ** 2)
        explained_ratio = jnp.sum(explained) / total
        print(f"[PCA] Top {n_components}/{latent_means.shape[1]} components explain "
              f"{float(explained_ratio) * 100:.1f}% of variance")

    return pca_mean, components


def encode_levels_to_means(vae_encode_fn, tokens, batch_size=256):
    """Encode token sequences through VAE and return latent means."""
    all_means = []
    for i in range(0, len(tokens), batch_size):
        m, _ = vae_encode_fn(tokens[i:i + batch_size])
        all_means.append(m)
    return jnp.concatenate(all_means, axis=0)

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    es_state: chex.ArrayTree = struct.field(pytree_node=True)
    es_state_full: chex.ArrayTree = struct.field(pytree_node=True)  # For pca_start_after: full-dim CMA-ES
    cenie_gmm_params: chex.ArrayTree = struct.field(pytree_node=True)
    pca_mean: chex.Array = struct.field(pytree_node=True)
    pca_components: chex.Array = struct.field(pytree_node=True)
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
        import subprocess, shutil
        gcloud_bin = shutil.which("gcloud") or "/cs/student/project_msc/2025/csml/rhautier/google-cloud-sdk/bin/gcloud"
        dest = f"gs://{gcs_bucket}/{gcs_path}"
        subprocess.run([gcloud_bin, "storage", "cp", local_path, dest], check=True)
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
            import subprocess, tempfile, shutil
            gcloud_bin = shutil.which("gcloud") or "/cs/student/project_msc/2025/csml/rhautier/google-cloud-sdk/bin/gcloud"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(config_json)
                tmp_path = f.name
            subprocess.run([gcloud_bin, "storage", "cp", tmp_path, f"{overall_save_dir}/config.json"], check=True)
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

def mna_score(dones, advantages, incomplete_value=-jnp.inf):
    """Maximum Novelty Approximation: mean of clipped negative advantages.

    MNA = mean(-min(advantages, 0)) over the trajectory.
    High score = agent performs worse than expected = novel/challenging level.
    From: "Dynamic Environment Generation for UED" (Mead et al.)
    """
    mean_scores, _, episode_count = accumulate_rollout_stats(
        dones, -jnp.minimum(advantages, 0), time_average=True
    )
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)

def compute_score(config, dones, values, max_returns, advantages):
    """Compute regret-based scores (MaxMC, PVL, or MNA). Used directly or as regret component for CENIE."""
    if config['score_function'] in ("MaxMC", "cenie"):
        # CENIE uses MaxMC as its regret component
        return max_mc(dones, values, max_returns)
    elif config['score_function'] == "pvl":
        return positive_value_loss(dones, advantages)
    elif config['score_function'] == "mna":
        return mna_score(dones, advantages)
    elif config['score_function'] == "sfl":
        # SFL doesn't use regret-based scores; return zeros as placeholder
        # (actual SFL scores computed separately via multi-rollout eval)
        return jnp.zeros(dones.shape[1])
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
    if config.get("use_dred"):
        tags.append("DRED")
    if config.get("score_function") == "sfl":
        tags.append("SFL")
    elif config.get("score_function") == "cenie":
        tags.append("CENIE")
    elif config.get("score_function") == "mna":
        tags.append("MNA")
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
    if config.get("use_dred"):
        wandb.define_metric("dred/*", step_metric="num_updates")

    # --- VAE setup (shared by CMA-ES and DRED) ---
    vae_decode_fn = None
    vae_encode_fn = None
    cmaes_mgr = None
    _needs_vae = config["use_cmaes"] or config.get("use_dred")
    if _needs_vae:
        assert config["vae_checkpoint_path"] is not None, "--vae_checkpoint_path required when --use_cmaes or --use_dred"
        assert config["vae_config_path"] is not None, "--vae_config_path required when --use_cmaes or --use_dred"

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

        # Build pure encode function: tokens (batch, seq_len) -> (mean, logvar)
        def vae_encode_fn(tokens):
            return vae.apply({"params": vae_params}, tokens, train=False, method=vae.encode)

    if config.get("use_dred"):
        print(f"[DRED] VAE loaded from {config['vae_checkpoint_path']}")
        print(f"[DRED] latent_dim={vae_cfg['latent_dim']}, interpolation-based level generation")

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


        # DRED metrics (averaged over the eval_freq training steps)
        if config.get("use_dred") and "dred/valid_structure_pct" in stats:
            valid_pct = np.array(stats["dred/valid_structure_pct"])
            dr_mask = valid_pct > 0
            if dr_mask.any():
                log_dict["dred/valid_structure_pct"] = float(valid_pct[dr_mask].mean())
                log_dict["dred/solvable_pct"] = float(np.array(stats["dred/solvable_pct"])[dr_mask].mean())
                log_dict["dred/mean_score"] = float(np.array(stats["dred/mean_score"])[dr_mask].mean())

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
                log_dict["cmaes/sigma_resets"] = float(np.array(stats["cmaes/sigma_reset"])[dr_mask].sum())
                log_dict["cmaes/periodic_resets"] = float(np.array(stats["cmaes/periodic_reset"])[dr_mask].sum())

        wandb.log(log_dict)
    
    # Setup the environment
    env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    eval_env = env
    sample_random_level = make_level_generator(env.max_height, env.max_width, config["n_walls"])
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    mutate_level = make_level_mutator_minimax(100)

    # --- Active latent dimension detection ---
    active_dims = None  # None = use all dims
    full_latent_dim = vae_cfg["latent_dim"] if _needs_vae else None
    if config.get("cmaes_delayed_start") and _needs_vae and config["use_cmaes"]:
        # Delayed start: skip KL filtering at init, will be computed when buffer is full
        print(f"[KL-filter] Deferred (will compute when PCA activates)")
    elif _needs_vae and config["use_cmaes"] and config["cmaes_kl_threshold"] > 0:
        kl_data_path = config.get("cmaes_kl_data")
        ws_buffer_path = config.get("warmstart_buffer")
        if kl_data_path and os.path.exists(kl_data_path):
            print(f"[KL-filter] Loading token data from {kl_data_path}...")
            sample_tokens = jnp.array(np.load(kl_data_path)[:config["cmaes_kl_samples"]])
        elif ws_buffer_path and os.path.exists(ws_buffer_path):
            print(f"[KL-filter] Using warm-start buffer for KL estimation...")
            ws_buf_kl = np.load(ws_buffer_path, allow_pickle=True)
            sample_tokens = jnp.array(ws_buf_kl["tokens"][:config["cmaes_kl_samples"]])
            print(f"[KL-filter] Loaded {sample_tokens.shape[0]} tokens from buffer")
        else:
            if kl_data_path:
                print(f"[KL-filter] WARNING: {kl_data_path} not found, falling back to random levels")
            print(f"[KL-filter] Generating {config['cmaes_kl_samples']} random levels...")
            sample_rng = jax.random.PRNGKey(0)
            sample_levels = jax.vmap(sample_random_level)(jax.random.split(sample_rng, config["cmaes_kl_samples"]))
            sample_tokens = jax.vmap(level_to_tokens)(sample_levels)  # (n_samples, 52)
        print(f"[KL-filter] Estimating per-dim KL with {sample_tokens.shape[0]} levels...")
        # Encode in batches to avoid OOM
        batch_size = 256
        all_means, all_logvars = [], []
        for i in range(0, len(sample_tokens), batch_size):
            m, lv = vae_encode_fn(sample_tokens[i:i+batch_size])
            all_means.append(m)
            all_logvars.append(lv)
        mean_enc = jnp.concatenate(all_means, axis=0)
        logvar_enc = jnp.concatenate(all_logvars, axis=0)
        # Per-dim KL: -0.5 * (1 + logvar - mean² - exp(logvar)), averaged over samples
        kl_per_dim = -0.5 * (1 + logvar_enc - mean_enc**2 - jnp.exp(logvar_enc))
        kl_per_dim = jnp.mean(kl_per_dim, axis=0)  # (latent_dim,)

        active_mask = kl_per_dim > config["cmaes_kl_threshold"]
        active_dims = jnp.where(active_mask)[0]
        n_active = int(active_dims.shape[0])
        n_total = int(full_latent_dim)

        print(f"[KL-filter] Per-dim KL stats: min={float(kl_per_dim.min()):.4f}, "
              f"max={float(kl_per_dim.max()):.4f}, mean={float(kl_per_dim.mean()):.4f}")
        print(f"[KL-filter] Active dims: {n_active}/{n_total} (threshold={config['cmaes_kl_threshold']})")
        print(f"[KL-filter] Active indices: {np.array(active_dims).tolist()}")
        print(f"[KL-filter] Dead dims KL: {np.array(kl_per_dim[~active_mask]).tolist()}")

        if n_active == 0:
            print("[KL-filter] WARNING: No active dims found! Falling back to all dims.")
            active_dims = None
        elif n_active == n_total:
            print("[KL-filter] All dims active, no filtering needed.")
            active_dims = None

        # Log to wandb
        wandb.run.summary["kl_per_dim"] = np.array(kl_per_dim).tolist()
        wandb.run.summary["active_dims"] = np.array(active_dims).tolist() if active_dims is not None else list(range(n_total))
        wandb.run.summary["n_active_dims"] = n_active if active_dims is not None else n_total

    # Wrap vae_decode_fn to expand from active subspace to full latent space
    if active_dims is not None:
        _original_decode_fn = vae_decode_fn
        _active_dims = active_dims
        _full_dim = full_latent_dim

        def vae_decode_fn(z):
            """Expand reduced latent vector to full space, then decode."""
            if z.ndim == 1:
                z_full = jnp.zeros(_full_dim)
                z_full = z_full.at[_active_dims].set(z)
            else:
                z_full = jnp.zeros((z.shape[0], _full_dim))
                z_full = z_full.at[:, _active_dims].set(z)
            return _original_decode_fn(z_full)

        cmaes_latent_dim = int(active_dims.shape[0])
    else:
        cmaes_latent_dim = vae_cfg["latent_dim"] if _needs_vae else None

    # --- PCA dimensionality reduction (overrides KL filtering if both set) ---
    pca_mean_init = None
    pca_components_init = None
    use_pca = _needs_vae and config["use_cmaes"] and config.get("cmaes_pca_dims", 0) > 0
    if use_pca:
        pca_dims = config["cmaes_pca_dims"]
        # Skip initial PCA fit if delayed start or warm-start buffer (PCA will be fit later)
        if config.get("cmaes_delayed_start") or config.get("cmaes_pca_start_after", 0) > 0:
            _start_msg = f"after {config['cmaes_pca_start_after']} updates" if config.get("cmaes_pca_start_after", 0) > 0 else "when buffer is full"
            print(f"[PCA] Deferred ({_start_msg})")
            full_d = vae_cfg["latent_dim"]
            pca_mean_init = jnp.zeros(full_d)
            pca_components_init = jnp.eye(pca_dims, full_d)
        elif config.get("warmstart_buffer"):
            print(f"[PCA] Deferring initial PCA fit to warm-start buffer")
            # Create placeholder PCA arrays with correct shapes (will be overwritten)
            # If KL filtering is active, PCA operates in active-dim subspace
            pca_space_dim = len(active_dims) if active_dims is not None else vae_cfg["latent_dim"]
            pca_mean_init = jnp.zeros(pca_space_dim)
            pca_components_init = jnp.eye(pca_dims, pca_space_dim)
        else:
            pca_data_path = config.get("cmaes_pca_data") or config.get("cmaes_kl_data")
            n_pca_samples = config.get("cmaes_kl_samples", 5000)
            if pca_data_path and os.path.exists(pca_data_path):
                print(f"[PCA] Loading token data from {pca_data_path}...")
                pca_tokens = jnp.array(np.load(pca_data_path)[:n_pca_samples])
            else:
                if pca_data_path:
                    print(f"[PCA] WARNING: {pca_data_path} not found, falling back to random levels")
                print(f"[PCA] Generating {n_pca_samples} random levels...")
                sample_rng = jax.random.PRNGKey(0)
                sample_levels = jax.vmap(sample_random_level)(jax.random.split(sample_rng, n_pca_samples))
                pca_tokens = jax.vmap(level_to_tokens)(sample_levels)

            print(f"[PCA] Encoding {pca_tokens.shape[0]} levels through VAE...")
            pca_latent_means = encode_levels_to_means(vae_encode_fn, pca_tokens)
            # If KL filtering is active, fit PCA only on active dims
            if active_dims is not None:
                print(f"[PCA] Restricting PCA to {len(active_dims)} KL-active dims")
                pca_latent_means = pca_latent_means[:, active_dims]
            pca_mean_init, pca_components_init = fit_pca(pca_latent_means, pca_dims)

        # PCA sets the CMA-ES search dim
        cmaes_latent_dim = pca_dims
        # NOTE: if KL filtering is active, keep active_dims so the decode wrapper
        # expands from active subspace to full 64 dims. PCA will operate within
        # the active subspace (not full latent space).

        wandb.run.summary["pca_dims"] = pca_dims
        if not config.get("warmstart_buffer") and not config.get("cmaes_delayed_start") and not config.get("cmaes_pca_start_after", 0) > 0:
            # Log explained variance (only when we have pca_latent_means from initial fit)
            wandb.run.summary["pca_explained_var"] = float(
                jnp.sum(jnp.linalg.svd(pca_latent_means - pca_mean_init, full_matrices=False)[1][:pca_dims]**2)
                / jnp.sum(jnp.linalg.svd(pca_latent_means - pca_mean_init, full_matrices=False)[1]**2)
            )

    # --- CMA-ES setup ---
    cmaes_mgr = None
    cmaes_mgr_full = None  # For pca_start_after: full-dim CMA-ES before PCA activates
    if config["use_cmaes"]:
        cmaes_mgr = CMAESManager(
            popsize=config["num_train_envs"],
            latent_dim=cmaes_latent_dim,
            sigma_init=config["cmaes_sigma_init"],
        )
        # For pca_start_after: also create a full-dim CMA-ES for phase 1
        if config.get("cmaes_pca_start_after", 0) > 0 and use_pca:
            kl_dim = len(active_dims) if active_dims is not None else vae_cfg["latent_dim"]
            cmaes_mgr_full = CMAESManager(
                popsize=config["num_train_envs"],
                latent_dim=kl_dim,
                sigma_init=config["cmaes_sigma_init"],
            )
            print(f"[CMA-ES] Phase 1: full search_dim={kl_dim} (before PCA at {config['cmaes_pca_start_after']} updates)")
            print(f"[CMA-ES] Phase 2: PCA search_dim={cmaes_latent_dim}")
        else:
            print(f"[CMA-ES] search_dim={cmaes_latent_dim}, full_latent_dim={full_latent_dim}, popsize={config['num_train_envs']}")
        print(f"[CMA-ES] VAE loaded from {config['vae_checkpoint_path']}")
        if active_dims is not None:
            print(f"[CMA-ES] KL filtering: {len(active_dims)}/{full_latent_dim} active dims")
        if use_pca and not config.get("cmaes_pca_start_after", 0) > 0:
            pca_input_dim = len(active_dims) if active_dims is not None else full_latent_dim
            print(f"[CMA-ES] PCA: {pca_input_dim} dims -> {pca_dims} PCs")

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
    
    # --- SFL: multi-rollout learnability scoring ---
    def compute_sfl_scores(rng, train_state, levels, max_returns):
        """Estimate learnability = p * (1-p) via multi-rollout evaluation.

        Uses the training rollout result (max_returns > 0) plus additional
        evaluation rollouts to estimate the agent's success rate p on each level.
        """
        # Success from the training rollout
        train_success = (max_returns > 0).astype(jnp.float32)

        # Additional eval rollouts using the unwrapped env
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

        # Combine training rollout + eval rollouts
        all_successes = jnp.concatenate([train_success[None], eval_successes], axis=0)
        p = all_successes.mean(axis=0)
        return p * (1 - p)

    # --- CENIE: novelty + regret scoring (pure JAX) ---
    cenie_scorer = None
    cenie_obs_act_dim = config["agent_view_size"] ** 2 * 3 + 1  # flattened obs image + 1 action
    if config["score_function"] == "cenie":
        cenie_scorer = CENIEScorer(
            buffer_size=config["cenie_buffer_size"],
            n_components=config["cenie_num_components"],
            alpha=config["cenie_alpha"],
            temperature=config["temperature"],
        )

    # Initialize CENIE GMM params (placeholder zeros; updated on host between eval steps)
    if config["score_function"] == "cenie":
        cenie_gmm_init = {
            'means': jnp.zeros((config["cenie_num_components"], cenie_obs_act_dim)),
            'log_vars': jnp.zeros((config["cenie_num_components"], cenie_obs_act_dim)),
            'log_weights': jnp.full((config["cenie_num_components"],),
                                     -jnp.log(config["cenie_num_components"])),  # uniform
            'fitted': jnp.bool_(False),
        }
    else:
        cenie_gmm_init = None

    def _gmm_log_likelihood_diag(x, means, log_vars, log_weights):
        """Compute log p(x) under a diagonal-covariance GMM. Pure JAX.

        Args:
            x: (D,) single data point
            means: (K, D) component means
            log_vars: (K, D) log-variances per component
            log_weights: (K,) log mixing weights (log-normalized)
        Returns:
            scalar log p(x)
        """
        diff = x[None, :] - means  # (K, D)
        # Clip log_vars for numerical stability: var in [exp(-20), exp(20)]
        log_vars_safe = jnp.clip(log_vars, -20.0, 20.0)
        inv_vars = jnp.exp(-log_vars_safe)  # (K, D)
        D = means.shape[1]
        log_norm = -0.5 * (D * jnp.log(2 * jnp.pi) + log_vars_safe.sum(axis=-1))  # (K,)
        log_exp = -0.5 * (diff ** 2 * inv_vars).sum(axis=-1)  # (K,)
        return jax.scipy.special.logsumexp(log_weights + log_norm + log_exp)

    def _jax_rankdata(scores):
        """Rank scores descending (highest score = rank 1). Pure JAX."""
        order = jnp.argsort(-scores)
        ranks = jnp.zeros_like(order).at[order].set(jnp.arange(1, scores.shape[0] + 1))
        return ranks.astype(jnp.float32)

    def compute_cenie_scores(obs, actions, regret_scores, cenie_params):
        """Compute CENIE combined novelty+regret scores. Pure JAX, zero callbacks.

        Buffer accumulation happens via metrics pipeline (between eval steps on host).
        Only the NLL scoring runs inside JIT.
        """
        # Flatten obs to (T, N, D) and concatenate with actions
        obs_flat = obs.image.reshape(config["num_steps"], config["num_train_envs"], -1)
        actions_float = actions[..., None].astype(jnp.float32)
        obs_actions = jnp.concatenate([obs_flat, actions_float], axis=-1)  # (T, N, D)

        # Compute per-env novelty as mean NLL under GMM (pure JAX)
        means = cenie_params['means']
        log_vars = cenie_params['log_vars']
        log_weights = cenie_params['log_weights']
        fitted = cenie_params['fitted']

        # Compute NLL for each (timestep, env) pair, then average over timesteps per env
        T, N, D = config["num_steps"], config["num_train_envs"], cenie_obs_act_dim
        flat_points = obs_actions.reshape(-1, D)  # (T*N, D)
        nll_flat = -jax.vmap(
            lambda x: _gmm_log_likelihood_diag(x, means, log_vars, log_weights)
        )(flat_points)  # (T*N,)
        nll_per_env = nll_flat.reshape(T, N).mean(axis=0)  # (N,)

        # Rank-based combination (CENIE Eq. 4-5)
        alpha = config["cenie_alpha"]
        temperature = config["temperature"]
        r_regret = _jax_rankdata(regret_scores)
        r_novelty = _jax_rankdata(nll_per_env)

        p_r = (1.0 / r_regret) ** (1.0 / temperature)
        p_r = p_r / p_r.sum()
        p_n = (1.0 / r_novelty) ** (1.0 / temperature)
        p_n = p_n / p_n.sum()

        combined = alpha * p_n + (1 - alpha) * p_r

        # If GMM not fitted yet, fall back to pure regret
        return jnp.where(fitted, combined, regret_scores)

    def compute_level_scores(rng, train_state, levels, obs, actions,
                             dones, values, max_returns, advantages):
        """Unified score computation dispatching to MaxMC/PVL, SFL, or CENIE."""
        if config["score_function"] == "sfl":
            return compute_sfl_scores(rng, train_state, levels, max_returns)
        elif config["score_function"] == "cenie":
            regret_scores = compute_score(config, dones, values, max_returns, advantages)
            return compute_cenie_scores(obs, actions, regret_scores, train_state.cenie_gmm_params)
        else:
            return compute_score(config, dones, values, max_returns, advantages)

    # --- CMA-ES population archive callback (runs on host via jax.debug.callback) ---
    def _save_cmaes_population(z_population, scores, es_mean, num_dr_updates, should_reset):
        """Save CMA-ES population archive before reset. Called from inside JIT via jax.debug.callback."""
        if not bool(should_reset):
            return
        dr_num = int(num_dr_updates)
        save_dir = os.path.join("/tmp", "cmaes_populations", f"{config['run_name']}", str(config["seed"]))
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"pre_reset_{dr_num}.npz")
        np.savez_compressed(path,
                            z_population=np.asarray(z_population),
                            scores=np.asarray(scores),
                            es_mean=np.asarray(es_mean))
        print(f"[CMA-ES] Population archive saved: {path} ({len(scores)} candidates)")
        if config.get("gcs_bucket"):
            gcs_base = f"{config['gcs_prefix']}/cmaes_populations/{config['run_name']}/{config['seed']}"
            _upload_to_gcs(path, config["gcs_bucket"], f"{gcs_base}/pre_reset_{dr_num}.npz")

    # Initialize PCA state OUTSIDE jit
    if pca_mean_init is None:
        # Dummy PCA arrays when PCA is not used (shape doesn't matter, never read)
        _pca_mean_init = jnp.zeros(1)
        _pca_components_init = jnp.zeros((1, 1))
    else:
        _pca_mean_init = pca_mean_init
        _pca_components_init = pca_components_init

    # Initialize CMA-ES state OUTSIDE jit to avoid tracing issues with evosax
    es_state_full_init = None
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
        # For pca_start_after: initialize full-dim CMA-ES state for phase 1
        if cmaes_mgr_full is not None:
            es_state_full_init = cmaes_mgr_full.initialize(jax.random.PRNGKey(42))
            print(f"[CMA-ES] Initialized es_state_full: mean.shape={es_state_full_init.mean.shape}")
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
            es_state_full=es_state_full_init if es_state_full_init is not None else es_state_init,
            cenie_gmm_params=cenie_gmm_init,
            pca_mean=_pca_mean_init,
            pca_components=_pca_components_init,
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

                def _cmaes_generate(rng_ask):
                    z_pop, es_new = cmaes_mgr.ask(rng_ask, es_state)
                    if use_pca:
                        z_full = train_state.pca_mean + z_pop @ train_state.pca_components
                        levels = decode_latent_to_levels(vae_decode_fn, z_full, rng_decode)
                    else:
                        levels = decode_latent_to_levels(vae_decode_fn, z_pop, rng_decode)
                    return levels, es_new, z_pop

                def _random_generate(rng_ask):
                    levels = jax.vmap(sample_random_level)(jax.random.split(rng_ask, config["num_train_envs"]))
                    dummy_z = jnp.zeros((config["num_train_envs"], cmaes_latent_dim))
                    return levels, es_state, dummy_z

                if config.get("cmaes_pca_start_after", 0) > 0 and cmaes_mgr_full is not None:
                    # Phase 1: CMA-ES searches full KL-filtered space
                    # Phase 2: CMA-ES searches PCA space
                    pca_active = train_state.num_dr_updates >= config["cmaes_pca_start_after"]

                    def _cmaes_full_generate(rng_ask):
                        es_full = train_state.es_state_full
                        z_pop, es_full_new = cmaes_mgr_full.ask(rng_ask, es_full)
                        levels = decode_latent_to_levels(vae_decode_fn, z_pop, rng_decode)
                        # Return dummy pca-dim z and unchanged es_state for the pca manager
                        dummy_z = jnp.zeros((config["num_train_envs"], cmaes_latent_dim))
                        return levels, es_state, dummy_z, es_full_new

                    def _cmaes_pca_generate(rng_ask):
                        z_pop, es_new = cmaes_mgr.ask(rng_ask, es_state)
                        z_full = train_state.pca_mean + z_pop @ train_state.pca_components
                        levels = decode_latent_to_levels(vae_decode_fn, z_full, rng_decode)
                        return levels, es_new, z_pop, train_state.es_state_full

                    new_levels, es_state, z_population, es_state_full = jax.lax.cond(
                        pca_active, _cmaes_pca_generate, _cmaes_full_generate, rng_ask)
                elif config.get("cmaes_delayed_start"):
                    cmaes_ready = sampler["size"] >= config["level_buffer_capacity"]
                    new_levels, es_state, z_population = jax.lax.cond(
                        cmaes_ready, _cmaes_generate, _random_generate, rng_ask)
                    es_state_full = train_state.es_state_full
                else:
                    new_levels, es_state, z_population = _cmaes_generate(rng_ask)
                    es_state_full = train_state.es_state_full
            elif config.get("use_dred"):
                # DRED: interpolate pairs from buffer in VAE latent space
                # Fall back to random generation if buffer has < 2 levels
                def dred_interpolate(rng):
                    rng, rng_idx_a, rng_idx_b, rng_alpha, rng_z, rng_decode = jax.random.split(rng, 6)
                    N = config["num_train_envs"]
                    buf_size = jnp.maximum(sampler["size"], 1)

                    # Sample two sets of indices for pairs
                    idx_a = jax.random.randint(rng_idx_a, (N,), 0, buf_size)
                    idx_b = jax.random.randint(rng_idx_b, (N,), 0, buf_size)

                    # Extract levels and convert to tokens
                    levels_a = jax.tree_util.tree_map(lambda x: x[idx_a], sampler["levels"])
                    levels_b = jax.tree_util.tree_map(lambda x: x[idx_b], sampler["levels"])
                    tokens_a = jax.vmap(level_to_tokens)(levels_a)  # (N, 52)
                    tokens_b = jax.vmap(level_to_tokens)(levels_b)  # (N, 52)

                    # Encode to latent space
                    mean_a, logvar_a = vae_encode_fn(tokens_a)  # each (N, 64)
                    mean_b, logvar_b = vae_encode_fn(tokens_b)  # each (N, 64)

                    # Random interpolation coefficient per pair
                    alpha = jax.random.uniform(rng_alpha, (N, 1))

                    # Interpolate latent distributions
                    mean_interp = alpha * mean_a + (1 - alpha) * mean_b
                    logvar_interp = alpha * logvar_a + (1 - alpha) * logvar_b

                    # Sample from interpolated distribution
                    std_interp = jnp.exp(0.5 * logvar_interp)
                    eps = jax.random.normal(rng_z, mean_interp.shape)
                    z = mean_interp + eps * std_interp

                    return decode_latent_to_levels(vae_decode_fn, z, rng_decode)

                def random_generate(rng):
                    return jax.vmap(sample_random_level)(jax.random.split(rng, config["num_train_envs"]))

                rng, rng_gen = jax.random.split(rng)
                new_levels = jax.lax.cond(
                    sampler["size"] >= 2,
                    dred_interpolate,
                    random_generate,
                    rng_gen,
                )
            else:
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
            rng, rng_score = jax.random.split(rng)
            scores = compute_level_scores(rng_score, train_state, new_levels, obs, actions,
                                          dones, values, max_returns, advantages)

            # CMA-ES: tell fitness and insert into buffer
            if config["use_cmaes"]:
                # CMA-ES minimizes; negate scores so high-regret = low fitness
                rng, rng_tell = jax.random.split(rng)

                def _cmaes_tell(args):
                    rng_tell, z_pop, scores_neg, es, num_dr = args
                    es = cmaes_mgr.tell(rng_tell, z_pop, scores_neg, es)

                    # Reset on sigma collapse (adaptive) or fixed interval (fallback)
                    sigma_collapsed = (config["cmaes_sigma_min"] > 0) & (es.std < config["cmaes_sigma_min"])
                    periodic_reset = (num_dr % config["cmaes_reset_interval"]) == 0
                    should_reset = sigma_collapsed | periodic_reset

                    # Archive population before reset for latent visualization
                    if config.get("save_cmaes_populations", True):
                        jax.debug.callback(
                            _save_cmaes_population,
                            z_pop, -scores_neg, es.mean,
                            num_dr, should_reset,
                        )

                    rng_reset_es = jax.random.fold_in(rng_tell, 999)
                    fresh_es = cmaes_mgr.initialize(rng_reset_es)
                    es = jax.tree_util.tree_map(
                        lambda fresh, old: jnp.where(should_reset, fresh, old),
                        fresh_es, es
                    )
                    return es

                def _skip_tell(args):
                    return args[3]  # return es_state unchanged

                if config.get("cmaes_pca_start_after", 0) > 0 and cmaes_mgr_full is not None:
                    pca_active = train_state.num_dr_updates >= config["cmaes_pca_start_after"]

                    def _cmaes_full_tell(args):
                        rng_t, z_pop_full, scores_neg, es_full, num_dr = args
                        # z_pop_full is dummy (pca-dim) — we need the actual full-dim z
                        # But we can't access it here. Instead, skip tell for pca es_state
                        # and tell the full es_state separately
                        return es_full  # unchanged — full tell handled below

                    # Phase 2: tell PCA CMA-ES
                    es_state = jax.lax.cond(
                        pca_active, _cmaes_tell, _skip_tell,
                        (rng_tell, z_population, -scores, es_state, train_state.num_dr_updates))

                    # Phase 1: tell full CMA-ES (need full-dim z_population from _cmaes_full_generate)
                    # We re-ask to get the z_population used (deterministic given same rng)
                    # Actually, the full es_state is updated in the generate step already via es_state_full
                    # We need to tell es_state_full with the scores
                    def _full_tell(args):
                        rng_t, scores_neg, es_full, num_dr = args
                        # Re-ask to get z_population (same rng produces same z)
                        z_pop, _ = cmaes_mgr_full.ask(rng_ask, train_state.es_state_full)
                        es_full = cmaes_mgr_full.tell(rng_t, z_pop, scores_neg, es_full)
                        sigma_collapsed = (config["cmaes_sigma_min"] > 0) & (es_full.std < config["cmaes_sigma_min"])
                        periodic_reset = (num_dr % config["cmaes_reset_interval"]) == 0
                        should_reset = sigma_collapsed | periodic_reset
                        rng_reset_es = jax.random.fold_in(rng_t, 999)
                        fresh_es = cmaes_mgr_full.initialize(rng_reset_es)
                        es_full = jax.tree_util.tree_map(
                            lambda fresh, old: jnp.where(should_reset, fresh, old),
                            fresh_es, es_full)
                        return es_full

                    def _full_skip(args):
                        return args[2]  # return es_state_full unchanged

                    es_state_full = jax.lax.cond(
                        pca_active, _full_skip, _full_tell,
                        (rng_tell, -scores, es_state_full, train_state.num_dr_updates))

                elif config.get("cmaes_delayed_start"):
                    cmaes_ready = sampler["size"] >= config["level_buffer_capacity"]
                    es_state = jax.lax.cond(
                        cmaes_ready, _cmaes_tell, _skip_tell,
                        (rng_tell, z_population, -scores, es_state, train_state.num_dr_updates))
                else:
                    es_state = _cmaes_tell(
                        (rng_tell, z_population, -scores, es_state, train_state.num_dr_updates))

            # DRED: only insert solvable levels (agent got positive reward at least once)
            if config.get("use_dred"):
                is_solvable = max_returns > 0
                scores = jnp.where(is_solvable, scores, -jnp.inf)

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

            # DRED monitoring metrics
            if config.get("use_dred"):
                is_solvable = max_returns > 0
                metrics["dred/valid_structure_pct"] = is_valid.mean() * 100
                metrics["dred/solvable_pct"] = is_solvable.mean() * 100
                metrics["dred/mean_score"] = scores.mean()

            # CMA-ES monitoring metrics
            if config["use_cmaes"]:
                metrics["cmaes/valid_structure_pct"] = is_valid.mean() * 100
                metrics["cmaes/mean_fitness"] = scores.mean()
                metrics["cmaes/mean_episode_length"] = dones.sum(axis=0).mean()
                # For pca_start_after: use full es_state during phase 1, pca es_state during phase 2
                if config.get("cmaes_pca_start_after", 0) > 0 and cmaes_mgr_full is not None:
                    pca_active = train_state.num_dr_updates >= config["cmaes_pca_start_after"]
                    _report_sigma = jnp.where(pca_active, es_state.std, es_state_full.std)
                else:
                    _report_sigma = es_state.std
                # Step size (sigma) — tracks exploration vs convergence
                metrics["cmaes/sigma"] = _report_sigma
                # Spread of population in latent space (std of z-vectors across candidates)
                # For pca_start_after phase 1: z_population is dummy, use es_state_full.mean norm instead
                if config.get("cmaes_pca_start_after", 0) > 0 and cmaes_mgr_full is not None:
                    _pop_spread = jnp.where(pca_active, z_population.std(), es_state_full.std)
                    _z_norm = jnp.where(pca_active,
                        jnp.linalg.norm(z_population, axis=-1).mean(),
                        jnp.linalg.norm(es_state_full.mean))
                else:
                    _pop_spread = z_population.std()
                    _z_norm = jnp.linalg.norm(z_population, axis=-1).mean()
                metrics["cmaes/pop_spread"] = _pop_spread
                # Mean norm of latent vectors (how far from origin)
                metrics["cmaes/mean_z_norm"] = _z_norm
                # Track sigma-triggered resets (1.0 if reset happened this step, 0.0 otherwise)
                _sigma_collapsed = (config["cmaes_sigma_min"] > 0) & (_report_sigma < config["cmaes_sigma_min"])
                _periodic_reset = (train_state.num_dr_updates % config["cmaes_reset_interval"]) == 0
                metrics["cmaes/sigma_reset"] = jnp.where(_sigma_collapsed, 1.0, 0.0)
                metrics["cmaes/periodic_reset"] = jnp.where(_periodic_reset & ~_sigma_collapsed, 1.0, 0.0)

            # CENIE: pass subsampled obs_actions through metrics (no callbacks)
            if config["score_function"] == "cenie":
                obs_flat_last = obs.image[-1].reshape(config["num_train_envs"], -1)
                act_last = actions[-1, ..., None].astype(jnp.float32)
                metrics["cenie_sample"] = jnp.concatenate([obs_flat_last, act_last], axis=-1)  # (N, D)

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                es_state=es_state,
                es_state_full=es_state_full if config.get("cmaes_pca_start_after", 0) > 0 else train_state.es_state_full,
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
            rng, rng_score = jax.random.split(rng)
            scores = compute_level_scores(rng_score, train_state, levels, obs, actions,
                                          dones, values, max_returns, advantages)
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
            if config.get("use_dred"):
                metrics["dred/valid_structure_pct"] = jnp.float32(0.0)
                metrics["dred/solvable_pct"] = jnp.float32(0.0)
                metrics["dred/mean_score"] = jnp.float32(0.0)
            if config["use_cmaes"]:
                metrics["cmaes/valid_structure_pct"] = jnp.float32(0.0)
                metrics["cmaes/mean_fitness"] = jnp.float32(0.0)
                metrics["cmaes/mean_episode_length"] = jnp.float32(0.0)
                metrics["cmaes/sigma"] = jnp.float32(0.0)
                metrics["cmaes/pop_spread"] = jnp.float32(0.0)
                metrics["cmaes/mean_z_norm"] = jnp.float32(0.0)
                metrics["cmaes/sigma_reset"] = jnp.float32(0.0)
                metrics["cmaes/periodic_reset"] = jnp.float32(0.0)

            # CENIE: pass subsampled obs_actions through metrics (no callbacks)
            if config["score_function"] == "cenie":
                obs_flat_last = obs.image[-1].reshape(config["num_train_envs"], -1)
                act_last = actions[-1, ..., None].astype(jnp.float32)
                metrics["cenie_sample"] = jnp.concatenate([obs_flat_last, act_last], axis=-1)

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
            rng, rng_score = jax.random.split(rng)
            scores = compute_level_scores(rng_score, train_state, child_levels, obs, actions,
                                          dones, values, max_returns, advantages)
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
            if config.get("use_dred"):
                metrics["dred/valid_structure_pct"] = jnp.float32(0.0)
                metrics["dred/solvable_pct"] = jnp.float32(0.0)
                metrics["dred/mean_score"] = jnp.float32(0.0)
            if config["use_cmaes"]:
                metrics["cmaes/valid_structure_pct"] = jnp.float32(0.0)
                metrics["cmaes/mean_fitness"] = jnp.float32(0.0)
                metrics["cmaes/mean_episode_length"] = jnp.float32(0.0)
                metrics["cmaes/sigma"] = jnp.float32(0.0)
                metrics["cmaes/pop_spread"] = jnp.float32(0.0)
                metrics["cmaes/mean_z_norm"] = jnp.float32(0.0)
                metrics["cmaes/sigma_reset"] = jnp.float32(0.0)
                metrics["cmaes/periodic_reset"] = jnp.float32(0.0)

            # CENIE: pass subsampled obs_actions through metrics (no callbacks)
            if config["score_function"] == "cenie":
                obs_flat_last = obs.image[-1].reshape(config["num_train_envs"], -1)
                act_last = actions[-1, ..., None].astype(jnp.float32)
                metrics["cenie_sample"] = jnp.concatenate([obs_flat_last, act_last], axis=-1)

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

    # --- Warm-start: load agent params and/or buffer from a previous run ---
    if config.get("warmstart_checkpoint"):
        print(f"[Warmstart] Loading agent params from {config['warmstart_checkpoint']}...")
        ws_manager = ocp.CheckpointManager(
            config["warmstart_checkpoint"],
            item_handlers=ocp.StandardCheckpointHandler(),
        )
        ws_step = ws_manager.latest_step()
        ws_ckpt = ws_manager.restore(ws_step)
        ws_params = ws_ckpt["params"] if isinstance(ws_ckpt, dict) and "params" in ws_ckpt else ws_ckpt.params
        train_state = train_state.replace(params=ws_params)
        print(f"[Warmstart] Loaded agent params from step {ws_step}")

    if config.get("warmstart_buffer"):
        print(f"[Warmstart] Loading buffer from {config['warmstart_buffer']}...")
        ws_buf = np.load(config["warmstart_buffer"], allow_pickle=True)
        ws_tokens = ws_buf["tokens"]  # (N, 52)
        ws_scores = ws_buf["scores"]  # (N,)
        ws_size = int(ws_buf["size"])
        print(f"[Warmstart] Buffer has {ws_size} levels")

        # Convert tokens back to levels and insert into sampler in batches
        ws_levels = jax.vmap(tokens_to_level)(jnp.array(ws_tokens))
        sampler = train_state.sampler
        batch_sz = config["num_train_envs"]
        for start in range(0, ws_size, batch_sz):
            end = min(start + batch_sz, ws_size)
            batch_levels = jax.tree_util.tree_map(lambda x: x[start:end], ws_levels)
            batch_scores = jnp.array(ws_scores[start:end])
            # Pad to batch_sz if last batch is smaller
            if end - start < batch_sz:
                pad = batch_sz - (end - start)
                batch_levels = jax.tree_util.tree_map(
                    lambda x: jnp.concatenate([x, jnp.repeat(x[:1], pad, axis=0)]), batch_levels)
                batch_scores = jnp.concatenate([batch_scores, jnp.full(pad, -jnp.inf)])
            batch_max_returns = jnp.zeros(batch_sz)
            sampler, _ = level_sampler.insert_batch(sampler, batch_levels, batch_scores, {"max_return": batch_max_returns})
        train_state = train_state.replace(sampler=sampler)
        print(f"[Warmstart] Inserted {ws_size} levels into buffer")

        # If PCA is enabled, refit PCA on the warm-start buffer
        if use_pca and vae_encode_fn is not None:
            print(f"[Warmstart] Refitting PCA on {ws_size} buffer levels...")
            ws_means = encode_levels_to_means(vae_encode_fn, jnp.array(ws_tokens))
            # If KL filtering is active, restrict to active dims
            if active_dims is not None:
                print(f"[Warmstart PCA] Restricting to {len(active_dims)} KL-active dims")
                ws_means = ws_means[:, active_dims]
            ws_fitness = jnp.array(ws_scores) if config.get("cmaes_pca_fitness_aware") else None
            ws_pca_mean, ws_pca_components = fit_pca(ws_means, config["cmaes_pca_dims"], fitness_scores=ws_fitness)

            # Diagnostic: verify PCA projection
            print(f"[PCA debug] pca_mean shape: {ws_pca_mean.shape}, norm: {float(jnp.linalg.norm(ws_pca_mean)):.4f}")
            print(f"[PCA debug] pca_components shape: {ws_pca_components.shape}")
            for i in range(ws_pca_components.shape[0]):
                pc = ws_pca_components[i]
                print(f"[PCA debug]   PC{i}: norm={float(jnp.linalg.norm(pc)):.4f}, "
                      f"top-3 dims={np.argsort(np.abs(np.array(pc)))[-3:][::-1].tolist()}")
            # Test projection: project buffer means into PC space and back
            ws_projected = (ws_means - ws_pca_mean) @ ws_pca_components.T  # (N, n_pcs)
            ws_reconstructed = ws_pca_mean + ws_projected @ ws_pca_components  # (N, 64)
            recon_error = float(jnp.mean(jnp.linalg.norm(ws_means - ws_reconstructed, axis=1)))
            print(f"[PCA debug] Mean reconstruction error (buffer->PCA->back): {recon_error:.4f}")
            print(f"[PCA debug] PC-space range: min={float(ws_projected.min()):.2f}, max={float(ws_projected.max()):.2f}, "
                  f"std={float(ws_projected.std()):.2f}")

            train_state = train_state.replace(
                pca_mean=ws_pca_mean,
                pca_components=ws_pca_components,
            )
            # Reset CMA-ES for the new PCA basis
            new_es = cmaes_mgr.initialize(jax.random.PRNGKey(999))
            train_state = train_state.replace(es_state=new_es)

            # Check if sigma_init is appropriate for PC-space scale
            pc_std_per_dim = float(ws_projected.std())
            print(f"[Warmstart] CMA-ES sigma_init={config['cmaes_sigma_init']}, "
                  f"PC-space std={pc_std_per_dim:.2f}")
            if config["cmaes_sigma_init"] < pc_std_per_dim * 0.1:
                print(f"[Warmstart] WARNING: sigma_init is much smaller than PC-space spread! "
                      f"Consider --cmaes_sigma_init {pc_std_per_dim:.1f}")
            print(f"[Warmstart] PCA refit on buffer, CMA-ES reset")

    # Apply update counter offset for wandb continuity
    if config.get("warmstart_updates", 0) > 0:
        offset = config["warmstart_updates"]
        train_state = train_state.replace(
            num_dr_updates=train_state.num_dr_updates + offset,
            num_replay_updates=train_state.num_replay_updates,
            num_mutation_updates=train_state.num_mutation_updates,
        )
        print(f"[Warmstart] Update counter offset by {offset} (wandb starts at {offset})")

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

    # Track whether delayed-start KL+PCA has been initialized
    _delayed_start_initialized = False
    _delayed_active_dims = None  # set when delayed-start KL is computed

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

        # CENIE: accumulate obs_actions from metrics, refit GMM, inject params
        if config["score_function"] == "cenie" and cenie_scorer is not None:
            # Extract subsampled obs_actions from metrics (stacked by scan: eval_freq x N x D)
            if "cenie_sample" in metrics:
                cenie_samples = np.asarray(metrics["cenie_sample"])  # (eval_freq, N, D)
                # Reshape to (eval_freq * N, D) and add to buffer
                cenie_scorer.add_to_buffer(cenie_samples.reshape(1, -1, cenie_samples.shape[-1]))
            if (eval_step + 1) % config["cenie_refit_interval"] == 0:
                cenie_scorer.refit_gmm()
            if cenie_scorer.gmm is not None:
                new_gmm_params = cenie_scorer.get_jax_params(config["cenie_num_components"])
                rng_cur, ts_cur = runner_state
                ts_cur = ts_cur.replace(cenie_gmm_params=new_gmm_params)
                runner_state = (rng_cur, ts_cur)

        # Delayed-start / pca_start_after: initialize KL filtering + PCA
        _pca_start_after = config.get("cmaes_pca_start_after", 0)
        _use_delayed = config.get("cmaes_delayed_start") or _pca_start_after > 0
        if _use_delayed and use_pca and not _delayed_start_initialized:
            rng_cur, ts_cur = runner_state
            buf_size = int(ts_cur.sampler["size"])
            updates_so_far = (eval_step + 1) * config["eval_freq"]
            if _pca_start_after > 0:
                should_activate = updates_so_far >= _pca_start_after and buf_size >= config["cmaes_pca_dims"]
            else:
                should_activate = buf_size >= config["level_buffer_capacity"]
            if should_activate:
                print(f"[Delayed start] Buffer full ({buf_size} levels). Computing KL + PCA on buffer...")

                # Encode buffer levels
                buf_levels = jax.tree_util.tree_map(lambda x: x[:buf_size], ts_cur.sampler["levels"])
                buf_tokens = jax.vmap(level_to_tokens)(buf_levels)
                buf_means = encode_levels_to_means(vae_encode_fn, buf_tokens)

                # KL filtering on buffer
                full_d = vae_cfg["latent_dim"]
                if config["cmaes_kl_threshold"] > 0:
                    batch_size = 256
                    all_means_kl, all_logvars_kl = [], []
                    for i in range(0, len(buf_tokens), batch_size):
                        m, lv = vae_encode_fn(buf_tokens[i:i+batch_size])
                        all_means_kl.append(m)
                        all_logvars_kl.append(lv)
                    mean_enc = jnp.concatenate(all_means_kl, axis=0)
                    logvar_enc = jnp.concatenate(all_logvars_kl, axis=0)
                    kl_per_dim = -0.5 * (1 + logvar_enc - mean_enc**2 - jnp.exp(logvar_enc))
                    kl_per_dim = jnp.mean(kl_per_dim, axis=0)

                    delayed_active_mask = kl_per_dim > config["cmaes_kl_threshold"]
                    delayed_active_dims = jnp.where(delayed_active_mask)[0]
                    n_active = int(delayed_active_dims.shape[0])
                    print(f"[Delayed start KL] Active dims: {n_active}/{full_d} (threshold={config['cmaes_kl_threshold']})")
                    print(f"[Delayed start KL] Active indices: {np.array(delayed_active_dims).tolist()}")

                    if n_active == 0 or n_active == full_d:
                        print(f"[Delayed start KL] {'No active' if n_active == 0 else 'All'} dims — skipping KL filter")
                        delayed_active_dims = None
                    wandb.run.summary["delayed_kl_per_dim"] = np.array(kl_per_dim).tolist()
                    wandb.run.summary["delayed_active_dims"] = np.array(delayed_active_dims).tolist() if delayed_active_dims is not None else list(range(full_d))
                else:
                    delayed_active_dims = None

                # Fit PCA on active dims, embed back into full space
                pca_input = buf_means
                if delayed_active_dims is not None:
                    pca_input = buf_means[:, delayed_active_dims]

                pca_fitness = None
                if config.get("cmaes_pca_fitness_aware"):
                    pca_fitness = jnp.array(np.asarray(ts_cur.sampler["scores"][:buf_size]))

                new_pca_mean_sub, new_pca_components_sub = fit_pca(pca_input, config["cmaes_pca_dims"], fitness_scores=pca_fitness)

                # Embed back into full 64-dim space
                if delayed_active_dims is not None:
                    new_pca_mean = jnp.zeros(full_d).at[delayed_active_dims].set(new_pca_mean_sub)
                    new_pca_components = jnp.zeros((config["cmaes_pca_dims"], full_d)).at[:, delayed_active_dims].set(new_pca_components_sub)
                    print(f"[Delayed start PCA] {len(delayed_active_dims)} active dims -> {config['cmaes_pca_dims']} PCs (embedded in {full_d}D)")
                else:
                    new_pca_mean = new_pca_mean_sub
                    new_pca_components = new_pca_components_sub
                    print(f"[Delayed start PCA] {full_d} dims -> {config['cmaes_pca_dims']} PCs")

                # Reset CMA-ES and update PCA in train_state
                new_es = cmaes_mgr.initialize(jax.random.PRNGKey(eval_step + 2000))
                ts_cur = ts_cur.replace(
                    pca_mean=new_pca_mean,
                    pca_components=new_pca_components,
                    es_state=new_es,
                )
                runner_state = (rng_cur, ts_cur)
                _delayed_start_initialized = True
                _delayed_active_dims = delayed_active_dims
                print(f"[Delayed start] CMA-ES + PCA activated at eval_step={eval_step+1}")

        # PCA refit
        if use_pca and config.get("cmaes_pca_refit_interval", 0) > 0:
            if (eval_step + 1) % config["cmaes_pca_refit_interval"] == 0:
                rng_cur, ts_cur = runner_state
                buf_size = int(ts_cur.sampler["size"])
                if buf_size >= config["cmaes_pca_dims"]:
                    # Encode buffer levels
                    buf_levels = jax.tree_util.tree_map(lambda x: x[:buf_size], ts_cur.sampler["levels"])
                    buf_tokens = jax.vmap(level_to_tokens)(buf_levels)
                    buf_means = encode_levels_to_means(vae_encode_fn, buf_tokens)
                    # Determine which dims to use for PCA
                    # delayed-start uses _delayed_active_dims, normal mode uses active_dims
                    _refit_active = _delayed_active_dims if config.get("cmaes_delayed_start") else active_dims
                    if _refit_active is not None:
                        buf_means = buf_means[:, _refit_active]

                    if config.get("cmaes_pca_buffer_only", False):
                        # Buffer-only PCA
                        print(f"[PCA refit @ eval_step={eval_step+1}] Using {buf_size} buffer levels only")
                        refit_means = buf_means
                    else:
                        # Buffer + random levels
                        n_random = min(buf_size, config.get("cmaes_kl_samples", 5000))
                        print(f"[PCA refit @ eval_step={eval_step+1}] Using {buf_size} buffer + {n_random} random levels")
                        rng_pca = jax.random.PRNGKey(eval_step)
                        rand_levels = jax.vmap(sample_random_level)(jax.random.split(rng_pca, n_random))
                        rand_tokens = jax.vmap(level_to_tokens)(rand_levels)
                        rand_means = encode_levels_to_means(vae_encode_fn, rand_tokens)
                        if _refit_active is not None:
                            rand_means = rand_means[:, _refit_active]
                        refit_means = jnp.concatenate([buf_means, rand_means], axis=0)

                    # Use buffer fitness scores for fitness-aware PCA
                    refit_fitness = None
                    if config.get("cmaes_pca_fitness_aware"):
                        buf_scores = np.asarray(ts_cur.sampler["scores"][:buf_size])
                        if config.get("cmaes_pca_buffer_only", False):
                            refit_fitness = jnp.array(buf_scores)
                        else:
                            # Random levels get score 0 (neutral)
                            refit_fitness = jnp.concatenate([
                                jnp.array(buf_scores),
                                jnp.zeros(len(refit_means) - buf_size)
                            ])
                    new_pca_mean, new_pca_components = fit_pca(refit_means, config["cmaes_pca_dims"], fitness_scores=refit_fitness)

                    # For delayed-start mode, embed PCA back into full 64-dim space
                    if config.get("cmaes_delayed_start") and _refit_active is not None:
                        full_d = vae_cfg["latent_dim"]
                        new_pca_mean = jnp.zeros(full_d).at[_refit_active].set(new_pca_mean)
                        new_pca_components = jnp.zeros((config["cmaes_pca_dims"], full_d)).at[:, _refit_active].set(new_pca_components)

                    # Reset CMA-ES (coordinate system changed)
                    new_es = cmaes_mgr.initialize(jax.random.PRNGKey(eval_step + 1000))
                    ts_cur = ts_cur.replace(
                        pca_mean=new_pca_mean,
                        pca_components=new_pca_components,
                        es_state=new_es,
                    )
                    runner_state = (rng_cur, ts_cur)
                else:
                    print(f"[PCA refit] Skipped — buffer too small ({buf_size} < {config['cmaes_pca_dims']})")

        # Periodic buffer dump at configured intervals
        updates_so_far = (eval_step + 1) * config["eval_freq"]
        if config["buffer_dump_interval"] > 0 and updates_so_far % config["buffer_dump_interval"] == 0:
            dump_buffer(runner_state[1], updates_so_far // 1000)

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
    group.add_argument("--score_function", type=str, default="MaxMC",
                       choices=["MaxMC", "pvl", "sfl", "cenie", "mna"],
                       help="Level scoring function: MaxMC (regret), pvl (positive value loss), "
                            "sfl (learnability p*(1-p)), cenie (novelty+regret)")
    group.add_argument("--num_sfl_rollouts", type=int, default=10,
                       help="Number of evaluation rollouts for SFL learnability estimation")
    group.add_argument("--cenie_alpha", type=float, default=0.5,
                       help="CENIE novelty weight (0=pure regret, 1=pure novelty)")
    group.add_argument("--cenie_buffer_size", type=int, default=50000,
                       help="CENIE state-action coverage buffer size (FIFO)")
    group.add_argument("--cenie_num_components", type=int, default=10,
                       help="CENIE GMM number of components")
    group.add_argument("--cenie_refit_interval", type=int, default=5,
                       help="Refit CENIE GMM every N eval steps")
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
    # === DRED CONFIG ===
    group.add_argument("--use_dred", action=argparse.BooleanOptionalAction, default=False,
                       help="Use DRED: generate new levels by interpolating buffer levels in VAE latent space")
    # === CMA-ES + VAE CONFIG ===
    group.add_argument("--use_cmaes", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--vae_checkpoint_path", type=str, default=None,
                       help="Path to VAE .pkl checkpoint file")
    group.add_argument("--vae_config_path", type=str, default=None,
                       help="Path to VAE config.yaml (run directory)")
    group.add_argument("--cmaes_sigma_init", type=float, default=1.0)
    group.add_argument("--cmaes_popsize", type=int, default=None,
                       help="CMA-ES population size. Overrides num_train_envs when set (they must be equal).")
    group.add_argument("--cmaes_reset_interval", type=int, default=500,
                       help="Reset CMA-ES every N DR updates to prevent stagnation")
    group.add_argument("--cmaes_sigma_min", type=float, default=0.0,
                       help="Reset CMA-ES when sigma drops below this threshold (0 = disabled)")
    group.add_argument("--cmaes_kl_threshold", type=float, default=0.0,
                       help="Only search latent dims with per-dim KL > threshold (0 = use all dims)")
    group.add_argument("--cmaes_kl_samples", type=int, default=5000,
                       help="Number of levels to encode for estimating per-dim KL divergence")
    group.add_argument("--cmaes_kl_data", type=str, default=None,
                       help="Path to .npy token file for KL estimation (e.g. val set). Falls back to random levels.")
    group.add_argument("--cmaes_pca_dims", type=int, default=0,
                       help="Number of PCA dimensions for CMA-ES search (0 = disabled, overrides KL filtering)")
    group.add_argument("--cmaes_pca_data", type=str, default=None,
                       help="Path to .npy token file for PCA fitting. Falls back to --cmaes_kl_data or random levels.")
    group.add_argument("--cmaes_pca_refit_interval", type=int, default=0,
                       help="Refit PCA every N eval steps using buffer + random levels (0 = never, static PCA)")
    group.add_argument("--cmaes_pca_buffer_only", action=argparse.BooleanOptionalAction, default=False,
                       help="Refit PCA on buffer levels only (no random levels mixed in)")
    group.add_argument("--cmaes_pca_fitness_aware", action=argparse.BooleanOptionalAction, default=False,
                       help="Fitness-aware PCA: PC1 = direction of max fitness change, rest = max variance orthogonal to it")
    group.add_argument("--cmaes_delayed_start", action=argparse.BooleanOptionalAction, default=False,
                       help="Start with standard ACCEL (random levels), switch to CMA-ES+PCA once buffer is full. "
                            "KL filtering and PCA are computed from buffer levels at that point.")
    group.add_argument("--cmaes_pca_start_after", type=int, default=0,
                       help="Number of updates before switching from free CMA-ES to PCA-guided CMA-ES. "
                            "Before this point, CMA-ES searches the full KL-filtered space. "
                            "At this point, PCA is fit on the buffer and CMA-ES switches to PCA space. (0 = disabled)")
    group.add_argument("--warmstart_checkpoint", type=str, default=None,
                       help="Path to Orbax checkpoint dir to warm-start agent params from (e.g. checkpoints/run/0/models)")
    group.add_argument("--warmstart_buffer", type=str, default=None,
                       help="Path to buffer dump .npz to warm-start PLR buffer from. "
                            "If PCA is enabled, PCA is fit on these buffer levels.")
    group.add_argument("--warmstart_updates", type=int, default=0,
                       help="Offset for update counter so wandb logs continue from this step")
    group.add_argument("--save_cmaes_populations", action=argparse.BooleanOptionalAction, default=True,
                       help="Save CMA-ES population archive before each reset for latent visualization")
    # === GCS CONFIG ===
    group.add_argument("--gcs_bucket", type=str, default=None,
                       help="GCS bucket name for saving checkpoints/artifacts (e.g. 'ucl-ued-project-bucket')")
    group.add_argument("--gcs_prefix", type=str, default="accel",
                       help="Prefix path within GCS bucket")
    group.add_argument("--buffer_dump_interval", type=int, default=10000,
                       help="Dump PLR buffer (VAE token format) every N updates. 0 to disable periodic dumps.")
    group.add_argument("--skip_post_eval", action="store_true", default=False,
                       help="Skip post-training buffer evaluation, rendering, and PCA (run evaluate_buffer.py separately)")

    config = vars(parser.parse_args())

    # CMA-ES popsize overrides num_train_envs (they must match for parallel rollouts)
    if config["use_cmaes"] and config["cmaes_popsize"] is not None:
        print(f"[CMA-ES] Setting num_train_envs={config['cmaes_popsize']} (from --cmaes_popsize)")
        config["num_train_envs"] = config["cmaes_popsize"]

    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])
    config["group_name"] = ''.join([str(config[key]) for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])])
    
    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'
    
    # wandb.login()
    main(config, project=config["project"])
