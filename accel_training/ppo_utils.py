"""
PPO utilities for ACCEL training.

Adapted from examples/maze_plr.py (jaxued). Pure JAX functions —
no wandb, no side-effects.

Key functions:
    compute_gae                 GAE advantage estimation
    sample_trajectories_rnn     Collect rollout with LSTM agent
    update_actor_critic_rnn     PPO gradient update (multiple epochs)
"""

from typing import Tuple
import jax
import jax.numpy as jnp
import chex


def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Generalised Advantage Estimation.

    Args:
        gamma: Discount factor.
        lambd: GAE lambda.
        last_value: V(s_T), shape (num_envs,).
        values:  V(s_t) for t in [0, T-1], shape (num_steps, num_envs).
        rewards: r_t for t in [0, T-1], shape (num_steps, num_envs).
        dones:   d_t for t in [0, T-1], shape (num_steps, num_envs).

    Returns:
        advantages: shape (num_steps, num_envs).
        targets:    shape (num_steps, num_envs).  (= advantages + values)
    """
    def _gae_step(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _gae_step,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values


def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env,
    env_params,
    train_state,
    init_hstate: chex.ArrayTree,
    init_obs,
    init_env_state,
    num_envs: int,
    max_episode_length: int,
):
    """Collect a rollout with an LSTM-based actor-critic.

    Expects train_state.apply_fn to have the signature:
        (params, (obs, dones), hidden) -> (hidden, pi, value)

    Returns:
        carry: (rng, train_state, hstate, last_obs, last_env_state, last_value)
        traj:  (obs, actions, rewards, dones, log_probs, values, info)
               each of shape (num_steps, num_envs, ...).
    """
    def _step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda a: a[None, ...], (obs, last_done))
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
        _step,
        (rng, train_state, init_hstate, init_obs, init_env_state, jnp.zeros(num_envs, dtype=bool)),
        None,
        length=max_episode_length,
    )

    x = jax.tree_util.tree_map(lambda a: a[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)
    last_value = last_value.squeeze(0)

    return (rng, train_state, hstate, last_obs, last_env_state, last_value), traj


def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool = True,
):
    """PPO update for an LSTM actor-critic over multiple epochs + minibatches.

    Args:
        batch: (obs, actions, dones, log_probs, values, targets, advantages)
        update_grad: If False the parameters are not updated (exploratory rollout).

    Returns:
        (rng, train_state), losses
        losses: (total_loss, (l_vf, l_clip, entropy))
    """
    obs, actions, dones, log_probs, values, targets, advantages = batch
    # Shift dones so we know which timestep the RNN should reset at
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages

    def _epoch(carry, _):
        def _minibatch(train_state, mb):
            hstate_mb, obs_mb, actions_mb, last_dones_mb, log_probs_mb, values_mb, targets_mb, advantages_mb = mb

            def _loss(params):
                _, pi, values_pred = train_state.apply_fn(params, (obs_mb, last_dones_mb), hstate_mb)
                log_probs_pred = pi.log_prob(actions_mb)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs_mb)
                A = (advantages_mb - advantages_mb.mean()) / (advantages_mb.std() + 1e-5)
                l_clip = (
                    -jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)
                ).mean()

                v_clipped = values_mb + (values_pred - values_mb).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum(
                    (values_pred - targets_mb) ** 2,
                    (v_clipped - targets_mb) ** 2,
                ).mean()

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy
                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(_loss, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, ts = carry
        rng, rng_perm = jax.random.split(rng)
        perm = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_util.tree_map(
                lambda x: jnp.take(x, perm, axis=0).reshape(n_minibatch, -1, *x.shape[1:]),
                init_hstate,
            ),
            *jax.tree_util.tree_map(
                lambda x: jnp.take(x, perm, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        ts, losses = jax.lax.scan(_minibatch, ts, minibatches)
        return (rng, ts), losses

    return jax.lax.scan(_epoch, (rng, train_state), None, n_epochs)
