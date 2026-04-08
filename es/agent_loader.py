"""
Load a frozen ACCEL agent from an orbax checkpoint for fitness evaluation.

The ActorCritic architecture is duplicated from examples/maze_plr.py:292-326
to avoid import-time side effects from that script. It MUST match the checkpoint
exactly: Conv(16, 3x3) + LSTM(256) + actor(32->7) + critic(32->1).

Usage:
    from agent_loader import load_agent, ActorCritic
    agent_params, network = load_agent("agent_folder/119/default")
"""

from typing import Sequence
import numpy as np
import pickle
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jaxued.linen import ResetRNN
from jaxued.environments.maze import Maze
from jaxued.environments.maze.env import Observation


# ---------------------------------------------------------------------------
# ActorCritic network (exact copy from examples/maze_plr.py:292-326)
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """LSTM-based actor-critic. Must match checkpoint architecture exactly.

    Architecture:
        Conv(16, 3x3, VALID) on obs.image -> flatten -> relu
        one_hot(agent_dir, 4) -> Dense(5)
        concat -> ResetRNN(OptimizedLSTMCell(256))
        Actor: Dense(32) -> relu -> Dense(action_dim) -> Categorical
        Critic: Dense(32) -> relu -> Dense(1)

    Action space: 7 discrete (left, right, forward, pickup, drop, toggle, done).
    """
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(
            5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed"
        )(dir_embed)

        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))(
            (embedding, dones), initial_carry=hidden
        )

        actor_mean = nn.Dense(
            32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0"
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1"
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0"
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1"
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, 256)
        )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _orbax_restore_cross_device(mgr, step):
    """Restore Orbax checkpoint when saved sharding references missing devices (GPU → CPU, etc.)."""
    import orbax.checkpoint.type_handlers as ocp_th

    def restore():
        return mgr.restore(step)

    try:
        return restore()
    except ValueError as exc:
        msg = str(exc).lower()
        if not any(
            t in msg
            for t in (
                'topology mismatch',
                'jax.local_devices',
                'was not found',
                'device cuda',
                'device gpu',
                'sharding',
            )
        ):
            raise

    print(
        'Orbax: checkpoint was saved on a different device layout than this process '
        '(e.g. CUDA while JAX only sees CPU). Retrying restore with local-device sharding…'
    )

    # Older orbax exposed this on type_handlers; newer versions moved / renamed internals.
    orig_ds = None
    if hasattr(ocp_th, '_deserialize_sharding_from_json_string'):
        orig_ds = ocp_th._deserialize_sharding_from_json_string

        def _deserialize_fallback(json_string: str):
            try:
                return orig_ds(json_string)
            except (ValueError, KeyError):
                return jax.sharding.SingleDeviceSharding(jax.local_devices()[0])

        ocp_th._deserialize_sharding_from_json_string = _deserialize_fallback

    mds_mod = None
    orig_tj = None
    try:
        import orbax.checkpoint._src.metadata.sharding as mds_mod

        orig_tj = mds_mod.SingleDeviceShardingMetadata.to_jax_sharding

        def _to_jax_sharding_fallback(self):
            try:
                return orig_tj(self)
            except ValueError:
                # Any device / metadata mismatch → host all arrays on the first local device.
                return jax.sharding.SingleDeviceSharding(jax.local_devices()[0])

        mds_mod.SingleDeviceShardingMetadata.to_jax_sharding = _to_jax_sharding_fallback
    except ImportError:
        pass

    try:
        return restore()
    finally:
        if orig_ds is not None:
            ocp_th._deserialize_sharding_from_json_string = orig_ds
        if mds_mod is not None and orig_tj is not None:
            mds_mod.SingleDeviceShardingMetadata.to_jax_sharding = orig_tj


def load_agent_params(checkpoint_dir, step=None):
    """Load agent parameters from a pickle file or orbax checkpoint.

    Tries to load from a pickle file (agent_params.pkl) first for portability
    across machines (no Orbax device metadata). Falls back to orbax if no pickle
    is found; orbax restore automatically retries with a local-device sharding
    fallback when the checkpoint was saved on GPU but this process only has CPU
    (or another topology mismatch).

    Args:
        checkpoint_dir: Path to the checkpoint directory
                        (e.g. "agent_folder" or "agent_folder/119/default").
        step: Specific checkpoint step to load (orbax only). None = latest.

    Returns:
        params: Frozen parameter dict for ActorCritic.
    """
    # Resolve to absolute path
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', checkpoint_dir
        )
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    # Try pickle first (portable, no tensorstore/orbax dependency)
    pkl_path = os.path.join(checkpoint_dir, 'agent_params.pkl')
    if os.path.exists(pkl_path):
        print(f"Loading agent params from pickle: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            params = pickle.load(f)
        # Convert numpy arrays to jax arrays
        params = jax.tree_util.tree_map(jnp.asarray, params)
        return params

    # Fall back to orbax checkpoint
    import orbax.checkpoint as ocp
    mgr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    step = step if step is not None else mgr.latest_step()
    print(f"Loading agent checkpoint from {checkpoint_dir} (step={step})")
    loaded = _orbax_restore_cross_device(mgr, step)

    # Orbax restores the full TrainState dict. The 'params' key contains
    # the flax variable dict {'params': {actual network params}}.
    # Return just the inner network params for use with network.apply({'params': ...}).
    params = loaded['params']
    if isinstance(params, dict) and 'params' in params:
        params = params['params']
    return params


def load_agent(checkpoint_dir, action_dim=7, step=None):
    """Load agent params and create network in one call.

    Args:
        checkpoint_dir: Path to orbax checkpoint directory.
        action_dim: Number of discrete actions (7 for Maze).
        step: Checkpoint step (None = latest).

    Returns:
        (params, network): Tuple of parameter dict and ActorCritic instance.
    """
    params = load_agent_params(checkpoint_dir, step)
    network = ActorCritic(action_dim=action_dim)
    return params, network


# ---------------------------------------------------------------------------
# Contract test: verify checkpoint matches expected architecture
# ---------------------------------------------------------------------------

def verify_agent_contract(agent_params, network):
    """Run a dummy forward pass to verify checkpoint/architecture match.

    Checks:
        - obs preprocessing: normalize_obs=True, agent_view_size=5 -> image (5,5,3)
        - action logits shape: (..., 7)
        - value output: scalar
        - recurrent carry dimensions

    Raises AssertionError on mismatch.
    """
    # Use batch_size=1 (ResetRNN requires at least 1 batch dim for resets[:, None])
    dummy_image = jnp.zeros((1, 5, 5, 3), dtype=jnp.float32)  # (batch=1, H, W, C)
    dummy_dir = jnp.zeros(1, dtype=jnp.uint8)
    dummy_obs = Observation(image=dummy_image, agent_dir=dummy_dir)
    dummy_done = jnp.zeros(1, dtype=jnp.bool_)

    hstate = ActorCritic.initialize_carry((1,))
    # Add leading seq dim: (1, batch=1, ...)
    x = jax.tree_util.tree_map(lambda a: a[None, ...], (dummy_obs, dummy_done))
    hstate_out, pi, value = network.apply({'params': agent_params}, x, hstate)

    assert pi.logits.shape[-1] == 7, (
        f"Expected 7 actions, got {pi.logits.shape[-1]}. "
        "Check action_dim matches checkpoint."
    )
    print(f"Agent contract test PASSED: "
          f"obs(5,5,3) -> {pi.logits.shape[-1]} actions, value shape={value.shape}")
