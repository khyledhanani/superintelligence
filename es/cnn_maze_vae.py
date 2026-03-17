"""Helpers for using the CNN maze VAE inside PLWM-style pipelines."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from vae.cnn_vae_level_utils import decode_latent_to_levels_grid
from vae.cnn_vae_model import CnnEncoder, CnnLstmDecoder


def load_cnn_vae_params(checkpoint_dir: str | Path) -> dict:
    handler = ocp.CompositeCheckpointHandler(default=ocp.StandardCheckpointHandler())
    checkpointer = ocp.Checkpointer(handler)
    state = checkpointer.restore(str(checkpoint_dir))
    for _ in range(4):
        if hasattr(state, "keys") and "params" in state:
            break
        if hasattr(state, "keys") and "default" in state:
            state = state["default"]
            continue
        break
    if not (hasattr(state, "keys") and "params" in state):
        keys = list(state.keys()) if hasattr(state, "keys") else []
        raise KeyError(f"Checkpoint missing `params`. Top keys: {keys}")
    return state["params"]


def encode_cnn_vae_levels(full_params: dict, grids: jnp.ndarray) -> jnp.ndarray:
    """Encode (B,H,W,3) grids to CNN-VAE posterior means."""
    encoder = CnnEncoder(name="encoder")
    mean_kernel = jnp.asarray(full_params["mean_layer"]["kernel"], dtype=jnp.float32)
    mean_bias = jnp.asarray(full_params["mean_layer"]["bias"], dtype=jnp.float32)
    h = encoder.apply({"params": full_params["encoder"]}, grids)
    return jnp.tanh(h @ mean_kernel + mean_bias) * 4.0


def decode_cnn_vae_latents(
    full_params: dict,
    z: jnp.ndarray,
    rng_key: jax.Array | None = None,
):
    """Decode a batch of latents to Maze Levels using deterministic wall/position heads."""
    latent_dim = int(full_params["mean_layer"]["kernel"].shape[1])
    decoder = CnnLstmDecoder(latent_dim=latent_dim, name="decoder")

    @jax.jit
    def _decode_single(z_single: jnp.ndarray):
        wall_logits, goal_logits, agent_logits = decoder.apply(
            {"params": full_params["decoder"]}, z_single[None, :]
        )
        return wall_logits[0], goal_logits[0], agent_logits[0]

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    return decode_latent_to_levels_grid(_decode_single, z, rng_key)


def latent_dim_from_params(full_params: dict) -> int:
    return int(np.asarray(full_params["mean_layer"]["kernel"]).shape[1])
