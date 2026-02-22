"""
Standalone CLUTTR VAE encoder for PLR-Weighted Latent Mutation (PLWM).

Extracts the encoder portion from a trained CluttrVAE checkpoint and maps
52-element integer maze sequences to 64-dim tanh-scaled latent vectors.

Encoder architecture (mirrors train_vae.py):
    x (batch, 52)  -> Embed(170, 300)          [Embed_0]
                   -> HighwayStage(300)          [HighwayStage_0]
                   -> HighwayStage(300)          [HighwayStage_1]
                   -> BiLSTM(300) last hidden    [LSTMCell_0, LSTMCell_1]
                   -> Dense(128)                 [Dense_0]
                   -> split -> mean (64)
                   -> tanh * 4.0
                   -> z (batch, 64)

Full VAE param tree (flat top-level keys):
    Encoder: Embed_0, HighwayStage_0, HighwayStage_1, LSTMCell_0, LSTMCell_1, Dense_0
    Decoder: LSTMCell_2, LSTMCell_3, LSTMCell_4, LSTMCell_5, Dense_1

No remapping is needed: the standalone CluttrEncoder defines modules in the
same order as the full VAE encoder, so Flax assigns identical parameter names.
"""

import os
import pickle

import jax
import jax.numpy as jnp
import yaml
from flax import linen as nn

# Load config from the vae directory (sibling to es/)
_config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'vae', 'vae_train_config.yml'
)
with open(_config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Highway network — must exactly match train_vae.py to preserve param names
# ---------------------------------------------------------------------------

class HighwayStage(nn.Module):
    """Highway network matching train_vae.py architecture exactly."""
    dim: int = 300

    @nn.compact
    def __call__(self, x):
        g = nn.Dense(self.dim)(x)
        g = nn.relu(g)
        f_g_x = nn.relu(nn.Dense(self.dim)(g))
        q_x = nn.Dense(self.dim)(nn.relu(nn.Dense(self.dim)(x)))
        gate = nn.sigmoid(nn.Dense(self.dim)(x))
        return gate * f_g_x + (1.0 - gate) * q_x


# ---------------------------------------------------------------------------
# Standalone encoder
# ---------------------------------------------------------------------------

class CluttrEncoder(nn.Module):
    """Standalone encoder mirroring the encoder portion of CluttrVAE.

    Module creation order matches train_vae.py so Flax's nn.compact assigns
    identical parameter names. Full VAE encoder params can be used directly
    without any key remapping.
    """

    @nn.compact
    def __call__(self, x):
        # x: (batch, seq_len=52) integer tokens in [0, vocab_size-1]
        x = nn.Embed(CONFIG["vocab_size"], CONFIG["embed_dim"])(x)  # Embed_0
        x = HighwayStage(CONFIG["embed_dim"])(x)                    # HighwayStage_0
        x = HighwayStage(CONFIG["embed_dim"])(x)                    # HighwayStage_1
        outputs = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(300)),                               # LSTMCell_0
            nn.RNN(nn.LSTMCell(300)),                               # LSTMCell_1
        )(x)
        h = outputs[:, -1, :]                      # last timestep: (batch, 600)
        stats = nn.Dense(CONFIG["latent_dim"] * 2)(h)              # Dense_0
        mean, _ = jnp.split(stats, 2, axis=-1)                    # (batch, 64)
        return jnp.tanh(mean) * 4.0                                # tanh*4 scaled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENCODER_KEYS = frozenset({
    'Embed_0', 'HighwayStage_0', 'HighwayStage_1', 'LSTMCell_0', 'LSTMCell_1', 'Dense_0',
})


def extract_encoder_params(full_params: dict) -> dict:
    """Extract encoder parameters from a full CluttrVAE checkpoint.

    The encoder keys are identical in the full VAE and the standalone
    CluttrEncoder, so no remapping is needed.

    Args:
        full_params: Top-level parameter dict from the full VAE checkpoint.

    Returns:
        Dict containing only encoder keys.
    """
    return {k: v for k, v in full_params.items() if k in _ENCODER_KEYS}


def encode_levels_to_latents(encoder_params: dict, sequences: jnp.ndarray) -> jnp.ndarray:
    """Encode a batch of 52-element CLUTTR sequences to latent vectors.

    Args:
        encoder_params: Extracted encoder parameters (from extract_encoder_params).
        sequences: Integer array of shape (batch_size, 52), values in [0, 169].

    Returns:
        Latent vectors of shape (batch_size, 64), values in [-4, 4] (tanh*4).
    """
    return CluttrEncoder().apply({'params': encoder_params}, sequences)
