"""
Standalone CLUTTR VAE encoder for PLR-Weighted Latent Mutation (PLWM).

Extracts the encoder portion from a trained CluttrVAE checkpoint and maps
52-element integer maze sequences to 64-dim tanh-scaled latent vectors.

Encoder architecture (actual checkpoint layout):
    x (batch, 52)  -> Embed(170, 300)          [Embed_0]
                   -> HighwayStage(300)          [HighwayStage_0]
                   -> HighwayStage(300)          [HighwayStage_1]
                   -> BiLSTM(300) last hidden    [LSTMCell_0, LSTMCell_1]
                   -> Dense(64, name='mean_layer')  [mean_layer]
                   -> tanh * 4.0
                   -> z (batch, 64)

Full VAE param tree (actual flat top-level keys in checkpoint):
    Encoder: Embed_0, HighwayStage_0, HighwayStage_1, LSTMCell_0, LSTMCell_1,
             mean_layer (600->64), logvar_layer (600->64)
    Decoder: LSTMCell_2, LSTMCell_3, LSTMCell_4, LSTMCell_5, Dense_0 (800->170)

The encoder bottleneck uses explicitly named Dense layers (mean_layer, logvar_layer)
rather than auto-numbered Dense modules. This means Dense_0 in the checkpoint belongs
to the decoder. The standalone CluttrEncoder must use the same named layers to match
the checkpoint parameter keys.
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
# Highway network — must exactly match the trained checkpoint's architecture
# ---------------------------------------------------------------------------

class HighwayStage(nn.Module):
    """Highway network matching the CLUTTR VAE checkpoint architecture."""
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
    """Standalone encoder matching the encoder portion of the CLUTTR VAE checkpoint.

    Uses name='mean_layer' for the bottleneck Dense so its parameter key matches
    the actual checkpoint (which stores encoder mean/logvar as named layers rather
    than auto-numbered Dense_0/Dense_1).
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
        # Named Dense so params are stored as 'mean_layer', matching checkpoint
        mean = nn.Dense(CONFIG["latent_dim"], name='mean_layer')(h)  # mean_layer
        return jnp.tanh(mean) * 4.0                                  # tanh*4 scaled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENCODER_KEYS = frozenset({
    'Embed_0', 'HighwayStage_0', 'HighwayStage_1',
    'LSTMCell_0', 'LSTMCell_1',
    'mean_layer', 'logvar_layer',   # named bottleneck layers (logvar included for completeness)
})


def extract_encoder_params(full_params: dict) -> dict:
    """Extract encoder parameters from a full CluttrVAE checkpoint.

    The encoder keys in the checkpoint are:
        Embed_0, HighwayStage_0, HighwayStage_1, LSTMCell_0, LSTMCell_1,
        mean_layer, logvar_layer

    These map directly to the standalone CluttrEncoder's parameter names
    (no remapping needed).

    Args:
        full_params: Top-level parameter dict from the full VAE checkpoint.

    Returns:
        Dict containing only encoder keys (logvar_layer is included but unused
        during the forward pass since CluttrEncoder only reads mean_layer).
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
