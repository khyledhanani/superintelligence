"""CNN+LSTM VAE model for 13x13x3 maze grids.

Checkpoint key naming contract:
- mean_layer and logvar_layer are Dense layers defined directly inside
  CnnLstmVAE.__call__, NOT inside CnnEncoder.
  This ensures params/mean_layer/... and params/logvar_layer/... are at the
  TOP LEVEL of the param tree, matching the existing ES pipeline checkpoint.
- CnnEncoder is instantiated as name='encoder', producing params/encoder/...
- CnnLstmDecoder is instantiated as name='decoder', producing params/decoder/...

Do NOT nest mean_layer or logvar_layer inside CnnEncoder — that would produce
params/encoder/mean_layer/... which breaks Phase 3 compatibility.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


class CnnEncoder(nn.Module):
    """CNN encoder with LSTM bridge.

    Converts (B, 13, 13, 3) maze grid to (B, 512) feature vector.

    Spatial downsampling chain (verified shapes):
      Input:  (B, 13, 13, 3)
      conv1:  (B, 7, 7, 64)   -- stride 2, SAME padding
      conv2:  (B, 4, 4, 128)  -- stride 2, SAME padding
      conv3:  (B, 2, 2, 256)  -- stride 2, SAME padding
      reshape: (B, 4, 256)    -- row-major flatten of spatial dims
      LSTM:   (B, 512)        -- final hidden state
      bridge: (B, 512)        -- Dense projection with ReLU
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b = x.shape[0]

        # 3 strided Conv layers with ReLU: 13x13 -> 7x7 -> 4x4 -> 2x2
        h = nn.relu(nn.Conv(64, (3, 3), strides=(2, 2), padding='SAME', name='conv1')(x))
        h = nn.relu(nn.Conv(128, (3, 3), strides=(2, 2), padding='SAME', name='conv2')(h))
        h = nn.relu(nn.Conv(256, (3, 3), strides=(2, 2), padding='SAME', name='conv3')(h))

        # Row-major flatten: (B, 2, 2, 256) -> (B, 4, 256)
        h_seq = h.reshape(b, 4, 256)

        # nn.scan LSTM bridge (hidden=512)
        # CRITICAL: initialize_carry requires (batch_size, input_dim), not just (batch_size,)
        ScanLSTM = nn.scan(
            nn.LSTMCell,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=1,   # scan over sequence dim (axis 1)
            out_axes=1,
        )
        init_carry = nn.LSTMCell(512).initialize_carry(
            jax.random.PRNGKey(0),
            (b, 256)  # MUST include input_dim
        )
        carry, _ = ScanLSTM(features=512, name='enc_lstm')(init_carry, h_seq)
        final_h = carry[1]  # carry = (c_state, h_state); take h_state

        # Bridge Dense with ReLU
        return nn.relu(nn.Dense(512, name='enc_bridge')(final_h))


def free_bits_kl(mu: jnp.ndarray, logvar: jnp.ndarray, free_bits: float = 0.5) -> jnp.ndarray:
    """Compute KL divergence with per-dimension free bits floor.

    Args:
        mu: (B, latent_dim) mean
        logvar: (B, latent_dim) log variance (already clipped)
        free_bits: per-dimension KL floor (default 0.5 nats)

    Returns:
        Scalar KL loss (mean over batch, sum over dimensions, with floor applied).
    """
    kl_per_dim = -0.5 * (1 + logvar - jnp.square(mu) - jnp.exp(logvar))
    kl_with_floor = jnp.maximum(kl_per_dim, free_bits)
    return jnp.mean(jnp.sum(kl_with_floor, axis=-1))


class CnnLstmDecoder(nn.Module):
    """CNN decoder with LSTM spatial unfold for 13x13x3 maze grids.

    Design A: Non-autoregressive LSTM spatial unfold.
    The LSTM input is z tiled to (B, 4, latent_dim) — the LSTM never sees
    its previous output tokens, which prevents posterior collapse.

    Spatial upsample chain (nearest-neighbor, NO ConvTranspose):
      LSTM out: (B, 4, 256)
      dec_proj:  (B, 4, 128) -> reshape (B, 2, 2, 128)
      dec_conv1: 2x2 -> 4x4, 192 channels
      dec_conv2: 4x4 -> 7x7, 128 channels
      dec_conv3: 7x7 -> 13x13, 128 channels
      heads:     (B, 13, 13) x 3  -- wall, goal, agent logits
    """

    latent_dim: int = 64

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Decode latent vector to wall, goal, agent logits.

        Args:
            z: (B, latent_dim) latent vector

        Returns:
            Tuple of (wall_logits, goal_logits, agent_logits), each (B, 13, 13)
        """
        b = z.shape[0]

        # Non-autoregressive LSTM spatial unfold: tile z, LSTM never sees output
        z_seq = jnp.tile(z[:, None, :], (1, 4, 1))  # (B, 4, latent_dim)
        ScanLSTM = nn.scan(
            nn.LSTMCell,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=1,
            out_axes=1,
        )
        init_carry = nn.LSTMCell(256).initialize_carry(
            jax.random.PRNGKey(0), (b, self.latent_dim)
        )
        _, lstm_out = ScanLSTM(features=256, name='dec_lstm')(init_carry, z_seq)
        # lstm_out: (B, 4, 256)

        # Project and reshape to spatial feature map
        d = nn.Dense(128, name='dec_proj')(lstm_out)   # (B, 4, 128)
        d = d.reshape(b, 2, 2, 128)                     # (B, 2, 2, 128)

        # Nearest-neighbor upsample chain (NO ConvTranspose — checkerboard artifacts)
        # Step 1: 2x2 -> 4x4 (keep 128 channels for resize, then apply conv -> 192 channels)
        d = nn.relu(nn.Conv(192, (3, 3), padding='SAME', name='dec_conv1')(
            jax.image.resize(d, (b, 4, 4, 128), method='nearest')))
        # d is now (B, 4, 4, 192)

        # Step 2: 4x4 -> 7x7 (use 192 channels for resize — channel count changed!)
        d = nn.relu(nn.Conv(128, (3, 3), padding='SAME', name='dec_conv2')(
            jax.image.resize(d, (b, 7, 7, 192), method='nearest')))
        # d is now (B, 7, 7, 128)

        # Step 3: 7x7 -> 13x13 (use 128 channels for resize)
        d = nn.relu(nn.Conv(128, (3, 3), padding='SAME', name='dec_conv3')(
            jax.image.resize(d, (b, 13, 13, 128), method='nearest')))
        # d is now (B, 13, 13, 128)

        # 3-head output (logits, not probabilities — loss functions apply sigmoid/softmax)
        wall_logits = nn.Conv(1, (1, 1), padding='SAME', name='wall_head')(d)[..., 0]   # (B, 13, 13)
        goal_logits = nn.Conv(1, (1, 1), padding='SAME', name='goal_head')(d)[..., 0]   # (B, 13, 13)
        agent_logits = nn.Conv(1, (1, 1), padding='SAME', name='agent_head')(d)[..., 0] # (B, 13, 13)

        return wall_logits, goal_logits, agent_logits


class CnnLstmVAE(nn.Module):
    """CNN+LSTM Variational Autoencoder for 13x13x3 maze grids.

    Checkpoint key naming contract:
    - params/encoder/...       -- CnnEncoder sub-module (conv1, conv2, conv3, enc_lstm, enc_bridge)
    - params/mean_layer/...    -- Dense(latent_dim) directly in this module's __call__
    - params/logvar_layer/...  -- Dense(latent_dim) directly in this module's __call__
    - params/decoder/...       -- CnnLstmDecoder sub-module (added in plan 01-03)

    The mean_layer and logvar_layer MUST remain at the TOP LEVEL of this module
    to match the existing ES pipeline checkpoint key convention.
    """

    latent_dim: int = 64

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        z_rng: jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass through encoder, bottleneck, and decoder.

        Args:
            x: (B, 13, 13, 3) input maze grid
            z_rng: JAX random key for reparameterization

        Returns:
            Tuple of (wall_logits, goal_logits, agent_logits, mu, logvar)
            - wall_logits:  (B, 13, 13)
            - goal_logits:  (B, 13, 13)
            - agent_logits: (B, 13, 13)
            - mu:           (B, latent_dim)
            - logvar:       (B, latent_dim)
        """
        # Encoder: produces params/encoder/...
        h = CnnEncoder(name='encoder')(x)

        # Bottleneck AT TOP LEVEL: produces params/mean_layer/... and params/logvar_layer/...
        # Do NOT move these inside CnnEncoder — key naming must stay at top level.
        mu = jnp.tanh(nn.Dense(self.latent_dim, name='mean_layer')(h)) * 4.0
        logvar = jnp.clip(nn.Dense(self.latent_dim, name='logvar_layer')(h), -10, 2)

        # Reparameterization (no Distrax per requirements)
        z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(z_rng, mu.shape)

        # Decoder: produces params/decoder/...
        wall_logits, goal_logits, agent_logits = CnnLstmDecoder(
            latent_dim=self.latent_dim, name='decoder'
        )(z)

        return wall_logits, goal_logits, agent_logits, mu, logvar
