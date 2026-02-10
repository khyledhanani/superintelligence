from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass(frozen=True)
class VAEConfig:
    seq_len: int = 52
    vocab_size: int = 170
    embed_dim: int = 300
    latent_dim: int = 64
    recon_weight: float = 500.0
    anneal_steps: int = 100_000
    learning_rate: float = 5e-5
    batch_size: int = 32
    dropout_rate: float = 0.1


class HighwayStage(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        g = nn.relu(nn.Dense(self.dim)(x))
        f_g_x = nn.relu(nn.Dense(self.dim)(g))
        q_x = nn.Dense(self.dim)(nn.relu(nn.Dense(self.dim)(x)))
        gate = nn.sigmoid(nn.Dense(self.dim)(x))
        return gate * f_g_x + (1.0 - gate) * q_x


class CluttrVAE(nn.Module):
    config: VAEConfig

    @nn.compact
    def encode_stats(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cfg = self.config

        # Explicit module names are aligned to the original training code's
        # auto-generated compact names to preserve checkpoint compatibility.
        x = nn.Embed(cfg.vocab_size, cfg.embed_dim, name="Embed_0")(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not train)
        x = HighwayStage(cfg.embed_dim, name="HighwayStage_0")(x)
        x = HighwayStage(cfg.embed_dim, name="HighwayStage_1")(x)
        outputs = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(300)),
            nn.RNN(nn.LSTMCell(300)),
            name="Bidirectional_0",
        )(x)
        outputs = nn.Dropout(rate=cfg.dropout_rate)(outputs, deterministic=not train)

        fwd_out = outputs[:, :, :300]
        bwd_out = outputs[:, :, 300:]
        h = jnp.concatenate([fwd_out[:, -1, :], bwd_out[:, 0, :]], axis=-1)

        mean = nn.Dense(cfg.latent_dim, name="mean_layer")(h)
        logvar = nn.Dense(cfg.latent_dim, name="logvar_layer")(h)
        mean = jnp.tanh(mean) * 4.0
        return mean, logvar

    @nn.compact
    def decode(self, z: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        del train
        cfg = self.config
        z_seq = jnp.tile(z[:, jnp.newaxis, :], (1, cfg.seq_len, 1))
        d_out = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(400)),
            nn.RNN(nn.LSTMCell(400)),
            name="Bidirectional_1",
        )(z_seq)
        d_out = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(400)),
            nn.RNN(nn.LSTMCell(400)),
            name="Bidirectional_2",
        )(d_out)
        logits = nn.Dense(cfg.vocab_size, name="Dense_0")(d_out)
        return logits

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        z_rng: jax.Array,
        train: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        cfg = self.config

        x = nn.Embed(cfg.vocab_size, cfg.embed_dim, name="Embed_0")(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not train)
        x = HighwayStage(cfg.embed_dim, name="HighwayStage_0")(x)
        x = HighwayStage(cfg.embed_dim, name="HighwayStage_1")(x)
        outputs = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(300)),
            nn.RNN(nn.LSTMCell(300)),
            name="Bidirectional_0",
        )(x)
        outputs = nn.Dropout(rate=cfg.dropout_rate)(outputs, deterministic=not train)

        fwd_out = outputs[:, :, :300]
        bwd_out = outputs[:, :, 300:]
        h = jnp.concatenate([fwd_out[:, -1, :], bwd_out[:, 0, :]], axis=-1)

        mean = nn.Dense(cfg.latent_dim, name="mean_layer")(h)
        logvar = nn.Dense(cfg.latent_dim, name="logvar_layer")(h)
        mean = jnp.tanh(mean) * 4.0

        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(z_rng, mean.shape)
        z = mean + eps * std

        z_seq = jnp.tile(z[:, jnp.newaxis, :], (1, cfg.seq_len, 1))
        d_out = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(400)),
            nn.RNN(nn.LSTMCell(400)),
            name="Bidirectional_1",
        )(z_seq)
        d_out = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(400)),
            nn.RNN(nn.LSTMCell(400)),
            name="Bidirectional_2",
        )(d_out)
        logits = nn.Dense(cfg.vocab_size, name="Dense_0")(d_out)
        return logits, mean, logvar, z


def kl_weight(step: int, anneal_steps: int) -> jnp.ndarray:
    return jnp.minimum(1.0, step / float(anneal_steps))
