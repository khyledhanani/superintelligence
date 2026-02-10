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

    def setup(self) -> None:
        cfg = self.config

        self.embed = nn.Embed(cfg.vocab_size, cfg.embed_dim)
        self.dropout1 = nn.Dropout(rate=cfg.dropout_rate)
        self.dropout2 = nn.Dropout(rate=cfg.dropout_rate)

        self.highway_1 = HighwayStage(cfg.embed_dim)
        self.highway_2 = HighwayStage(cfg.embed_dim)

        self.enc_bidir = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(300)),
            nn.RNN(nn.LSTMCell(300)),
        )

        self.mean_layer = nn.Dense(cfg.latent_dim, name="mean_layer")
        self.logvar_layer = nn.Dense(cfg.latent_dim, name="logvar_layer")

        self.dec_bidir_1 = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(400)),
            nn.RNN(nn.LSTMCell(400)),
        )
        self.dec_bidir_2 = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(400)),
            nn.RNN(nn.LSTMCell(400)),
        )
        self.out_layer = nn.Dense(cfg.vocab_size)

    def encode_stats(self, x: jnp.ndarray, train: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = self.embed(x)
        x = self.dropout1(x, deterministic=not train)
        x = self.highway_1(x)
        x = self.highway_2(x)

        outputs = self.enc_bidir(x)
        outputs = self.dropout2(outputs, deterministic=not train)

        fwd_out = outputs[:, :, :300]
        bwd_out = outputs[:, :, 300:]
        h = jnp.concatenate([fwd_out[:, -1, :], bwd_out[:, 0, :]], axis=-1)

        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        mean = jnp.tanh(mean) * 4.0
        return mean, logvar

    def decode(self, z: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        del train  # Decoder currently has no train-time branches.
        z_seq = jnp.tile(z[:, jnp.newaxis, :], (1, self.config.seq_len, 1))
        d_out = self.dec_bidir_1(z_seq)
        d_out = self.dec_bidir_2(d_out)
        logits = self.out_layer(d_out)
        return logits

    def __call__(
        self,
        x: jnp.ndarray,
        z_rng: jax.Array,
        train: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mean, logvar = self.encode_stats(x, train=train)
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(z_rng, mean.shape)
        z = mean + eps * std
        logits = self.decode(z, train=train)
        return logits, mean, logvar, z


def kl_weight(step: int, anneal_steps: int) -> jnp.ndarray:
    return jnp.minimum(1.0, step / float(anneal_steps))
