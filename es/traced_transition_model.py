"""Transition-prediction utilities for TRACED-style task scoring."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class TransitionImageEncoder(nn.Module):
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(32, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(self.hidden_dim)(x)


class TransitionImageDecoder(nn.Module):
    hidden_dim: int = 128
    out_channels: int = 3

    @nn.compact
    def __call__(self, z: jnp.ndarray, height: int, width: int) -> jnp.ndarray:
        x = nn.Dense(height * width * 64)(z)
        x = nn.relu(x)
        x = x.reshape((z.shape[0], height, width, 64))
        x = nn.Conv(32, (3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(self.out_channels, (3, 3), padding="SAME")(x)
        return nn.sigmoid(x)


class ImageTransitionPredictionModel(nn.Module):
    """Predict next observations from image observations and one-hot actions.

    Inputs:
        images: (batch, time, height, width, channels)
        actions: (batch, time, action_dim)
    Output:
        predicted next images: (batch, time, height, width, channels)
    """

    action_dim: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, images: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        batch, time, height, width, channels = images.shape

        encoder = TransitionImageEncoder(hidden_dim=self.hidden_dim)
        decoder = TransitionImageDecoder(
            hidden_dim=self.hidden_dim,
            out_channels=channels,
        )

        flat_images = images.reshape((batch * time, height, width, channels))
        encoded = encoder(flat_images).reshape((batch, time, -1))
        lstm_inputs = jnp.concatenate([encoded, actions], axis=-1)

        hidden = nn.RNN(nn.LSTMCell(self.hidden_dim))(lstm_inputs)
        flat_hidden = hidden.reshape((batch * time, self.hidden_dim))
        decoded = decoder(flat_hidden, height, width)
        return decoded.reshape((batch, time, height, width, channels))
