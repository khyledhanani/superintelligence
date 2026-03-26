"""
Supervised trajectory encoder: learns a representation from raw position
sequences that predicts regret.

Architecture:
    position sequence (N, 250) → Embed → BiLSTM → masked mean pool → z (repr_dim)
    z → linear → regret prediction

The bottleneck z is the learned representation — optimized to encode
whatever spatial behavior patterns are predictive of regret.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn


class TrajectoryEncoder(nn.Module):
    """BiLSTM encoder over position trajectories → regret predictor."""
    vocab_size: int = 170       # 0=pad, 1-169=grid positions
    embed_dim: int = 64
    hidden_dim: int = 128       # per-direction LSTM hidden size
    repr_dim: int = 32          # bottleneck representation dimension
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Args:
            x: (batch, max_steps) int32 position tokens
        Returns:
            regret_pred: (batch,) predicted regret
            z: (batch, repr_dim) learned representation
        """
        z = self.encode(x, train=train)
        regret_pred = nn.Dense(1, name="regret_head")(z).squeeze(-1)
        return regret_pred, z

    def encode(self, x, train: bool = True):
        """Encode position sequence to representation vector.

        Args:
            x: (batch, max_steps) int32 position tokens
        Returns:
            z: (batch, repr_dim)
        """
        mask = (x > 0).astype(jnp.float32)  # (batch, max_steps)

        h = nn.Embed(self.vocab_size, self.embed_dim, name="embed")(x)
        h = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(h)

        h = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(self.hidden_dim)),
            nn.RNN(nn.LSTMCell(self.hidden_dim)),
        )(h)  # (batch, max_steps, 2*hidden_dim)

        # Masked mean pooling
        mask_exp = mask[:, :, None]
        h = (h * mask_exp).sum(axis=1) / jnp.maximum(mask_exp.sum(axis=1), 1.0)

        # Bottleneck
        z = nn.Dense(self.repr_dim, name="bottleneck")(h)
        z = nn.relu(z)

        return z
