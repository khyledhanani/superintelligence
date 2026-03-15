"""
Trajectory VAE: encodes agent position trajectories through mazes.

Instead of encoding *what a maze looks like* (level VAE), this encodes
*how the agent behaves* in a maze — the sequence of grid positions visited.

Hypothesis: trajectory latent space correlates with regret more naturally
than level latent space, because regret is fundamentally about agent behavior.

Trajectory format:
    [pos_0, pos_1, ..., pos_T, 0, 0, ...] padded to max_steps (250)
    pos_i = y * 13 + x + 1  (1-based, 0 = padding)
    Same vocabulary as level VAE (170 tokens).
"""
import jax
import jax.numpy as jnp
from flax import linen as nn


class TrajectoryVAE(nn.Module):
    """BiLSTM VAE over agent position trajectories.

    Architecture mirrors CluttrVAE but operates on longer sequences (250 vs 52)
    and encodes behavior rather than structure.
    """
    vocab_size: int = 170        # 0=pad, 1-169=grid positions
    embed_dim: int = 128         # smaller than level VAE (less complex per-token)
    latent_dim: int = 32         # smaller latent: trajectories are lower-entropy
    max_steps: int = 250         # max episode length

    def setup(self):
        # Encoder
        self.embed = nn.Embed(self.vocab_size, self.embed_dim)
        self.enc_drop = nn.Dropout(rate=0.1)
        self.enc_bilstm = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(self.embed_dim)),
            nn.RNN(nn.LSTMCell(self.embed_dim)),
        )
        self.mean_layer = nn.Dense(self.latent_dim)
        self.logvar_layer = nn.Dense(self.latent_dim)

        # Decoder
        self.dec_bilstm1 = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(256)), nn.RNN(nn.LSTMCell(256)),
        )
        self.dec_bilstm2 = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(256)), nn.RNN(nn.LSTMCell(256)),
        )
        self.dec_output = nn.Dense(self.vocab_size)

    def __call__(self, x, z_rng, train: bool = True):
        mean, logvar = self.encode(x, train=train)
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(z_rng, mean.shape)
        z = mean + eps * std
        logits = self.decode(z)
        return logits, mean, logvar

    def encode(self, x, train: bool = True):
        """Encode position sequence to (mean, logvar).

        Args:
            x: (batch, max_steps) int32 position tokens.
        Returns:
            mean: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        # Compute mask for non-padding positions
        mask = (x > 0).astype(jnp.float32)  # (batch, max_steps)

        x = self.embed(x)  # (batch, max_steps, embed_dim)
        x = self.enc_drop(x, deterministic=not train)
        outputs = self.enc_bilstm(x)  # (batch, max_steps, 2*embed_dim)

        # Masked mean pooling instead of last/first hidden
        # (more robust to variable-length sequences)
        mask_expanded = mask[:, :, None]  # (batch, max_steps, 1)
        pooled = (outputs * mask_expanded).sum(axis=1) / jnp.maximum(mask_expanded.sum(axis=1), 1.0)

        mean = jnp.tanh(self.mean_layer(pooled)) * 6.0
        logvar = jnp.clip(self.logvar_layer(pooled), -10.0, 4.0)
        return mean, logvar

    def decode(self, z):
        """Decode latent vector to position sequence logits.

        Args:
            z: (batch, latent_dim) or (latent_dim,)
        Returns:
            logits: (batch, max_steps, vocab_size)
        """
        squeeze = False
        if z.ndim == 1:
            z = z[jnp.newaxis, :]
            squeeze = True

        z_seq = jnp.tile(z[:, jnp.newaxis, :], (1, self.max_steps, 1))
        d_out = self.dec_bilstm1(z_seq)
        d_out = self.dec_bilstm2(d_out)
        logits = self.dec_output(d_out)

        if squeeze:
            logits = logits[0]
        return logits


def trajectory_vae_loss(logits, targets, mean, logvar, kl_weight=1.0):
    """Compute VAE loss with position-aware weighting.

    Args:
        logits: (batch, max_steps, vocab_size) predicted logits
        targets: (batch, max_steps) ground truth position tokens
        mean: (batch, latent_dim) encoder mean
        logvar: (batch, latent_dim) encoder log-variance
        kl_weight: KL annealing weight

    Returns:
        total_loss, (recon_loss, kl_loss)
    """
    # Mask: only compute loss on actual trajectory steps (non-padding)
    mask = (targets > 0).astype(jnp.float32)

    # Per-token cross-entropy
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    ce = -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)  # (batch, max_steps)

    # Masked mean
    recon_loss = (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    # KL divergence
    kl_loss = -0.5 * jnp.mean(jnp.sum(1 + logvar - mean ** 2 - jnp.exp(logvar), axis=-1))

    total = recon_loss + kl_weight * kl_loss
    return total, (recon_loss, kl_loss)
