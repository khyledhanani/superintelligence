"""
Standalone VAE model definition for decoding latent vectors to maze token sequences.
No training code, no global CONFIG — all dimensions passed as constructor args.
Compatible with checkpoints saved by train_vae.py.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn


class HighwayStage(nn.Module):
    dim: int = 300

    def setup(self):
        self.dense_g = nn.Dense(self.dim)
        self.dense_fg = nn.Dense(self.dim)
        self.dense_q1 = nn.Dense(self.dim)
        self.dense_q2 = nn.Dense(self.dim)
        self.dense_gate = nn.Dense(self.dim)

    def __call__(self, x):
        g = nn.relu(self.dense_g(x))
        f_g_x = nn.relu(self.dense_fg(g))
        q_x = self.dense_q2(nn.relu(self.dense_q1(x)))
        gate = nn.sigmoid(self.dense_gate(x))
        return gate * f_g_x + (1.0 - gate) * q_x


class CluttrVAE(nn.Module):
    vocab_size: int = 170
    embed_dim: int = 300
    latent_dim: int = 64
    seq_len: int = 52
    enc_lstm_dim: int = 300
    dec_lstm_dim: int = 400

    def setup(self):
        # Encoder
        self.embed = nn.Embed(self.vocab_size, self.embed_dim)
        self.enc_drop1 = nn.Dropout(rate=0.1)
        self.enc_hw1 = HighwayStage(self.embed_dim)
        self.enc_hw2 = HighwayStage(self.embed_dim)
        self.enc_bilstm = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(self.enc_lstm_dim)),
            nn.RNN(nn.LSTMCell(self.enc_lstm_dim)),
        )
        self.enc_drop2 = nn.Dropout(rate=0.1)
        self.mean_layer = nn.Dense(self.latent_dim)
        self.logvar_layer = nn.Dense(self.latent_dim)

        # Decoder
        self.dec_bilstm1 = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(self.dec_lstm_dim)), nn.RNN(nn.LSTMCell(self.dec_lstm_dim)),
        )
        self.dec_bilstm2 = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(self.dec_lstm_dim)), nn.RNN(nn.LSTMCell(self.dec_lstm_dim)),
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
        x = self.embed(x)
        x = self.enc_drop1(x, deterministic=not train)
        x = self.enc_hw1(x)
        x = self.enc_hw2(x)
        outputs = self.enc_bilstm(x)
        outputs = self.enc_drop2(outputs, deterministic=not train)

        fwd_out = outputs[:, :, :self.enc_lstm_dim]
        bwd_out = outputs[:, :, self.enc_lstm_dim:]
        h = jnp.concatenate([fwd_out[:, -1, :], bwd_out[:, 0, :]], axis=-1)

        mean = jnp.tanh(self.mean_layer(h)) * 6.0
        logvar = jnp.clip(self.logvar_layer(h), -10.0, 4)
        return mean, logvar

    def decode(self, z):
        squeeze = False
        if z.ndim == 1:
            z = z[jnp.newaxis, :]
            squeeze = True

        z_seq = jnp.tile(z[:, jnp.newaxis, :], (1, self.seq_len, 1))
        d_out = self.dec_bilstm1(z_seq)
        d_out = self.dec_bilstm2(d_out)
        logits = self.dec_output(d_out)

        if squeeze:
            logits = logits[0]
        return logits
