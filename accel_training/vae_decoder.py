"""
Standalone VAE decoder extraction and CLUTTR environment utilities.

Extracts the decoder portion from a trained CluttrVAE checkpoint and provides
functions for decoding latent vectors to valid CLUTTR environment sequences.

Decoder architecture (mirrors train_vae.py lines 55-59):
    z (batch, 64) -> tile to (batch, 52, 64)
    -> BiLSTM(400) -> BiLSTM(400) -> Dense(170)
    -> argmax -> integer sequence (batch, 52)

Two checkpoint formats are supported (auto-detected by extract_decoder_params):

Format A — flat numbered keys (checkpoint_420000.pkl and earlier):
    Encoder: Embed_0, HighwayStage_0/1, LSTMCell_0/1, mean_layer (600->64), logvar_layer (600->64)
    Decoder: LSTMCell_2/3 (BiLSTM 1), LSTMCell_4/5 (BiLSTM 2), Dense_0 (800->170)
    Mapping:  LSTMCell_2->LSTMCell_0, LSTMCell_3->LSTMCell_1,
              LSTMCell_4->LSTMCell_2, LSTMCell_5->LSTMCell_3, Dense_0->Dense_0

Format B — named BiLSTM keys (checkpoint_final.pkl):
    Decoder: dec_bilstm1/{forward_rnn,backward_rnn}/cell, dec_bilstm2/..., dec_output
    Mapping:  dec_bilstm1/forward_rnn/cell->LSTMCell_0, dec_bilstm1/backward_rnn/cell->LSTMCell_1,
              dec_bilstm2/forward_rnn/cell->LSTMCell_2, dec_bilstm2/backward_rnn/cell->LSTMCell_3,
              dec_output->Dense_0
"""

import jax
import jax.numpy as jnp
import pickle
import yaml
import os
from flax import linen as nn

# Load config from the vae directory (sibling to es)
_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vae', 'vae_train_config.yml')
with open(_config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)


class CluttrDecoder(nn.Module):
    """Standalone decoder that mirrors the decoder portion of CluttrVAE."""
    @nn.compact
    def __call__(self, z):
        # z: (batch, latent_dim=64)
        z_seq = jnp.tile(z[:, jnp.newaxis, :], (1, CONFIG["seq_len"], 1))
        d_out = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(400)),
            nn.RNN(nn.LSTMCell(400))
        )(z_seq)
        d_out = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(400)),
            nn.RNN(nn.LSTMCell(400))
        )(d_out)
        logits = nn.Dense(CONFIG["vocab_size"])(d_out)
        return logits  # (batch, seq_len=52, vocab_size=170)


def load_vae_params(checkpoint_path):
    """Load the full VAE parameters from a pickle checkpoint."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    return data['params']


def extract_decoder_params(full_params):
    """Extract decoder parameters from the full VAE checkpoint.

    Auto-detects the checkpoint format and remaps to CluttrDecoder's expected keys
    (LSTMCell_0..3, Dense_0). See module docstring for format details.
    """
    if 'LSTMCell_2' in full_params:
        # Format A: flat numbered keys (checkpoint_420000.pkl and earlier)
        key_map = {
            'LSTMCell_2': 'LSTMCell_0',
            'LSTMCell_3': 'LSTMCell_1',
            'LSTMCell_4': 'LSTMCell_2',
            'LSTMCell_5': 'LSTMCell_3',
            'Dense_0': 'Dense_0',
        }
        return {key_map[k]: v for k, v in full_params.items() if k in key_map}
    elif 'dec_bilstm1' in full_params:
        # Format B: named BiLSTM keys (checkpoint_final.pkl)
        return {
            'LSTMCell_0': full_params['dec_bilstm1']['forward_rnn']['cell'],
            'LSTMCell_1': full_params['dec_bilstm1']['backward_rnn']['cell'],
            'LSTMCell_2': full_params['dec_bilstm2']['forward_rnn']['cell'],
            'LSTMCell_3': full_params['dec_bilstm2']['backward_rnn']['cell'],
            'Dense_0': full_params['dec_output'],
        }
    else:
        raise ValueError(
            f"Unknown VAE checkpoint format. Top-level param keys: {list(full_params.keys())}"
        )


def decode_latent_to_env(decoder_params, z, rng_key=None, temperature=1.0):
    """Decode a batch of latent vectors to CLUTTR environment sequences.

    Args:
        decoder_params: Extracted decoder parameters (remapped keys).
        z: Latent vectors, shape (batch_size, 64).
        rng_key: Optional PRNG key. If provided, use Gumbel-max sampling.
                 If None, use deterministic argmax (backward compatible).
        temperature: Sampling temperature for logits when rng_key is provided.
                     Lower values approach argmax; must be > 0 for sampling.

    Returns:
        Integer sequences, shape (batch_size, 52), values in [0, 169].
    """
    logits = CluttrDecoder().apply({'params': decoder_params}, z)
    if rng_key is None or temperature <= 0:
        sequences = jnp.argmax(logits, axis=-1)  # (batch, 52)
        return sequences

    # Gumbel-max: argmax(logits / T + gumbel) samples from softmax(logits / T).
    temp = jnp.asarray(temperature, dtype=logits.dtype)
    u = jax.random.uniform(rng_key, logits.shape, minval=1e-6, maxval=1.0 - 1e-6)
    gumbel = -jnp.log(-jnp.log(u))
    sequences = jnp.argmax((logits / temp) + gumbel, axis=-1)
    return sequences


def repair_cluttr_sequence(seq):
    """Post-process a decoded 52-element sequence to enforce CLUTTR constraints.

    Constraints enforced:
        1. All values clamped to [0, 169].
        2. Goal (idx 50) and agent (idx 51) clamped to [1, 169].
        3. If goal == agent, agent is shifted by +1 (wrapping within [1, 169]).
        4. Obstacles colliding with goal/agent are zeroed out.
        5. Obstacles (positions 0-49) sorted ascending (zeros first).

    Args:
        seq: Integer array of shape (52,).

    Returns:
        Repaired integer array of shape (52,).
    """
    seq = jnp.clip(seq, 0, 169)

    goal = jnp.clip(seq[50], 1, 169)
    agent = jnp.clip(seq[51], 1, 169)
    # Resolve collision: shift agent by 1, wrapping within [1, 169]
    agent = jnp.where(goal == agent, (agent % 169) + 1, agent)

    obstacles = seq[:50]
    obstacles = jnp.where(obstacles == goal, 0, obstacles)
    obstacles = jnp.where(obstacles == agent, 0, obstacles)
    obstacles = jnp.sort(obstacles)

    return jnp.concatenate([obstacles, jnp.array([goal, agent])])
