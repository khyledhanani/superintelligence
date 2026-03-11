"""
Grid-based Convolutional Autoencoder with SIGReg for PLWM.

Architecture:
    Encoder: (B,13,13,3) -> Conv stack -> Dense -> z (B, latent_dim)
    Decoder: z -> Dense -> DeConv stack -> wall_logits (B,13,13)
                                         goal_logits  (B,169)
                                         agent_logits (B,169)

Latent regularization: SIGReg (Sketched Isotropic Gaussian Regularization)
from LeJEPA (Balestriero & LeCun, 2025). Projects z along M random unit
vectors and matches the resulting 1-D distributions to N(0,1) via the
Epps-Pulley test on empirical characteristic functions. This enforces a
full isotropic Gaussian on the latent space with provably bounded gradients —
avoiding the posterior collapse that plagues beta-VAE on sparse grids.

Checkpoint structure (pickle):
    {'params': {'MazeEncoder_0': {...}, 'MazeDecoder_0': {...}}, 'step': int}

PLWM usage:
    encoder_params = extract_maze_encoder_params(load_maze_ae_params(path))
    decoder_params = extract_maze_decoder_params(load_maze_ae_params(path))
    grids = jax.vmap(maze_level_to_grid)(wall_maps, goal_positions, agent_positions)
    z     = encode_maze_levels(encoder_params, grids)          # (B, 64)
    z_    = z + sigma * jax.random.normal(rng, z.shape)
    levels= decode_maze_latents(decoder_params, z_, rng_keys)  # (B,) Level
"""

import os
import pickle

import jax
import jax.numpy as jnp
from flax import linen as nn

from jaxued.environments.maze.level import Level


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class MazeEncoder(nn.Module):
    latent_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, 3)
        h = nn.Conv(32, (3, 3), padding='SAME')(x)
        h = nn.leaky_relu(h, negative_slope=0.2)

        h = nn.Conv(64, (3, 3), strides=(2, 2), padding='SAME')(h)   # (B, 7, 7, 64)
        h = nn.leaky_relu(h, negative_slope=0.2)

        h = nn.Conv(128, (3, 3), strides=(2, 2), padding='SAME')(h)  # (B, 4, 4, 128)
        h = nn.leaky_relu(h, negative_slope=0.2)

        h = h.reshape(h.shape[0], -1)          # (B, 2048)
        h = nn.Dense(512)(h)
        h = nn.leaky_relu(h, negative_slope=0.2)

        z = nn.Dense(self.latent_dim)(h)       # (B, latent_dim) — no tanh, SIGReg handles dist
        return z


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class MazeDecoder(nn.Module):
    height: int = 13
    width: int = 13

    @nn.compact
    def __call__(self, z):
        # z: (B, latent_dim)
        H, W = self.height, self.width

        h = nn.Dense(512)(z)
        h = nn.leaky_relu(h, negative_slope=0.2)

        h = nn.Dense(4 * 4 * 128)(h)
        h = nn.leaky_relu(h, negative_slope=0.2)

        h = h.reshape(h.shape[0], 4, 4, 128)

        h = nn.ConvTranspose(64, (3, 3), strides=(2, 2), padding='SAME')(h)  # (B, 8, 8, 64)
        h = nn.leaky_relu(h, negative_slope=0.2)

        h = nn.ConvTranspose(32, (3, 3), strides=(2, 2), padding='SAME')(h)  # (B, 16, 16, 32)
        h = nn.leaky_relu(h, negative_slope=0.2)

        h = h[:, :H, :W, :]   # crop to (B, 13, 13, 32)

        # Wall head
        wh = nn.Conv(16, (3, 3), padding='SAME')(h)
        wh = nn.leaky_relu(wh, negative_slope=0.2)
        wall_logits = nn.Conv(1, (1, 1))(wh)[:, :, :, 0]   # (B, H, W)

        # Position heads: global-average-pool spatial features → Dense
        gap = h.mean(axis=(1, 2))              # (B, 32)
        ph = nn.Dense(256)(gap)
        ph = nn.leaky_relu(ph, negative_slope=0.2)
        goal_logits  = nn.Dense(H * W)(ph)    # (B, H*W)
        agent_logits = nn.Dense(H * W)(ph)    # (B, H*W)

        return wall_logits, goal_logits, agent_logits


# ---------------------------------------------------------------------------
# Full AE (used only during training)
# ---------------------------------------------------------------------------

class MazeAE(nn.Module):
    latent_dim: int = 64
    height: int = 13
    width: int = 13

    @nn.compact
    def __call__(self, x):
        z = MazeEncoder(self.latent_dim)(x)
        wall_logits, goal_logits, agent_logits = MazeDecoder(self.height, self.width)(z)
        return z, wall_logits, goal_logits, agent_logits


# ---------------------------------------------------------------------------
# SIGReg loss (Balestriero & LeCun, 2025 — LeJEPA)
# ---------------------------------------------------------------------------

def sigreg_loss(
    z: jnp.ndarray,
    rng: jax.Array,
    n_directions: int = 64,
    n_t: int = 64,
    t_max: float = 4.0,
) -> jnp.ndarray:
    """Sketched Isotropic Gaussian Regularization.

    Projects the batch of latent vectors along M random unit directions and
    measures how far each 1-D projected distribution deviates from N(0,1)
    using the Epps-Pulley test on empirical characteristic functions (ECFs).

    EP(X) = N * integral |ECF_X(t) - exp(-t^2/2)|^2 dt
          ≈ N * dt * sum_t [ (mean cos(t*x) - exp(-t^2/2))^2
                            + (mean sin(t*x))^2 ]

    SIGReg = (1/M) * sum_m EP(z @ a_m)
    where a_m are unit vectors resampled each call (Cramér-Wold theorem).

    Args:
        z:            (batch, latent_dim) latent vectors.
        rng:          JAX PRNG key. Directions are resampled every call.
        n_directions: M — number of random projections.
        n_t:          Number of discrete t values for the integral.
        t_max:        Integration range [-t_max, t_max].

    Returns:
        Scalar SIGReg loss value.
    """
    batch, dim = z.shape

    # Sample M unit directions on S^{dim-1}
    a = jax.random.normal(rng, (n_directions, dim))
    a = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)   # (M, D)

    # Project: (M, batch)
    proj = jnp.einsum('md,nd->mn', a, z)

    # Discrete t grid for integral approximation
    t = jnp.linspace(-t_max, t_max, n_t)    # (T,)
    dt = 2.0 * t_max / n_t

    # ECF at each t: (M, N, T) -> mean over N -> (M, T)
    pt = proj[:, :, None] * t[None, None, :]   # (M, N, T)
    ecf_real = jnp.cos(pt).mean(axis=1)         # (M, T)
    ecf_imag = jnp.sin(pt).mean(axis=1)         # (M, T)

    # N(0,1) characteristic function: phi(t) = exp(-t^2/2)  [real-valued]
    gaussian_cf = jnp.exp(-0.5 * t ** 2)        # (T,)

    # |ECF(t) - phi(t)|^2 = (ecf_real - phi)^2 + ecf_imag^2
    diff_sq = (ecf_real - gaussian_cf[None, :]) ** 2 + ecf_imag ** 2  # (M, T)

    # EP per direction: N * integral ~ N * dt * sum_t diff_sq
    ep_per_dir = batch * dt * diff_sq.sum(axis=-1)   # (M,)

    return ep_per_dir.mean()


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_maze_ae_params(checkpoint_path: str) -> dict:
    """Load MazeAE parameters from a pickle checkpoint."""
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    return data['params']


def extract_maze_encoder_params(full_params: dict) -> dict:
    """Extract MazeEncoder sub-params from the full MazeAE checkpoint."""
    return full_params['MazeEncoder_0']


def extract_maze_decoder_params(full_params: dict) -> dict:
    """Extract MazeDecoder sub-params from the full MazeAE checkpoint."""
    return full_params['MazeDecoder_0']


# ---------------------------------------------------------------------------
# PLWM helpers — JIT/vmap compatible
# ---------------------------------------------------------------------------

def maze_level_to_grid(
    wall_map: jnp.ndarray,
    goal_pos: jnp.ndarray,
    agent_pos: jnp.ndarray,
    height: int = 13,
    width: int = 13,
) -> jnp.ndarray:
    """Convert Level components to (H, W, 3) float32 grid.

    JIT and vmap compatible. Channel layout:
        0: wall map   (1.0 = wall, 0.0 = free)
        1: goal       (1.0 at goal cell)
        2: agent      (1.0 at agent cell)

    Args:
        wall_map:  (H, W) bool.
        goal_pos:  [col, row] uint32 (Level convention).
        agent_pos: [col, row] uint32 (Level convention).

    Returns:
        (H, W, 3) float32 grid.
    """
    grid = jnp.zeros((height, width, 3), dtype=jnp.float32)
    grid = grid.at[:, :, 0].set(wall_map.astype(jnp.float32))
    grid = grid.at[goal_pos[1],  goal_pos[0],  1].set(1.0)
    grid = grid.at[agent_pos[1], agent_pos[0], 2].set(1.0)
    return grid


def encode_maze_levels(
    encoder_params: dict,
    grids: jnp.ndarray,
) -> jnp.ndarray:
    """Encode a batch of (H, W, 3) grids to latent vectors.

    Args:
        encoder_params: MazeEncoder parameters (from extract_maze_encoder_params).
        grids:          (B, H, W, 3) float32 grids.

    Returns:
        (B, latent_dim) latent vectors.
    """
    return MazeEncoder().apply({'params': encoder_params}, grids)


def _decode_single_level(
    wall_logits: jnp.ndarray,
    goal_logits: jnp.ndarray,
    agent_logits: jnp.ndarray,
    rng: jax.Array,
    wall_threshold: float,
    temperature: float,
    height: int,
    width: int,
) -> Level:
    """Decode single-sample decoder outputs to a jaxued Level.

    Uses Gumbel-max sampling for goal/agent positions so mutation diversity
    is controlled by `temperature` (lower → more deterministic / argmax-like).
    """
    rng_goal, rng_agent, rng_dir = jax.random.split(rng, 3)

    # ---- Wall map ----
    wall_map = jax.nn.sigmoid(wall_logits) > wall_threshold   # (H, W)

    # ---- Goal position: Gumbel-max over H*W cells ----
    gumbel_g = jax.random.gumbel(rng_goal, (height * width,))
    goal_flat = jnp.argmax(goal_logits / temperature + gumbel_g)
    goal_col  = (goal_flat % width).astype(jnp.uint32)
    goal_row  = (goal_flat // width).astype(jnp.uint32)

    # ---- Agent position: Gumbel-max, goal cell masked ----
    agent_mask = jnp.zeros(height * width).at[goal_flat].set(-jnp.inf)
    gumbel_a   = jax.random.gumbel(rng_agent, (height * width,))
    agent_flat = jnp.argmax((agent_logits + agent_mask) / temperature + gumbel_a)
    agent_col  = (agent_flat % width).astype(jnp.uint32)
    agent_row  = (agent_flat // width).astype(jnp.uint32)

    # ---- Repair: clear walls at goal/agent cells ----
    wall_map = wall_map.at[goal_row,  goal_col ].set(False)
    wall_map = wall_map.at[agent_row, agent_col].set(False)

    agent_dir = jax.random.randint(rng_dir, (), 0, 4).astype(jnp.uint8)

    return Level(
        wall_map=wall_map,
        goal_pos=jnp.array([goal_col,  goal_row],  dtype=jnp.uint32),
        agent_pos=jnp.array([agent_col, agent_row], dtype=jnp.uint32),
        agent_dir=agent_dir,
        width=width,
        height=height,
    )


def decode_maze_latents(
    decoder_params: dict,
    z: jnp.ndarray,
    rng_keys: jnp.ndarray,
    wall_threshold: float = 0.5,
    temperature: float = 0.25,
    height: int = 13,
    width: int = 13,
) -> Level:
    """Decode a batch of latent vectors to jaxued Level objects.

    Args:
        decoder_params: MazeDecoder parameters (from extract_maze_decoder_params).
        z:              (B, latent_dim) perturbed latents.
        rng_keys:       (B, 2) per-sample RNG keys.
        wall_threshold: Sigmoid threshold for binarizing wall logits.
        temperature:    Gumbel-max temperature for goal/agent sampling.

    Returns:
        Batched Level with fields of shape (B, ...).
    """
    wall_logits, goal_logits, agent_logits = MazeDecoder(height, width).apply(
        {'params': decoder_params}, z
    )
    return jax.vmap(
        lambda wl, gl, al, rng: _decode_single_level(
            wl, gl, al, rng, wall_threshold, temperature, height, width
        )
    )(wall_logits, goal_logits, agent_logits, rng_keys)
