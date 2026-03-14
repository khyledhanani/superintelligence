"""
Adapter MLP and Regret Predictor for fitness-aware latent space correction.

The adapter learns to correct VAE latent vectors so that environments with
similar regret end up nearby, making CMA-ES search more efficient.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class RegretPredictor(nn.Module):
    """MLP that predicts regret from a latent vector z.

    Trained on (z, regret) pairs, then frozen during adapter training.
    """
    hidden_dim: int = 128
    n_layers: int = 2

    @nn.compact
    def __call__(self, z):
        x = z
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)  # scalar regret per sample


class AdapterMLP(nn.Module):
    """MLP that predicts a correction delta_z to the latent vector.

    z' = z + adapter(z), where z' is passed to the frozen VAE decoder.
    No output activation — delta can be positive or negative.
    """
    latent_dim: int = 64
    hidden_dim: int = 128
    n_layers: int = 2

    @nn.compact
    def __call__(self, z):
        x = z
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        delta_z = nn.Dense(self.latent_dim)(x)
        return delta_z


def create_predictor(latent_dim, hidden_dim=128, n_layers=2):
    """Create a RegretPredictor and initialize its parameters."""
    model = RegretPredictor(hidden_dim=hidden_dim, n_layers=n_layers)
    dummy_z = jnp.zeros((1, latent_dim))
    params = model.init(jax.random.PRNGKey(0), dummy_z)
    return model, params


def create_adapter(latent_dim, hidden_dim=128, n_layers=2):
    """Create an AdapterMLP and initialize its parameters."""
    model = AdapterMLP(latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    dummy_z = jnp.zeros((1, latent_dim))
    params = model.init(jax.random.PRNGKey(0), dummy_z)
    return model, params
