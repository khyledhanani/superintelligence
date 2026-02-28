"""ESStrategy Protocol — the common interface for all ES strategies."""

from __future__ import annotations
from typing import Protocol
import jax
import jax.numpy as jnp


class ESStrategy(Protocol):
    """Structural protocol for ES strategies.

    Any class implementing init_state, ask, and tell satisfies this protocol
    without explicit inheritance.

    Candidates shape: (pop_size, param_dim) — directly usable with vmap.
    State: plain Python dict containing JAX arrays and evosax PyTree structs.
    """

    def init_state(self, rng: jax.Array, config: dict) -> dict:
        """Initialize ES state from config. Returns a state dict."""
        ...

    def ask(self, state: dict, rng: jax.Array) -> tuple[jnp.ndarray, dict]:
        """Generate pop_size candidate parameter vectors.
        Returns: (candidates shape (pop_size, param_dim), new_state)
        """
        ...

    def tell(self, state: dict, candidates: jnp.ndarray, fitness: jnp.ndarray) -> dict:
        """Update ES state given evaluated candidates and fitness scores.
        Returns: new_state
        """
        ...
