"""
CMA-ES manager for searching the VAE latent space.
Thin wrapper around evosax with init/ask/tell API.

evosax MINIMIZES fitness — callers should negate scores
when higher is better (e.g., pass -regret so CMA-ES finds high-regret levels).

NOTE: The new evosax API decorates methods with @partial(jax.jit, static_argnames=("self",))
which breaks Python's descriptor protocol — instance.method() does NOT auto-bind self.
We must call methods via the CLASS and pass the instance explicitly.
"""
import jax
import jax.numpy as jnp

try:
    from evosax.algorithms import CMA_ES
    _NEW_API = True
except ImportError:
    from evosax import CMA_ES
    _NEW_API = False

print(f"[CMAESManager] evosax API: {'new (evosax.algorithms)' if _NEW_API else 'old (evosax)'}")


class CMAESManager:
    """Manages CMA-ES search in the VAE latent space.

    The population size should match num_train_envs so each CMA-ES candidate
    maps to one environment slot for parallel rollouts.
    """

    def __init__(self, popsize, latent_dim=64, sigma_init=1.0):
        self.popsize = popsize
        self.latent_dim = latent_dim
        self.sigma_init = sigma_init

        if _NEW_API:
            dummy = jnp.zeros(latent_dim)
            self.strategy = CMA_ES(population_size=popsize, solution=dummy)
            self.es_params = self.strategy.default_params
            # Verify evosax inferred the correct dimensionality
            assert self.strategy.num_dims == latent_dim, (
                f"evosax num_dims={self.strategy.num_dims} != latent_dim={latent_dim}. "
                f"solution_flat.size={self.strategy.solution_flat.size}"
            )
        else:
            self.strategy = CMA_ES(popsize=popsize, num_dims=latent_dim)
            self.es_params = self.strategy.default_params

        print(f"[CMAESManager] popsize={popsize}, latent_dim={latent_dim}, "
              f"num_dims={getattr(self.strategy, 'num_dims', 'N/A')}")

    def initialize(self, rng):
        """Initialize CMA-ES state. Returns a JAX pytree."""
        if _NEW_API:
            # Must call via class — jax.jit wrapper breaks descriptor protocol
            return CMA_ES.init(self.strategy, rng, self.es_params)
        else:
            return self.strategy.initialize(rng, self.es_params)

    def ask(self, rng, es_state):
        """Sample population from CMA-ES.

        Returns:
            population: (popsize, latent_dim) candidate latent vectors.
            es_state: updated state.
        """
        if _NEW_API:
            return CMA_ES.ask(self.strategy, rng, es_state, self.es_params)
        else:
            return self.strategy.ask(rng, es_state, self.es_params)

    def tell(self, rng, population, fitness, es_state):
        """Update CMA-ES with fitness values.

        Args:
            rng: PRNGKey (required by new evosax API, ignored by old API).
            population: (popsize, latent_dim) from ask().
            fitness: (popsize,) scalar fitness per candidate.
                     CMA-ES minimizes, so pass -regret for maximizing regret.
            es_state: current state.

        Returns:
            Updated es_state.
        """
        if _NEW_API:
            # New API: tell(self, key, population, fitness, state, params) -> (state, metrics)
            state, _metrics = CMA_ES.tell(
                self.strategy, rng, population, fitness, es_state, self.es_params
            )
            return state
        else:
            # Old API: tell(population, fitness, state, params) -> state
            return self.strategy.tell(population, fitness, es_state, self.es_params)
