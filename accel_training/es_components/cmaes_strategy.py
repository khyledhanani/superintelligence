"""CMAESStrategy — thin wrapper around evosax CMA_ES satisfying ESStrategy Protocol.

evosax API:  ask(key, state, params)  |  tell(key, population, fitness, state, params)
             init(key, mean, params)  — 3 args (mean sets initial distribution center)
Protocol:    ask(state, rng)          |  tell(state, candidates, fitness)
             init_state(rng, config)

Absorbs evosax params into the state dict to satisfy the 2-arg Protocol interface.
Zero behavioral change to the underlying CMA-ES algorithm.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp

# sys.path may need es/ on path — use the same pattern as accel_training/train.py
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
sys.path.insert(0, os.path.join(_ROOT, 'es'))


class CMAESStrategy:
    """Thin wrapper around evosax CMA_ES satisfying the ESStrategy Protocol.

    Construction:
        strategy = CMAESStrategy(param_dim=64, pop_size=32)
        state = strategy.init_state(rng, config={"sigma_init": 0.5})
        candidates, state = strategy.ask(state, rng)
        state = strategy.tell(state, candidates, fitness)

    State dict keys:
        "es_state":  evosax EvoState (flax.struct.dataclass, registered JAX PyTree)
        "es_params": evosax EvoParams (flax.struct.dataclass, registered JAX PyTree)

    Note on evosax init API:
        evosax CMA_ES.init(key, mean, params) requires 3 args.
        The mean vector sets the initial distribution center (matches evolve_envs.py usage).
        We default mean to jnp.zeros(param_dim) for a zero-centered start.
    """

    def __init__(self, param_dim: int, pop_size: int):
        from evosax.algorithms import CMA_ES
        dummy_solution = jnp.zeros(param_dim)
        self._es = CMA_ES(population_size=pop_size, solution=dummy_solution)
        self._param_dim = param_dim
        self._pop_size = pop_size
        self._dummy_solution = dummy_solution

    def init_state(self, rng: jax.Array, config: dict) -> dict:
        """Initialize ES state. Optionally override sigma_init from config.

        evosax init: 3 args (rng, mean, params) — mean sets distribution center.
        """
        es_params = self._es.default_params
        if "sigma_init" in config:
            es_params = es_params.replace(std_init=config["sigma_init"])
        # evosax init: init(key, mean, params) — mean defaults to zeros
        mean = config.get("mean", self._dummy_solution)
        es_state = self._es.init(rng, mean, es_params)
        return {"es_state": es_state, "es_params": es_params}

    def ask(self, state: dict, rng: jax.Array) -> tuple[jnp.ndarray, dict]:
        """Generate candidates. Returns (candidates shape (pop_size, param_dim), new_state).

        evosax ask: ask(rng, state, params) — key is FIRST arg.
        """
        population, new_es_state = self._es.ask(rng, state["es_state"], state["es_params"])
        new_state = {**state, "es_state": new_es_state}
        return population, new_state

    def tell(self, state: dict, candidates: jnp.ndarray, fitness: jnp.ndarray) -> dict:
        """Update ES state. fitness is (pop_size,) — lower is better (evosax minimizes).

        evosax tell: tell(key, population, fitness, state, params) — 5 args.
        """
        dummy_key = jax.random.PRNGKey(0)
        new_es_state, _ = self._es.tell(
            dummy_key, candidates, fitness, state["es_state"], state["es_params"]
        )
        return {**state, "es_state": new_es_state}
