"""NSESStrategy — NS-ES as a composite-fitness variant of CMAESStrategy.

NS-ES (Novelty Search Evolution Strategy) extends CMA-ES by replacing the raw
fitness signal with a composite measure:

    F = alpha * regret + beta * novelty

where novelty is computed via k-NN distance against behavior signatures stored
in the PLR replay buffer.  The ask() method is IDENTICAL to CMAESStrategy —
NS-ES distinction is entirely in the fitness computation inside tell().

Per locked CONTEXT.md decision: "Novelty source: NSESStrategy reads behavior
signatures from the replay buffer for k-NN novelty computation. No separate
novelty archive — one data structure, no duplication." Buffer state (buffer_sigs
and valid_mask) is passed into tell() at call time.

Usage:
    strategy = NSESStrategy(param_dim=64, pop_size=32)
    state = strategy.init_state(rng, config={"sigma_init": 0.5})
    candidates, state = strategy.ask(state, rng)
    new_state, mean_novelty = strategy.tell(
        state, candidates, regrets,
        candidate_sigs, buffer_sigs, valid_mask,
        alpha=0.8, beta=0.2, k=5,
    )
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
sys.path.insert(0, os.path.join(_ROOT, 'es'))

import jax
import jax.numpy as jnp


class NSESStrategy:
    """NS-ES as a composite-fitness variant of CMAESStrategy.

    Construction is identical to CMAESStrategy.  The ask() method is also
    identical.  The tell() method extends the minimal Protocol surface with
    novelty-related inputs; see NSESStrategy.tell() docstring.

    State dict keys (same as CMAESStrategy):
        "es_state":  evosax EvoState (flax.struct.dataclass, registered JAX PyTree)
        "es_params": evosax EvoParams (flax.struct.dataclass, registered JAX PyTree)
    """

    def __init__(self, param_dim: int, pop_size: int):
        from evosax.algorithms import CMA_ES
        dummy_solution = jnp.zeros(param_dim)
        self._es = CMA_ES(population_size=pop_size, solution=dummy_solution)
        self._param_dim = param_dim
        self._pop_size = pop_size
        self._dummy_solution = dummy_solution

    def init_state(self, rng: jax.Array, config: dict) -> dict:
        """Initialize ES state.  Optionally override sigma_init from config.

        evosax init: 3 args (rng, mean, params) — mean sets distribution center.
        """
        es_params = self._es.default_params
        if "sigma_init" in config:
            es_params = es_params.replace(std_init=config["sigma_init"])
        mean = config.get("mean", self._dummy_solution)
        es_state = self._es.init(rng, mean, es_params)
        return {"es_state": es_state, "es_params": es_params}

    def ask(self, state: dict, rng: jax.Array) -> tuple[jnp.ndarray, dict]:
        """Generate candidates.  Returns (candidates shape (pop_size, param_dim), new_state).

        Behavior is IDENTICAL to CMAESStrategy.ask().  The NS-ES distinction is
        entirely in the fitness computation inside tell().

        evosax ask: ask(rng, state, params) — key is FIRST arg.
        """
        population, new_es_state = self._es.ask(rng, state["es_state"], state["es_params"])
        new_state = {**state, "es_state": new_es_state}
        return population, new_state

    def tell(
        self,
        state: dict,
        candidates: jnp.ndarray,
        regrets: jnp.ndarray,
        candidate_sigs: jnp.ndarray,
        buffer_sigs: jnp.ndarray,
        valid_mask: jnp.ndarray,
        alpha: float,
        beta: float,
        k: int = 5,
        candidate_valid: jnp.ndarray | None = None,
    ) -> tuple[dict, float]:
        """Update ES state using composite fitness F = alpha*regret + beta*novelty.

        Note on Protocol surface:
            NSESStrategy.tell() extends the ESStrategy Protocol minimum surface
            (which has 3 args: state, candidates, fitness) with novelty inputs.
            The caller in train.py uses the concrete NSESStrategy type, NOT the
            Protocol abstraction, so the extra arguments are safe.

        Args:
            state:          ES state dict with 'es_state' and 'es_params' keys.
            candidates:     (pop_size, param_dim) float32 — population from ask().
            regrets:        (pop_size,) float32 — raw positive MaxMC regret per candidate.
            candidate_sigs: (pop_size, D) float32 — behavior signatures of candidates.
            buffer_sigs:    (capacity, D) float32 — all buffer behavior signatures.
            valid_mask:     (capacity,) bool — True for filled buffer slots.
            alpha:          float — weight for regret component.
            beta:           float — weight for novelty component.
            k:              int — number of nearest neighbors for novelty (default 5).
            candidate_valid: (pop_size,) bool or None — True for valid candidates.
                            Invalid candidates receive worst composite fitness so
                            CMA-ES ranks them last.

        Returns:
            (new_state, mean_novelty_float) where:
                new_state is the updated ES state dict.
                mean_novelty_float is a Python float (mean k-NN novelty across population).
        """
        # Lazy imports inside tell() to avoid circular import between submodules.
        from .novelty import compute_novelty_batch
        from .fitness import compute_fitness_batch

        # Step 1: compute per-candidate novelty via k-NN against PLR buffer.
        novelties = compute_novelty_batch(candidate_sigs, buffer_sigs, valid_mask, k=k)

        # Step 2: compute composite fitness F = alpha*regret + beta*novelty.
        composite = compute_fitness_batch(regrets, novelties, alpha=alpha, beta=beta)

        # Step 2b: validity masking — penalize invalid candidates.
        # composite is "higher = better". Invalid get -10.0 (worst).
        # After negation: invalid → +10.0 = worst for evosax minimizer.
        if candidate_valid is not None:
            composite = jnp.where(candidate_valid, composite, -10.0)

        # Step 3: negate for evosax (which minimizes; we want to MAXIMIZE composite).
        fitness_for_evosax = -composite

        # Step 4: update CMA-ES distribution.  dummy_key satisfies evosax API.
        dummy_key = jax.random.PRNGKey(0)
        new_es_state, _ = self._es.tell(
            dummy_key, candidates, fitness_for_evosax,
            state["es_state"], state["es_params"]
        )

        # Step 5: compute mean novelty as a Python float for logging.
        mean_novelty = float(jnp.mean(novelties))

        return {**state, "es_state": new_es_state}, mean_novelty
