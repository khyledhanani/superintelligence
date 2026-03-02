"""ES strategy components for ES-ACCEL integration.

Public exports:
    ESStrategy      — typing.Protocol defining the ask/tell/init_state interface
    CMAESStrategy   — thin wrapper around evosax CMA_ES satisfying ESStrategy
    NSESStrategy    — NS-ES as composite-fitness variant of CMAESStrategy

Usage:
    from accel_training.es_components import ESStrategy, CMAESStrategy
    strategy = CMAESStrategy(param_dim=64, pop_size=32)
    state = strategy.init_state(rng, config={})
    candidates, state = strategy.ask(state, rng)
    state = strategy.tell(state, candidates, fitness)

    from accel_training.es_components import NSESStrategy
    ns_strategy = NSESStrategy(param_dim=64, pop_size=32)
    state = ns_strategy.init_state(rng, config={"sigma_init": 0.5})
    candidates, state = ns_strategy.ask(state, rng)
    new_state, mean_novelty = ns_strategy.tell(
        state, candidates, regrets,
        candidate_sigs, buffer_sigs, valid_mask,
        alpha=0.8, beta=0.2, k=5,
    )
"""

from .interface import ESStrategy
from .cmaes_strategy import CMAESStrategy
from .nses_strategy import NSESStrategy

__all__ = ["ESStrategy", "CMAESStrategy", "NSESStrategy"]
