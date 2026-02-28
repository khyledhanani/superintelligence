"""ES strategy components for ES-ACCEL integration.

Public exports:
    ESStrategy      — typing.Protocol defining the ask/tell/init_state interface
    CMAESStrategy   — thin wrapper around evosax CMA_ES satisfying ESStrategy

Usage:
    from accel_training.es_components import ESStrategy, CMAESStrategy
    strategy = CMAESStrategy(param_dim=64, pop_size=32)
    state = strategy.init_state(rng, config={})
    candidates, state = strategy.ask(state, rng)
    state = strategy.tell(state, candidates, fitness)
"""

from .interface import ESStrategy
from .cmaes_strategy import CMAESStrategy

__all__ = ["ESStrategy", "CMAESStrategy"]
