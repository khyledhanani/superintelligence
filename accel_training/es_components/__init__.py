"""ES strategy components for ES-ACCEL integration.

Public exports:
    ESStrategy        — typing.Protocol defining the ask/tell/init_state interface
    CMAESStrategy     — thin wrapper around evosax CMA_ES satisfying ESStrategy
    NSESStrategy      — NS-ES as composite-fitness variant of CMAESStrategy
    SVCMAESStrategy   — SV-CMA-ES: N-particle CMA-ES with Stein repulsion in behavior space

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

    from accel_training.es_components import SVCMAESStrategy
    sv_strategy = SVCMAESStrategy(param_dim=64, pop_size=16)
    state = sv_strategy.init_state(rng, config={"sv_n_particles": 4, "sigma_init": 0.5})
    candidates, state = sv_strategy.ask(state, rng)
    new_state, metrics = sv_strategy.tell(
        state, pre_cands, pre_bsigs, post_cands, post_regrets, post_bsigs, epsilon=0.01
    )
    # metrics: {"sv_behavior_dist_pre": float, "sv_behavior_dist_post": float}
"""

from .interface import ESStrategy
from .cmaes_strategy import CMAESStrategy
from .nses_strategy import NSESStrategy
from .svcmaes_strategy import SVCMAESStrategy

__all__ = ["ESStrategy", "CMAESStrategy", "NSESStrategy", "SVCMAESStrategy"]
