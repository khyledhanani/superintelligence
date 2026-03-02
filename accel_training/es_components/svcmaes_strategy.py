"""SVCMAESStrategy — N independent CMA-ES particles with Stein repulsion in behavior space.

SV-CMA-ES (Stein Variational CMA-ES) is the primary thesis contribution. N CMA-ES
particles explore in parallel, maintaining behavioral diversity by applying SVGD repulsion
(Liu & Wang 2016) to their latent-space means after each tell() step.

Algorithm per step (caller in train.py handles the two-pass evaluation loop):
  1. All N particles call ask() -> N*pop_size candidate latents (concatenated batch)
  2. Evaluate all N*pop_size candidates (decode -> rollout -> behavior sigs + regrets)
  3. Apply Stein gradient to candidate latents (nudge by epsilon) — done by train.py
  4. Re-evaluate repelled latents (second eval pass) -> final behavior sigs + regrets
  5. Each particle calls tell() with its own repelled candidates and regrets
  6. After tell(), update each particle's CMA mean using the Stein gradient

State structure:
  {"particles": [
    {"es_state": EvoState_0, "es_params": EvoParams_0},
    {"es_state": EvoState_1, "es_params": EvoParams_1},
    ...  # N total
  ]}

tell() receives pre- and post-repulsion arrays (two-pass eval done in train.py):
  - pre_cands / pre_bsigs: first-pass results, used for the pre-repulsion diversity metric
  - post_cands / post_regrets / post_bsigs: second-pass results, used to update each particle

Fitness for each particle's tell(): pure negated regret (evosax minimizes; we maximize regret).
Stein repulsion replaces the novelty bonus from NS-ES — no composite fitness here.

N=1 degrades gracefully to plain CMA-ES: repulsion step is skipped entirely.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
sys.path.insert(0, os.path.join(_ROOT, 'es'))

import jax
import jax.numpy as jnp


class SVCMAESStrategy:
    """N independent CMA-ES particles with Stein repulsion in behavior space.

    Construction:
        strategy = SVCMAESStrategy(param_dim=64, pop_size=32)
        state = strategy.init_state(rng, config={"sv_n_particles": 4, "sigma_init": 0.5})
        candidates, state = strategy.ask(state, rng)
        new_state, metrics = strategy.tell(
            state, pre_cands, pre_bsigs, post_cands, post_regrets, post_bsigs, epsilon=0.01
        )

    State dict keys:
        "particles": list of N dicts, each with:
            "es_state":  evosax EvoState (flax.struct.dataclass, registered JAX PyTree)
            "es_params": evosax EvoParams (flax.struct.dataclass, registered JAX PyTree)

    Config keys (passed to init_state):
        "sv_n_particles": int — number of particles (default 2)
        "sigma_init":     float — initial CMA-ES sigma / step size (default 0.5)

    Note on particle mean initialization:
        Particle means are initialized from N(0, sigma_init) with distinct RNG keys —
        NOT zeros. Zero initialization causes zero Stein gradient (all means identical),
        defeating the purpose of multi-particle diversity. Locked CONTEXT.md decision.
    """

    def __init__(self, param_dim: int, pop_size: int):
        from evosax.algorithms import CMA_ES
        self._param_dim = param_dim
        self._pop_size = pop_size
        self._es_template = CMA_ES(
            population_size=pop_size,
            solution=jnp.zeros(param_dim),
        )

    def init_state(self, rng: jax.Array, config: dict) -> dict:
        """Initialize N particles with distinct random means.

        Each particle gets its own RNG key so means differ from the start.
        All particles share sigma_init but are independently seeded.

        Args:
            rng:    JAX PRNGKey used to generate N particle keys.
            config: dict with optional "sv_n_particles" (int, default 2)
                    and "sigma_init" (float, default 0.5).

        Returns:
            {"particles": [{"es_state": ..., "es_params": ...}, ...]}  — N entries.
        """
        N = config.get("sv_n_particles", 2)
        sigma_init = config.get("sigma_init", 0.5)

        particles = []
        for i in range(N):
            rng, rng_i = jax.random.split(rng)
            es_params = self._es_template.default_params.replace(std_init=sigma_init)
            # Random mean init (NOT zeros) — locked CONTEXT decision.
            # Zero means would cause K_rowsum * mean - K @ mean = 0 for all particles,
            # producing zero Stein gradient at the very first step.
            mean_i = jax.random.normal(rng_i, (self._param_dim,)) * sigma_init
            es_state = self._es_template.init(rng_i, mean_i, es_params)
            particles.append({"es_state": es_state, "es_params": es_params})

        return {"particles": particles}

    def ask(self, state: dict, rng: jax.Array) -> tuple[jnp.ndarray, dict]:
        """Generate candidates for all N particles, concatenated into a single batch.

        Each particle gets its own RNG key to ensure independent sampling.

        Args:
            state: state dict with "particles" list of N particle state dicts.
            rng:   JAX PRNGKey split into N sub-keys.

        Returns:
            (candidates, new_state) where:
                candidates: (N*pop_size, param_dim) float32 — all particle populations
                             concatenated in particle order (particle 0 first, then 1, ...).
                new_state:  updated state dict with post-ask es_state per particle.
        """
        N = len(state["particles"])
        all_pops = []
        new_particles = []

        for i, p in enumerate(state["particles"]):
            rng, rng_i = jax.random.split(rng)
            # evosax ask: ask(rng, state, params) — rng is FIRST arg
            pop_i, new_es_state = self._es_template.ask(rng_i, p["es_state"], p["es_params"])
            all_pops.append(pop_i)
            new_particles.append({"es_state": new_es_state, "es_params": p["es_params"]})

        # Concatenate all particle populations: (N*pop_size, param_dim)
        all_cands = jnp.concatenate(all_pops, axis=0)
        new_state = {"particles": new_particles}
        return all_cands, new_state

    def tell(
        self,
        state: dict,
        pre_cands: jnp.ndarray,
        pre_bsigs: jnp.ndarray,
        post_cands: jnp.ndarray,
        post_regrets: jnp.ndarray,
        post_bsigs: jnp.ndarray,
        epsilon: float,
    ) -> tuple[dict, dict]:
        """Update all N particle CMA states and apply Stein repulsion to means.

        Two-pass tell() design: train.py performs two evaluation rounds per step.
        Pre-pass results are used only for the diversity metric (dist_pre). Post-pass
        results (repelled candidates and regrets) are used for the actual CMA update
        and the Stein mean adjustment.

        Fitness sign: regret is negated before passing to evosax (evosax minimizes;
        we maximize regret). Stein repulsion is applied AFTER tell() — per CONTEXT
        step order 6: update CMA means using Stein gradient computed from post-bsigs.

        Args:
            state:        ES state dict with "particles" list of N dicts.
            pre_cands:    (N*pop_size, param_dim) — first-pass candidates.
            pre_bsigs:    (N*pop_size, D_bsig) — first-pass behavior signatures.
            post_cands:   (N*pop_size, param_dim) — second-pass (repelled) candidates.
            post_regrets: (N*pop_size,) — second-pass regrets (positive; will be negated).
            post_bsigs:   (N*pop_size, D_bsig) — second-pass behavior signatures.
            epsilon:      float — Stein repulsion step size.

        Returns:
            (new_state, metrics_dict) where:
                new_state:    updated state dict with N particles.
                metrics_dict: {"sv_behavior_dist_pre": float, "sv_behavior_dist_post": float}
        """
        # Lazy imports to avoid circular import between submodules.
        from .stein import compute_stein_repulsion, mean_pairwise_behavior_dist

        N = len(state["particles"])
        pop_size = self._pop_size
        dummy_key = jax.random.PRNGKey(0)

        # --- Pre-repulsion diversity metric ---
        # Average behavior sig per particle over its pop_size candidates.
        particle_pre_bsigs = jnp.stack([
            jnp.mean(pre_bsigs[i * pop_size:(i + 1) * pop_size], axis=0)
            for i in range(N)
        ])  # (N, D_bsig)
        dist_pre = mean_pairwise_behavior_dist(particle_pre_bsigs)

        # --- Update each particle with post-repulsion data ---
        new_particles = []
        for i, p in enumerate(state["particles"]):
            cands_i = post_cands[i * pop_size:(i + 1) * pop_size]     # (pop_size, param_dim)
            regrets_i = post_regrets[i * pop_size:(i + 1) * pop_size] # (pop_size,)
            fitness_i = -regrets_i  # negate: evosax minimizes, we maximize regret

            # evosax tell: tell(key, population, fitness, state, params) — 5 args
            new_es_state, _ = self._es_template.tell(
                dummy_key, cands_i, fitness_i, p["es_state"], p["es_params"]
            )
            new_particles.append({"es_state": new_es_state, "es_params": p["es_params"]})

        # --- Apply Stein repulsion to updated means (N > 1 only) ---
        # Stein step is after tell() — CONTEXT order step 6.
        # For N=1: skip repulsion (single particle, no peers, gradient is exactly zero).
        if N > 1:
            current_means = jnp.stack([
                p["es_state"].mean for p in new_particles
            ])  # (N, param_dim)

            # Per-particle post-repulsion behavior sigs (average over each particle's pop)
            particle_post_bsigs = jnp.stack([
                jnp.mean(post_bsigs[i * pop_size:(i + 1) * pop_size], axis=0)
                for i in range(N)
            ])  # (N, D_bsig)

            repulsion = compute_stein_repulsion(current_means, particle_post_bsigs, epsilon)

            for i in range(N):
                new_mean_i = new_particles[i]["es_state"].mean + repulsion[i]
                updated_es_state = new_particles[i]["es_state"].replace(mean=new_mean_i)
                new_particles[i] = {**new_particles[i], "es_state": updated_es_state}

        # --- Post-repulsion diversity metric ---
        if N > 1:
            # Reuse particle_post_bsigs computed above (already averaged per particle)
            particle_post_bsigs_final = jnp.stack([
                jnp.mean(post_bsigs[i * pop_size:(i + 1) * pop_size], axis=0)
                for i in range(N)
            ])  # (N, D_bsig)
            dist_post = mean_pairwise_behavior_dist(particle_post_bsigs_final)
        else:
            dist_post = 0.0

        new_state = {"particles": new_particles}
        metrics = {
            "sv_behavior_dist_pre": dist_pre,
            "sv_behavior_dist_post": dist_post,
        }
        return new_state, metrics
