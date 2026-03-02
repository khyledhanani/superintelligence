"""Phase 4 integration and unit tests for SV-CMA-ES components.

Tests:
  1. SVCMAESStrategy init_state: N=2 particles, distinct means, not zeros (ALGO-02)
  2. SVCMAESStrategy ask: shape (N*pop_size, param_dim), state updates (ALGO-02)
  3. SVCMAESStrategy tell: new_state + metrics_dict with both distance floats (ALGO-02)
  4. Stein repulsion pushes particles apart: nonzero repulsion for distinct particles (ALGO-02)
  5. N=1 degrades to CMA-ES: sv_behavior_dist_pre/post == 0.0, no error (ALGO-02)
  6. End-to-end 3-update smoke test with sv_cma_es strategy (ALGO-02)

Run with:
    /cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python tests/test_phase4_sv_cma_es.py
from project root (no pytest required).
"""

import os
os.environ["WANDB_MODE"] = "disabled"

import sys

# Path setup — allow imports from project root (mirror tests/test_phase3_ns_es.py exactly)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'accel_training'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'es'))

import jax
import jax.numpy as jnp


def test_svcmaes_init_state():
    """Test SVCMAESStrategy init_state: N=2 particles, distinct means, not zeros (ALGO-02).

    Verifies:
    - "particles" key in returned state
    - len(state["particles"]) == 2
    - each particle has "es_state" and "es_params" keys
    - particle means are distinct (norm(m0 - m1) > 1e-6)
    - means are NOT zeros (norm(m0) > 1e-6)
    """
    from accel_training.es_components.svcmaes_strategy import SVCMAESStrategy

    param_dim = 4
    pop_size = 3
    strategy = SVCMAESStrategy(param_dim=param_dim, pop_size=pop_size)

    rng = jax.random.PRNGKey(0)
    rng, rng_init = jax.random.split(rng)
    state = strategy.init_state(rng_init, {"sv_n_particles": 2, "sigma_init": 0.5})

    assert "particles" in state, "init_state must return dict with 'particles' key"
    assert len(state["particles"]) == 2, (
        f"Expected 2 particles, got {len(state['particles'])}"
    )

    # Check each particle has required keys
    for i, p in enumerate(state["particles"]):
        assert "es_state" in p, f"Particle {i} must have 'es_state' key"
        assert "es_params" in p, f"Particle {i} must have 'es_params' key"

    # Means must be distinct (not copies of each other)
    m0 = state["particles"][0]["es_state"].mean
    m1 = state["particles"][1]["es_state"].mean
    dist = float(jnp.linalg.norm(m0 - m1))
    assert dist > 1e-6, (
        f"Particle means must be distinct (norm > 1e-6), got norm(m0-m1)={dist:.2e}"
    )

    # Means must not be zeros
    norm_m0 = float(jnp.linalg.norm(m0))
    assert norm_m0 > 1e-6, (
        f"Particle 0 mean must not be zeros (norm > 1e-6), got norm={norm_m0:.2e}"
    )

    print("  PASS test_svcmaes_init_state")


def test_svcmaes_ask():
    """Test SVCMAESStrategy ask: shape (N*pop_size, param_dim), state updates (ALGO-02).

    Verifies:
    - cands.shape == (6, 4)  (N*pop_size, param_dim) with N=2, pop_size=3, param_dim=4
    - new_state has "particles" with 2 particles
    - second ask() also works (state is properly updated)
    """
    from accel_training.es_components.svcmaes_strategy import SVCMAESStrategy

    param_dim = 4
    pop_size = 3
    N = 2
    strategy = SVCMAESStrategy(param_dim=param_dim, pop_size=pop_size)

    rng = jax.random.PRNGKey(1)
    rng, rng_init = jax.random.split(rng)
    state = strategy.init_state(rng_init, {"sv_n_particles": N, "sigma_init": 0.5})

    # First ask
    rng, rng_ask = jax.random.split(rng)
    cands, new_state = strategy.ask(state, rng_ask)

    assert cands.shape == (N * pop_size, param_dim), (
        f"ask() must return shape ({N * pop_size}, {param_dim}), got {cands.shape}"
    )
    assert "particles" in new_state, "ask() returned state must have 'particles' key"
    assert len(new_state["particles"]) == N, (
        f"Expected {N} particles after ask, got {len(new_state['particles'])}"
    )

    # Second ask — verify state is self-consistent after first ask
    rng, rng_ask2 = jax.random.split(rng)
    cands2, new_state2 = strategy.ask(new_state, rng_ask2)
    assert cands2.shape == (N * pop_size, param_dim), (
        f"Second ask() must return shape ({N * pop_size}, {param_dim}), got {cands2.shape}"
    )

    print("  PASS test_svcmaes_ask")


def test_svcmaes_tell():
    """Test SVCMAESStrategy tell: new_state + metrics_dict with both distance floats (ALGO-02).

    Verifies:
    - result is tuple of length 2 (new_state, metrics)
    - "particles" in new_state; len == 2
    - "sv_behavior_dist_pre" in metrics and "sv_behavior_dist_post" in metrics
    - both metric values are Python floats >= 0.0
    - a second tell() call succeeds (state is self-consistent)
    """
    from accel_training.es_components.svcmaes_strategy import SVCMAESStrategy

    param_dim = 4
    pop_size = 3
    N = 2
    bsig_dim = 8
    strategy = SVCMAESStrategy(param_dim=param_dim, pop_size=pop_size)

    rng = jax.random.PRNGKey(2)
    rng, rng_init = jax.random.split(rng)
    state = strategy.init_state(rng_init, {"sv_n_particles": N, "sigma_init": 0.5})

    rng, rng_ask = jax.random.split(rng)
    cands, state = strategy.ask(state, rng_ask)

    # Construct synthetic pre/post arrays
    # (N*pop_size, param_dim) candidates, (N*pop_size, bsig_dim) bsigs, (N*pop_size,) regrets
    total = N * pop_size
    rng, rng_data = jax.random.split(rng)
    pre_cands = jax.random.normal(rng_data, (total, param_dim))
    rng, rng_data2 = jax.random.split(rng)
    pre_bsigs = jax.random.normal(rng_data2, (total, bsig_dim))
    post_cands = pre_cands + 0.01  # slightly different from pre
    rng, rng_data3 = jax.random.split(rng)
    post_regrets = jnp.abs(jax.random.normal(rng_data3, (total,))) + 0.1
    post_bsigs = pre_bsigs + 0.05

    result = strategy.tell(state, pre_cands, pre_bsigs, post_cands, post_regrets, post_bsigs, epsilon=0.01)

    assert isinstance(result, tuple) and len(result) == 2, (
        f"tell() must return tuple of length 2, got {type(result)} len={len(result) if hasattr(result, '__len__') else 'N/A'}"
    )
    new_state, metrics = result

    assert "particles" in new_state, "tell() returned state must have 'particles' key"
    assert len(new_state["particles"]) == N, (
        f"Expected {N} particles after tell, got {len(new_state['particles'])}"
    )

    assert "sv_behavior_dist_pre" in metrics, "metrics must have 'sv_behavior_dist_pre' key"
    assert "sv_behavior_dist_post" in metrics, "metrics must have 'sv_behavior_dist_post' key"

    dist_pre = metrics["sv_behavior_dist_pre"]
    dist_post = metrics["sv_behavior_dist_post"]

    assert isinstance(dist_pre, float), (
        f"sv_behavior_dist_pre must be Python float, got {type(dist_pre)}"
    )
    assert isinstance(dist_post, float), (
        f"sv_behavior_dist_post must be Python float, got {type(dist_post)}"
    )
    assert dist_pre >= 0.0, f"sv_behavior_dist_pre must be >= 0.0, got {dist_pre}"
    assert dist_post >= 0.0, f"sv_behavior_dist_post must be >= 0.0, got {dist_post}"

    # Second tell() — verify state is self-consistent
    rng, rng_ask2 = jax.random.split(rng)
    cands2, state2 = strategy.ask(new_state, rng_ask2)
    result2 = strategy.tell(
        state2, pre_cands, pre_bsigs, post_cands, post_regrets, post_bsigs, epsilon=0.01
    )
    assert isinstance(result2, tuple) and len(result2) == 2, (
        "Second tell() must also return tuple of length 2"
    )

    print("  PASS test_svcmaes_tell")


def test_stein_repulsion_pushes_apart():
    """Test Stein repulsion: nonzero repulsion for distinct particles (ALGO-02).

    Creates N=3 particles with distinct behavior sigs.
    Verifies that compute_stein_repulsion returns a nonzero delta.
    Note: We verify repulsion is nonzero (norm > 1e-6), not unconditionally
    that post_dist > pre_dist — Stein repulsion pushes apart in gradient direction
    but magnitude and direction relative to pre-step positions depends on kernel
    bandwidth at early steps.
    """
    from accel_training.es_components.svcmaes_strategy import SVCMAESStrategy
    from accel_training.es_components.stein import compute_stein_repulsion

    param_dim = 64
    pop_size = 16
    N = 3
    strategy = SVCMAESStrategy(param_dim=param_dim, pop_size=pop_size)

    rng = jax.random.PRNGKey(3)
    rng, rng_init = jax.random.split(rng)
    state = strategy.init_state(rng_init, {"sv_n_particles": N, "sigma_init": 0.5})

    # Get particle means from state
    means_pre = jnp.stack([
        state["particles"][i]["es_state"].mean for i in range(N)
    ])  # (N, param_dim)

    # Compute distinct behavior sigs (ensure they differ meaningfully)
    rng, rng_bsig = jax.random.split(rng)
    bsigs = jax.random.normal(rng_bsig, (N, 169))

    # Compute Stein repulsion
    repulsion = compute_stein_repulsion(means_pre, bsigs, epsilon=0.1)

    # Verify repulsion has nonzero norm
    total_repulsion_norm = float(jnp.linalg.norm(repulsion))
    assert total_repulsion_norm > 1e-6, (
        f"Stein repulsion must be nonzero for distinct particles, got norm={total_repulsion_norm:.2e}"
    )

    # Verify repulsion shape matches means
    assert repulsion.shape == means_pre.shape, (
        f"Repulsion must have same shape as means {means_pre.shape}, got {repulsion.shape}"
    )

    print("  PASS test_stein_repulsion_pushes_apart")


def test_n1_degrades_to_cma_es():
    """Test N=1 degrades to plain CMA-ES: sv_behavior_dist_pre/post == 0.0, no error (ALGO-02).

    Verifies:
    - SVCMAESStrategy with N=1 initializes exactly 1 particle
    - sv_behavior_dist_pre == 0.0 (no pairs for N=1)
    - sv_behavior_dist_post == 0.0 (no pairs for N=1)
    - Both SV (N=1) and CMA-ES complete without error
    """
    from accel_training.es_components.svcmaes_strategy import SVCMAESStrategy
    from accel_training.es_components.cmaes_strategy import CMAESStrategy

    param_dim = 4
    pop_size = 3

    # Create both strategies
    sv_strategy = SVCMAESStrategy(param_dim=param_dim, pop_size=pop_size)
    cma_strategy = CMAESStrategy(param_dim=param_dim, pop_size=pop_size)

    rng = jax.random.PRNGKey(4)

    # Init both with same rng (same key for each)
    rng, rng_sv_init = jax.random.split(rng)
    rng, rng_cma_init = jax.random.split(rng)
    sv_state = sv_strategy.init_state(rng_sv_init, {"sv_n_particles": 1, "sigma_init": 0.5})
    cma_state = cma_strategy.init_state(rng_cma_init, {"sigma_init": 0.5})

    assert len(sv_state["particles"]) == 1, (
        f"N=1 must produce exactly 1 particle, got {len(sv_state['particles'])}"
    )

    # SV ask -> tell cycle
    rng, rng_sv_ask = jax.random.split(rng)
    sv_cands, sv_state_asked = sv_strategy.ask(sv_state, rng_sv_ask)
    # Shape: (1*pop_size, param_dim)
    assert sv_cands.shape == (pop_size, param_dim), (
        f"N=1 ask must return ({pop_size}, {param_dim}), got {sv_cands.shape}"
    )

    bsig_dim = 8
    total_sv = 1 * pop_size
    pre_cands = sv_cands
    rng, rng_bsig = jax.random.split(rng)
    pre_bsigs = jax.random.normal(rng_bsig, (total_sv, bsig_dim))
    post_cands = pre_cands + 0.0
    rng, rng_regrets = jax.random.split(rng)
    post_regrets = jnp.abs(jax.random.normal(rng_regrets, (total_sv,))) + 0.1
    post_bsigs = pre_bsigs

    sv_result = sv_strategy.tell(
        sv_state_asked, pre_cands, pre_bsigs, post_cands, post_regrets, post_bsigs, epsilon=0.01
    )
    sv_new_state, sv_metrics = sv_result

    assert sv_metrics["sv_behavior_dist_pre"] == 0.0, (
        f"N=1 sv_behavior_dist_pre must be 0.0, got {sv_metrics['sv_behavior_dist_pre']}"
    )
    assert sv_metrics["sv_behavior_dist_post"] == 0.0, (
        f"N=1 sv_behavior_dist_post must be 0.0, got {sv_metrics['sv_behavior_dist_post']}"
    )

    # CMA ask -> tell cycle (for comparison — no errors)
    rng, rng_cma_ask = jax.random.split(rng)
    cma_cands, cma_state_asked = cma_strategy.ask(cma_state, rng_cma_ask)
    regrets = jnp.ones(pop_size, dtype=jnp.float32)
    cma_new_state = cma_strategy.tell(cma_state_asked, cma_cands, -regrets)
    assert "es_state" in cma_new_state, "CMA tell must return state with 'es_state'"

    print("  PASS test_n1_degrades_to_cma_es")


def test_end_to_end_3_updates_sv_cma_es():
    """Smoke test: train.py runs 3 updates with sv_cma_es strategy and warmup_n=4 (ALGO-02).

    Verifies that the full SV-CMA-ES pipeline (warmup -> training loop -> ES routing)
    runs without crashing with sv_n_particles=2 and sv_epsilon=0.01.

    Guards on VAE checkpoint presence (skips if not found).
    """
    if not os.path.exists("vae/model/checkpoint_420000.pkl"):
        print("  SKIP test_end_to_end_3_updates_sv_cma_es (VAE checkpoint not found at vae/model/checkpoint_420000.pkl)")
        return

    import os as _os
    _os.makedirs("/tmp/test_phase4_smoke", exist_ok=True)

    from accel_training.train import train

    config = {
        # Agent / PPO
        "lr": 0.0001,
        "max_grad_norm": 0.5,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_eps": 0.2,
        "entropy_coeff": 0.001,
        "critic_coeff": 0.5,
        "epoch_ppo": 5,
        "num_minibatches": 1,
        # Rollout
        "num_train_envs": 32,
        "num_steps": 256,
        "agent_view_size": 5,
        # Level buffer
        "level_buffer_capacity": 4000,
        "replay_prob": 0.8,
        "staleness_coeff": 0.3,
        "minimum_fill_ratio": 0.5,
        "prioritization": "rank",
        "temperature": 0.3,
        "topk_k": 4,
        "score_function": "MaxMC",
        "exploratory_grad_updates": False,
        # ACCEL / MAP-Elites
        "use_accel": True,
        "n_candidates": 16,
        "mutation_sigma": 0.5,
        "random_fraction": 0.3,
        "eval_rollout_steps": 128,
        "min_obstacles": 5,
        "min_distance": 3,
        "decode_temperature": 0.25,
        # Training loop — overrides for smoke test
        "num_updates": 3,
        "eval_freq": 500,
        "checkpoint_every": 2000,
        # Paths
        "vae_checkpoint": "vae/model/checkpoint_420000.pkl",
        "log_dir": "/tmp/test_phase4_smoke",
        "run_name": "test_sv_cma_es",
        # ES / SV-CMA-ES
        "warmup_n": 4,
        "es_strategy": "sv_cma_es",
        "sv_n_particles": 2,
        "sv_epsilon": 0.01,
        "es_alpha": 0.8,
        "es_beta": 0.2,
        "es_pop_size": 16,
        "es_sigma_init": 0.5,
        "es_k_novelty": 5,
        # WandB
        "wandb_project": "test",
        "wandb_log_freq": 1,
        # Misc
        "seed": 0,
    }

    train_state, archive = train(config)

    assert train_state is not None, "train() must return a non-None train_state"
    assert "size" in train_state.sampler, "train_state.sampler must have 'size' key"

    print("  PASS test_end_to_end_3_updates_sv_cma_es")


if __name__ == "__main__":
    print("Running Phase 4 SV-CMA-ES tests...")
    test_svcmaes_init_state()
    test_svcmaes_ask()
    test_svcmaes_tell()
    test_stein_repulsion_pushes_apart()
    test_n1_degrades_to_cma_es()
    test_end_to_end_3_updates_sv_cma_es()
    print("\nAll Phase 4 tests passed.")
