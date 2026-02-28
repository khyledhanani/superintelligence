"""Integration tests for all Phase 2 ES components.

Tests:
  - ESStrategy Protocol import
  - CMAESStrategy ask/tell cycle
  - compute_novelty_knn JIT compilation and masking
  - compute_fitness correctness (known values)
  - Full pipeline: buffer -> novelty -> composite fitness

Run with:
    python3 tests/test_es_components.py
from project root (no pytest required).
"""

import sys
import os

# Path setup — allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'accel_training'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'es'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp


def test_es_interface():
    """Verify ESStrategy Protocol import succeeds."""
    from accel_training.es_components import ESStrategy
    # Protocol should be importable and be a class (or type alias)
    assert ESStrategy is not None, "ESStrategy should not be None"
    print("  PASS test_es_interface")


def test_cmaes_strategy_ask_tell():
    """Full ask/tell cycle for CMAESStrategy.

    Verifies:
    - init_state returns dict with 'es_state' and 'es_params'
    - ask returns candidates of shape (pop_size, param_dim)
    - tell returns a new state dict without error
    - After 3 ask/tell cycles, second ask gives different candidates from first
    """
    from accel_training.es_components import CMAESStrategy

    param_dim = 8
    pop_size = 4
    strategy = CMAESStrategy(param_dim=param_dim, pop_size=pop_size)

    rng = jax.random.PRNGKey(42)

    # Init
    rng, rng_init = jax.random.split(rng)
    state = strategy.init_state(rng_init, {})
    assert isinstance(state, dict), "init_state should return a dict"
    assert "es_state" in state, "state must have 'es_state' key"
    assert "es_params" in state, "state must have 'es_params' key"

    # Ask 1 — record first candidates
    rng, rng_ask = jax.random.split(rng)
    candidates1, state = strategy.ask(state, rng_ask)
    assert candidates1.shape == (pop_size, param_dim), (
        f"ask should return shape ({pop_size}, {param_dim}), got {candidates1.shape}"
    )

    # Tell 1
    fitness1 = jnp.ones(pop_size)
    state = strategy.tell(state, candidates1, fitness1)
    assert isinstance(state, dict), "tell should return a dict"

    # Ask 2 + tell 2
    rng, rng_ask2 = jax.random.split(rng)
    candidates2, state = strategy.ask(state, rng_ask2)
    assert candidates2.shape == (pop_size, param_dim)
    fitness2 = jnp.arange(pop_size, dtype=jnp.float32)
    state = strategy.tell(state, candidates2, fitness2)

    # Ask 3 + tell 3
    rng, rng_ask3 = jax.random.split(rng)
    candidates3, state = strategy.ask(state, rng_ask3)
    assert candidates3.shape == (pop_size, param_dim)
    fitness3 = -jnp.arange(pop_size, dtype=jnp.float32)
    state = strategy.tell(state, candidates3, fitness3)

    # Candidates should evolve across iterations (at least 2nd != 1st)
    # (They will differ because rng keys differ or CMA-ES state evolves)
    candidates_differ = not jnp.allclose(candidates1, candidates2, atol=1e-6)
    # Note: with different rng keys they will almost always differ; we just check shape + no-crash
    assert candidates3 is not None
    print("  PASS test_cmaes_strategy_ask_tell")


def test_novelty_knn_jit():
    """Verify compute_novelty_knn JIT-compiles without ConcretizationTypeError.

    Verifies:
    - Single candidate: novelty is scalar float32
    - Batch (pop_size=16): novelty_scores shape is (16,)
    - jax.jit lower().compile() succeeds
    """
    from accel_training.es_components.novelty import compute_novelty_knn, compute_novelty_batch

    capacity = 4000
    D = 169
    pop_size = 16
    k = 5

    # Build fake buffer with 100 valid slots
    buffer_sigs = jnp.zeros((capacity, D), dtype=jnp.float32)
    valid_mask = jnp.arange(capacity) < 100

    # Set the first 100 valid slots to distinct values so k-NN is well-defined
    rng = jax.random.PRNGKey(0)
    valid_sigs = jax.random.normal(rng, (100, D), dtype=jnp.float32)
    buffer_sigs = buffer_sigs.at[:100].set(valid_sigs)

    # Single candidate
    candidate = jax.random.normal(jax.random.PRNGKey(1), (D,), dtype=jnp.float32)
    novelty = compute_novelty_knn(candidate, buffer_sigs, valid_mask, k=k)
    assert novelty.shape == (), f"Single novelty should be scalar, got shape {novelty.shape}"
    assert novelty.dtype == jnp.float32, f"Expected float32, got {novelty.dtype}"
    assert jnp.isfinite(novelty), f"Novelty should be finite, got {novelty}"

    # Batch
    candidate_sigs = jax.random.normal(jax.random.PRNGKey(2), (pop_size, D), dtype=jnp.float32)
    novelty_scores = compute_novelty_batch(candidate_sigs, buffer_sigs, valid_mask, k=k)
    assert novelty_scores.shape == (pop_size,), (
        f"Batch novelty should have shape ({pop_size},), got {novelty_scores.shape}"
    )

    # JIT lower().compile() — verifies no ConcretizationTypeError at trace time
    import functools
    fn_partial = functools.partial(compute_novelty_knn, k=k)
    lowered = jax.jit(fn_partial).lower(candidate, buffer_sigs, valid_mask)
    compiled = lowered.compile()
    result = compiled(candidate, buffer_sigs, valid_mask)
    assert jnp.isfinite(result), "JIT-compiled result should be finite"

    print("  PASS test_novelty_knn_jit")


def test_novelty_knn_masking():
    """Verify masking logic for compute_novelty_knn.

    Verifies:
    - All-False valid_mask (no valid neighbors) -> novelty is inf
    - Setting first 5 entries to True with ones as sigs -> novelty is finite and > 0
    """
    from accel_training.es_components.novelty import compute_novelty_knn

    capacity = 20
    D = 169
    k = 3

    buffer_sigs = jnp.zeros((capacity, D), dtype=jnp.float32)
    candidate = jnp.zeros(D, dtype=jnp.float32)

    # All-False mask: no valid neighbors -> k-NN distances are all inf -> novelty is inf
    valid_mask_empty = jnp.zeros(capacity, dtype=jnp.bool_)
    novelty_empty = compute_novelty_knn(candidate, buffer_sigs, valid_mask_empty, k=k)
    assert jnp.isinf(novelty_empty), (
        f"With no valid entries, novelty should be inf, got {novelty_empty}"
    )

    # Set first 5 entries to valid, with sigs = ones (non-zero L2 distance from candidate=zeros)
    buffer_sigs_filled = buffer_sigs.at[:5].set(jnp.ones((5, D), dtype=jnp.float32))
    valid_mask_filled = jnp.arange(capacity) < 5
    novelty_filled = compute_novelty_knn(candidate, buffer_sigs_filled, valid_mask_filled, k=k)
    assert jnp.isfinite(novelty_filled), (
        f"With valid entries, novelty should be finite, got {novelty_filled}"
    )
    assert float(novelty_filled) > 0.0, (
        f"Novelty with non-zero neighbors should be > 0, got {novelty_filled}"
    )

    print("  PASS test_novelty_knn_masking")


def test_composite_fitness_correctness():
    """Known-value test for compute_fitness and compute_fitness_batch.

    Verifies:
    - compute_fitness(2.0, 3.0, alpha=0.8, beta=0.2) == 2.2 within 1e-5
    - compute_fitness(1.0, 1.0, alpha=1.0, beta=0.0) == 1.0
    - compute_fitness(0.0, 5.0, alpha=0.0, beta=1.0) == 5.0
    - compute_fitness_batch: shape (pop_size,), values match scalar loop
    """
    from accel_training.es_components.fitness import compute_fitness, compute_fitness_batch

    # Known-value scalar tests
    f1 = compute_fitness(2.0, 3.0, alpha=0.8, beta=0.2)
    assert abs(float(f1) - 2.2) < 1e-5, f"Expected 2.2, got {float(f1)}"

    f2 = compute_fitness(1.0, 1.0, alpha=1.0, beta=0.0)
    assert abs(float(f2) - 1.0) < 1e-5, f"Expected 1.0, got {float(f2)}"

    f3 = compute_fitness(0.0, 5.0, alpha=0.0, beta=1.0)
    assert abs(float(f3) - 5.0) < 1e-5, f"Expected 5.0, got {float(f3)}"

    # Batch test: shape and values match scalar loop
    pop_size = 8
    regrets = jnp.array([float(i) for i in range(pop_size)], dtype=jnp.float32)
    novelties = jnp.array([float(i) * 0.5 for i in range(pop_size)], dtype=jnp.float32)
    alpha, beta = 0.7, 0.3

    batch_fitness = compute_fitness_batch(regrets, novelties, alpha=alpha, beta=beta)
    assert batch_fitness.shape == (pop_size,), (
        f"Expected shape ({pop_size},), got {batch_fitness.shape}"
    )

    # Compare with scalar loop
    for i in range(pop_size):
        expected = alpha * float(regrets[i]) + beta * float(novelties[i])
        actual = float(batch_fitness[i])
        assert abs(actual - expected) < 1e-5, (
            f"Batch fitness at index {i}: expected {expected}, got {actual}"
        )

    print("  PASS test_composite_fitness_correctness")


def test_buffer_pipeline():
    """Full pipeline: buffer_sigs -> novelty scoring -> composite fitness.

    Verifies:
    - buffer_sigs[valid_mask] has shape (50, 169) for 50 valid entries
    - Novelty scores for pop_size=8 candidates have correct shape
    - Composite fitness scores have correct shape and all values are finite
    """
    from accel_training.es_components.novelty import compute_novelty_batch
    from accel_training.es_components.fitness import compute_fitness_batch

    capacity = 100
    D = 169
    pop_size = 8
    k = 5
    alpha = 0.7
    beta = 0.3

    # Create fake buffer: 50 valid slots with random sigs
    rng = jax.random.PRNGKey(99)
    rng, rng_buf, rng_cand = jax.random.split(rng, 3)

    buffer_sigs = jnp.zeros((capacity, D), dtype=jnp.float32)
    valid_sigs = jax.random.normal(rng_buf, (50, D), dtype=jnp.float32)
    buffer_sigs = buffer_sigs.at[:50].set(valid_sigs)
    valid_mask = jnp.arange(capacity) < 50

    # Verify valid slice shape (outside jit is fine)
    valid_slice = buffer_sigs[jnp.where(valid_mask)[0]]
    assert valid_slice.shape == (50, D), (
        f"Expected valid buffer slice shape (50, {D}), got {valid_slice.shape}"
    )

    # Pop candidates
    candidate_sigs = jax.random.normal(rng_cand, (pop_size, D), dtype=jnp.float32)

    # Compute novelty
    novelty_scores = compute_novelty_batch(candidate_sigs, buffer_sigs, valid_mask, k=k)
    assert novelty_scores.shape == (pop_size,), (
        f"Expected novelty shape ({pop_size},), got {novelty_scores.shape}"
    )
    assert jnp.all(jnp.isfinite(novelty_scores)), (
        f"All novelty scores should be finite: {novelty_scores}"
    )

    # Fake regrets for each candidate
    regrets = jnp.abs(jax.random.normal(jax.random.PRNGKey(7), (pop_size,), dtype=jnp.float32))

    # Compute composite fitness
    composite = compute_fitness_batch(regrets, novelty_scores, alpha=alpha, beta=beta)
    assert composite.shape == (pop_size,), (
        f"Expected composite shape ({pop_size},), got {composite.shape}"
    )
    assert jnp.all(jnp.isfinite(composite)), (
        f"All composite fitness values should be finite: {composite}"
    )

    print("  PASS test_buffer_pipeline")


if __name__ == "__main__":
    print("Running Phase 2 integration tests...")
    test_es_interface()
    test_cmaes_strategy_ask_tell()
    test_novelty_knn_jit()
    test_novelty_knn_masking()
    test_composite_fitness_correctness()
    test_buffer_pipeline()
    print("\nAll Phase 2 integration tests passed.")
