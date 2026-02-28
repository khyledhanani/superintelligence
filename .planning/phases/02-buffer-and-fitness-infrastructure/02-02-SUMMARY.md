---
phase: 02-buffer-and-fitness-infrastructure
plan: 02
subsystem: infra
tags: [jax, jit, knn, novelty, fitness, curriculum, behavioral-diversity]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: behavior signature (169-dim L1-normalized histogram) consumed by k-NN distance computation

provides:
  - JIT-compatible k-NN novelty scoring over the replay buffer (compute_novelty_knn, compute_novelty_batch)
  - Composite fitness function F = alpha*regret + beta*novelty (compute_fitness, compute_fitness_batch)
  - accel_training/es_components/ package with __init__.py

affects:
  - Phase 3 (NS-ES wiring): novelty.py and fitness.py are the mathematical primitives wired into the ES training loop
  - Phase 5 (SV-CMA-ES): fitness function signature must remain stable

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Masked k-NN: use jnp.where(valid_mask, sq_dists, jnp.inf) to exclude empty buffer slots — JIT-safe, no dynamic slicing"
    - "jax.lax.top_k negate trick: negate distances to get k-smallest via top_k (which returns largest)"
    - "Static k via functools.partial + static_argnames so k is a compile-time constant under jit"
    - "vmap over candidate batch axis with buffer/mask broadcast (in_axes=(0, None, None))"
    - "Sign convention: compute_fitness returns higher-is-better; caller negates before passing to evosax"

key-files:
  created:
    - accel_training/__init__.py
    - accel_training/es_components/__init__.py
    - accel_training/es_components/novelty.py
    - accel_training/es_components/fitness.py
  modified: []

key-decisions:
  - "k=5 nearest neighbors as default (from research recommendation); exposed as static parameter"
  - "No normalization of regret or novelty in compute_fitness — raw combination, caller responsible for negating for evosax"
  - "alpha and beta are Python floats passed at call time, not stored in JAX state — keeps function pure and avoids retracing"

patterns-established:
  - "Pattern: Masked brute-force k-NN with jnp.where + inf masking for JIT-safe variable-fill buffers"
  - "Pattern: top_k negate trick for k-smallest distances without dynamic indexing"
  - "Pattern: vmap(partial(jit_fn, k=k), in_axes=(0, None, None)) for batched novelty scoring"

requirements-completed: [INFRA-03, INFRA-04]

# Metrics
duration: 8min
completed: 2026-02-28
---

# Phase 02 Plan 02: Novelty Scoring and Fitness Function Summary

**Masked k-NN novelty scoring (brute-force L2, JIT-safe via jnp.where+inf) and composite fitness F = alpha*regret + beta*novelty for behavioral diversity curriculum**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-28T18:51:07Z
- **Completed:** 2026-02-28T18:59:00Z
- **Tasks:** 2
- **Files modified:** 4 (created)

## Accomplishments

- JIT-compatible k-NN novelty scorer using masked brute-force L2 distance; no ConcretizationTypeError; handles variable buffer fill via jnp.where masking
- Batched novelty via vmap over pop_size candidates with buffer/mask broadcast
- Pure composite fitness function with no normalization and explicit sign convention (caller negates for evosax)
- Established accel_training/es_components/ package structure

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement JIT-compatible k-NN novelty scorer** - `d696695` (feat)
2. **Task 2: Implement composite fitness function** - `5f2f3b7` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `accel_training/__init__.py` - Package init for accel_training module
- `accel_training/es_components/__init__.py` - Package init for es_components subpackage
- `accel_training/es_components/novelty.py` - compute_novelty_knn (JIT, masked k-NN) and compute_novelty_batch (vmap)
- `accel_training/es_components/fitness.py` - compute_fitness and compute_fitness_batch (F = alpha*regret + beta*novelty)

## Decisions Made

- k=5 as default following research recommendation; k is static (functools.partial + static_argnames) so JAX can compile without retracing per value
- No normalization in compute_fitness: raw combination keeps the function pure and easy to test; caller is responsible for sign convention (negate before evosax)
- alpha and beta are plain Python floats (not JAX arrays): avoids JAX state management complexity and matches ES config dict structure planned for Phase 3

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created missing accel_training package init files**
- **Found during:** Task 1 (novelty scorer implementation)
- **Issue:** accel_training had no __init__.py and es_components directory did not exist; Python imports would fail
- **Fix:** Created accel_training/__init__.py and accel_training/es_components/__init__.py as package markers
- **Files modified:** accel_training/__init__.py, accel_training/es_components/__init__.py
- **Verification:** Import `from accel_training.es_components.novelty import compute_novelty_knn` succeeds
- **Committed in:** d696695 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking — missing package structure)
**Impact on plan:** Required for Python package imports to work. No scope creep.

## Issues Encountered

- Default `python3` on the machine does not have JAX. Used `/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python` (jax_env conda environment, JAX 0.5.3) for all verification commands.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- novelty.py and fitness.py are the mathematical primitives needed for NS-ES wiring in Phase 3
- compute_novelty_batch(candidate_sigs, buffer_sigs, valid_mask, k=5) produces (pop_size,) novelty scores ready for fitness combination
- compute_fitness_batch(regrets, novelties, alpha, beta) produces (pop_size,) composite scores; caller must negate before passing to evosax
- Buffer infrastructure (plan 02-01) provides the buffer_sigs and valid_mask arrays consumed here

---
*Phase: 02-buffer-and-fitness-infrastructure*
*Completed: 2026-02-28*

## Self-Check: PASSED

- FOUND: accel_training/es_components/novelty.py
- FOUND: accel_training/es_components/fitness.py
- FOUND: .planning/phases/02-buffer-and-fitness-infrastructure/02-02-SUMMARY.md
- FOUND: commit d696695 (Task 1 - k-NN novelty scorer)
- FOUND: commit 5f2f3b7 (Task 2 - composite fitness function)
