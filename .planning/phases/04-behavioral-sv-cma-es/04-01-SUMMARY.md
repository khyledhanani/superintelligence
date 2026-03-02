---
phase: 04-behavioral-sv-cma-es
plan: 01
subsystem: es-algorithm
tags: [jax, evosax, cma-es, svgd, stein-repulsion, behavior-diversity, latent-space]

# Dependency graph
requires:
  - phase: 03-ns-es-integration
    provides: CMAESStrategy and NSESStrategy patterns; evosax CMA_ES.tell/ask/init API verified

provides:
  - compute_stein_repulsion(): RBF kernel in behavior space, gradient applied to latent means
  - mean_pairwise_behavior_dist(): diversity scalar metric as Python float
  - SVCMAESStrategy class with init_state/ask/tell for N-particle SVGD-augmented CMA-ES

affects:
  - 04-02 (train.py wiring for sv_cma_es routing and two-pass eval loop)
  - 04-03 (ablation and plotting using SVCMAESStrategy metrics)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SVGD repulsion in behavior space (D=169) applied to latent means (D=64) — distinct kernel and gradient spaces"
    - "N=1 short-circuit: if N == 1 return zeros_like(means) before any JAX computation (avoids log(1)=0 NaN)"
    - "Median heuristic bandwidth: h = median(sq_dists) / log(N+1) with 1e-8 floor"
    - "tell() receives pre- and post-repulsion arrays; Stein mean update applied after evosax tell()"
    - "Particle means initialized N(0, sigma_init) per particle — not zeros — to ensure nonzero initial gradient"

key-files:
  created:
    - accel_training/es_components/stein.py
    - accel_training/es_components/svcmaes_strategy.py
  modified: []

key-decisions:
  - "N=1 short-circuit in compute_stein_repulsion: return zeros_like before log(N+1) to avoid float32 precision NaN (log(float32(1)+1e-8)=0 in JAX)"
  - "Bandwidth formula uses log(N+1) not log(N): ensures denominator >= log(2) > 0 for all N >= 1"
  - "Fitness for each particle tell() is pure negated regret — no composite fitness; Stein repulsion replaces novelty bonus"
  - "Stein mean update applied after evosax tell() — CONTEXT step 6 order honored"
  - "particle_post_bsigs recomputed for dist_post metric (minor redundancy accepted for clarity)"

patterns-established:
  - "SVCMAESStrategy state: {'particles': [{'es_state': ..., 'es_params': ...}, ...]} — mirrors CMAESStrategy/NSESStrategy pattern"
  - "Lazy imports inside tell() for circular import safety (from .stein import ...)"
  - "sys.path setup at module level matching nses_strategy.py convention"

requirements-completed: [ALGO-02]

# Metrics
duration: 8min
completed: 2026-03-02
---

# Phase 4 Plan 01: Behavioral SV-CMA-ES Core Algorithm Summary

**SVGD-augmented N-particle CMA-ES with RBF kernel in behavior space (D=169) and Stein mean updates in latent space (D=64) after each tell() step**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-02T20:31:27Z
- **Completed:** 2026-03-02T20:39:00Z
- **Tasks:** 2
- **Files modified:** 2 (both created)

## Accomplishments

- Pure JAX stein.py implementing SVGD (Liu & Wang 2016) with median-heuristic RBF kernel; N=1 returns zeros safely without NaN
- SVCMAESStrategy with N-particle init (distinct random means), concatenated ask(), and two-pass tell() with post-tell Stein mean adjustment
- Full shape contract verified: N=2, pop_size=3, param_dim=4; N=1 no-NaN guarantee; diversity metrics as Python floats

## Task Commits

Each task was committed atomically:

1. **Task 1: stein.py — pure Stein repulsion functions** - `394d2eb` (feat)
2. **Task 2: svcmaes_strategy.py — SVCMAESStrategy class** - `26af432` (feat)

## Files Created/Modified

- `accel_training/es_components/stein.py` - compute_stein_repulsion() and mean_pairwise_behavior_dist() pure JAX functions
- `accel_training/es_components/svcmaes_strategy.py` - SVCMAESStrategy class wrapping N evosax CMA_ES instances

## Decisions Made

- **N=1 short-circuit:** Added `if N == 1: return jnp.zeros_like(means)` at the top of compute_stein_repulsion(). Without this, `jnp.log(jnp.float32(1) + 1e-8)` evaluates to exactly 0.0 in float32 (1e-8 < float32 machine epsilon ~1.2e-7), causing 0/0 = NaN. The plan suggested using `log(N + 1e-8)` as the guard but this is insufficient in float32.
- **Bandwidth denominator uses log(N+1):** Changed from plan's `log(N + 1e-8)` to `log(N + 1.0)` for N >= 2, ensuring the denominator is always >= log(3) > 0. This is a numerically stable approximation of the standard median heuristic that matches the plan's intent.
- **particle_post_bsigs recomputed for dist_post:** The post-bsig average is computed twice (once for Stein kernel, once for dist_post metric). Minor redundancy kept for clarity rather than introducing shared mutable state.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed N=1 NaN in compute_stein_repulsion**
- **Found during:** Task 1 verification
- **Issue:** Plan's `jnp.log(N_f + 1e-8)` guard fails in float32: `float32(1) + 1e-8 = float32(1)` exactly, so `log(1.0) = 0.0`, then `median(0)/0.0 = nan`
- **Fix:** Added early return `if N == 1: return jnp.zeros_like(means)` before any computation; changed bandwidth to `log(N+1.0)` for N >= 2
- **Files modified:** accel_training/es_components/stein.py
- **Verification:** `compute_stein_repulsion(means[:1], bsigs[:1], 0.01)` returns shape (1,64) with no NaN
- **Committed in:** 394d2eb (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - float32 precision bug in N=1 NaN guard)
**Impact on plan:** Required for correctness; the plan's guard was mathematically correct in exact arithmetic but failed in float32. Fix preserves intended behavior (N=1 repulsion = zero) with guaranteed no-NaN.

## Issues Encountered

- float32 machine epsilon prevents `1.0 + 1e-8 > 1.0` in JAX — the CONTEXT doc noted "Floor prevents NaN when all bsigs identical (median=0)" but the issue for N=1 is the log denominator, not the median numerator. Fixed by N=1 short-circuit.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- stein.py and SVCMAESStrategy are ready for integration into train.py in Plan 04-02
- train.py needs: sv_cma_es branch in ES routing, two-pass eval loop (ask -> eval -> repel candidates -> re-eval -> tell), WandB logging of sv_behavior_dist_pre/post
- SVCMAESStrategy.tell() signature fully specified and verified — Plan 04-02 can wire it directly

## Self-Check: PASSED

- FOUND: accel_training/es_components/stein.py
- FOUND: accel_training/es_components/svcmaes_strategy.py
- FOUND: .planning/phases/04-behavioral-sv-cma-es/04-01-SUMMARY.md
- FOUND commit: 394d2eb (feat: stein.py)
- FOUND commit: 26af432 (feat: svcmaes_strategy.py)

---
*Phase: 04-behavioral-sv-cma-es*
*Completed: 2026-03-02*
