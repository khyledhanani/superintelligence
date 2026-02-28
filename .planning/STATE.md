# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-26)

**Core value:** The ES module must find diverse, high-regret environments that continuously challenge the agent — without collapsing to a single mode — so the agent develops generalizable skills through open-ended curriculum learning.
**Current focus:** Phase 2 — Buffer and Fitness Infrastructure

## Current Position

Phase: 2 of 5 (Buffer and Fitness Infrastructure)
Plan: 2 of 2 in current phase
Status: In progress
Last activity: 2026-02-28 — Completed 02-02 (Novelty scoring and fitness function)

Progress: [████░░░░░░] 40%

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 8 min
- Total execution time: 0.52 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation | 2 | 13 min | 7 min |
| 02-buffer-and-fitness-infrastructure | 2 | 18 min | 9 min |

**Recent Trend:**
- Last 5 plans: 01-01 (10 min), 01-02 (3 min), 02-01 (~10 min), 02-02 (8 min)
- Trend: Consistent ~8 min/plan

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Pre-phase]: Behavioral diversity over latent-space diversity — fixed axes become obsolete as agent learns
- [Pre-phase]: Unified buffer as replay + novelty archive — eliminates redundant data structures
- [Pre-phase]: Modular ES interface — need to test CMA-ES / NS-ES / SV-CMA-ES; NS-ES is MVP first
- [Pre-phase]: JAX-first — all new code must be JIT-compatible; no scikit-learn or FAISS
- [01-01]: gae_lambda=0.98 classified potential-bug (DCD uses 0.95); monitor training stability
- [01-01]: entropy_coeff=1e-3 classified intentional (DCD uses 0.0); promotes exploration in sparse-reward maze
- [01-01]: score_function=MaxMC confirmed correct — matches DCD ACCEL config
- [01-01]: MAP-Elites/ES mutation confirmed INTENTIONAL — thesis contribution replacing minimax mutation
- [01-02]: Behavior signature v1 is 169-cell L1-normalized visit-count histogram; EXPERIMENTAL, revisit after Phase 3 NS-ES validation (see .planning/DECISIONS.md DECISION-01)
- [02-02]: k=5 nearest neighbors as default for novelty k-NN; k is static (functools.partial + static_argnames) so JAX compiles without retracing per value
- [02-02]: No normalization in compute_fitness — raw combination F = alpha*regret + beta*novelty; caller negates before passing to evosax (which minimizes)
- [02-02]: alpha and beta are plain Python floats (not JAX arrays) — avoids JAX state management complexity, matches ES config dict structure planned for Phase 3

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3 flag]: First end-to-end ES wiring into JAX training loop has broad integration surface — run /gsd:research-phase before planning Phase 3
- [Phase 4 flag]: Stein kernel implementation and multi-particle evosax state management are novel — run /gsd:research-phase before planning Phase 4
- [Phase 1 resolved]: Behavior signature dimensionality fixed at 13x13=169 cells for v1 (full resolution, no lossy binning); revisit criteria documented in DECISIONS.md
- [Phase 2 note]: Default python3 on machine lacks JAX; use /cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python for all JAX verification

## Session Continuity

Last session: 2026-02-28
Stopped at: Completed 02-02-PLAN.md (Novelty scoring and fitness function)
Resume file: None
