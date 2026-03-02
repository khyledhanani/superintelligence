# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-26)

**Core value:** The ES module must find diverse, high-regret environments that continuously challenge the agent — without collapsing to a single mode — so the agent develops generalizable skills through open-ended curriculum learning.
**Current focus:** Phase 3 — NS-ES Integration

## Current Position

Phase: 3 of 5 (NS-ES Integration)
Plan: 2 of 3 in current phase
Status: In Progress
Last activity: 2026-03-02 — Completed 03-02 (NS-ES wired into training loop)

Progress: [████████░░] 73%

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 4 min
- Total execution time: 0.60 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation | 2 | 13 min | 7 min |
| 02-buffer-and-fitness-infrastructure | 3 | 20 min | 7 min |
| 03-ns-es-integration | 2 | 4 min | 2 min |

**Recent Trend:**
- Last 5 plans: 02-02 (8 min), 02-03 (2 min), 03-01 (2 min), 03-02 (2 min)
- Trend: Fast execution for well-specified plans

*Updated after each plan completion*
| Phase 02-buffer-and-fitness-infrastructure P01 | 3 | 2 tasks | 3 files |
| Phase 02-buffer-and-fitness-infrastructure P03 | 2 | 2 tasks | 2 files |
| Phase 03-ns-es-integration P01 | 2 | 2 tasks | 2 files |
| Phase 03-ns-es-integration P02 | 2 | 2 tasks | 2 files |

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
- [Phase 02-buffer-and-fitness-infrastructure]: evosax CMA_ES.init() requires 3 args (key, mean, params); mean sets distribution center; plan comment was incorrect
- [Phase 02-buffer-and-fitness-infrastructure]: CMAESStrategy stores es_params inside state dict — callers never touch evosax internals, clean Protocol surface
- [02-03]: Only one insert_batch call in train.py (NEW/mutate branch); REPLAY branch uses update_batch — assertion guard applied only at insert_batch site
- [02-03]: behavior_sig missing from level_extra at insertion is INTENTIONAL in Phase 2; assert is API contract for Phase 3 (which adds behavior_sig extraction)
- [03-01]: NSESStrategy.tell() extends ESStrategy Protocol minimum surface with novelty inputs; caller uses concrete NSESStrategy type not Protocol abstraction
- [03-01]: No separate novelty archive — buffer_sigs and valid_mask passed into tell() at call time (locked CONTEXT.md decision)
- [03-01]: mean_novelty returned as Python float (float() conversion inside tell()) for logging convenience in train.py
- [03-02]: ES strategy routing via config['es_strategy'] string: ns_es -> NSESStrategy, cma_es -> CMAESStrategy at train() startup
- [03-02]: run_archive_warmup triggered when es_strategy != cma_es OR warmup_n > 0 — allows warm-up for CMA-ES baseline too
- [03-02]: ES tell() placed after insert_batch in NEW/mutate branch so buffer_sigs reflect just-inserted state
- [03-02]: mean_novelty initialized to 0.0; only updated in NS-ES NEW/mutate branch; WandB logs last known value for replay steps

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3 flag]: First end-to-end ES wiring into JAX training loop has broad integration surface — run /gsd:research-phase before planning Phase 3
- [Phase 4 flag]: Stein kernel implementation and multi-particle evosax state management are novel — run /gsd:research-phase before planning Phase 4
- [Phase 1 resolved]: Behavior signature dimensionality fixed at 13x13=169 cells for v1 (full resolution, no lossy binning); revisit criteria documented in DECISIONS.md
- [Phase 2 note]: Default python3 on machine lacks JAX; use /cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python for all JAX verification

## Session Continuity

Last session: 2026-03-02
Stopped at: Completed 03-02-PLAN.md (NS-ES wired into training loop)
Resume file: None
