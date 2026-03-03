# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-26)

**Core value:** The ES module must find diverse, high-regret environments that continuously challenge the agent — without collapsing to a single mode — so the agent develops generalizable skills through open-ended curriculum learning.
**Current focus:** Phase 5 — Ablations and Analysis

## Current Position

Phase: 5 of 5 (Ablations and Analysis)
Plan: 1 of ? in current phase (Plan 1 COMPLETE)
Status: Phase 5 In Progress
Last activity: 2026-03-03 — Completed 05-01 (Phase 5 Plan 1: Clean Two-Mode Train Pipeline — all 12 tests PASS)

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Average duration: 4 min
- Total execution time: 0.75 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-foundation | 2 | 13 min | 7 min |
| 02-buffer-and-fitness-infrastructure | 3 | 20 min | 7 min |
| 03-ns-es-integration | 2 | 4 min | 2 min |

**Recent Trend:**
- Last 5 plans: 03-01 (2 min), 03-02 (2 min), 03-03 (11 min), 04-01 (8 min), 04-02 (3 min), 04-03 (2 min)
- Trend: Fast execution for well-specified plans; test-only plans consistently 2 min

*Updated after each plan completion*
| Phase 02-buffer-and-fitness-infrastructure P01 | 3 | 2 tasks | 3 files |
| Phase 02-buffer-and-fitness-infrastructure P03 | 2 | 2 tasks | 2 files |
| Phase 03-ns-es-integration P01 | 2 | 2 tasks | 2 files |
| Phase 03-ns-es-integration P02 | 2 | 2 tasks | 2 files |
| Phase 03-ns-es-integration P03 | 11 | 1 task | 3 files |
| Phase 04-behavioral-sv-cma-es P01 | 8 | 2 tasks | 2 files |
| Phase 04-behavioral-sv-cma-es P02 | 3 | 1 task | 2 files |
| Phase 04-behavioral-sv-cma-es P03 | 2 | 1 task | 1 file |
| Phase 05-ablations-and-analysis P01 | 9 | 2 tasks | 4 files |

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
- [03-03]: Flax variable dict convention: train_state.params is {'params': {...}}; pass directly to network.apply() — do NOT wrap again
- [03-03]: AutoReplayState nests inner EnvState: agent_pos is at state.env_state.agent_pos, not state.agent_pos
- [03-03]: LevelSampler extras stored at sampler['levels_extra'], not sampler['extra']
- [03-03]: All 3 bugs (double-params, agent_pos, sampler key) were pre-existing Phase 2 code triggered first time by Phase 3 full integration path
- [Phase 04-behavioral-sv-cma-es]: N=1 short-circuit in compute_stein_repulsion: return zeros_like before log computation to avoid float32 precision NaN
- [Phase 04-behavioral-sv-cma-es]: Bandwidth uses log(N+1) not log(N+1e-8): ensures denominator >= log(2) in float32 for all N >= 1
- [Phase 04-behavioral-sv-cma-es]: SVCMAESStrategy tell() fitness: pure negated regret (evosax minimizes); Stein repulsion applied post-tell() to CMA means
- [04-02]: Post-repulsion PLR buffer uses second insert_batch (overwrites first-pass data with post-repulsion levels/regrets/bsigs — richer quality data)
- [04-02]: max_returns2 from second eval pass stored in post-repulsion level_extra (not reused from first pass — more accurate for post-repulsion levels)
- [04-02]: sv_behavior_dist_pre/post always logged (0.0 for non-sv_cma_es strategies) — avoids conditional logging, enables cross-run comparison
- [04-03]: Repulsion test uses norm assertion (not post_dist > pre_dist): correct because Stein direction is sound but mean pairwise distance increase is not monotonically guaranteed from random means
- [04-03]: N=1 assertion uses exact Python 0.0: mean_pairwise_behavior_dist() early-returns Python float 0.0 for N <= 1 — exact equality is safe and expressive
- [Phase 05-ablations-and-analysis]: train() returns train_state only (not tuple) — archive concept is gone from the API surface
- [Phase 05-ablations-and-analysis]: bootstrap_min=50 default; bootstrap loop does not count toward num_updates; fills PLR buffer before main loop
- [Phase 05-ablations-and-analysis]: replay_ratio key replaces replay_prob (backward-compatible via config.get fallback)

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3 resolved]: First end-to-end ES wiring into JAX training loop complete; 3 integration bugs found and fixed; all 4 Phase 3 requirements proven by tests
- [Phase 4 flag]: Stein kernel implementation and multi-particle evosax state management are novel — run /gsd:research-phase before planning Phase 4
- [Phase 1 resolved]: Behavior signature dimensionality fixed at 13x13=169 cells for v1 (full resolution, no lossy binning); revisit criteria documented in DECISIONS.md
- [Phase 2 note]: Default python3 on machine lacks JAX; use /cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python for all JAX verification

## Session Continuity

Last session: 2026-03-03
Stopped at: Completed 05-01-PLAN.md (Phase 5 Plan 1: Clean Two-Mode Train Pipeline — all 12 tests PASS)
Resume file: None
