# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-23)

**Core value:** LLM-generated mazes must measurably improve agent generalization (solve rate on held-out benchmarks) compared to ACCEL-only and CMA-ES-only baselines.
**Current focus:** Phase 2 — Grid Adapter

## Current Position

Phase: 2 of 4 (Grid Adapter)
Plan: 2 of 3 in current phase (complete, paused at human-verify checkpoint)
Status: In progress — Plan 02-02 tasks 1+2 complete, paused at Task 3 checkpoint:human-verify
Last activity: 2026-03-23 — Completed 02-02 tasks 1+2: gated injection pipeline, smoke test script

Progress: [████░░░░░░] 40%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 3.3 min
- Total execution time: 0.17 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-checkpoint | 2 | 7 min | 3.5 min |
| 02-grid-adapter | 2 | 8 min | 4 min |

**Recent Trend:**
- Last 5 plans: 01-01 (2 min), 01-02 (5 min), 02-01 (3 min), 02-02 (5 min)
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Pre-phase]: Periodic batch injection chosen over async — simpler, sufficient for research
- [Pre-phase]: LLM used for seed generation only, not mutation — cost constraint
- [Pre-phase]: Reuse existing DecisionGate from friend's code — already implements diversity+learnability filtering
- [Pre-phase]: Threading vs synchronous injection is an open question — measure LLM latency in Phase 1 before deciding
- [01-01]: score_seeds_with_rollout defaults False in Phase 1 — AgentEvaluator rollout scoring deferred to Phase 2
- [01-01]: n_raw field maps to --llm_batch_size CLI flag per plan spec; naming reflects what user controls
- [01-01]: BufferStatsExtractor is self-contained, does not import from test_generator.py — replaces file-based flow
- [01-02]: validate_llm_level() border wall check is hard reject — LLM mazes without full borders are structurally invalid
- [01-02]: Mutation amplification uses num_edits=3 (not max_num_edits=100) — creates nearby variants rather than completely random mutations
- [01-02]: Plan test maze was unsolvable (goal trapped in disconnected compartment) — used actual Labyrinth prefab for verification instead
- [02-01]: AgentEvaluator uses None-sentinel for JIT invalidation — _rollout_fn=None forces retrace with fresh params; JAX re-uses trace by shape so cost is minimal
- [02-01]: from_checkpoint() defers cross_evaluate import to call site — avoids loading checkpoint infra on every llm.agent_evaluator import
- [02-01]: Gate defaults encode CONTEXT.md locked decisions: gate_enabled=True, difficulty_threshold=0.6, min_diversity=0.02, diversity_metric=td_error_emd, n_rollouts_gate=100
- [02-02]: Gate acceptance detected via empty result.diversity_issues — generate_with_feedback() always returns but marks unresolved issues when exhausted retries with gate failure
- [02-02]: Reference trajectories computed once per injection event, not per seed — avoids N redundant rollouts across seeds in same event
- [02-02]: Phase 1 fallback path preserved exactly with --no-llm_gate — generate() + validate_llm_level() for backward compat

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 2 prep — RESOLVED in 02-01]: Gate threshold calibration needed — locked to difficulty_threshold=0.6, min_diversity=0.02 per CONTEXT.md decisions
- [Phase 4 prep]: Confirm whether existing 50k `accel-baseline` runs (JAXUED_50K) are sufficient as control condition or whether fresh control runs are needed for identical experimental conditions

## Session Continuity

Last session: 2026-03-23
Stopped at: Completed 02-02 Tasks 1+2 (gated injection pipeline + smoke test). Paused at Task 3 checkpoint:human-verify — awaiting user verification of gate CLI flags and optionally smoke test on GPU.
Resume file: None
