---
phase: 02-grid-adapter
plan: 01
subsystem: llm
tags: [jax, agent-evaluator, injection-config, buffer-stats, decision-gate, policy-rollout]

# Dependency graph
requires:
  - phase: 01-checkpoint
    provides: LLMInjectionManager, BufferStatsExtractor, AgentEvaluator skeleton
provides:
  - AgentEvaluator with direct-param construction (apply_fn + params, no file I/O)
  - update_params() method invalidating JIT rollout cache on each injection event
  - AgentEvaluator.from_checkpoint() classmethod for backward compat
  - LLMInjectionConfig gate fields locked to Phase 2 decisions (gate_enabled, difficulty_threshold, min_diversity, etc.)
  - BufferStatsExtractor.extract_references_with_levels() returning (ReferenceMaze list, Level list)
  - BufferStatsExtractor.extract_global_metrics() converting buffer summary to MetricEntry list
affects:
  - 02-02: injector wires AgentEvaluator.update_params(), extract_references_with_levels(), extract_global_metrics() into _do_injection()

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Direct-param evaluator: live training injects current policy params, no checkpoint files needed mid-run"
    - "JIT invalidation via None sentinel: update_params sets _rollout_fn=None, forcing retrace with fresh params on next call"
    - "DRY buffer extraction: extract_references() delegates to extract_references_with_levels() to avoid duplicate selection logic"

key-files:
  created: []
  modified:
    - llm/agent_evaluator.py
    - llm/injection_config.py
    - llm/buffer_stats.py
    - llm/test_generator.py

key-decisions:
  - "AgentEvaluator uses None-sentinel pattern for JIT invalidation — _rollout_fn=None forces retrace; JAX re-uses trace cache by shape so cost is minimal (one retrace per num_levels shape)"
  - "from_checkpoint() classmethod defers cross_evaluate import to call site — no top-level import needed, avoids loading checkpoint infrastructure unless actually loading from file"
  - "extract_references() now delegates to extract_references_with_levels() — single source of truth for selection logic"

patterns-established:
  - "Direct-param evaluator pattern: AgentEvaluator(apply_fn, params, env_params) for live use; from_checkpoint() for file-based use"
  - "Gate config defaults mirror CONTEXT.md locked decisions — gate_enabled=True, difficulty_threshold=0.6, min_diversity=0.02, diversity_metric=td_error_emd"

requirements-completed: [GATE-02, GATE-04]

# Metrics
duration: 3min
completed: 2026-03-23
---

# Phase 02 Plan 01: Grid Adapter Prerequisites Summary

**AgentEvaluator refactored for live-param injection (apply_fn+params direct), gate config fields locked to Phase 2 decisions, BufferStatsExtractor extended with Level extraction and MetricEntry conversion**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-23T17:06:11Z
- **Completed:** 2026-03-23T17:09:30Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- AgentEvaluator now accepts policy params directly — no checkpoint file needed for live training integration; update_params() invalidates JIT rollout cache at each injection event
- LLMInjectionConfig gate fields updated to Phase 2 locked decisions: gate_enabled=True, difficulty_threshold=0.6, min_diversity=0.02, diversity_metric="td_error_emd", max_diversity_retries=2, n_rollouts_gate=100
- BufferStatsExtractor extended with extract_references_with_levels() (single-pass Level+ReferenceMaze extraction) and extract_global_metrics() (buffer summary to MetricEntry list)

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor AgentEvaluator for direct param passing** - `bc598ae` (refactor)
2. **Task 2: Add gate config fields to LLMInjectionConfig and extend BufferStatsExtractor** - `99fce59` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified
- `llm/agent_evaluator.py` - Refactored: new __init__(apply_fn, params, env_params), update_params(), from_checkpoint() classmethod, _build_rollout_fn() with num_levels caching
- `llm/injection_config.py` - Updated: Phase 2 gate fields replacing Phase 1 placeholders, from_config_dict() mapping all new CLI flags
- `llm/buffer_stats.py` - Extended: extract_references_with_levels(), extract_references() delegating to it, extract_global_metrics() static method, Tuple import
- `llm/test_generator.py` - Updated: two AgentEvaluator() call sites changed to AgentEvaluator.from_checkpoint()

## Decisions Made
- AgentEvaluator uses None-sentinel pattern for JIT invalidation: update_params() sets _rollout_fn=None, forcing retrace on next _evaluate_batch() call. JAX re-uses trace by shape so cost is minimal (one retrace per distinct num_levels value).
- from_checkpoint() classmethod defers `from cross_evaluate import load_agent` to call time — removes top-level import that would pull in checkpoint infrastructure on every import.
- extract_references() now delegates to extract_references_with_levels() to avoid duplicate buffer selection logic (DRY). Performance is identical since level_objects are just discarded.
- Gate defaults directly encode CONTEXT.md locked decisions: difficulty_threshold=0.6 (regret-based learnability filter), min_diversity=0.02 (td_error_emd threshold), n_rollouts_gate=100.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All three files are ready for Plan 02-02 (injector gate wiring)
- AgentEvaluator.update_params() provides the hook for injector to refresh policy at each injection event
- extract_references_with_levels() provides Level objects for trajectory computation in the diversity gate
- extract_global_metrics() provides MetricEntry list for prompt global context
- from_config_dict() maps all gate CLI flags so maze_plr.py can control gate thresholds at launch time

---
*Phase: 02-grid-adapter*
*Completed: 2026-03-23*

## Self-Check: PASSED

- FOUND: llm/agent_evaluator.py
- FOUND: llm/injection_config.py
- FOUND: llm/buffer_stats.py
- FOUND: .planning/phases/02-grid-adapter/02-01-SUMMARY.md
- FOUND commit: bc598ae (Task 1 - AgentEvaluator refactor)
- FOUND commit: 99fce59 (Task 2 - config and buffer stats updates)
