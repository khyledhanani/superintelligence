---
phase: 01-checkpoint
plan: 02
subsystem: llm-injection
tags: [python, jax, plr, buffer, llm, injection, mutation, bfs, wandb]

# Dependency graph
requires:
  - phase: 01-01
    provides: LLMInjectionConfig dataclass and BufferStatsExtractor for live buffer reference extraction
provides:
  - LLMInjectionManager orchestrator class in llm/injector.py encapsulating full injection pipeline
  - _bfs_path_length() standalone BFS utility for solvability validation
  - validate_llm_level() with border wall check and BFS path length > 5
  - Training loop injection hook in examples/maze_plr.py via maybe_inject()
affects:
  - 02-grid-adapter (uses LLMInjectionManager, may extend validation)
  - 03-integration (full wiring builds on injection pipeline established here)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "LLMInjectionManager.maybe_inject(runner_state, eval_step) is the single injection call site in training loop"
    - "Injection scheduling: current_step % injection_interval == 0 check in maybe_inject()"
    - "Border wall hard reject: all 4 outer edges must be walls (LLM-generated mazes without borders rejected)"
    - "BFS path length > 5 hard reject: short paths filter trivial/degenerate mazes"
    - "Mutation amplification via jax.vmap(lambda r: mutate_level(r, seed, 3))(mut_rngs)"
    - "Single batch insert: level_sampler.insert_batch(sampler, levels_batch, scores_batch) never loop of insert()"
    - "Max-priority injection: inject_score = max_buffer_score + 1e-4 forces immediate replay"
    - "On API failure: RuntimeError raised immediately, no silent skip"

key-files:
  created:
    - llm/injector.py
  modified:
    - examples/maze_plr.py

key-decisions:
  - "validate_llm_level() border wall check is hard reject — LLM mazes without full borders are structurally invalid for the training environment"
  - "Mutation amplification uses jax.vmap with num_edits=3 (not max_num_edits=100) — 3 edits creates nearby variants rather than completely random mutations"
  - "Plan test maze (Labyrinth with '#' top border) was unsolvable due to trapped goal — used actual Labyrinth prefab (open borders) from level.py for BFS verification instead"

patterns-established:
  - "LLMInjectionManager is the only LLM call site: training loop calls maybe_inject() once, all scheduling/validation/amplification/insertion is internal"
  - "_bfs_path_length(wall_map_np, start, goal) -> int: standalone numpy BFS returning -1 on unsolvable"
  - "validate_llm_level(level) -> (bool, str): hard validation returning rejection reason string"

requirements-completed: [INTG-01, INTG-02, INTG-03, INTG-05]

# Metrics
duration: 5min
completed: 2026-03-23
---

# Phase 1 Plan 02: LLMInjectionManager and Training Loop Integration Summary

**LLMInjectionManager orchestrator in llm/injector.py with full injection pipeline (generate, validate with border wall + BFS check, mutate via jax.vmap, batch-insert with max-priority), wired into maze_plr.py training loop via single maybe_inject() call with WandB logging**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-23T15:25:50Z
- **Completed:** 2026-03-23T15:30:53Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created LLMInjectionManager with full injection pipeline: scheduling → reference extraction via BufferStatsExtractor → n_raw LLM generations (crash on failure) → validate_llm_level() (border walls + BFS > 5) → jax.vmap mutation amplification with solvability check → level_sampler.insert_batch() with max-priority score → WandB logging of 10 llm/* metrics
- Implemented standalone _bfs_path_length() using collections.deque BFS for Python-side solvability checks
- Wired LLMInjectionManager into maze_plr.py with exactly one injection call site in the eval loop, LLM injection setup block before the loop, and "llm" WandB tag

## Task Commits

Each task was committed atomically:

1. **Task 1: Create LLMInjectionManager orchestrator class** - `f94edef` (feat)
2. **Task 2: Wire LLMInjectionManager into maze_plr.py training loop** - `b94b07d` (feat)

## Files Created/Modified
- `llm/injector.py` - LLMInjectionManager orchestrator, validate_llm_level(), _bfs_path_length() — full injection pipeline from scheduling to buffer insertion
- `examples/maze_plr.py` - Added LLM imports, setup block before training loop, single injection hook in eval loop, "llm" WandB tag

## Decisions Made
- validate_llm_level() border wall check is a hard reject: LLM-generated mazes without full outer wall borders are structurally invalid and would cause issues with the maze environment assumptions
- Mutation amplification uses num_edits=3 in the jax.vmap call rather than the full max_num_edits=100 — 3 edits creates nearby structural variants of seeds; 100 edits would generate completely different mazes
- Plan's test verification maze (Labyrinth with solid '#' top border) was actually unsolvable because the goal cell was in a disconnected compartment. Used the actual Labyrinth prefab from level.py (which has open borders but is solvable via spiral path) for BFS testing instead

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan verification maze was unsolvable (goal trapped in disconnected compartment)**
- **Found during:** Task 1 (LLMInjectionManager verification)
- **Issue:** The plan's test maze string used `#############` on row 0 but had the goal G at (6,6) inside `#.#.#...#.#.#` which is a walled-off compartment. BFS returned -1 (no path). The plan intended to verify `path_len > 5` but the provided maze fails that test.
- **Fix:** Used the actual `Labyrinth` prefab from `level.py` (which has open outer borders but is solvable via a spiral path with path length 96) for verification. The implementation is correct — the test case in the plan spec was the bug.
- **Files modified:** No file change needed — used existing level.py prefab
- **Verification:** BFS on actual Labyrinth returns 96 > 5, assertion passes
- **Committed in:** f94edef (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in plan's test case)
**Impact on plan:** Minor — only affected which maze was used for verification. Implementation is correct and passes all 8 verification checks.

## Issues Encountered

The plan verification used a maze string that had `#############` on the top border (making it a hard-walled border) but then the goal was placed inside a fully enclosed inner compartment (`#.#.#.G.#.#.#` with no path in). This caused BFS to return -1. Diagnosed by tracing reachable cells from both agent and goal — they were in separate disconnected regions. Fixed by using the actual Labyrinth prefab which has the same maze structure but with open outer borders (`.............` on top row) allowing the spiral path to connect agent and goal.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- LLMInjectionManager and training loop hook are complete — running `--use_llm --llm_provider openrouter --llm_model <model>` will activate injection after warmup
- LLM API key must be set in environment (OPENROUTER_API_KEY or OLLAMA_API_KEY) before injection events fire
- Phase 2 (02-grid-adapter) can now extend validation or add AgentEvaluator rollout scoring by setting score_seeds_with_rollout=True
- No blockers — all INTG requirements (01, 02, 03, 05) satisfied

---
*Phase: 01-checkpoint*
*Completed: 2026-03-23*
