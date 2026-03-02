---
phase: 03-ns-es-integration
plan: 03
subsystem: testing
tags: [jax, ns-es, plr, behavior-signature, warmup, integration-tests, regret-fitness]

# Dependency graph
requires:
  - phase: 03-ns-es-integration
    plan: 01
    provides: NSESStrategy class with ask/tell interface and composite-fitness computation
  - phase: 03-ns-es-integration
    plan: 02
    provides: run_archive_warmup, ES strategy routing, behavior_sig extraction in train.py
  - phase: 02-buffer-and-fitness-infrastructure
    provides: LevelSampler with levels_extra, compute_novelty_batch, compute_fitness_batch
  - phase: 01-foundation
    provides: extract_behavior_signature, rollout_agent_on_levels_with_positions
provides:
  - 6 passing integration/unit tests in tests/test_phase3_ns_es.py covering ALGO-01, INTEG-01, INTEG-02, INTEG-03
  - Proof that NSESStrategy ask/tell produces correct shapes and types
  - Proof that tell() novelty varies with actual k-NN distance (not a constant)
  - Proof that extract_behavior_signature returns (pop_size, 169) L1-normalized float32
  - Proof that empty PLR buffer falls back to frontier (sample_replay_decision returns False)
  - Proof that run_archive_warmup() populates sampler['size'] > 0 when warmup_n > 0
  - Proof that train() runs 3 updates with ns_es strategy and warmup_n=4 without crashing
  - Bug fixes in regret_fitness.py (double-params, agent_pos access) and train.py (sampler key)
affects:
  - 04-sv-cmaes (will use the same regret_fitness.py rollout functions now that bugs are fixed)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Flax variable dict convention: train_state.params is {'params': {...}}; pass directly to network.apply(), do NOT wrap in {'params': ...} again"
    - "AutoReplayState nesting: agent_pos lives at state.env_state.agent_pos, not state.agent_pos"
    - "LevelSampler extra key: sampler['levels_extra'] not sampler['extra']"
    - "Test pattern: os.environ['WANDB_MODE'] = 'disabled' at very top before any imports"

key-files:
  created:
    - tests/test_phase3_ns_es.py
  modified:
    - es/regret_fitness.py
    - accel_training/train.py

key-decisions:
  - "All 3 bugs (double-params wrap, agent_pos path, sampler key) were pre-existing in Phase 2 code — Phase 3 tests are the first full integration exercising all three code paths simultaneously"
  - "Test 4 (empty buffer guard) uses manually constructed Level (no VAE) — proves the guard works independently of VAE checkpoint"
  - "Test 5 guard on vae_checkpoint existence: test MUST PASS (not skip) since VAE is present; guard is for CI portability only"
  - "WANDB_MODE=disabled set at very top of test file (before any imports) to prevent wandb network calls during tests"

patterns-established:
  - "Integration test pattern for JAX+Flax: use jax_env python, assert-based, no pytest, WANDB_MODE=disabled at top"
  - "Bug fix tracking: auto-fixed bugs documented with Rule tag in commit message for traceability"

requirements-completed: [ALGO-01, INTEG-01, INTEG-02, INTEG-03]

# Metrics
duration: 11min
completed: 2026-03-02
---

# Phase 3 Plan 03: NS-ES Tests Summary

**6 passing Phase 3 integration tests covering NSESStrategy, behavior_sig extraction, two-bucket empty-buffer guard, archive warmup buffer population, and 3-update end-to-end smoke run — plus 3 auto-fixed bugs found during testing**

## Performance

- **Duration:** 11 min
- **Started:** 2026-03-02T17:44:21Z
- **Completed:** 2026-03-02T17:55:33Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments

- Wrote and verified tests/test_phase3_ns_es.py with all 6 tests passing under jax_env python
- Found and fixed 3 pre-existing bugs in regret_fitness.py and train.py discovered during test execution
- test_archive_warmup_populates_buffer confirms run_archive_warmup() inserts 4 entries (warmup_n=4) into PLR buffer
- test_end_to_end_3_updates confirms full NS-ES pipeline (warmup -> 3 updates -> checkpoint) runs without crashing

## Task Commits

Each task was committed atomically:

1. **Task 1: Write tests/test_phase3_ns_es.py with 6 passing tests** - `9d704e6` (feat)
   - Includes 3 auto-fixed bugs (Rule 1) in the same commit

**Plan metadata:** TBD (docs commit)

## Files Created/Modified

- `tests/test_phase3_ns_es.py` - 6 integration/unit tests: NSESStrategy ask/tell shapes/types, composite fitness novelty variation, behavior_sig extraction shape/dtype/normalization, empty buffer frontier fallback, archive warmup buffer population, 3-update end-to-end smoke
- `es/regret_fitness.py` - Fixed network.apply() double-params wrap in both rollout functions; fixed agent_pos path from state.agent_pos to state.env_state.agent_pos
- `accel_training/train.py` - Fixed sampler key from sampler["extra"] to sampler["levels_extra"] in ES tell() block

## Decisions Made

- Test 4 constructs Level manually (without VAE) to prove the empty-buffer guard independently of VAE availability
- VAE checkpoint guard added to Tests 5 and 6 for CI portability, but both PASS (not SKIP) since checkpoint is present
- All bug fixes applied inline (not separate commits) since they were directly blocking the current task

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed network.apply() double-params wrap in regret_fitness.py**
- **Found during:** Task 1 (test_archive_warmup_populates_buffer execution)
- **Issue:** `rollout_agent_on_levels` and `rollout_agent_on_levels_with_positions` called `network.apply({'params': agent_params}, ...)` where `agent_params = train_state.params = {'params': {...}}`. This produced a double-wrapped `{'params': {'params': {...}}}` call, causing `ApplyScopeInvalidVariablesStructureError`.
- **Fix:** Changed both rollout functions to call `network.apply(agent_params, ...)` directly, matching the convention used in `build_eval_fn` (ued_interface.py).
- **Files modified:** es/regret_fitness.py (lines 98 and 169)
- **Verification:** test_archive_warmup_populates_buffer progressed past rollout call
- **Committed in:** 9d704e6 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed agent_pos path in rollout_agent_on_levels_with_positions**
- **Found during:** Task 1 (test_archive_warmup_populates_buffer execution, after fix #1)
- **Issue:** `rollout_agent_on_levels_with_positions` accessed `next_state.agent_pos` but `next_state` is `AutoReplayState` which wraps the inner `EnvState`. The actual agent_pos is at `next_state.env_state.agent_pos`.
- **Fix:** Changed `next_state.agent_pos` to `next_state.env_state.agent_pos` in the scan step function.
- **Files modified:** es/regret_fitness.py (line 185)
- **Verification:** test_archive_warmup_populates_buffer completed with PASS and buffer size 4
- **Committed in:** 9d704e6 (Task 1 commit)

**3. [Rule 1 - Bug] Fixed sampler key in train.py ES tell() block**
- **Found during:** Task 1 (test_end_to_end_3_updates execution)
- **Issue:** Line 500 in train.py accessed `train_state.sampler["extra"]["behavior_sig"]` but LevelSampler stores extras under `sampler["levels_extra"]`, not `sampler["extra"]`. This caused `KeyError: 'extra'` during the NS-ES training loop.
- **Fix:** Changed `sampler["extra"]` to `sampler["levels_extra"]` at line 500.
- **Files modified:** accel_training/train.py (line 500)
- **Verification:** test_end_to_end_3_updates completed 3 updates and checkpoint save successfully
- **Committed in:** 9d704e6 (Task 1 commit)

---

**Total deviations:** 3 auto-fixed (all Rule 1 - Bug)
**Impact on plan:** All 3 bugs were pre-existing in Phase 2 code but only triggered when the full integration path was exercised for the first time in Phase 3 tests. All fixes are correctness-critical and within the scope of making Phase 3 tests pass.

## Issues Encountered

Three pre-existing bugs discovered in sequence during test execution. Each was diagnosed from the JAX/Flax traceback and fixed immediately. All fixes were simple one-liner changes with clear root causes.

## User Setup Required

None - no external service configuration required. Tests use WANDB_MODE=disabled.

## Next Phase Readiness

- All 4 Phase 3 requirements proven: ALGO-01 (NSESStrategy), INTEG-01 (two-bucket guard), INTEG-02 (behavior_sig + warmup), INTEG-03 (end-to-end)
- Bug fixes in regret_fitness.py now enable Phase 4 to use rollout_agent_on_levels_with_positions correctly
- The sampler["levels_extra"] fix ensures Phase 4 SV-CMA-ES training loop won't hit the same KeyError
- Phase 4 (SV-CMA-ES) can build on the now-validated NS-ES pipeline

---
*Phase: 03-ns-es-integration*
*Completed: 2026-03-02*

## Self-Check: PASSED

- FOUND: tests/test_phase3_ns_es.py
- FOUND: es/regret_fitness.py
- FOUND: accel_training/train.py
- FOUND: .planning/phases/03-ns-es-integration/03-03-SUMMARY.md
- FOUND commit: 9d704e6 (feat(03-03): add Phase 3 NS-ES tests and fix rollout/sampler bugs)
