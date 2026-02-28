---
phase: 01-foundation
plan: "01"
subsystem: verification
tags: [ppo, accel, jax, jaxued, maxmc, smoke-test, hyperparameters]

# Dependency graph
requires: []
provides:
  - "AGENT_VERIFICATION.md: flat list of all PPO/ACCEL implementation differences vs DCD defaults"
  - "Smoke test confirmation: ACCEL+MaxMC pipeline runs 5 updates on CPU without training crash"
  - "Documented differences: gae_lambda=0.98 vs 0.95, entropy_coeff=1e-3 vs 0.0, MaxMC vs value_l1"
  - "Confirmed: MAP-Elites/ES level mutation is INTENTIONAL architectural difference (thesis contribution)"
affects:
  - "02-es-harness: ES harness wraps the verified ACCEL agent baseline"
  - "03-integration: end-to-end wiring uses agent with documented hyperparams"
  - "All phases: regret > 0 confirmed — MaxMC fitness signal is live"

# Tech tracking
tech-stack:
  added: [jax==0.5.3, jaxued, flax, optax, distrax, wandb==0.25.0, orbax-checkpoint]
  patterns:
    - "ACCEL training loop: new_levels -> rollout -> GAE -> score -> PLR insert -> PPO update"
    - "MaxMC regret: compute_max_returns(dones, rewards) then max_mc(dones, values, max_returns)"
    - "AutoReplayWrapper: agent replays same level on episode end within rollout"

key-files:
  created:
    - ".planning/phases/01-foundation/AGENT_VERIFICATION.md"
  modified: []

key-decisions:
  - "gae_lambda=0.98 (vs DCD 0.95): classified potential-bug, monitor training stability"
  - "entropy_coeff=1e-3 (vs DCD 0.0): classified intentional, promotes exploration in sparse-reward maze"
  - "score_function=MaxMC: matches DCD ACCEL config, jaxued max_mc is correct JAX equivalent"
  - "MAP-Elites/ES mutation: intentional thesis contribution, minimax fallback still available"

patterns-established:
  - "Verification-first: all implementation differences explicitly classified before building on baseline"
  - "Smoke test before integration: confirmed regret pipeline functional (regret range 0.126-0.210)"

requirements-completed: [FOUND-01]

# Metrics
duration: 10min
completed: 2026-02-28
---

# Phase 1 Plan 1: Agent Verification Summary

**ACCEL+MaxMC agent verified as valid baseline: 4 of 12 PPO params differ from DCD (2 intentional, 1 potential-bug, 1 matches-ACCEL-config); smoke test confirmed regret > 0 and changing across all 5 updates in 28s on CPU.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-02-28T17:18:20Z
- **Completed:** 2026-02-28T17:28:31Z
- **Tasks:** 2/2
- **Files modified:** 1 (AGENT_VERIFICATION.md created)

## Accomplishments

- Created AGENT_VERIFICATION.md with full hyperparameter comparison table (12 parameters, all filled with actual values from code)
- Classified every difference: gae_lambda=0.98 (potential-bug), entropy_coeff=1e-3 (intentional), MaxMC (matches-accel-config), MAP-Elites mutation (intentional thesis contribution)
- Ran 5 ACCEL+MaxMC training updates; confirmed regret > 0 for all updates (range: 0.126-0.210 mean), regret changing, no training crash
- Documented wandb.Video/moviepy logging-only crash — confirmed it does not affect training pipeline

## Task Commits

1. **Task 1: PPO Code Comparison** - `f37fab5` (feat) — AGENT_VERIFICATION.md created with hyperparameter table and structural differences
2. **Task 2: Smoke Test Results** - `6be393a` (feat) — Smoke test results appended, PASS verdict documented

## Files Created/Modified

- `.planning/phases/01-foundation/AGENT_VERIFICATION.md` — Full PPO/ACCEL comparison vs DCD defaults, GAE formula verification, regret computation path, and smoke test results with per-update metrics

## Decisions Made

- **gae_lambda=0.98:** Classified as potential-bug (higher than DCD's 0.95). Will monitor training stability. Does not block Phase 2.
- **entropy_coeff=1e-3:** Classified as intentional. Small entropy bonus is standard practice for sparse-reward maze navigation.
- **score_function=MaxMC:** Verified correct — matches DCD ACCEL-specific config (not DCD generic default).
- **jax_env conda environment:** Used for all JAX execution (`/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env`). Missing `wandb` was installed; `moviepy` remains missing (affects animation logging only).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed wandb in jax_env**
- **Found during:** Task 2 (Smoke Test)
- **Issue:** `wandb` not installed in jax_env environment, causing `ModuleNotFoundError` when running maze_plr.py
- **Fix:** Ran `pip install wandb` in jax_env; wandb 0.25.0 installed
- **Files modified:** jax_env site-packages (pip install, not tracked by git)
- **Verification:** `import wandb` succeeds; WandB offline run initialized successfully
- **Committed in:** 6be393a (Task 2 commit, documented in smoke test results)

**2. [Rule 3 - Blocking] Used direct Python script for metric capture after wandb.Video crash**
- **Found during:** Task 2 (Smoke Test)
- **Issue:** `wandb.Video` requires `moviepy` (not installed); crash occurred after training completed but prevented metric capture via `log_eval`
- **Fix:** Re-ran as direct Python script bypassing `log_eval`, capturing metrics (regret, solve rate, losses) directly to stdout
- **Files modified:** None (workaround, no code change)
- **Verification:** All 5 update metrics captured successfully; PASS verdict confirmed
- **Committed in:** 6be393a (documented in smoke test section)

---

**Total deviations:** 2 auto-fixed (both Rule 3 — blocking environment issues)
**Impact on plan:** Both fixes necessary to run smoke test. No scope creep. Training pipeline correctness unaffected.

## Issues Encountered

- **wandb.Video/moviepy:** The full maze_plr.py run crashes during `log_eval` when calling `wandb.Video(frames, fps=4)`. Missing `moviepy` dependency. Workaround: run without animation logging, or install `pip install moviepy`. Deferred to Phase 2+.
- **eval_freq default (250):** With `--num_updates 5`, the default `eval_freq=250` gives `range(0)` iterations — no training runs. Fixed by adding `--eval_freq 5` to force `5 // 5 = 1` training+eval cycle.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- ACCEL agent baseline confirmed: hyperparameters documented, pipeline functional
- FOUND-01 satisfied: we know the exact state of the baseline we are building ES around
- Two differences to track: gae_lambda=0.98 (monitor), wandb.Video crash (deferred)
- Ready to proceed to Phase 1 Plan 2 (behavior signatures / ES harness scaffolding)

## Self-Check: PASSED

- AGENT_VERIFICATION.md: FOUND
- 01-01-SUMMARY.md: FOUND
- Commit f37fab5: FOUND
- Commit 6be393a: FOUND

---
*Phase: 01-foundation*
*Completed: 2026-02-28*
