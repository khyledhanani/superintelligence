---
phase: 03-ns-es-integration
plan: 02
subsystem: training-loop
tags: [jax, ns-es, wandb, plr, behavior-signature, warmup, accel]

# Dependency graph
requires:
  - phase: 03-ns-es-integration
    plan: 01
    provides: NSESStrategy class with ask/tell interface and composite-fitness computation
  - phase: 01-foundation
    provides: rollout_agent_on_levels_with_positions and extract_behavior_signature in regret_fitness.py
  - phase: 02-buffer-and-fitness-infrastructure
    provides: LevelSampler.insert_batch with level_extra dict including behavior_sig slot
provides:
  - End-to-end NS-ES wired into ACCEL training loop via train.py
  - run_archive_warmup() with solvability gate pre-populating PLR buffer before step 0
  - behavior_sig extraction at every NEW/mutate insert_batch call (closes assert gap)
  - WandB init and per-update metric logging (regret, novelty_score, buffer stats)
  - ES strategy routing (ns_es -> NSESStrategy, cma_es -> CMAESStrategy) in train()
  - ES config block in config.yml for Phase 3 NS-ES run
affects:
  - 03-ns-es-integration (plan 03 — end-to-end integration testing)
  - 04-sv-cmaes (will build on same ES routing pattern)

# Tech tracking
tech-stack:
  added: [wandb]
  patterns:
    - "Solvability gate pattern: filter valid=True and ~nan_mask BEFORE tile/insert — never insert invalid levels"
    - "Separate rollout for behavior_sig: eval_fn does NOT return agent_positions; call rollout_agent_on_levels_with_positions separately"
    - "Warm-up synchronous timing: run_archive_warmup completes before training step 0, does not count toward num_updates"
    - "ES tell() in training loop: called after insert_batch in NEW/mutate branch; mean_novelty captured for WandB logging"
    - "WandB define_metric before loop: update as step_metric enables per-update x-axis in WandB dashboard"

key-files:
  created: []
  modified:
    - accel_training/train.py
    - accel_training/config.yml

key-decisions:
  - "ES strategy routing via config['es_strategy'] string: ns_es instantiates NSESStrategy, cma_es instantiates CMAESStrategy — clean branch at train() startup"
  - "run_archive_warmup called when es_strategy != cma_es OR warmup_n > 0 — allows warm-up for CMA-ES baseline too if warmup_n set"
  - "ES tell() placed after insert_batch in NEW/mutate branch — ES state and novelty updated after buffer state is current"
  - "mean_novelty initialized to 0.0 and only updated in NS-ES NEW/mutate branch — replay branch and CMA-ES retain last known value"

patterns-established:
  - "Solvability gate: valid_np = np.asarray(valid) & ~nan_mask; valid_indices = np.where(valid_np)[0] before any tile/insert"
  - "Behavior_sig extraction: separate rollout_agent_on_levels_with_positions call after eval_fn; extract_behavior_signature on agent_positions"

requirements-completed: [INTEG-01, INTEG-02, INTEG-03]

# Metrics
duration: 2min
completed: 2026-03-02
---

# Phase 3 Plan 02: NS-ES Integration Summary

**End-to-end NS-ES wired into ACCEL training loop: behavior_sig extraction at insert site, archive warm-up with solvability gate, WandB metrics, and ES strategy routing from config**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-02T17:34:04Z
- **Completed:** 2026-03-02T17:36:40Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Closed the assert gap: behavior_sig is now extracted via rollout_agent_on_levels_with_positions + extract_behavior_signature at every NEW/mutate insert_batch call
- Added run_archive_warmup() with mandatory solvability gate (valid_np + nan_mask filtering before tile/insert); called synchronously before training step 0
- Added ES strategy routing: config['es_strategy'] routes to NSESStrategy (ns_es) or CMAESStrategy (cma_es) with init_state at train() startup
- Added WandB initialization with define_metric and per-update logging (regret, novelty_score, replay_buffer_size, buffer_occupied, valid_fraction, mean_buffer_score)
- Added complete ES config block to config.yml with all Phase 3 NS-ES parameters

## Task Commits

Each task was committed atomically:

1. **Task 1: Add behavior_sig extraction, warm-up, ES routing, and WandB to train.py** - `0fda4a3` (feat)
2. **Task 2: Add ES config block to config.yml** - `269ce83` (feat)

## Files Created/Modified

- `accel_training/train.py` - Added imports (wandb, rollout_agent_on_levels_with_positions, extract_behavior_signature, NSESStrategy, CMAESStrategy); run_archive_warmup() module-level function; ES strategy instantiation; warmup call before loop; behavior_sig extraction at insert site; ES tell() after insert; WandB init and logging
- `accel_training/config.yml` - Added ES config block (es_strategy, es_alpha, es_beta, es_pop_size, es_sigma_init, es_k_novelty, warmup_n) and WandB config (wandb_project, wandb_log_freq)

## Decisions Made

- ES strategy routing uses config.get() with "cma_es" as safe default — train.py works without the new ES config keys if absent
- run_archive_warmup triggered when es_strategy != "cma_es" OR warmup_n > 0 — allows warm-up for CMA-ES baseline experiments too
- ES tell() placed after insert_batch so buffer_sigs reflect current state (just-inserted entries included in novelty archive)
- mean_novelty initialized to 0.0 at train() startup; only updated in NS-ES NEW/mutate branch; WandB logs last known value for replay steps

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. Both verification scripts passed on first run.

## User Setup Required

WandB logging requires a WandB account and API key. Before running train.py:
1. Install wandb: `pip install wandb`
2. Authenticate: `wandb login` (enter API key from https://wandb.ai/settings)
3. Or set environment variable: `WANDB_API_KEY=<your-key>`

The training script will log to project `es-accel` (configurable via `wandb_project` in config.yml).

## Next Phase Readiness

- End-to-end NS-ES pipeline is complete: warmup -> training loop -> ES routing -> WandB logging
- Phase 3 Plan 03 can run integration tests verifying the full pipeline end-to-end
- No blockers — both static checks pass, assert gap is closed

---
*Phase: 03-ns-es-integration*
*Completed: 2026-03-02*
