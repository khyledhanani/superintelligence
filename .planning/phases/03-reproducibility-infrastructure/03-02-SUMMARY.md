---
phase: 03-reproducibility-infrastructure
plan: 02
subsystem: infra
tags: [wandb, bash, launch-scripts, comparison-tooling, ablation]

# Dependency graph
requires:
  - phase: 03-01
    provides: LevelCache, wall-map hash logging, llm_inject_start_step rename
  - phase: 02-02
    provides: gated injection pipeline with --use_llm --use_accel flags
provides:
  - "launch_llm_injection.sh: ACCEL+LLM training launch script with 3 seeds, ablation-ready injection flags"
  - "launch_accel_only_control.sh: ACCEL-only control launch script with matching seeds and hyperparameters"
  - "compare_llm_results.py: WandB comparison table for accel-llm vs accel-only groups"
affects: [04-comparison-experiments]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Ablation parameters extracted as shell variables at top of launch script (INJECT_START, INJECT_INTERVAL, BATCH_SIZE)"
    - "COMMON variable shared between launch scripts enforces matching non-injection hyperparameters"
    - "WandB API pattern: api.runs(project) → group by run_name → final metric extraction"

key-files:
  created:
    - examples/launch_llm_injection.sh
    - examples/launch_accel_only_control.sh
    - scripts/compare_llm_results.py
  modified: []

key-decisions:
  - "Ablation parameters (INJECT_START, INJECT_INTERVAL, BATCH_SIZE) extracted as named shell variables at script top — changing one variable changes all uses"
  - "Both launch scripts target JAXUED_LLM WandB project with groups accel-llm and accel-only for side-by-side comparison"
  - "COMMON variable in both scripts holds all shared flags — ensures matching non-injection hyperparameters between conditions"

patterns-established:
  - "Ablation pattern: shell variables at script top, used in $PYTHON invocation — ablation = change variable value only"
  - "Control condition pattern: identical COMMON flags, same seeds, different group name and no LLM flags"

requirements-completed: [EXPT-02, EXPT-03]

# Metrics
duration: 15min
completed: 2026-03-24
---

# Phase 3 Plan 02: Reproducibility Infrastructure — Launch Scripts and Comparison Tooling Summary

**ACCEL+LLM injection and ACCEL-only control launch scripts with matching seeds and ablation-ready injection flags, plus a WandB comparison script for post-run analysis.**

## Performance

- **Duration:** 15 min (including checkpoint review)
- **Started:** 2026-03-24T10:30:00Z
- **Completed:** 2026-03-24T10:45:00Z
- **Tasks:** 3 (2 auto + 1 human-verify checkpoint)
- **Files modified:** 3 created, 0 modified

## Accomplishments

- Created `examples/launch_llm_injection.sh` with `--use_accel --use_llm`, ablation variables at top (INJECT_START, INJECT_INTERVAL, BATCH_SIZE), CUDA 13.1 LD_LIBRARY_PATH fix, and 3 seeds for JAXUED_LLM project group `accel-llm`
- Created `examples/launch_accel_only_control.sh` with `--use_accel` only, same COMMON flags as LLM script, same 3 seeds for JAXUED_LLM project group `accel-only` — matching conditions enforced via shared COMMON variable
- Created `scripts/compare_llm_results.py` querying `wandb.Api()` for JAXUED_LLM runs, grouping by run_name (accel-llm vs accel-only), printing mean+std solve rate and LLM-specific metrics (acceptance_rate, injected_count, diversity_score_mean)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create launch scripts for LLM injection and ACCEL-only control** - `3f64dc4` (feat)
2. **Task 2: Create WandB comparison script** - `d328d16` (feat)
3. **Task 3: Verify launch scripts and comparison tooling** - checkpoint:human-verify (user approved)

**Plan metadata:** TBD (this commit)

## Files Created/Modified

- `examples/launch_llm_injection.sh` - ACCEL+LLM injection launch script; 3 seeds, ablation variables (INJECT_START=5000, INJECT_INTERVAL=3000, BATCH_SIZE=25), CUDA LD_LIBRARY_PATH fix, WANDB_DIR, JAX_COMPILATION_CACHE_DIR, group `accel-llm` in `JAXUED_LLM`
- `examples/launch_accel_only_control.sh` - ACCEL-only control; same COMMON flags as LLM script, 3 matching seeds, no --use_llm, group `accel-only` in `JAXUED_LLM`
- `scripts/compare_llm_results.py` - WandB API comparison script; --project/--entity CLI args, groups by run_name, prints mean+std solve rate + acceptance_rate + injected_count + diversity_score_mean, per-run detail table, handles missing metrics as N/A

## Decisions Made

- Ablation parameters (INJECT_START, INJECT_INTERVAL, BATCH_SIZE) extracted as named shell variables at top of `launch_llm_injection.sh` — changing one variable changes all invocations, ablation requires editing only 1 line
- Both scripts share a COMMON variable to enforce matching non-injection hyperparameters (num_updates=50000, eval_freq=250, skip_video, skip_post_eval) — control condition integrity guaranteed structurally
- Both scripts target the same JAXUED_LLM WandB project with distinct group names (accel-llm, accel-only) so comparison script can query both conditions in one API call

## Deviations from Plan

None — plan executed exactly as written. Launch scripts and comparison script match all must_haves from plan frontmatter. User approved at checkpoint.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required. WandB credentials must already be present (from existing 50k experiment runs).

## Next Phase Readiness

- Phase 4 (Comparison Experiments) is unblocked: launch scripts are ready to run, comparison tooling exists
- User SSHes to albacore/smew/canada and runs `bash examples/launch_llm_injection.sh` for LLM condition; `bash examples/launch_accel_only_control.sh` for control
- After runs complete: `python scripts/compare_llm_results.py` prints the comparison table
- Ablation (different INJECT_START or INJECT_INTERVAL) requires editing only the variables at the top of launch_llm_injection.sh
- Open question from STATE.md: confirm whether existing JAXUED_50K accel-baseline runs are sufficient as control or whether fresh JAXUED_LLM control runs are needed for identical experimental conditions — scripts exist for both paths

---
*Phase: 03-reproducibility-infrastructure*
*Completed: 2026-03-24*
