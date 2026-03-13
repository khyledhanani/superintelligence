---
phase: 06-run-pca-space-cma-es-on-tpu-30k-steps-5-seeds-and-compare-solve-rate-against-phase-4-results
plan: 01
subsystem: infra
tags: [cma-es, pca, wandb, tpu, bash, experiment-orchestration]

# Dependency graph
requires:
  - phase: 05-pca-space-cma-es-search
    provides: PCA-space CMA-ES training code smoke-tested and ready (--use_pca_search flag)
  - phase: 03-integration
    provides: examples/maze_plr.py with --use_cmaes --use_accel entrypoints
provides:
  - examples/launch_pca_comparison.sh — 5-seed PCA-CMA-ES TPU launch script (group pca-cmaes-accel)
  - scripts/compare_phase4_results.py — WandB 3-way comparison: pca-cmaes-accel vs cmaes-cnn-vae-accel vs accel-baseline
affects:
  - phase 06 human-verify checkpoint (TPU run + comparison analysis)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "JAX_COMPILATION_CACHE_DIR reuse across seeds — set once in launch script, avoids recompilation cost per seed"
    - "Per-seed tee logging — 2>&1 | tee logs/pca_comparison_seed${seed}.log captures both stdout and stderr"
    - "WandB group = --run_name value — maze_plr.py passes run_name directly to wandb.init group param"

key-files:
  created:
    - examples/launch_pca_comparison.sh
    - scripts/compare_phase4_results.py
  modified: []

key-decisions:
  - "PCA launch script runs only 5 seeds (1 condition) — Phase 4 already has cmaes-cnn-vae-accel and accel-baseline data; no need to re-run baselines"
  - "WANDB_DIR=/tmp/wandb to avoid TPU local disk bloat (TPU VMs have small root volumes)"
  - "compare_phase4_results.py accepts --entity as optional arg (WandB infers from login if omitted)"
  - "Final value extraction via hist[metric].dropna().iloc[-1] — robust to sparse logging"

patterns-established:
  - "Experiment launch scripts: COMMON variable + per-seed loop + tee logging pattern"
  - "WandB analysis scripts: api.runs(project, filters={'group': group}) pattern for group queries"

requirements-completed: [RUN-01, RUN-03]

# Metrics
duration: 2min
completed: 2026-03-13
---

# Phase 06 Plan 01: PCA-space CMA-ES Experiment Orchestration Scripts Summary

**5-seed PCA-space CMA-ES TPU launch script and 3-way WandB comparison script (pca-cmaes-accel vs cmaes-cnn-vae-accel vs accel-baseline)**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-13T11:48:44Z
- **Completed:** 2026-03-13T11:50:10Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created `examples/launch_pca_comparison.sh`: runs 5 seeds of PCA-space CMA-ES at 30k updates, WandB group `pca-cmaes-accel`, XLA compilation cache reused across seeds
- Created `scripts/compare_phase4_results.py`: queries WandB API for 3 groups, prints mean ± std + per-seed values table, handles missing groups gracefully

## Task Commits

Each task was committed atomically:

1. **Task 1: Create examples/launch_pca_comparison.sh** - `4570fd4` (feat)
2. **Task 2: Create scripts/compare_phase4_results.py** - `fc171ee` (feat)

## Files Created/Modified
- `examples/launch_pca_comparison.sh` - 5-seed PCA-CMA-ES experiment launcher for TPU; --use_pca_search --pca_stage2_step 10000 --pca_components 20 --pca_sigma_init 0.5; WandB group pca-cmaes-accel
- `scripts/compare_phase4_results.py` - WandB API query script; compares solve_rate/mean across 3 groups; --entity/--project/--metric CLI args; graceful missing-data handling

## Decisions Made
- PCA launch script runs only 5 seeds (1 condition) — Phase 4 already has cmaes-cnn-vae-accel and accel-baseline baselines; no need to re-run them
- WANDB_DIR=/tmp/wandb to avoid TPU local disk bloat
- compare_phase4_results.py accepts --entity as optional arg (WandB infers from login if omitted)
- Final value extraction via hist[metric].dropna().iloc[-1] — robust to sparse WandB logging

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required beyond what Phase 4 already set up. Scripts are ready to sync to TPU VM and run.

## Next Phase Readiness
- Both scripts ready to sync to TPU VM: `gcloud compute tpus tpu-vm scp examples/launch_pca_comparison.sh scripts/compare_phase4_results.py cma-es-v4:~/superintelligence/ --zone us-central2-b`
- Run `bash examples/launch_pca_comparison.sh` from `~/superintelligence/` on TPU VM
- After all 5 seeds finish, run `python scripts/compare_phase4_results.py` (add `--entity <username>` if needed)
- Check WandB `JAXUED_COMPARISON` project, group `pca-cmaes-accel`, average `solve_rate/mean` at step 30k

---
*Phase: 06-run-pca-space-cma-es-on-tpu-30k-steps-5-seeds-and-compare-solve-rate-against-phase-4-results*
*Completed: 2026-03-13*

## Self-Check: PASSED

- FOUND: examples/launch_pca_comparison.sh
- FOUND: scripts/compare_phase4_results.py
- FOUND: 06-01-SUMMARY.md
- FOUND commit 4570fd4 (feat(06-01): add PCA-space CMA-ES launch script)
- FOUND commit fc171ee (feat(06-01): add WandB 3-way comparison script)
