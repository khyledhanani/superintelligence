---
phase: 05-pca-space-cma-es-search
plan: 03
subsystem: testing
tags: [pca, cma-es, cnn-vae, smoke-test, validation, stage1, stage2, weight-norm-pruning]

# Dependency graph
requires:
  - phase: 05-pca-space-cma-es-search
    plan: 01
    provides: vae/cnn_vae_pca_utils.py with 5 PCA utility functions
  - phase: 05-pca-space-cma-es-search
    plan: 02
    provides: examples/maze_plr.py with --use_pca_search flag and two-stage integration

provides:
  - scripts/smoke_test_pca_search.py with Phase A (offline decode validation) and Phase B (500-step training run)
  - PCA-08 validated: Stage 1 K=55, Stage 2 K=20 (71.7% variance), valid_structure_pct=100%, sigma=0.963

affects: []  # Phase 5 complete after this plan

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Smoke test Phase A/B split: Phase A (offline CPU validation of decode wrappers) + Phase B (subprocess 500-step training run)"
    - "Subprocess approach for Phase B: subprocess.run with capture_output=True; check returncode, search stdout for Stage 1 message"
    - "Stage 1 K=55 (upper boundary): cumulative 86.5% of weight norms; valid range [15, 55] confirmed"
    - "Two-phase test design: offline validation can run anywhere; Phase B needs GPU (or slow CPU)"

key-files:
  created:
    - scripts/smoke_test_pca_search.py
  modified: []

key-decisions:
  - "Stage 1 K=55 (86.5% cumulative norm threshold) — at upper boundary of expected [15, 55] but PASS"
  - "Phase B run on CPU (blaze-equivalent sumida): 500 steps completed in ~31 min (vs ~2 min on GPU); valid"
  - "pca_stage2_step=99999 keeps Stage 1 active for full 500 steps so Phase B tests Stage 1 path only"
  - "Stage 2 explains 71.7% variance with K=20 PCs (> 50% requirement satisfied)"

patterns-established:
  - "Smoke test script is standalone (sys.path manipulation) with no circular imports"
  - "Phase B uses subprocess.run with capture_output=True; parse combined stdout+stderr for Stage 1 message"
  - "NaN check searches for 'nan' in lines containing fitness/sigma/valid/reward/loss keywords only"

requirements-completed: [PCA-08]

# Metrics
duration: 36min
completed: 2026-03-13
---

# Phase 5 Plan 03: PCA-Space CMA-ES Smoke Test Summary

**End-to-end PCA-space CMA-ES smoke test: Stage 1 K=55 weight-norm pruned search active from step 0, 500-step training run exits 0 with sigma=0.963 and valid_structure_pct=100%**

## Performance

- **Duration:** ~36 min (~5 min script + 31 min 500-step CPU run)
- **Started:** 2026-03-13T10:41:50Z
- **Completed:** 2026-03-13T11:17:00Z
- **Tasks:** 1
- **Files created:** 1

## Accomplishments

- Created `scripts/smoke_test_pca_search.py` with Phase A (offline decode validation) and Phase B (500-step training run via subprocess)
- Phase A validated: Stage 1 compute_active_dims yields K=55 from weight norms alone (no dataset needed), Stage 2 encode_mazes_to_mu + compute_pca_axes yields K=20 PCs explaining 71.7% variance
- Phase A validated: both Stage 1 and Stage 2 decode wrappers produce 100% valid Levels from z=zeros and random z_batch of 32
- Phase B validated: 500-step CMA-ES with `--use_pca_search` exits 0; Stage 1 activates (`[PCA Stage 1] Keeping 55 of 64 dims`); sigma=0.963 (no collapse); valid_structure_pct=100%
- PCA-08 result comment recorded in script header with actual measured values

## Task Commits

Each task was committed atomically:

1. **Task 1: Write scripts/smoke_test_pca_search.py and run 500-step PCA-space CMA-ES** - `ea54245` (feat)

## Files Created/Modified

- `scripts/smoke_test_pca_search.py` - Two-phase smoke test: Phase A (offline decode validation for Stage 1 and Stage 2 wrappers) + Phase B (500-step CMA-ES training subprocess), with PCA-08 result comment

## Decisions Made

- **Stage 1 K=55 at upper boundary**: cumulative norm threshold 0.85 yields K=55 from this checkpoint's mean_layer weights — technically at the assertion boundary `15 <= K1 <= 55`, but all tests pass. The threshold adapts to checkpoint quality.
- **Phase B run on CPU**: GPU nodes (sideswipe, prowl) were not accessible via SSH from the head node. The 500-step run completed on CPU in ~31 minutes (vs ~2 min on GPU). All functional requirements verified (exit code 0, sigma stability, valid_structure_pct).
- **pca_stage2_step=99999**: Stage 2 transition is suppressed for the smoke test — tests Stage 1 path end-to-end without requiring dataset GCS download or long encoding step.
- **Subprocess approach for Phase B**: `subprocess.run` with `capture_output=True` captures combined stdout+stderr, allowing regex search for Stage 1 message and NaN checks.

## Deviations from Plan

None - plan executed exactly as written. The Phase A and Phase B checks match the plan specification. The only practical difference is that Phase B ran on CPU (slower but equivalent functional validation).

## Issues Encountered

None. All checks passed on first attempt:
- Stage 1: K=55 (at expected [15, 55] upper bound), 100% batch validity
- Stage 2: 71.7% variance, 100% batch validity
- 500-step run: exit 0, sigma=0.963, valid_structure_pct=100%, no NaN

## User Setup Required

None - no external service configuration required. Dataset was already present at `/tmp/train_1M_envs.npy`.

## Next Phase Readiness

- Phase 5 is complete: PCA utilities (Plan 01) + maze_plr.py integration (Plan 02) + smoke test validation (Plan 03) all done
- To run the full PCA-space CMA-ES experiment: `python examples/maze_plr.py --use_cmaes --use_pca_search --num_updates 30000 --pca_stage2_step 10000 --run_name pca_search_exp --project JAXUED_COMPARISON --seed 42` on a GPU node
- Stage 2 transition will fire at step 10000: encodes buffer levels -> PCA -> new K=20 dim search space

## Self-Check: PASSED

- FOUND: scripts/smoke_test_pca_search.py
- FOUND commit: ea54245 (feat(05-03): add smoke_test_pca_search.py for PCA-08 validation)
- FOUND: PCA-08 RESULT comment in smoke_test_pca_search.py
- Phase A Stage 1: K=55, 100% valid (validated inline)
- Phase A Stage 2: K=20, 71.7% variance, 100% valid (validated inline)
- Phase B: 500-step run exit 0, sigma=0.963, valid_structure_pct=100% (validated via /tmp/smoke_phase_b.log)

---
*Phase: 05-pca-space-cma-es-search*
*Completed: 2026-03-13*
