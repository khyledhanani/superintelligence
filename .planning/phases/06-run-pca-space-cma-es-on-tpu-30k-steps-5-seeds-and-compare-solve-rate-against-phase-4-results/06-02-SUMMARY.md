---
phase: 06-run-pca-space-cma-es-on-tpu-30k-steps-5-seeds-and-compare-solve-rate-against-phase-4-results
plan: 02
subsystem: infra
tags: [tpu, gcloud, scp, pca, cma-es, maze_plr, wandb]

# Dependency graph
requires:
  - phase: 06-01
    provides: "launch_pca_comparison.sh and compare_phase4_results.py scripts"
  - phase: 05-pca-space-cma-es-search
    provides: "PCA-space CMA-ES implementation in maze_plr.py and cnn_vae_pca_utils.py"
provides:
  - "All Phase 5+6 code synced to TPU VM cma-es-v4 (maze_plr.py, cnn_vae_pca_utils.py, launch_pca_comparison.sh, compare_phase4_results.py)"
  - "Human checkpoint: TPU VM free check and pca_run tmux session launch"
affects: [06-03, compare-results]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "gcloud scp for remote file sync (gcloud compute tpus tpu-vm scp)"
    - "wc -l verification of synced file integrity"

key-files:
  created: []
  modified: []

key-decisions:
  - "gcloud binary is at /cs/student/project_msc/2025/csml/gmaralla/home/google-cloud-sdk/bin/gcloud (not in PATH by default — use full path)"
  - "SCP 'Attempting to connect to worker 0...' output is normal — gcloud SSH tunnel init, not an error"
  - "Task 1 has no local git-trackable changes (pure remote sync) — SUMMARY commit captures plan completion"

patterns-established:
  - "Pattern 1: Always use full gcloud path when running from non-interactive shells"

requirements-completed: [RUN-01, RUN-02]

# Metrics
duration: 4min
completed: 2026-03-13
---

# Phase 6 Plan 02: TPU File Sync and Experiment Launch Summary

**Phase 5+6 code (maze_plr.py, cnn_vae_pca_utils.py, launch_pca_comparison.sh, compare_phase4_results.py) synced to TPU VM cma-es-v4; human checkpoint reached to verify TPU is free and launch pca_run tmux session**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-13T11:53:41Z
- **Completed:** 2026-03-13T11:55:34Z
- **Tasks:** 1 complete, 1 checkpoint (awaiting human)
- **Files modified:** 0 (pure remote sync)

## Accomplishments
- Synced all 4 Phase 5+6 files to TPU VM cma-es-v4 via gcloud scp
- Verified file integrity via wc -l (all line counts match local: 1874+241+35+88 = 2238 total)
- Reached checkpoint: human must SSH to TPU, check comparison3 session is done, and launch pca_run

## Task Commits

No local file changes in this plan — all work was remote TPU sync operations.

**Plan metadata:** (pending — blocked at checkpoint)

## Files Created/Modified
- None — this plan syncs existing local files to remote TPU VM

## Decisions Made
- gcloud binary path: `/cs/student/project_msc/2025/csml/gmaralla/home/google-cloud-sdk/bin/gcloud` (not in PATH by default on head node blaze)
- SCP commands displayed "Attempting to connect to worker 0..." which is normal gcloud SSH tunnel initialization, not an error; verification via ssh command confirmed all files landed correctly

## Deviations from Plan

None - plan executed exactly as written. The only discovery was the gcloud binary location (not in PATH), which was auto-resolved by finding it at the expected home directory location.

## Issues Encountered
- `gcloud` not in PATH on head node (blaze). Found at `/cs/student/project_msc/2025/csml/gmaralla/home/google-cloud-sdk/bin/gcloud`. Used full path for all subsequent commands.
- SCP commands showed "Using scp batch size of 1. Attempting to connect to worker 0..." without a completion message, but subsequent SSH verification confirmed all 4 files synced correctly.

## User Setup Required

**Human action required.** SSH to TPU VM `cma-es-v4` and:

1. Check if Phase 4 experiment (`tmux ls` for `comparison3` session) is complete
2. If TPU is free, start the PCA experiment:
   ```bash
   gcloud compute tpus tpu-vm ssh cma-es-v4 --zone us-central2-b
   cd ~/superintelligence
   mkdir -p logs
   tmux new-session -d -s pca_run 'bash examples/launch_pca_comparison.sh 2>&1 | tee logs/pca_comparison_main.log'
   tmux attach -t pca_run
   ```
3. Verify seed 0 shows `[PCA Stage 1] Keeping 55 of 64 dims` in output
4. Detach with Ctrl+B, D
5. Confirm WandB group `pca-cmaes-accel` has an active run at https://wandb.ai

**Resume signal:** Type "running" once seed 0 is past Stage 1 init, or "waiting" if Phase 4 is still running.

## Next Phase Readiness
- Files are on the TPU VM and ready
- Human must confirm TPU is free and start the tmux session
- Once pca_run is active and past Stage 1, Phase 6 Plan 03 (monitoring + comparison) can begin

---
*Phase: 06-run-pca-space-cma-es-on-tpu-30k-steps-5-seeds-and-compare-solve-rate-against-phase-4-results*
*Completed: 2026-03-13*
