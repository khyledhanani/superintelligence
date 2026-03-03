---
phase: 05-ablations-and-analysis
plan: "02"
subsystem: experiment-infrastructure
tags: [wandb, jupyter, matplotlib, bash, es, cma_es, ns_es, sv_cma_es, accel]

# Dependency graph
requires:
  - phase: 05-01
    provides: Refactored two-mode train.py with --es_strategy/--run_name/--group/--seed/--num_updates CLI flags

provides:
  - Executable bash launcher script running pre-launch smoke + four 20k-update experiments sequentially
  - Jupyter notebook pulling WandB data and producing thesis-quality comparison figures

affects:
  - Phase 06 ablations (notebook has placeholder cell for phase6-ablations group)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sequential experiment launcher with set -e and read -r smoke-test gate"
    - "WandB api.runs() with group filter for multi-run data retrieval"
    - "Rolling mean smoothing (window=50) on regret curves for thesis figures"

key-files:
  created:
    - scripts/run_phase5_comparison.sh
    - notebooks/phase5_comparison.ipynb
  modified: []

key-decisions:
  - "Sequential execution (no backgrounding) avoids JAX GPU OOM on shared machine"
  - "read -r pause after smoke test lets user verify buf_score > 0.004 before committing to 20k runs"
  - "ACCEL baseline runs examples/maze_plr.py black-box; WandB name/group managed via UI rename after run"
  - "set -e halts launcher on first failure, preventing cascading wasted compute"
  - "matplotlib.use('Agg') non-interactive backend for server environment notebook execution"

patterns-established:
  - "Launcher pattern: smoke test gate -> full experiments -> notebook analysis"
  - "Notebook pattern: WandB API pull -> pandas smoothing -> matplotlib figure -> savefig PDF+PNG"

requirements-completed: [COMP-01]

# Metrics
duration: 6min
completed: 2026-03-03
---

# Phase 5 Plan 02: Experiment Launcher and Thesis Notebook Summary

**Sequential four-way comparison launcher (smoke + 4x20k runs) and WandB-backed Jupyter notebook producing thesis-quality regret curves for ACCEL, CMA-ES, NS-ES, SV-CMA-ES.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-03T17:31:07Z
- **Completed:** 2026-03-03T17:37:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Bash launcher script runs pre-launch smoke test (SV-CMA-ES, 1k updates) with read -r gate, then sequentially runs CMA-ES, NS-ES, SV-CMA-ES (each 20k updates), then ACCEL baseline black-box
- All experiment runs configured with seed=42, group=phase5-comparison for reproducible WandB grouping
- Jupyter notebook pulls WandB data via wandb.Api(), applies 50-step rolling mean, saves Figure 1 (four-method comparison) to figures/phase5_comparison.{pdf,png}; Figure 2 cell is a Phase 6 placeholder

## Task Commits

Each task was committed atomically:

1. **Task 1: Create experiment launcher script** - `c9e2c92` (feat)
2. **Task 2: Create Jupyter notebook for thesis-quality comparison plots** - `b29a442` (feat)

## Files Created/Modified

- `scripts/run_phase5_comparison.sh` - Sequential launcher: smoke test gate + four 20k-update experiments + ACCEL baseline
- `notebooks/phase5_comparison.ipynb` - WandB API data pull, smoothed regret curves, Figure 1 + Phase 6 placeholder

## Decisions Made

- Sequential execution (no `&` backgrounding) chosen to avoid JAX GPU OOM on shared research machine
- `read -r` pause after smoke test provides manual verification gate before committing hours of compute
- ACCEL baseline runs `examples/maze_plr.py` completely unmodified (black-box per user decision from Phase 5 research); WandB run name/group require manual adjustment in WandB UI after run completes
- `set -e` ensures launcher halts on first error rather than silently continuing to next experiment
- `matplotlib.use("Agg")` non-interactive backend selected for server-side notebook execution compatibility

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. WandB credentials assumed pre-configured from Phase 05-01.

**Note for ACCEL baseline:** After running `examples/maze_plr.py`, the resulting WandB run will need its name set to `accel-baseline` and group tagged as `phase5-comparison` via the WandB UI, so the notebook's `api.runs()` group filter picks it up correctly.

## Next Phase Readiness

- Launcher script ready to execute in tmux when experiments are to begin
- Notebook is a complete template ready to execute once WandB data from experiments is available
- Phase 6 ablation placeholder cell is in place in the notebook
- Ready for Phase 6: ablation studies (alpha/beta sweep for SV-CMA-ES, group=phase6-ablations)

---
*Phase: 05-ablations-and-analysis*
*Completed: 2026-03-03*

## Self-Check: PASSED

- FOUND: scripts/run_phase5_comparison.sh
- FOUND: notebooks/phase5_comparison.ipynb
- FOUND commit: c9e2c92 (Task 1 launcher script)
- FOUND commit: b29a442 (Task 2 notebook)
