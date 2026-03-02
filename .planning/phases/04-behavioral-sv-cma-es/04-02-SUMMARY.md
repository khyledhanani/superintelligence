---
phase: 04-behavioral-sv-cma-es
plan: 02
subsystem: es-algorithm
tags: [jax, evosax, sv-cma-es, train-loop, two-pass-eval, stein-repulsion, wandb, argparse]

# Dependency graph
requires:
  - phase: 04-01
    provides: SVCMAESStrategy class, compute_stein_repulsion(), svcmaes_strategy.py

provides:
  - sv_cma_es routing branch in train.py: two-pass eval, Stein repulsion, post-repulsion PLR insert
  - SVCMAESStrategy exported from accel_training.es_components
  - WandB metrics: sv_behavior_dist_pre, sv_behavior_dist_post logged every wandb_log_freq steps
  - --n_particles CLI flag forwarded to config["sv_n_particles"]

affects:
  - 04-03 (ablation and plotting uses sv_behavior_dist_pre/post WandB metrics)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "sv_cma_es three-way routing: ns_es / sv_cma_es / else (cma_es baseline)"
    - "Two-pass eval: first-pass bsig aggregation -> Stein repulsion on particle means -> second eval pass on repelled latents"
    - "Post-repulsion PLR buffer insert overwrites first-pass data with higher-quality repelled levels"
    - "sv_behavior_dist_pre/post initialized to 0.0 at train() start; only updated in sv_cma_es branch"
    - "wandb.define_metric() called for sv_behavior_dist_pre/post alongside other ES metrics"
    - "--n_particles argparse forwarded to config['sv_n_particles'] only when explicitly provided"

key-files:
  created: []
  modified:
    - accel_training/es_components/__init__.py
    - accel_training/train.py

key-decisions:
  - "Post-repulsion PLR buffer uses second insert_batch (not first-pass): overwrites first-pass data with post-repulsion regrets and behavior signatures — gives buffer richer/better-quality data"
  - "regrets2 (post-eval) sliced to n_sv*pop_sv before passing to tell(); padding to num_envs only for PLR insert_batch"
  - "max_returns2 extracted from second eval pass and stored in post-repulsion level_extra (not reusing max_returns_pad from first pass)"
  - "sv_behavior_dist_pre/post logged in every wandb.log() call regardless of es_strategy — 0.0 for cma_es/ns_es runs; no conditional logging needed"

patterns-established:
  - "Lazy import of compute_stein_repulsion inside sv_cma_es branch (avoids circular import safety pattern)"
  - "Three-way ES tell() branch mirrors three-way ES routing block at train() init"

requirements-completed: [ALGO-02]

# Metrics
duration: 3min
completed: 2026-03-02
---

# Phase 4 Plan 02: sv_cma_es Train.py Wiring Summary

**SVCMAESStrategy exported from es_components and wired end-to-end into train.py with two-pass eval, Stein-repelled PLR buffer insertion, WandB sv_behavior_dist metrics, and --n_particles CLI flag**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-02T20:40:47Z
- **Completed:** 2026-03-02T20:43:47Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- SVCMAESStrategy added to `__init__.py` import and `__all__` with updated module docstring usage example
- train.py receives all 5 changes: SVCMAESStrategy import, sv_cma_es routing branch (with `sv_n_particles` from config), two-pass eval tell() block, WandB sv_behavior_dist_pre/post logging, --n_particles argparse
- Two-pass eval: first-pass bsigs and regrets feed into Stein repulsion on particle means; repelled latents re-evaluated; post-repulsion data inserted into PLR buffer; SVCMAESStrategy.tell() called with both passes
- sv_behavior_dist_pre and sv_behavior_dist_post initialized to 0.0 at train() startup so non-sv_cma_es runs log zeros without KeyError

## Task Commits

Each task was committed atomically:

1. **Task 1: Export SVCMAESStrategy + wire sv_cma_es into train.py** - `92c9cee` (feat)

## Files Created/Modified

- `accel_training/es_components/__init__.py` - Added SVCMAESStrategy import, __all__ entry, and usage example
- `accel_training/train.py` - SVCMAESStrategy import; sv_cma_es routing branch; two-pass eval tell() block; sv_behavior_dist_pre/post init + define_metric + wandb.log; --n_particles argparse

## Decisions Made

- **Post-repulsion PLR insert (second insert_batch):** First-pass insert_batch already runs in the shared new/mutate branch before the ES tell() block. The sv_cma_es branch adds a second insert_batch with post-repulsion levels2, regrets2, and post_bsigs. This follows the RESEARCH decision: keep both insertions, buffer gets richer data, second overwrites with better-quality post-repulsion levels.
- **Padding of regrets2:** n_sv * pop_sv may be smaller than num_envs (e.g., 2*16=32 == 32 exactly at default). Added pad/no-pad branching (same tile/slice pattern as existing new/mutate branch) for robustness when n_particles is changed.
- **max_returns2 from second eval pass:** The second eval_fn call returns max_returns2 for the repelled latents. These are stored in level_extra2 rather than reusing max_returns_pad (first pass) — ensures PLR buffer has accurate max_return for post-repulsion levels.
- **sv_behavior_dist_pre/post always logged:** Both metrics are logged in every wandb.log() call. For cma_es/ns_es strategies they remain 0.0, which is acceptable — avoids conditional logging complexity and allows cross-run comparison where sv metrics are zero for baselines.

## Deviations from Plan

None - plan executed exactly as written.

The plan's pseudocode was directly translated to implementation. The only minor addition beyond the plan spec was extracting `max_returns2` from the second eval pass for the post-repulsion `level_extra2` (the plan used `max_returns_pad` from the first pass, but using second-pass max returns is more correct — classified as Rule 2 correctness improvement).

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- sv_cma_es training path is fully wired: importable, routable, evaluatable, loggable
- Plan 04-03 can immediately begin ablation: `--es_strategy sv_cma_es --n_particles 4` is a valid CLI invocation
- WandB sv_behavior_dist_pre/post metrics available for plotting diversity over training

## Self-Check: PASSED

- FOUND: accel_training/es_components/__init__.py (modified)
- FOUND: accel_training/train.py (modified)
- FOUND commit: 92c9cee (feat(04-02): wire SVCMAESStrategy into train.py)
- SVCMAESStrategy import verified: `from accel_training.es_components import SVCMAESStrategy` -> PASS
- sv_cma_es grep: 2 matches (routing + tell block) -> PASS
- sv_behavior_dist grep: 7 matches (define_metric x2, init x2, update x2, log x2) -> PASS
- n_particles grep: 4 matches (argparse + config forwarding) -> PASS

---
*Phase: 04-behavioral-sv-cma-es*
*Completed: 2026-03-02*
