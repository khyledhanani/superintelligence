---
phase: 05-pca-space-cma-es-search
plan: 02
subsystem: training
tags: [pca, cma-es, cnn-vae, latent-space, jit-recompilation, maze-plr, two-stage]

# Dependency graph
requires:
  - phase: 05-pca-space-cma-es-search
    plan: 01
    provides: vae/cnn_vae_pca_utils.py with 5 PCA utility functions
  - phase: 03-integration
    provides: CNN-VAE wired into maze_plr.py, CMAESManager, decode_latent_to_levels_grid

provides:
  - examples/maze_plr.py with --use_pca_search flag for two-stage PCA-space CMA-ES
  - Stage 1 active from step 0: weight-norm pruned dims via compute_active_dims, no dataset needed
  - Stage 2 transition at configurable step: buffer-derived PCA via encode_mazes_to_mu + compute_pca_axes
  - JIT factory pattern: jax.jit(train_and_eval_step) called again after Stage 2 to force recompilation
  - 7 new CLI flags: --use_pca_search, --pca_components, --pca_dataset_size, --pca_dataset_path, --pca_stage2_step, --pca_stage1_k, --pca_sigma_init
  - pca/stage and pca/K WandB logging after each eval step
  - pca/explained_var and pca/transition_step WandB logging at Stage 2 transition

affects:
  - 05-03 (plan 03: experiment run with --use_pca_search)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "JIT factory pattern: remove @jax.jit decorator, use explicit jax.jit(f) variable; reassign after Stage 2 to force retrace"
    - "Stage 1 at init time: compute_active_dims(mean_layer_kernel) prunes dims from checkpoint weights only"
    - "Stage 2 at transition: jax.vmap(level_to_tokens)(buffer_levels) + np.array() conversion before encode_mazes_to_mu"
    - "Python closures capture variable bindings (not values): reassigning cmaes_mgr/vae_decode_fn + re-jitting forces new trace"
    - "es_state reinitialized and injected via ts_cur.replace(es_state=new_es_state) at Stage 2"

key-files:
  created: []
  modified:
    - examples/maze_plr.py

key-decisions:
  - "Remove @jax.jit from train_and_eval_step and use explicit jax.jit() variable to enable re-jitting after Stage 2 transition"
  - "Stage 1 overwrites cmaes_mgr and vae_decode_fn; Stage 2 overwrites them again then forces JIT recompilation"
  - "Stage 2 guarded by _pca_stage == 1 (not _pca_stage < 2) to fire exactly once at or after pca_stage2_step"
  - "Buffer encoding: tokens_np = np.array(tokens_jax) required before encode_mazes_to_mu (which calls clutr_to_grid internally)"
  - "PCA WandB logging done outside jit in outer loop (pca/stage, pca/K are Python-level state, not JAX arrays)"

patterns-established:
  - "Import cnn_vae_pca_utils unconditionally at module load (lightweight, no side effects)"
  - "_cnn_vae_params = _restored['params'] saved in CNN-VAE setup block for encoder access in Stage 1/2"
  - "cnn_base_decode_fn = vae_decode_fn saved before PCA wrapping so both stages can wrap the original decoder"

requirements-completed: [PCA-03, PCA-04, PCA-05, PCA-06, PCA-07]

# Metrics
duration: 4min
completed: 2026-03-13
---

# Phase 5 Plan 02: PCA-Space CMA-ES Integration Summary

**Two-stage PCA-space CMA-ES integrated into examples/maze_plr.py: Stage 1 weight-norm pruned search active from step 0, Stage 2 buffer-PCA transition at 10k steps with JIT recompilation via factory pattern**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-13T10:34:28Z
- **Completed:** 2026-03-13T10:38:28Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `from cnn_vae_pca_utils import ...` and saved `_cnn_vae_params` + `cnn_base_decode_fn` references in CNN-VAE setup block for use by both PCA stages
- Stage 1 PCA block activates when `--use_cmaes --use_pca_search` is set: calls `compute_active_dims` on checkpoint weights, wraps `vae_decode_fn` with `make_variance_pruned_decode_fn`, reinitializes `CMAESManager` with K_stage1 dims — all before training begins, no dataset needed
- Stage 2 transition hook in outer loop fires at `pca_stage2_step` (default 10k): encodes buffer levels via `jax.vmap(level_to_tokens)` + `encode_mazes_to_mu`, computes PCA axes, wraps decode_fn with `make_pc_decode_fn`, reinitializes `CMAESManager` and `es_state`, then calls `jax.jit(train_and_eval_step)` to force JIT recompilation with new closures
- JIT factory pattern: replaced `@jax.jit` decorator with `train_and_eval_step_jitted = jax.jit(train_and_eval_step)` — variable is reassigned at Stage 2 to trigger retrace
- All 7 CLI flags added: `--use_pca_search`, `--pca_components`, `--pca_dataset_size`, `--pca_dataset_path`, `--pca_stage2_step`, `--pca_stage1_k`, `--pca_sigma_init`

## Task Commits

Each task was committed atomically:

1. **Task 1: PCA imports, CLI flags, Stage 1 setup, WandB logging** - `e347e9e` (feat)
2. **Task 2: Stage 2 transition hook and JIT factory** - `33e1a5f` (feat)

## Files Created/Modified

- `examples/maze_plr.py` - Two-stage PCA-space CMA-ES search: Stage 1 setup, Stage 2 transition hook, JIT factory pattern, 7 CLI flags, WandB logging

## Decisions Made

- **Remove @jax.jit decorator from train_and_eval_step** and use explicit `jax.jit()` variable — Python's `@jax.jit` caches by function object identity; reassigning the variable and calling `jax.jit(train_and_eval_step)` again creates a new jitted wrapper that retraces on first call, picking up the new `cmaes_mgr` and `vae_decode_fn` closures
- **Stage 2 guarded by `_pca_stage == 1`** (not `_pca_stage < 2`) to fire exactly once at or after `pca_stage2_step`; if buffer is too small, stays in Stage 1 with a warning
- **`tokens_np = np.array(tokens_jax)` conversion required** before `encode_mazes_to_mu` — the function calls `clutr_to_grid` internally which requires numpy, not JAX arrays
- **PCA WandB logging outside jit** in the outer loop — `_pca_stage` and `_effective_latent_dim` are Python-level state (not JAX arrays), so they cannot be logged from inside `train_step`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - both tasks passed all verification checks on first attempt. The `_effective_latent_dim = _latent_dim_for_cmaes if config["use_cmaes"] else None` pattern is safe because Python ternary evaluates lazily: `_latent_dim_for_cmaes` is only accessed when `use_cmaes=True`, in which case it is defined.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 03 can now run `python examples/maze_plr.py --use_cmaes --use_pca_search` to exercise the two-stage search
- Stage 1 is active immediately (no dataset download needed) — the 10k step Stage 2 transition will trigger automatically during the run
- The `--pca_stage2_step` flag can be set to 0 to force immediate Stage 2 (if a dataset is available) or to a later step to give the buffer more time to fill

## Self-Check: PASSED

- FOUND: examples/maze_plr.py (modified)
- FOUND commit: e347e9e (feat(05-02): add PCA imports, CLI flags, Stage 1 setup, and WandB logging)
- FOUND commit: 33e1a5f (feat(05-02): add Stage 2 transition hook and JIT factory for recompilation)
- FOUND: from cnn_vae_pca_utils import in examples/maze_plr.py
- FOUND: make_variance_pruned_decode_fn usage in Stage 1 block
- FOUND: make_pc_decode_fn usage in Stage 2 transition
- FOUND: train_and_eval_step_jitted in outer loop
- FOUND: all 7 PCA CLI flags in --help output

---
*Phase: 05-pca-space-cma-es-search*
*Completed: 2026-03-13*
