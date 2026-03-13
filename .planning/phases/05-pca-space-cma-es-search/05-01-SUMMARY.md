---
phase: 05-pca-space-cma-es-search
plan: 01
subsystem: vae
tags: [pca, cma-es, cnn-vae, latent-space, dimensionality-reduction, numpy-svd, jax-vmap]

# Dependency graph
requires:
  - phase: 03-integration
    provides: CNN-VAE wired into maze_plr.py, decode_latent_to_levels_grid, CMAESManager

provides:
  - vae/cnn_vae_pca_utils.py with 5 functions for two-stage PCA-space CMA-ES search
  - scripts/download_pca_dataset.py for end-to-end validation on GPU nodes
  - encode_mazes_to_mu: CLUTR seqs -> (N, 64) mu vectors via CnnEncoder + mean_layer
  - compute_active_dims: weight-norm pruning for Stage 1 without dataset encoding
  - compute_pca_axes: truncated SVD returning mu_mean, pc_axes, pc_stds, evr
  - make_variance_pruned_decode_fn: K-dim Stage 1 closure (JIT/vmap compatible)
  - make_pc_decode_fn: K-dim whitened PCA Stage 2 closure (JIT/vmap compatible)

affects:
  - 05-02 (plan 02: integrate into maze_plr.py with --use_pca_search flag)
  - 05-03 (plan 03: Stage 2 buffer PCA encoding at 10k steps)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Weight-norm pruning: np.linalg.norm(kernel, axis=0) on mean_layer (512, 64) identifies active dims without dataset"
    - "Whitened inverse PCA: z_latent = mu_mean + (z_pc * pc_stds) @ pc_axes maps unit-variance search to latent space"
    - "Closure pattern: make_*_decode_fn captures jnp arrays (not numpy) for JIT/vmap compatibility"
    - "Batch encoding: @jax.jit inner function processes grids in batches of 256 for efficiency"
    - "params['mean_layer'] at TOP LEVEL (NOT inside params['encoder']) — checked at every usage"

key-files:
  created:
    - vae/cnn_vae_pca_utils.py
    - scripts/download_pca_dataset.py
  modified: []

key-decisions:
  - "cumulative norm threshold 0.85 for Stage 1 K selection (not fixed K) — adapts to any checkpoint"
  - "full_matrices=False in np.linalg.svd is essential for efficiency when N >> 64"
  - "Closures capture jnp arrays (not numpy) so they are JIT/vmap compatible without conversion"
  - "Stage 1 uses VAE prior mean (zeros) as mu_mean baseline — dataset not needed at search time"
  - "pc_stds used for whitening in make_pc_decode_fn: z_pc=1 corresponds to 1 std deviation"

patterns-established:
  - "PCA utilities are self-contained in vae/cnn_vae_pca_utils.py (no other files modified)"
  - "Imports use bare module names (not vae.module) matching existing vae/ file convention"
  - "compute_active_dims returns kept_dims sorted ascending for stable index-based operations"

requirements-completed: [PCA-01, PCA-02]

# Metrics
duration: 3min
completed: 2026-03-13
---

# Phase 5 Plan 01: PCA Utilities Summary

**Five-function PCA utility library for two-stage CMA-ES latent-space search: weight-norm pruning (Stage 1, no data needed) and whitened SVD-based PCA (Stage 2, buffer-derived)**

## Performance

- **Duration:** 3 min (~148s)
- **Started:** 2026-03-13T10:29:00Z
- **Completed:** 2026-03-13T10:32:00Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments

- Created `vae/cnn_vae_pca_utils.py` with all 5 exported functions for two-stage dimensionality reduction
- Stage 1 function `compute_active_dims` works from checkpoint weights alone — no dataset encoding needed for CMA-ES initialization
- Both decode wrappers (`make_variance_pruned_decode_fn`, `make_pc_decode_fn`) produce JIT and vmap-compatible closures that integrate with the existing `decode_latent_to_levels_grid` function unchanged
- Created `scripts/download_pca_dataset.py` as integration test script that validates all functions end-to-end on GPU nodes

## Task Commits

Each task was committed atomically:

1. **Task 1: Create vae/cnn_vae_pca_utils.py** - `e02de7b` (feat)
2. **Task 2: Create scripts/download_pca_dataset.py** - `0bd3647` (feat)

## Files Created/Modified

- `vae/cnn_vae_pca_utils.py` - Five PCA utility functions: encode_mazes_to_mu, compute_active_dims, compute_pca_axes, make_variance_pruned_decode_fn, make_pc_decode_fn
- `scripts/download_pca_dataset.py` - End-to-end validation script: GCS download + 10k encoding + Stage 1/2 analysis + vmap decode test

## Decisions Made

- **Cumulative norm threshold 0.85** for Stage 1 K selection rather than fixed K — adapts to any trained checkpoint; expected K ~25-35 dims out of 64
- **full_matrices=False** in np.linalg.svd — essential for efficiency when N (10k) >> latent_dim (64); avoids (N, N) U matrix
- **Closures capture jnp arrays** not numpy arrays — required for JAX JIT/vmap tracing; plain numpy inside closures would fail at trace time
- **Stage 1 mu_mean = zeros** as default — matches VAE prior N(0,I); dataset encoding is optional for Stage 1 (by design)
- **pc_stds whitening** in make_pc_decode_fn — ensures CMA-ES sees unit-variance dimensions, sigma_init=0.5 becomes meaningful

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - both tasks passed all verification checks on first attempt.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02 can now integrate these utilities into `examples/maze_plr.py` with `--use_pca_search` flag
- `compute_active_dims` is ready to call at training start (no data needed) for immediate Stage 1
- `encode_mazes_to_mu` and `compute_pca_axes` are ready for Stage 2 buffer encoding at 10k steps
- `scripts/download_pca_dataset.py` should be run on sideswipe or prowl to validate end-to-end before Plan 02 integration

## Self-Check: PASSED

- FOUND: vae/cnn_vae_pca_utils.py
- FOUND: scripts/download_pca_dataset.py
- FOUND: .planning/phases/05-pca-space-cma-es-search/05-01-SUMMARY.md
- FOUND commit: e02de7b (feat(05-01): create vae/cnn_vae_pca_utils.py)
- FOUND commit: 0bd3647 (feat(05-01): create scripts/download_pca_dataset.py)

---
*Phase: 05-pca-space-cma-es-search*
*Completed: 2026-03-13*
