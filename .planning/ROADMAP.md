# Roadmap: CNN-VAE CMA-ES Integration

## Overview

Four sequential phases that take the project from zero code to a running 20k-update comparison experiment. Each phase is a hard prerequisite for the next: checkpoint loading enables adapter implementation, a verified adapter enables safe training integration, and a passing smoke test enables the GPU experiment launch. No phase can be parallelized with another.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Checkpoint** - Download CNN-VAE checkpoint from GCS and verify decoder param tree
- [x] **Phase 2: Grid Adapter** - Implement and unit-test `decode_latent_to_levels_grid()` in isolation
- [x] **Phase 3: Integration** - Wire CNN-VAE as default decoder in `maze_plr.py` and pass smoke tests
- [ ] **Phase 4: Experiment** - Launch 20k-update CNN-VAE CMA-ES vs ACCEL comparison run
- [ ] **Phase 5: PCA-Space CMA-ES Search** - Two-stage dimensionality reduction for CMA-ES (variance-pruned + buffer PCA)

## Phase Details

### Phase 1: Checkpoint
**Goal**: CNN-VAE decoder params are loaded locally and confirmed correct before any code is written
**Depends on**: Nothing (first phase)
**Requirements**: CKPT-01, CKPT-02, CKPT-03
**Success Criteria** (what must be TRUE):
  1. `vae/checkpoints/cnn_vae/` exists locally with Orbax checkpoint files from GCS run10/step200000
  2. `decoder_params` dict can be extracted and its keys match the documented `CnnLstmDecoder` param tree (`dec_lstm`, `dec_proj`, `dec_conv1-3`, `wall_head`, `goal_head`, `agent_head`)
  3. `decode_fn(z_zeros)` runs without error and returns `(wall_logits, goal_logits, agent_logits)` each of shape `(13, 13)`
**Plans**: 1 plan
- [x] 01-01-PLAN.md — Download CNN-VAE checkpoint from GCS and verify decoder params + decode_fn output

### Phase 2: Grid Adapter
**Goal**: `decode_latent_to_levels_grid()` correctly converts CNN-VAE output to valid Level objects, verified in isolation before touching the training loop
**Depends on**: Phase 1
**Requirements**: GRID-01, GRID-02, GRID-03, GRID-04, GRID-05, GRID-06, GRID-07, GRID-08, GRID-09
**Success Criteria** (what must be TRUE):
  1. Decoding `z=zeros(64)` produces a Level that passes `level.is_well_formatted()` with correct field shapes and dtypes (`wall_map: bool`, `goal_pos/agent_pos: uint32`, `agent_dir: uint8`)
  2. `level.to_str()` visual output confirms goal and agent are placed at non-wall cells and their coordinates are not row/col inverted
  3. A batch of 1000 random-z decodes produces zero goal-agent collisions (same flat index) and zero wall-cell placements
  4. `jax.jit(decode_latent_to_levels_grid)` compiles and runs without error (JIT compatibility confirmed)
**Plans**: 2 plans
- [x] 02-01-PLAN.md — Implement vae/cnn_vae_level_utils.py with decode_latent_to_levels_grid (GRID-01..08)
- [x] 02-02-PLAN.md — Write and run scripts/test_grid_adapter.py verifying all GRID-01..09 against real checkpoint

### Phase 3: Integration
**Goal**: CNN-VAE is the default decode path in `maze_plr.py`, CluttrVAE fallback is preserved, and a 1000-step CMA-ES run passes with no errors and acceptable valid-structure rate
**Depends on**: Phase 2
**Requirements**: INTG-01, INTG-02, INTG-03, INTG-04, VALD-01, VALD-02, VALD-03, VALD-04
**Success Criteria** (what must be TRUE):
  1. `python examples/maze_plr.py --use_cmaes` launches with CNN-VAE as decoder (no flag required) without error
  2. `python examples/maze_plr.py --use_cmaes --use_clutr_vae` launches with original CluttrVAE token decoder and completes without error
  3. A 1000-step CMA-ES run with CNN-VAE decoder completes with `cmaes/valid_structure_pct > 90%` and no NaN fitness values logged to WandB
  4. BFS solvability check confirms generated levels are navigable (at least one path from agent to goal exists)
**Plans**: 2 plans
- [x] 03-01-PLAN.md — Add --use_clutr_vae flag, CNN-VAE conditional setup block, and decode dispatch in maze_plr.py
- [x] 03-02-PLAN.md — Write and run scripts/smoke_test_integration.py (VALD-01/03/04) + 1000-step CMA-ES run (VALD-02)

### Phase 4: Experiment
**Goal**: A full 20k-update CNN-VAE CMA-ES vs ACCEL comparison run completes and results are logged to WandB for write-up
**Depends on**: Phase 3
**Requirements**: EXPT-01, EXPT-02, EXPT-03
**Success Criteria** (what must be TRUE):
  1. The adapted launch script runs both ACCEL and CNN-VAE CMA-ES jobs sequentially or in parallel with WandB logging under a consistent group name
  2. Both runs reach 20k updates without crashing; WandB shows complete training curves for solve rate and max returns
  3. CNN-VAE CMA-ES results are directly comparable to ACCEL baseline (same metrics, same update count, same evaluation protocol)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in strict sequential order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Checkpoint | 1/1 | Complete | 2026-03-11 |
| 2. Grid Adapter | 2/2 | Complete | 2026-03-11 |
| 3. Integration | 2/2 | Complete | 2026-03-11 |
| 4. Experiment | 0/? | Not started | - |
| 5. PCA-Space CMA-ES | 1/3 | In Progress|  |

### Phase 5: PCA-Space CMA-ES Search
**Goal**: Two-stage dimensionality reduction for CMA-ES: Stage 1 (weight-norm pruned latent search, ~30 dims from step 0 using checkpoint weights) and Stage 2 (buffer PCA search, K components after 10k steps). CMA-ES searches in a reduced subspace that reflects the valid maze manifold, giving it free covariance structure and O(K^2) sample complexity instead of O(64^2).
**Depends on**: Phase 3 (CNN-VAE integrated into maze_plr.py)
**Requirements**: PCA-01, PCA-02, PCA-03, PCA-04, PCA-05, PCA-06, PCA-07, PCA-08
**Success Criteria** (what must be TRUE):
  1. `vae/cnn_vae_pca_utils.py` exists with `encode_mazes_to_mu`, `compute_active_dims`, `compute_pca_axes`, `make_variance_pruned_decode_fn`, `make_pc_decode_fn`
  2. `python examples/maze_plr.py --use_cmaes --use_pca_search` starts with Stage 1 (weight-norm pruned dims from checkpoint) from step 0 and CMAESManager uses K_stage1 < 64
  3. At the Stage 2 transition step (default 10k), buffer levels are encoded to PCA, CMAESManager reinitialized with K_stage2 dims, and `train_and_eval_step` is recompiled
  4. A 500-step CMA-ES run with `--use_pca_search` completes with exit code 0, valid_structure_pct > 90%, no NaN
  5. z=zeros(K) through both Stage 1 and Stage 2 decode wrappers produces valid Levels
**Plans**: 3 plans
- [ ] 05-01-PLAN.md — Create vae/cnn_vae_pca_utils.py (5 functions) + scripts/download_pca_dataset.py
- [ ] 05-02-PLAN.md — Integrate Stage 1 + Stage 2 into maze_plr.py (flags, setup, transition hook, JIT factory)
- [ ] 05-03-PLAN.md — Write and run smoke_test_pca_search.py (PCA-08: offline + 500-step training)
