# Requirements: CNN-VAE CMA-ES Integration

**Defined:** 2026-03-11
**Core Value:** CMA-ES with CNN-VAE produces valid, solvable maze Levels and runs a complete 20k training experiment comparable to the ACCEL baseline.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Checkpoint Loading

- [x] **CKPT-01**: CNN-VAE Orbax checkpoint downloaded from GCS to `vae/checkpoints/cnn_vae/`
- [x] **CKPT-02**: Decoder params loaded via Orbax (PyTreeCheckpointer or CheckpointManager) and verified against `CnnLstmDecoder.init()` param tree
- [x] **CKPT-03**: `decode_fn` closure created: `z (latent_dim,) → (wall_logits, goal_logits, agent_logits)` each `(13, 13)`

### Grid-to-Level Adapter

- [x] **GRID-01**: Wall map derived from wall logits: `wall_map = sigmoid(wall_logits) > 0.5` (threshold at logit=0)
- [x] **GRID-02**: Goal/agent logits masked at wall positions (set to -1e9) before argmax to prevent placement on walls
- [x] **GRID-03**: Goal position computed from masked argmax: `flat_idx → (x=col=flat%13, y=row=flat//13)`
- [x] **GRID-04**: Agent position computed from masked argmax with same coordinate transform
- [x] **GRID-05**: Goal/agent collision resolved when argmax produces same flat index
- [x] **GRID-06**: Wall cells cleared at goal/agent positions (ensure `wall_map[row, col] = False`)
- [x] **GRID-07**: Agent direction randomized (0-3) per sample using provided RNG key
- [x] **GRID-08**: `decode_latent_to_levels_grid()` is JIT-compatible via `jax.vmap` over single-sample function
- [x] **GRID-09**: Generated levels pass `Level.is_well_formatted()` validation

### Training Integration

- [x] **INTG-01**: CNN-VAE is the default decoder path in `examples/maze_plr.py` when `use_cmaes=True`
- [x] **INTG-02**: `--use_clutr_vae` flag falls back to original CluttrVAE token-based decoder
- [x] **INTG-03**: CluttrVAE path remains fully functional (no breaking changes)
- [x] **INTG-04**: `decode_latent_to_levels_grid()` drops into existing CMA-ES ask/decode/tell loop

### Validation

- [x] **VALD-01**: Smoke test: decode `z=zeros(64)` → valid Level with correct field shapes and dtypes
- [x] **VALD-02**: Short CMA-ES run (1000 steps) completes without errors
- [x] **VALD-03**: Generated levels verified solvable via BFS pathfinding check
- [x] **VALD-04**: Coordinate convention verified: Level positions match expected grid locations

### Experiment

- [ ] **EXPT-01**: `scripts/launch_sfl_cenie.sh` adapted to use CNN-VAE decoder instead of CluttrVAE (CNN-VAE checkpoint download, `--use_cmaes` path uses CNN-VAE by default)
- [ ] **EXPT-02**: Script runs both ACCEL + CMA-ES+CNN-VAE together with WandB logging (same pattern as existing ACCEL vs CluttrVAE runs)
- [ ] **EXPT-03**: 20k-update comparison run launched via adapted script

### PCA-Space CMA-ES Search

- [x] **PCA-01**: `vae/cnn_vae_pca_utils.py` created with `encode_mazes_to_mu`, `compute_active_dims`, `compute_pca_axes`, `make_variance_pruned_decode_fn`, `make_pc_decode_fn`
- [x] **PCA-02**: `compute_active_dims` uses mean_layer weight norms from checkpoint to select active dims (no dataset encoding needed for Stage 1); `scripts/download_pca_dataset.py` for offline validation only
- [ ] **PCA-03**: Stage 1 weight-norm pruned decode function wraps base decode_fn; CMA-ES searches in ~30 reduced dims from step 0 using `make_variance_pruned_decode_fn` — checkpoint weights only, no data dependency
- [ ] **PCA-04**: Stage 2 PCA computed from replay buffer levels' mu vectors at configurable transition step (default 10k updates)
- [ ] **PCA-05**: Stage 2 decode function uses whitened PCA projection via `make_pc_decode_fn`; CMAESManager reinitialized with K_stage2 dims and `pca_sigma_init`
- [ ] **PCA-06**: JIT recompilation handled at Stage 1->2 transition via factory pattern (`jax.jit(train_and_eval_step)` called fresh after reassigning closures)
- [ ] **PCA-07**: CLI flags added to `maze_plr.py`: `--use_pca_search`, `--pca_components`, `--pca_dataset_size`, `--pca_dataset_path`, `--pca_stage2_step`, `--pca_stage1_k`, `--pca_sigma_init`
- [ ] **PCA-08**: Smoke test passes: z=zeros(K) decodes to valid Level through both wrappers; 500-step CMA-ES run with `--use_pca_search` completes with exit 0, valid_structure_pct > 90%, no NaN

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Analysis

- **ANLZ-01**: Detailed comparison report: CNN-VAE CMA-ES vs ACCEL (solve rates, level diversity, training curves)
- **ANLZ-02**: Visualization of generated maze levels from CNN-VAE decoder
- **ANLZ-03**: Latent space interpolation analysis for CNN-VAE

## Out of Scope

| Feature | Reason |
|---------|--------|
| CluttrVAE vs CNN-VAE comparison | User explicitly not interested in comparing the two VAEs |
| CNN-VAE retraining | Checkpoint is fixed (run10, step 200000) |
| NS-ES / SV-CMA-ES strategies | Only CMA-ES for this integration |
| CMAESManager changes | Already latent-dim agnostic |
| Temperature-based sampling | Would make CMA-ES fitness non-deterministic |
| ConvTranspose decoder variant | CNN-VAE uses nearest-neighbor upsampling by design |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CKPT-01 | Phase 1 | Pending |
| CKPT-02 | Phase 1 | Pending |
| CKPT-03 | Phase 1 | Pending |
| GRID-01 | Phase 2 | Complete |
| GRID-02 | Phase 2 | Complete |
| GRID-03 | Phase 2 | Complete |
| GRID-04 | Phase 2 | Complete |
| GRID-05 | Phase 2 | Complete |
| GRID-06 | Phase 2 | Complete |
| GRID-07 | Phase 2 | Complete |
| GRID-08 | Phase 2 | Complete |
| GRID-09 | Phase 2 | Complete |
| INTG-01 | Phase 3 | Complete |
| INTG-02 | Phase 3 | Complete |
| INTG-03 | Phase 3 | Complete |
| INTG-04 | Phase 3 | Complete |
| VALD-01 | Phase 3 | Complete |
| VALD-02 | Phase 3 | Complete |
| VALD-03 | Phase 3 | Complete |
| VALD-04 | Phase 3 | Complete |
| EXPT-01 | Phase 4 | Pending |
| EXPT-02 | Phase 4 | Pending |
| EXPT-03 | Phase 4 | Pending |
| PCA-01 | Phase 5 | Complete |
| PCA-02 | Phase 5 | Complete |
| PCA-03 | Phase 5 | Pending |
| PCA-04 | Phase 5 | Pending |
| PCA-05 | Phase 5 | Pending |
| PCA-06 | Phase 5 | Pending |
| PCA-07 | Phase 5 | Pending |
| PCA-08 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 31 total
- Mapped to phases: 31
- Unmapped: 0

---
*Requirements defined: 2026-03-11*
*Last updated: 2026-03-12 — Phase 5 PCA-01..PCA-08 added for PCA-Space CMA-ES Search*
