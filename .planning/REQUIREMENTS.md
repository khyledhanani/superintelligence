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

- [ ] **GRID-01**: Wall map derived from wall logits: `wall_map = sigmoid(wall_logits) > 0.5` (threshold at logit=0)
- [ ] **GRID-02**: Goal/agent logits masked at wall positions (set to -1e9) before argmax to prevent placement on walls
- [ ] **GRID-03**: Goal position computed from masked argmax: `flat_idx → (x=col=flat%13, y=row=flat//13)`
- [ ] **GRID-04**: Agent position computed from masked argmax with same coordinate transform
- [ ] **GRID-05**: Goal/agent collision resolved when argmax produces same flat index
- [ ] **GRID-06**: Wall cells cleared at goal/agent positions (ensure `wall_map[row, col] = False`)
- [ ] **GRID-07**: Agent direction randomized (0-3) per sample using provided RNG key
- [ ] **GRID-08**: `decode_latent_to_levels_grid()` is JIT-compatible via `jax.vmap` over single-sample function
- [ ] **GRID-09**: Generated levels pass `Level.is_well_formatted()` validation

### Training Integration

- [ ] **INTG-01**: CNN-VAE is the default decoder path in `examples/maze_plr.py` when `use_cmaes=True`
- [ ] **INTG-02**: `--use_clutr_vae` flag falls back to original CluttrVAE token-based decoder
- [ ] **INTG-03**: CluttrVAE path remains fully functional (no breaking changes)
- [ ] **INTG-04**: `decode_latent_to_levels_grid()` drops into existing CMA-ES ask/decode/tell loop

### Validation

- [ ] **VALD-01**: Smoke test: decode `z=zeros(64)` → valid Level with correct field shapes and dtypes
- [ ] **VALD-02**: Short CMA-ES run (1000 steps) completes without errors
- [ ] **VALD-03**: Generated levels verified solvable via BFS pathfinding check
- [ ] **VALD-04**: Coordinate convention verified: Level positions match expected grid locations

### Experiment

- [ ] **EXPT-01**: `scripts/launch_sfl_cenie.sh` adapted to use CNN-VAE decoder instead of CluttrVAE (CNN-VAE checkpoint download, `--use_cmaes` path uses CNN-VAE by default)
- [ ] **EXPT-02**: Script runs both ACCEL + CMA-ES+CNN-VAE together with WandB logging (same pattern as existing ACCEL vs CluttrVAE runs)
- [ ] **EXPT-03**: 20k-update comparison run launched via adapted script

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
| GRID-01 | Phase 2 | Pending |
| GRID-02 | Phase 2 | Pending |
| GRID-03 | Phase 2 | Pending |
| GRID-04 | Phase 2 | Pending |
| GRID-05 | Phase 2 | Pending |
| GRID-06 | Phase 2 | Pending |
| GRID-07 | Phase 2 | Pending |
| GRID-08 | Phase 2 | Pending |
| GRID-09 | Phase 2 | Pending |
| INTG-01 | Phase 3 | Pending |
| INTG-02 | Phase 3 | Pending |
| INTG-03 | Phase 3 | Pending |
| INTG-04 | Phase 3 | Pending |
| VALD-01 | Phase 3 | Pending |
| VALD-02 | Phase 3 | Pending |
| VALD-03 | Phase 3 | Pending |
| VALD-04 | Phase 3 | Pending |
| EXPT-01 | Phase 4 | Pending |
| EXPT-02 | Phase 4 | Pending |
| EXPT-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 23 total
- Mapped to phases: 23
- Unmapped: 0

---
*Requirements defined: 2026-03-11*
*Last updated: 2026-03-11 — traceability filled after roadmap creation*
