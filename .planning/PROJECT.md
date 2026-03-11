# CNN-VAE CMA-ES Integration

## What This Is

Replace the CluttrVAE (LSTM token-sequence decoder) with a CNN-LSTM VAE (grid-based decoder) in the CMA-ES latent space search pipeline for procedural maze generation. The CNN-VAE outputs 13x13 grids directly instead of 52-token sequences, requiring a new grid-to-Level adapter and checkpoint loading path. The goal is to run a full 20k-update CMA-ES experiment with the CNN-VAE and compare solve rates against the vanilla ACCEL baseline.

## Core Value

CMA-ES with the CNN-VAE must produce valid, solvable maze Levels and run a complete 20k training experiment comparable to the ACCEL baseline.

## Requirements

### Validated

- ✓ CNN-VAE model files exist in repo — `vae/cnn_vae_model.py`, `vae/cnn_vae_data.py`, `vae/cnn_vae_losses.py` (commit 0b503eb)
- ✓ CMAESManager is latent-dim agnostic — no changes needed (`vae/cmaes_manager.py`)
- ✓ CluttrVAE integration works end-to-end in `examples/maze_plr.py`
- ✓ ACCEL baseline script works (`examples/maze_plr.py` with `use_cmaes=False`)

### Active

- [ ] Grid-to-Level adapter: `decode_latent_to_levels_grid()` converts CNN-VAE output `(wall_logits, goal_logits, agent_logits)` to batch of `Level` objects
- [ ] Wall masking: goal/agent argmax must avoid wall positions
- [ ] Checkpoint loading: download Orbax checkpoint to `vae/checkpoints/cnn_vae/`, load decoder params
- [ ] `maze_plr.py` integration: CNN-VAE as default decoder path, CluttrVAE kept as fallback via `--use_clutr_vae` flag
- [ ] Launch scripts: adapt existing scripts for CNN-VAE CMA-ES vs ACCEL comparison
- [ ] 20k-update comparison run: CNN-VAE CMA-ES vs ACCEL, logged to WandB
- [ ] Validation: smoke test (decode z=0 → valid Level), short CMA-ES run (1000 steps)

### Out of Scope

- CluttrVAE vs CNN-VAE comparison — not interested in comparing the two VAEs against each other
- Retraining the CNN-VAE — checkpoint is fixed (run10, step 200000)
- NS-ES / SV-CMA-ES strategies — only CMA-ES for this integration
- Changes to CMAESManager — it's latent-dim agnostic already
- Changes to the ACCEL baseline code — `examples/maze_plr.py` vanilla path stays untouched

## Context

**CNN-VAE architecture:**
- Encoder: 3 strided Conv layers (13x13→2x2) + LSTM bridge → 512-dim → mu/logvar (64-dim, bounded tanh*4)
- Decoder: LSTM spatial unfold → nearest-neighbor upsample (2x2→13x13) → 3 output heads
- Output: `(wall_logits, goal_logits, agent_logits)` each `(B, 13, 13)` — NOT token logits
- Latent dim: 64 (same as CluttrVAE)
- Performance: Wall IoU=0.860, prior solvability=96%, goal accuracy=100%

**Key difference from CluttrVAE:**
CluttrVAE decoder outputs `(seq_len=52, vocab_size=170)` token logits, converted to Levels via `tokens_to_level()`. CNN-VAE outputs `(13,13)` grids directly — needs a different conversion path (`decode_latent_to_levels_grid()`).

**Checkpoint:**
- Format: Orbax (not pickle)
- Location: GCS `gs://cnn-vae-maze-checkpoints/run10/` (step 200000)
- Local target: `vae/checkpoints/cnn_vae/`
- Param tree: `params/decoder/...` (dec_lstm, dec_proj, dec_conv1-3, wall_head, goal_head, agent_head)

**Coordinate convention:**
- Level positions: `(x, y) = (col, row)`
- Grid indexing: `grid[row, col]` (row-major)
- Conversion: `flat_idx = row * 13 + col`, then `pos = (col, row)` = `(flat_idx % 13, flat_idx // 13)`

**Existing comparison infrastructure:**
- `examples/maze_plr.py` supports both ACCEL and CMA-ES modes
- Existing launch scripts can be adapted
- WandB logging already set up
- 20k updates, same as Phase 5 runs

## Constraints

- **JIT compatibility**: `decode_latent_to_levels_grid()` must work inside `jax.jit` — no Python loops, use `jax.vmap`
- **Coordinate mapping**: Must correctly map grid `(row, col)` to Level `(x=col, y=row)` convention
- **No breaking changes**: CluttrVAE path must remain functional as `--use_clutr_vae` fallback
- **Environment**: GPU nodes sideswipe/prowl (CUDA 12), never set XLA_FLAGS, use jax_env conda env
- **Checkpoint format**: Orbax (not pickle) — requires `orbax.checkpoint` or compatible loader

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| CNN-VAE as default decoder | User wants CNN-VAE to be the primary path, not opt-in | — Pending |
| Download checkpoint locally | Simpler than GCS runtime loading, avoids auth complexity | — Pending |
| Reuse existing launch scripts | Minimize new infrastructure, adapt what works | — Pending |
| 20k updates for comparison | Matches Phase 5 run length, proven sufficient for convergence | — Pending |
| Local checkpoint path: `vae/checkpoints/cnn_vae/` | Keeps checkpoints organized within the vae directory | — Pending |

---
*Last updated: 2026-03-11 after initialization*
