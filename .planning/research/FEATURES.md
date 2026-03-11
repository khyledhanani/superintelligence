# Feature Research

**Domain:** VAE-to-game-level conversion pipeline (CNN-VAE + CMA-ES maze generation)
**Researched:** 2026-03-11
**Confidence:** HIGH (primary sources: existing codebase, vae_level_utils.py, cnn_vae_losses.py, maze_plr.py)

---

## Feature Landscape

### Table Stakes (Pipeline Breaks Without These)

Features the CMA-ES training loop depends on. Missing any one of these means `decode_latent_to_levels_grid()` either crashes or produces nonsense Levels that the environment rejects.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Wall sigmoid thresholding** | CNN-VAE outputs raw `wall_logits (B,13,13)` — must apply `sigmoid > 0.5` to get a binary `wall_map` before building Level | LOW | Direct port of training-time behavior; `jax.nn.sigmoid` + bool cast, fully jittable |
| **Goal/agent wall masking at decode time** | `cnn_vae_losses.py::apply_wall_mask` already implements this for training — must reuse the identical logic (`-1e9` at wall positions) before `argmax` to prevent goal/agent landing on a wall cell | LOW | `apply_wall_mask()` is already written and exported; just call it with predicted wall mask, not GT mask |
| **Goal/agent argmax placement** | Goal and agent positions come from `jnp.argmax` over the wall-masked `(B,169)` logits flat — no sampling, deterministic decode | LOW | Deterministic is correct here; sampling would require a temperature parameter and adds variance with no benefit |
| **Coordinate transform: flat-idx to (x,y)** | Level stores positions as `(x,y) = (col, row)` but grids are indexed `[row, col]`; `flat_idx → (col, row)` is non-trivial and already burned into `vae_level_utils.py::tokens_to_level` | LOW | Formula: `x = flat_idx % 13`, `y = flat_idx // 13`; must match existing convention exactly or agent/goal appear in wrong cells |
| **Collision resolution: goal != agent** | If argmax of goal and agent land on the same cell, the Level is invalid; must shift one to the nearest free cell | MEDIUM | The CluttrVAE path handles this in `repair_tokens()`; the CNN-VAE path must implement equivalent logic without token mutation — a `jnp.where` fallback to next non-wall cell suffices |
| **Defensive wall clearing at agent/goal cells** | Even after wall masking, the predicted `wall_logits` might still flag agent/goal cells as walls; must clear `wall_map[agent_y, agent_x]` and `wall_map[goal_y, goal_x]` before constructing Level | LOW | `tokens_to_level()` already does this as a final step; same pattern needed in grid decoder |
| **JAX vmap compatibility** | The entire decode must work inside `jax.vmap` over the batch dimension, matching how `decode_latent_to_levels()` is structured (one `_decode_single` vmapped over N) | MEDIUM | No Python-level control flow over N; all branching via `jnp.where`; shapes must be static |
| **Orbax checkpoint loading** | CNN-VAE checkpoint is Orbax format (not pickle) at `vae/checkpoints/cnn_vae/`; must load `params/decoder/...` subtree | MEDIUM | Current CluttrVAE path uses `pickle.load()`; CNN-VAE needs `orbax.checkpoint` restore; param subtree structure is documented in `cnn_vae_model.py` |
| **Decoder-only param extraction** | Only the `decoder` sub-tree of the checkpoint is needed for inference; extracting `params["decoder"]` and passing to `CnnLstmDecoder.apply({"params": decoder_params}, z)` | LOW | Must verify the Orbax-restored tree key name matches `params/decoder/...` — confirmed by `CnnLstmVAE` module naming convention |
| **`--use_clutr_vae` fallback flag** | CluttrVAE path in `maze_plr.py` must remain functional; CNN-VAE is the new default, CluttrVAE is opt-in | LOW | Gated by `if config.get("use_clutr_vae")` branch; no changes to existing CluttrVAE code |
| **Agent direction randomisation** | Level requires `agent_dir (uint8)` in `[0, 3]`; the existing `_decode_single` randomises direction with `jax.random.randint`; new grid decoder must do the same | LOW | Carry the rng split through vmap using the same pattern as `decode_latent_to_levels()` |

### Differentiators (Competitive Advantage for This Pipeline)

Features that go beyond "working" to "research-grade output that holds up to experimental scrutiny."

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **WandB validity rate logging** | `cmaes/valid_structure_pct` tracks `is_well_formatted()` pass rate per training step, letting us detect if CNN-VAE produces more or fewer invalid Levels than CluttrVAE during CMA-ES search | LOW | Already wired in `maze_plr.py`; the new decode path drops directly into the same `is_valid` check — no new logging code needed |
| **Smoke test: decode z=0 to valid Level** | A single `decode_latent_to_levels_grid(decode_fn, jnp.zeros((1,64)), rng)` call that returns a `Level` passing `is_well_formatted()` — proves the entire decode stack works before a 10-hour run | LOW | Include as a standalone script `scripts/smoke_test_cnn_vae.py`; catches shape bugs, checkpoint key mismatches, and coordinate errors immediately |
| **Short CMA-ES integration test (1000 steps)** | Verifies the full ask → decode → rollout → tell loop with CNN-VAE; confirms no XLA shape errors, no NaN fitness, and `valid_structure_pct > 0.9` | MEDIUM | Reuse `scripts/smoke_test_fixes.sh` pattern; add `--num_updates 1000 --use_cnn_vae` variant |
| **Solvability monitoring via episode reward** | `max_returns > 0` (agent reached goal at least once) reported as `cmaes/solvable_pct` — distinguishes structurally valid Levels from actually navigable ones | LOW | Extend existing `is_valid` metric pattern; `is_solvable = max_returns > 0` is already computed in DRED branch of `maze_plr.py`, just needs surfacing for CMA-ES path |
| **Wall density monitoring** | `mean_num_blocks = wall_map.sum() / num_envs` logged per step; CNN-VAE prior produces ~24% wall density (documented in training); CMA-ES might shift this — worth watching | LOW | Already logged as `mean_num_blocks` in metrics dict; no code change needed |

### Anti-Features (Build These and You Will Regret It)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Sampling from goal/agent logits with temperature** | "Adds diversity" in decoded positions | Adds stochasticity that makes CMA-ES fitness non-deterministic for the same z; CMA-ES already provides diversity via population search; double-sampling blurs the fitness signal | Use argmax (deterministic decode) — CMA-ES handles diversity |
| **Python-loop-based batch decoding** | "Easier to read" | Not jittable; breaks `jax.vmap`; each decode call recompiles; would make 20k-step training ~10x slower | Use `jax.vmap` over a single-example decode function, exactly as done in `decode_latent_to_levels()` |
| **Solvability filtering at decode time** | "Only pass valid mazes to the buffer" | Requires BFS/DFS which is not jittable; would need `jax.pure_callback` or host callback, breaking the JIT'd training loop; solvability is already implicitly selected for by CMA-ES fitness (regret = 0 for unsolvable mazes) | Let CMA-ES fitness signal drive selection; log `solvable_pct` for diagnostics only |
| **Retraining the CNN-VAE** | "Better reconstruction = better levels" | Out of scope; checkpoint is fixed at run10/step200000 with Wall IoU=0.860, prior solvability=96%; retraining costs 200k steps and risks losing the latent structure CMA-ES has already partially learned | Use fixed checkpoint; if solvability is inadequate, adjust CMA-ES fitness, not the VAE |
| **CluttrVAE vs CNN-VAE A/B comparison** | "Scientific rigor" | Not the research question; PROJECT.md explicitly flags this as out of scope; would require another full 20k run, doubling experiment time | Compare CNN-VAE CMA-ES vs vanilla ACCEL baseline (same as Phase 5) |
| **ConvTranspose upsampling in decoder** | "Standard for VAE decoders" | Already rejected at architecture design time due to checkerboard artifacts; `cnn_vae_model.py` uses nearest-neighbor + Conv which is correct for this grid size | Use the existing `CnnLstmDecoder` as-is |
| **Online checkpoint downloading from GCS during training** | "Always uses latest checkpoint" | Adds GCS auth complexity, network latency, and runtime failures if GCS is unavailable; LOCAL_CHECKPOINT_PATH is simpler and reproducible | Pre-download checkpoint to `vae/checkpoints/cnn_vae/` before launch; document the gsutil command |

---

## Feature Dependencies

```
[Orbax checkpoint loading]
    └──requires──> [Decoder-only param extraction]
                       └──requires──> [decode_fn: z -> (wall_logits, goal_logits, agent_logits)]
                                          └──requires──> [Wall sigmoid thresholding]
                                          └──requires──> [Goal/agent wall masking at decode time]
                                                             └──requires──> [Coordinate transform: flat-idx to (x,y)]
                                                             └──requires──> [Collision resolution: goal != agent]
                                                             └──requires──> [Defensive wall clearing]
                                                             └──requires──> [Agent direction randomisation]
                                                                                └──enables──> [JAX vmap compatibility]
                                                                                                  └──enables──> [Smoke test: decode z=0]
                                                                                                  └──enables──> [Short CMA-ES integration test]

[--use_clutr_vae fallback flag] ──conflicts──> [CNN-VAE as default path]
    (must coexist via conditional branch, not replace)

[Solvability monitoring] ──enhances──> [WandB validity rate logging]
    (validity = structural, solvability = navigational — both needed for full diagnostics)
```

### Dependency Notes

- **Orbax checkpoint loading requires decoder-only param extraction:** The full VAE params tree includes `encoder`, `mean_layer`, `logvar_layer`, and `decoder`. Only `decoder` is needed for inference; passing the full tree to `CnnLstmDecoder.apply()` would raise a key mismatch.

- **Wall masking requires knowing the predicted wall map first:** The wall logits must be thresholded to a binary mask before that mask can be used to block goal/agent logits. This creates a two-step ordering within the decode function: (1) wall sigmoid + threshold, (2) mask goal/agent, (3) argmax, (4) coordinate transform.

- **JAX vmap compatibility requires all features to be jittable:** Every feature in the table stakes list must be implemented with `jnp` ops only — no Python-level loops over batch, no `np` calls, no `if` statements on dynamic values. The `apply_wall_mask` function is already jittable and can be directly reused.

- **`--use_clutr_vae` flag conflicts with CNN-VAE default:** These two code paths share the `vae_decode_fn` variable in `maze_plr.py`. The CNN-VAE path sets `vae_decode_fn` to call `CnnLstmDecoder`; the CluttrVAE path sets it to call `CluttrVAE.decode`. They must be exclusive branches, not combined.

---

## MVP Definition

### Launch With (v1) — Minimum to run the 20k comparison experiment

- [x] **Orbax checkpoint load + decoder param extraction** — without this, there is no model
- [x] **`decode_latent_to_levels_grid()`** implementing all table stakes features (wall thresh, wall masking, argmax, coordinate transform, collision resolution, wall clearing, dir randomisation) — without this, the CMA-ES loop crashes
- [x] **JAX vmap over batch** — required for the training loop's `new_levels = decode_latent_to_levels_grid(...)` call to work inside JIT
- [x] **`--use_cnn_vae` flag (or CNN-VAE as default)** — wires the new decode path into `maze_plr.py` while keeping CluttrVAE accessible via `--use_clutr_vae`
- [x] **Smoke test script** — catch bugs before committing GPU time

### Add After Smoke Test Passes (v1.x)

- [ ] **Solvability monitoring (`cmaes/solvable_pct`)** — trigger: smoke test passes, 1000-step integration test passes; add before 20k run so we have the metric from the start
- [ ] **Short integration test (1000 steps)** — trigger: smoke test passes; confirms no shape/NaN issues at scale before 10-hour run

### Future Consideration (v2+)

- [ ] **Diversity metrics in latent space** — PCA / t-SNE visualisation of CMA-ES population trajectory; useful for the write-up but not needed for the 20k run itself
- [ ] **Latent space interpolation for level morphing** — interesting research direction but out of scope for this milestone

---

## Feature Prioritization Matrix

| Feature | Research Value | Implementation Cost | Priority |
|---------|----------------|---------------------|----------|
| `decode_latent_to_levels_grid()` (all table stakes) | HIGH | LOW | P1 |
| Orbax checkpoint loading | HIGH | MEDIUM | P1 |
| `--use_cnn_vae` / `--use_clutr_vae` flag wiring | HIGH | LOW | P1 |
| Smoke test script | HIGH | LOW | P1 |
| Short CMA-ES integration test | HIGH | LOW | P1 |
| Solvability monitoring | MEDIUM | LOW | P2 |
| WandB validity rate logging | MEDIUM | LOW | P2 (already mostly wired) |
| Wall density monitoring | LOW | LOW | P2 (already wired) |
| Latent space PCA visualisation | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for the 20k comparison run to launch
- P2: Should have — add before the run starts so metrics are captured from step 0
- P3: Nice to have — post-run analysis

---

## Competitor Feature Analysis

This is a research pipeline, not a commercial product. "Competitors" here are the prior implementations to match or exceed.

| Feature | CluttrVAE path (existing) | CNN-VAE path (to build) | Gap |
|---------|--------------------------|-------------------------|-----|
| Wall extraction | From token sort + scatter | Sigmoid threshold on logits | Simpler — no sort needed |
| Goal placement | `argmax(tokens[-2])` → 1-based idx | `argmax(masked_goal_logits)` → 0-based flat | Must add +1 offset removal (0-indexed now) |
| Agent placement | `argmax(tokens[-1])` → 1-based idx | `argmax(masked_agent_logits)` → 0-based flat | Same |
| Wall masking | Token-level: `walls = where(wall==goal, 0, walls)` | Logit-level: `apply_wall_mask(wall_map, goal_logits)` | CNN-VAE masking is cleaner — applied pre-argmax |
| Collision resolution | `agent = where(goal==agent, (agent%168)+1, agent)` | Need equivalent via `jnp.where` on flat index | Must implement; not yet in codebase |
| Checkpoint format | Pickle (`.pkl`) | Orbax (directory) | Different loading code required |
| Coordinate convention | 1-based → 0-based conversion | 0-based from argmax natively | CNN-VAE is simpler here |
| JIT compatibility | Full | Full (target) | Must verify |

---

## Sources

- `vae/vae_level_utils.py` — CluttrVAE conversion reference implementation (HIGH confidence, primary source)
- `vae/cnn_vae_losses.py` — `apply_wall_mask()` implementation to reuse (HIGH confidence, primary source)
- `vae/cnn_vae_model.py` — `CnnLstmDecoder` output shape and param tree naming (HIGH confidence, primary source)
- `examples/maze_plr.py` — Full CMA-ES integration pattern, existing CluttrVAE decode path, metric wiring (HIGH confidence, primary source)
- `.planning/PROJECT.md` — Scope, constraints, and out-of-scope items (HIGH confidence, authoritative)

---
*Feature research for: CNN-VAE to Level conversion pipeline*
*Researched: 2026-03-11*
