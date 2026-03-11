# Project Research Summary

**Project:** CNN-VAE Decoder Integration into JAX/CMA-ES Maze Generation Pipeline
**Domain:** Research ML pipeline — neural VAE decoder adapter for JAX/Flax training loop
**Researched:** 2026-03-11
**Confidence:** HIGH

## Executive Summary

This milestone integrates a pre-trained CNN-VAE grid decoder (CNN-LSTM architecture, run10/step200000, Wall IoU=0.860, prior solvability=96%) as the level-generation backend for the CMA-ES evolutionary strategy training loop in `examples/maze_plr.py`. The existing pipeline already uses a CluttrVAE token-based decoder; the CNN-VAE path is architecturally parallel but operates on spatial logit grids `(B, 13, 13)` instead of token sequences. The integration requires three new components: an Orbax checkpoint loader, a `vae/cnn_vae_level_utils.py` adapter module, and minimal additive changes to `maze_plr.py` to dispatch between decoder paths via a `--use_clutr_vae` flag.

The recommended approach is a strict phased build that front-loads verification. The checkpoint must be loaded and the decoder param tree confirmed before any adapter code is written. The single-sample `grid_to_level()` function must be unit-tested in isolation — with `level.is_well_formatted()`, `level.to_str()` visual checks, and a batch-1000 collision test — before it is lifted to batch via `jax.vmap` and wired into the JIT-compiled training loop. This order is dictated by the dependency graph: every downstream step depends on the correctness of coordinate convention and wall masking decisions made in `grid_to_level()`.

The central risk is silent correctness failures: coordinate inversion (row/col swap), missing wall masking, and goal-agent collision all produce Levels that can appear structurally valid while corrupting the RL training signal. A coordinate inversion discovered after a 20k-step run requires a full re-run. The mitigation is a mandatory smoke test suite (`scripts/smoke_test_cnn_vae.py`) that must pass before any GPU-hour commitment. All six critical pitfalls identified are preventable through targeted checks that take under an hour to write.

---

## Key Findings

### Recommended Stack

The existing JAX 0.5.3 / Flax 0.10.7 / evosax / JaxUED stack is unchanged. The three new technical surfaces are: (1) `orbax.checkpoint.PyTreeCheckpointer` (0.10.3) for loading the CNN-VAE checkpoint without requiring a target pytree — use `ocp.CheckpointManager` if the checkpoint directory contains step subdirectories; (2) decoder-only param extraction via `restored["params"]["decoder"]`, which maps directly to `CnnLstmDecoder`'s parameter tree without key remapping; and (3) `jax.vmap` over a pure per-sample `grid_to_level()` function, which is the only JIT-safe pattern for batch decoding. The GCS download (`gsutil cp gs://cnn-vae-maze-checkpoints/run10/200000/`) is a one-time pre-training step; `google-cloud-storage` must be installed into `jax_env` since `gsutil` is not present.

**Core technologies (new surfaces only):**
- `orbax.checkpoint.PyTreeCheckpointer` (0.10.3): Restore CNN-VAE checkpoint to nested dict — no target pytree required; verified working
- `CnnLstmDecoder` param subtree extraction: `restored["params"]["decoder"]` maps directly to module; no key remapping
- `jax.vmap` over `grid_to_level()`: Only JIT-safe batch decoding pattern; Python loops break `jax.jit` and inflate compile time ~100x

**What NOT to use:**
- `pickle.load()` for CNN-VAE checkpoint — it is Orbax format, not pickle
- Python loops over batch in adapter code — breaks JIT
- `ocp.StandardCheckpointer` without a target pytree — fails without matching target

### Expected Features

The feature set is cleanly divided by priority. Everything in the P1 category is a hard dependency for launching the 20k comparison run; P2 features should be wired before the run starts so metrics are captured from step 0; P3 features are post-run analysis.

**Must have (P1 — pipeline breaks without these):**
- `decode_latent_to_levels_grid()` implementing wall sigmoid thresholding, goal/agent wall masking, argmax placement, flat-idx to (x,y) coordinate transform, goal-agent collision resolution, defensive wall clearing, and agent direction randomisation
- Orbax checkpoint load + decoder param extraction
- `jax.vmap` over single-sample decode (JIT compatibility)
- `--use_cnn_vae` default / `--use_clutr_vae` fallback flag wiring in `maze_plr.py`
- Smoke test script (`scripts/smoke_test_cnn_vae.py`) — catches bugs before GPU time commitment

**Should have (P2 — add before 20k run):**
- Solvability monitoring (`cmaes/solvable_pct`) — distinguishes structurally valid from navigable levels
- Short CMA-ES integration test (1000 steps) — verifies no XLA shape errors or NaN fitness
- WandB validity rate logging (`cmaes/valid_structure_pct`) — already wired, just needs correct decode path

**Defer (v2+ / post-run):**
- Latent space PCA / t-SNE visualisation of CMA-ES population trajectory
- Latent space interpolation for level morphing
- CluttrVAE vs CNN-VAE A/B comparison (explicitly out of scope per PROJECT.md)

**Anti-features to reject explicitly:**
- Goal/agent position sampling with temperature — blurs CMA-ES fitness signal; argmax is correct
- Solvability filtering at decode time — requires non-jittable BFS; CMA-ES fitness signal already handles selection
- Retraining the CNN-VAE — out of scope; fixed at run10/step200000

### Architecture Approach

The architecture is a minimal additive adapter that slots into the existing CluttrVAE decode path. One new file (`vae/cnn_vae_level_utils.py`) provides the public API; changes to `maze_plr.py` are localized to the VAE setup block (~10 lines) and a one-line substitution in `on_new_levels()`. The frozen-closure pattern is critical: decoder params are loaded once at startup and captured by a Python lambda, making them compile-time constants inside `jax.jit` without polluting `TrainState`. The `Level` pytree (6 leaves) must be produced with exact dtype matching (`wall_map: bool`, `goal_pos/agent_pos: uint32`, `agent_dir: uint8`) to avoid silent Maze environment errors.

**Major components:**
1. `vae/cnn_vae_level_utils.py` (NEW) — `decode_latent_to_levels_grid()` (batch) and `grid_to_level()` (single-sample, vmapped); the sole new file containing all adapter logic
2. Orbax checkpoint loader (NEW, startup code in `maze_plr.py`) — runs once, extracts `params["decoder"]`, produces frozen dict closed over in `cnn_decode_fn`
3. `examples/maze_plr.py` flag dispatch (MODIFIED, additive only) — `--use_clutr_vae` flag selects decode path at Python level before JIT; `on_new_levels()` one-line change
4. `CnnLstmDecoder` (`vae/cnn_vae_model.py`, UNCHANGED) — existing Flax module; `z (B,64) -> (wall_logits, goal_logits, agent_logits)` each `(B,13,13)`
5. `CluttrVAE` + `decode_latent_to_levels()` (UNCHANGED) — kept fully intact as `--use_clutr_vae` fallback; no modifications permitted

### Critical Pitfalls

All 6 pitfalls are in the adapter implementation and checkpoint loading phases. None are in the training loop itself (which is unchanged). Three are silent correctness failures with HIGH recovery cost if discovered post-run.

1. **Row/col coordinate inversion** — `Level` expects `goal_pos = (col, row)` but grids are indexed `[row, col]`; swap silently produces mirror-reflected positions that pass `is_well_formatted()`; use `x = flat_idx % 13, y = flat_idx // 13`; verify with `level.to_str()` visual check before integration
2. **Wall masking omitted or applied at wrong point** — goal/agent argmax without masking places entities on wall cells, failing `is_well_formatted()`; apply `goal_logits + jnp.where(wall_map, -1e9, 0)` before argmax using predicted (not ground truth) wall mask
3. **Goal equals agent position** — independent argmax operations can produce the same flat index; add collision resolution `agent_flat = jnp.where(goal_flat == agent_flat, (agent_flat + 1) % 169, agent_flat)` before coordinate transform
4. **Orbax param key mismatch** — full checkpoint tree is `params/{encoder, mean_layer, logvar_layer, decoder}`; passing wrong subtree to `CnnLstmDecoder.apply()` raises KeyError or silently uses garbage params; print tree structure after restore before writing any model code
5. **JIT incompatibility from Python control flow** — Python loops, `np.where`, or `if array_value:` inside the adapter breaks `jax.jit`; use `jax.vmap` over a pure function and `jnp.where` for all conditionals; mandatory JIT smoke test before training loop integration

---

## Implications for Roadmap

Based on the dependency graph in ARCHITECTURE.md and the pitfall-to-phase mapping in PITFALLS.md, the build must follow strict sequential phases. Parallelization is not possible: each phase is a hard prerequisite for the next.

### Phase 1: Checkpoint Acquisition and Verification

**Rationale:** Every subsequent phase depends on having valid `decoder_params`. This phase has zero code changes and can be completed in under an hour. Resolving Orbax API variant (PyTreeCheckpointer vs CheckpointManager) and confirming the param tree structure here prevents wasted implementation effort on wrong assumptions.

**Delivers:** Local CNN-VAE checkpoint at `vae/checkpoints/cnn_vae/`; confirmed `decoder_params` dict matching documented tree structure; wall IoU spot-check against 0.860 reported metric.

**Addresses:** Orbax checkpoint loading (P1 feature); decoder-only param extraction (P1 feature)

**Avoids:** Pitfall 4 (Orbax param key mismatch), Pitfall 6 (Orbax API version mismatch)

**Research flag:** No additional research needed — STACK.md has verified API patterns for both checkpoint formats.

### Phase 2: grid_to_level() Adapter — Single Sample

**Rationale:** This is the critical-path component. The coordinate convention and wall masking logic here determines whether all generated levels are valid. Implementing and exhaustively verifying this in isolation (not inside the training loop) makes bugs cheap to catch and fix. The 1000-sample collision test and `level.to_str()` visual check must be passing before moving on.

**Delivers:** `vae/cnn_vae_level_utils.py` containing `grid_to_level()` and `decode_latent_to_levels_grid()`; unit test confirming `is_well_formatted()` on z=zeros and 100 random z; batch-1000 collision test (goal != agent); JIT smoke test passing.

**Addresses:** All table-stakes features in FEATURES.md (wall thresholding, wall masking, argmax, coordinate transform, collision resolution, wall clearing, direction randomisation, vmap compatibility)

**Avoids:** Pitfall 1 (coordinate inversion), Pitfall 3 (wall masking at wrong point), Pitfall 4 (goal-agent collision), Pitfall 5 (JIT incompatibility)

**Research flag:** No additional research needed — STACK.md and ARCHITECTURE.md provide complete verified implementation patterns.

### Phase 3: maze_plr.py Integration and Smoke Test

**Rationale:** With a verified adapter module, `maze_plr.py` changes are additive and low-risk. The `--use_clutr_vae` flag, frozen-closure wiring, and one-line `on_new_levels()` substitution are mechanical. The 1000-step CMA-ES integration test here catches any remaining shape/NaN issues before GPU-hour commitment.

**Delivers:** `maze_plr.py` with CNN-VAE as default decode path and `--use_clutr_vae` fallback; `scripts/smoke_test_cnn_vae.py`; 1000-step integration test passing with `cmaes/valid_structure_pct > 90%` and no NaN fitness.

**Addresses:** `--use_cnn_vae` flag wiring (P1 feature), smoke test (P1 feature), short CMA-ES integration test (P2 feature), WandB validity logging (P2 feature), solvability monitoring (P2 feature)

**Avoids:** Pitfall 2 (Orbax param key mismatch propagating to training), Pitfall 5 (JIT incompatibility discovered during first real run)

**Research flag:** No additional research needed — ARCHITECTURE.md documents exact lines to modify in `maze_plr.py` and the frozen-closure pattern.

### Phase 4: 20k Comparison Run

**Rationale:** Only when Phase 3 is fully verified (smoke test passing, `valid_structure_pct > 90%`, no NaN) should the 20k run be launched. This phase is identical to the existing Phase 5 run structure but substitutes the CNN-VAE decode path.

**Delivers:** WandB logs for CNN-VAE CMA-ES vs ACCEL baseline comparison; same metrics as Phase 5 (solvers rate, max_returns, level diversity).

**Addresses:** Research question (CNN-VAE CMA-ES vs vanilla ACCEL); forms write-up experimental data

**Avoids:** Coordinate inversion post-run (HIGH recovery cost per PITFALLS.md)

**Research flag:** No additional research needed — run structure mirrors existing Phase 5 scripts.

### Phase Ordering Rationale

- Phase 1 before Phase 2: Impossible to implement `grid_to_level()` without knowing the actual param tree structure and confirmed decoder output shapes.
- Phase 2 before Phase 3: Wiring an unverified adapter into a JIT-compiled training loop makes bugs exponentially harder to diagnose (XLA compile errors, silent bad levels in replay buffer).
- Phase 3 before Phase 4: A 10-hour GPU run on a buggy decode path wastes the entire run and potentially produces unrecoverable results (coordinate inversion requires full re-run per PITFALLS.md recovery table).
- No parallelization possible: the dependency chain is strictly linear.

### Research Flags

All phases are well-documented; no phase requires `/gsd:research-phase` during planning:

- **Phase 1:** Standard Orbax API patterns — verified and documented in STACK.md with both checkpoint format variants covered.
- **Phase 2:** Coordinate convention and wall masking patterns — verified against existing `tokens_to_level()` implementation; all JAX vmap patterns documented in STACK.md and ARCHITECTURE.md.
- **Phase 3:** Integration pattern — identical to existing CluttrVAE wiring; exact line references provided in ARCHITECTURE.md.
- **Phase 4:** Standard run — same script structure as Phase 5.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All API patterns verified by running code in the actual `jax_env` conda environment (JAX 0.5.3, Flax 0.10.7, Orbax 0.10.3); no theoretical research |
| Features | HIGH | Derived directly from existing codebase files (`vae_level_utils.py`, `cnn_vae_losses.py`, `cnn_vae_model.py`, `maze_plr.py`); primary source, not inferred |
| Architecture | HIGH | Derived from direct codebase inspection with line-level references; existing `tokens_to_level()` and CluttrVAE patterns verified as correct templates |
| Pitfalls | HIGH | All pitfalls derived from direct code analysis of the actual codebase; no generic advice; recovery costs validated against run history |

**Overall confidence:** HIGH

### Gaps to Address

- **Checkpoint directory structure unknown until downloaded:** STACK.md covers both PyTreeCheckpointer (flat directory) and CheckpointManager (step subdirectory) variants, but the correct one cannot be confirmed until `vae/checkpoints/cnn_vae/` is inspected. Phase 1 resolves this. Fallback is pickle extraction if Orbax loading fails.
- **GCS authentication:** `gsutil` is not installed; `google-cloud-storage` Python SDK must be installed into `jax_env`. If GCS auth is not pre-configured, the checkpoint must be downloaded manually or via an alternative path. This is a one-time blocker, not a runtime concern.
- **Actual decoder output quality on random z:** The 96% solvability figure is from the VAE training distribution. CMA-ES will search across the full latent space including out-of-distribution regions where wall masking may be more frequently needed. The `valid_structure_pct` metric in Phase 3 will confirm whether the adapter handles edge cases correctly.

---

## Sources

### Primary (HIGH confidence — direct codebase inspection)
- `vae/cnn_vae_model.py` — CnnLstmDecoder architecture, output shapes, param tree naming contract
- `vae/vae_level_utils.py` — `tokens_to_level()` reference implementation for coordinate convention
- `vae/cnn_vae_losses.py` — `apply_wall_mask()` implementation to reuse at inference
- `examples/maze_plr.py` — full CMA-ES integration pattern, CluttrVAE decode path, frozen-closure pattern, JIT context
- `src/jaxued/environments/maze/level.py` — `Level` dataclass; `is_well_formatted()` checks; `from_str()` confirms `(x, y)` convention
- `vae/cmaes_manager.py` — latent-dim agnostic; no changes needed
- `.planning/PROJECT.md` — scope, constraints, out-of-scope items, coordinate convention documentation

### Primary (HIGH confidence — verified by execution)
- Orbax 0.10.3 API patterns — verified by running `ocp.PyTreeCheckpointer().restore()` and `ocp.CheckpointManager().restore()` in `/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env`
- `jax.vmap` over `Level` construction — confirmed; Level has 6 pytree leaves, `width`/`height` are static ints and must not be batched
- `CnnLstmDecoder` param tree structure — confirmed from `decoder.init()` output
- Wall masking via logit subtraction (`- 1e9`) — verified JIT-safe via `jax.jit(jax.vmap(...))` test

---
*Research completed: 2026-03-11*
*Ready for roadmap: yes*
