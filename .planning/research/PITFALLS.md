# Pitfalls Research

**Domain:** CNN-VAE grid decoder integration into JAX/CMA-ES maze generation pipeline
**Researched:** 2026-03-11
**Confidence:** HIGH — all pitfalls derived from direct code analysis of the actual codebase, not generic advice

---

## Critical Pitfalls

### Pitfall 1: Row/Col vs X/Y Coordinate Inversion

**What goes wrong:**
The CNN-VAE decoder outputs `(B, 13, 13)` grids in NumPy/JAX row-major order: `grid[row, col]`. The `Level` dataclass uses `(x, y)` convention where `x=col, y=row`. When converting from argmax grid index to a Level position, swapping these produces goal/agent placed at the mirror-reflected position. Mazes look structurally valid but goal and agent are in wrong cells — the Level passes `is_well_formatted()` silently because the swapped position is still in-bounds and not on a wall (probability).

**Why it happens:**
The existing `tokens_to_level()` in `vae_level_utils.py` contains the correct pattern and is documented explicitly:
```python
agent_pos = jnp.array([agent_0 % GRID_SIZE, agent_0 // GRID_SIZE], dtype=jnp.uint32)
#                       ^--- x = col           ^--- y = row
```
The new `decode_latent_to_levels_grid()` must replicate this. The natural reading of a 2D array makes `grid[i, j]` feel like `pos = (i, j)`, but `Level.goal_pos` is `(col, row)` = `(j, i)`. The `Level.from_str()` and `Level.to_str()` methods confirm this: they iterate `for y, row in enumerate(rows): for x, c in enumerate(row)` and write `enc[y, x]`.

**How to avoid:**
In `decode_latent_to_levels_grid()`, when computing position from argmax:
```python
flat_idx = jnp.argmax(goal_logits.reshape(-1))   # 0-indexed, row-major
col = flat_idx % 13   # x
row = flat_idx // 13  # y
goal_pos = jnp.array([col, row], dtype=jnp.uint32)
```
Add an explicit correctness test: decode `z=zeros`, render with `Level.to_str()`, visually verify that `G` appears at the expected column/row.

**Warning signs:**
- Solve rate drops to near-zero despite wall IoU being correct
- Goal appears in visually wrong cell when rendering Level with `to_str()`
- `level.goal_pos` reads as `(row, col)` rather than `(col, row)` under inspection
- Agent solves Level when evaluated backwards (symmetric mazes mask the bug)

**Phase to address:** Adapter implementation phase — write correctness test before integration into training loop.

---

### Pitfall 2: Orbax Checkpoint Param Key Mismatch

**What goes wrong:**
The CNN-VAE checkpoint was saved with a specific param tree structure documented in `cnn_vae_model.py`:
- `params/encoder/...` — CnnEncoder
- `params/mean_layer/...` — Dense at top level of CnnLstmVAE
- `params/logvar_layer/...` — Dense at top level of CnnLstmVAE
- `params/decoder/...` — CnnLstmDecoder

Loading the checkpoint but applying it to a model where these sub-modules have different names (e.g., `decoder` renamed or `mean_layer` nested inside encoder) causes a silent key mismatch. Flax's `model.apply()` may either crash with a key error, or silently use wrong parameters if partial application is used.

**Why it happens:**
The existing CluttrVAE uses a completely different param tree (`embed`, `enc_bilstm`, `dec_bilstm1`, etc.) and is loaded via `pickle.load()`. The new CNN-VAE checkpoint is Orbax format — a different loading path entirely. There is no existing Orbax VAE loading code in `maze_plr.py` to copy from (the agent checkpoint saving uses Orbax, but VAE loading uses pickle). This means the CNN-VAE loading path must be written from scratch.

**How to avoid:**
1. After loading the Orbax checkpoint, print the top-level param keys before passing to `model.apply()`:
   ```python
   ckpt = ocp.PyTreeCheckpointer().restore(ckpt_path)
   print(jax.tree_util.tree_map(lambda x: x.shape, ckpt['params']))
   ```
2. Instantiate `CnnLstmDecoder` with the exact `name='decoder'` argument — matches the saved key `params/decoder/...`.
3. Only load decoder params (not full VAE params) since only the decoder is needed at inference:
   ```python
   decoder_params = ckpt['params']['decoder']
   decoder.apply({'params': decoder_params}, z)
   ```

**Warning signs:**
- `KeyError` on `params/decoder` during apply
- Output is garbage (all zeros or NaN) despite no error — wrong params silently applied
- Model loss/accuracy far worse than reported 96% solvability at step 200000

**Phase to address:** Checkpoint loading phase — verify by decoding `z=zeros` and checking wall IoU against known reference.

---

### Pitfall 3: Wall Masking Applied at Wrong Point (Logits vs Probabilities)

**What goes wrong:**
The CNN-VAE decoder outputs three independent logit heads: `wall_logits`, `goal_logits`, `agent_logits` — each `(B, 13, 13)`. During training, `apply_wall_mask()` in `cnn_vae_losses.py` masks goal/agent logits using ground truth wall masks. At inference (decoding latents to Levels), the wall mask must be derived from the decoded `wall_logits`, not from ground truth. If wall masking is skipped entirely, goal/agent may be placed on wall cells — the Level fails `is_well_formatted()` and the RL environment crashes or silently ignores it.

**Why it happens:**
The training loss code uses ground truth walls for masking to prevent self-referential masking. It is easy to carry this pattern to inference and forget to switch to predicted walls, or to simply omit wall masking at inference thinking argmax will naturally avoid walls (it won't — the model learned to rely on masking).

**How to avoid:**
In `decode_latent_to_levels_grid()`:
```python
wall_mask = (jax.nn.sigmoid(wall_logits) > 0.5).astype(jnp.float32)  # (B, 13, 13)
# mask goal/agent logits before argmax
goal_logits_masked = jnp.where(wall_mask, -1e9, goal_logits)
agent_logits_masked = jnp.where(wall_mask, -1e9, agent_logits)
goal_flat = jnp.argmax(goal_logits_masked.reshape(B, -1), axis=-1)    # (B,)
agent_flat = jnp.argmax(agent_logits_masked.reshape(B, -1), axis=-1)  # (B,)
```
Defensively clear wall at goal/agent positions (matching the pattern in `tokens_to_level()`):
```python
wall_map = wall_map.at[goal_pos[1], goal_pos[0]].set(False)
wall_map = wall_map.at[agent_pos[1], agent_pos[0]].set(False)
```

**Warning signs:**
- `is_well_formatted()` returns False on decoded levels (agent/goal on wall)
- Nonzero frequency of levels where `wall_map[goal_pos[1], goal_pos[0]]` is True
- Solvability below the 96% reported for the trained checkpoint

**Phase to address:** Adapter implementation phase — include in smoke test: decode z=zeros, call `level.is_well_formatted()` and assert True.

---

### Pitfall 4: Goal Equals Agent Position (Missing Collision Handling)

**What goes wrong:**
The wall-masked argmax for goal and agent logits are independent — they can produce the same flat index. If `goal_pos == agent_pos`, the Level fails `is_well_formatted()`. In the token-based pipeline, `repair_tokens()` explicitly handles this by shifting agent by 1. The grid-based adapter has no equivalent repair unless explicitly added.

**Why it happens:**
This is easier to overlook in a grid decoder because goal and agent are separate spatial distributions, so collision feels unlikely. But at the tail of the CMA-ES distribution (latent vectors near the boundary), the decoder may concentrate both distributions onto the same high-probability free cell.

**How to avoid:**
After computing `goal_flat` and `agent_flat`, add collision resolution before converting to (x, y):
```python
# If goal == agent, shift agent to next free cell (wrap)
collision = (goal_flat == agent_flat)
next_free = (agent_flat + 1) % (13 * 13)
# ensure next_free is not a wall either (simple: skip wall cells)
agent_flat = jnp.where(collision, next_free, agent_flat)
```
Or accept the defensive `wall_map.at[...].set(False)` pattern and add a goal/agent distinct check before calling `Level(...)`.

**Warning signs:**
- Occasional `is_well_formatted()` failures in batches of decoded levels
- Error rate correlated with CMA-ES convergence (narrower distribution = more collisions)
- Agent_pos equals goal_pos in any decoded Level

**Phase to address:** Adapter implementation phase — add to smoke test and validate with large batch (N=1000).

---

### Pitfall 5: JIT Incompatibility from Python Control Flow in Adapter

**What goes wrong:**
`decode_latent_to_levels_grid()` must run inside `jax.jit` (the entire `on_new_levels` function in `maze_plr.py` is JIT-compiled via `jax.lax.scan`). Any Python-level control flow that depends on array values — loops over batch, Python `if` on array shapes, `np.where` instead of `jnp.where` — will either trace incorrectly or raise a ConcretizationTypeError at JIT compile time.

**Why it happens:**
The existing `decode_latent_to_levels()` uses `jax.vmap` correctly. The new grid adapter must follow the same pattern. The temptation is to write a quick prototype using Python loops or NumPy for debugging, then forget to convert before integration.

The `repair_tokens()` function shows the correct approach: pure JAX, no Python loops, all conditions via `jnp.where`. The grid adapter must do the same.

**How to avoid:**
- Use `jax.vmap` over a single-example `_decode_single_grid()` function (matching the existing `_decode_single()` pattern in `vae_level_utils.py`).
- Use `jnp.where` not `np.where`, `jnp.argmax` not `np.argmax`.
- No `if array_value:` in the adapter — use `jnp.where` for all conditionals.
- Verify JIT compatibility before integration: `jax.jit(decode_latent_to_levels_grid)(decoder_fn, z_batch, rng)` must compile without error.

**Warning signs:**
- `ConcretizationTypeError` at JIT compile time
- `TracerBoolConversionError` from boolean Python `if` on JAX array
- Function works in eager mode but breaks inside `jax.lax.scan`

**Phase to address:** Adapter implementation phase — JIT smoke test is mandatory before plugging into training loop.

---

### Pitfall 6: Orbax Checkpoint Loading API Breaking Changes

**What goes wrong:**
Orbax has undergone significant API changes across versions. The existing training code uses `ocp.CheckpointManager` for saving/restoring agent TrainState. Loading a pre-trained VAE checkpoint (from an external training run on GCS) may use a different Orbax API — `ocp.PyTreeCheckpointer`, `ocp.StandardCheckpointer`, or raw `ocp.Checkpointer` — and the correct one depends on how the checkpoint was saved during CNN-VAE training.

**Why it happens:**
The CNN-VAE was trained in a separate repo (`cnn-vae-maze`). Its checkpoint format depends on which Orbax API was used there. The `maze_plr.py` uses `ocp.StandardCheckpointHandler` when restoring agent checkpoints. These APIs are not interchangeable — `StandardCheckpointer` saves metadata that `PyTreeCheckpointer` doesn't understand, and vice versa.

**How to avoid:**
1. Before writing loading code, inspect the checkpoint directory structure:
   ```bash
   ls vae/checkpoints/cnn_vae/
   # Orbax checkpoints contain: checkpoint, manifest.ocdbt, or orbax-checkpoint-tmp-*
   ```
2. Check if checkpoint has `_METADATA` file (StandardCheckpointer) or raw msgpack (PyTreeCheckpointer).
3. Try loading with `ocp.PyTreeCheckpointer().restore(path)` first; if that fails, try `ocp.StandardCheckpointer().restore(path)`.
4. If the CNN-VAE training used a different Orbax version, consider extracting params via the matching version, then pickling them as a fallback.

**Warning signs:**
- `FileNotFoundError` on `_METADATA`
- `KeyError` when accessing `ckpt['params']` — checkpoint may be flat (params at top level)
- `AttributeError` on `ocp.PyTreeCheckpointer` (API removed in newer Orbax)

**Phase to address:** Checkpoint loading phase — must resolve before any other work begins, since all subsequent steps depend on having valid decoder params.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Copy-pasting `decode_latent_to_levels()` and patching it for grids | Fast first draft | Two divergent decode paths that drift over time; bugs fixed in one not the other | Never — write a clean separate function from the start |
| Using `np.argmax` instead of `jnp.argmax` | Easier debugging with concrete values | Breaks inside JIT; must rewrite before training | Only in throwaway debug scripts, never in adapter code |
| Skipping wall masking at inference and relying on `is_well_formatted()` filter | Simpler code | Silently produces bad levels that get filtered, reducing effective batch size; CMA-ES fitness signal corrupted | Never — mask before argmax |
| Using pickle for CNN-VAE params instead of Orbax | Bypasses Orbax API complexity | One-time manual conversion step, but stable and tested | Acceptable if Orbax loading proves unreliable |
| Hardcoding `latent_dim=64` in adapter | Avoids passing config around | Breaks if CNN-VAE latent dim changes | Acceptable for this specific milestone since dim is fixed |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Orbax checkpoint restore | Calling `restore(step)` when CNN-VAE checkpoint uses step-less path | Check if checkpoint was saved with step index or as a flat directory; use `restore()` vs `restore(path)` accordingly |
| `jax.vmap` over `CnnLstmDecoder.apply` | Passing batched params when params are shared | Params must NOT be vmapped — only the input `z` is batched; use `vmap(lambda z: decoder.apply(params, z))` |
| `nn.scan` LSTM in decoder | Calling with wrong carry shape | `initialize_carry` requires `(batch_size, input_dim)` not just `(batch_size,)` — this is documented in `cnn_vae_model.py` and must be respected in any standalone decode call |
| GCS checkpoint download | Using `gsutil` which may not be authenticated | Use `gcloud storage cp` or `google.cloud.storage` Python SDK; test auth before starting work |
| CluttrVAE fallback path | Touching `decode_latent_to_levels` signature while adding CNN-VAE path | CNN-VAE and CluttrVAE must use separate decode functions; do not modify the existing `decode_latent_to_levels()` signature |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Decoding full (B, 13, 13) grid inside inner training loop without JIT | Decode takes ~100ms per step instead of <1ms | Ensure `decode_latent_to_levels_grid` is called from inside the JIT-compiled `on_new_levels` closure | From the first training step |
| `jax.image.resize` inside vmap | XLA compilation time explosion if batch size not static | Pass static batch dimension or use `jax.lax.dynamic_slice` for dynamic shapes | At first vmap call with non-static B |
| Loading CNN-VAE decoder params on every training step | params re-loaded from disk each call | Load params once before training loop, close over in decode function | Immediately |
| Not reusing the `vae_decode_fn` closure pattern | decode function re-traces on every call | Follow existing `vae_decode_fn = lambda z: vae.apply(...)` closure pattern from `maze_plr.py` | On second call to decoder |

---

## "Looks Done But Isn't" Checklist

- [ ] **Coordinate convention:** Decoded Level renders correctly with `level.to_str()` — G and agent appear at visually correct positions, not transposed
- [ ] **Wall masking:** `level.is_well_formatted()` returns True for all decoded levels from z=zeros and from 100 random z samples
- [ ] **Checkpoint loaded correctly:** Wall IoU on a known test input matches the 0.860 reported metric; solvability near 96%
- [ ] **JIT compatibility:** `jax.jit(decode_latent_to_levels_grid)(...)` compiles and runs without ConcretizationTypeError
- [ ] **CluttrVAE fallback:** `maze_plr.py --use_clutr_vae` still works after CNN-VAE integration — no import errors, no broken config paths
- [ ] **CMA-ES round-trip:** Full ask-decode-rollout-tell cycle runs for 10 steps without shape errors or NaN fitness
- [ ] **WandB logging:** `cmaes/valid_structure_pct` metric appears and is near 96%, not 0% or 100% (both indicate bugs)
- [ ] **Goal != Agent:** No levels in a batch of 1000 have `goal_pos == agent_pos`

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Coordinate inversion discovered after 20k run | HIGH | Re-run the full experiment; cannot fix results retroactively |
| Orbax loading failure | LOW | Fall back to pickle: extract params via matching Python env, save as `.pkl`, use existing CluttrVAE pickle loading pattern |
| JIT incompatibility | MEDIUM | Rewrite adapter as pure JAX (1-2 hours); no data loss |
| Wall masking omitted | MEDIUM | Add masking, re-run smoke test, re-start training from scratch |
| Param key mismatch | LOW | Remap keys manually: `{'params': ckpt_params['decoder']}` and re-verify |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Row/col coordinate inversion | Adapter implementation | `level.to_str()` visual check + `is_well_formatted()` on decoded levels |
| Orbax param key mismatch | Checkpoint loading | Print param tree shapes; compare to documented contract in `cnn_vae_model.py` |
| Wall masking at wrong point | Adapter implementation | `is_well_formatted()` on batch of 100 decoded levels; count failures = 0 |
| Goal equals agent collision | Adapter implementation | Decode 1000 random z; assert all have distinct goal/agent positions |
| JIT incompatibility | Adapter implementation | `jax.jit(decode_fn)(...)` must compile before integration |
| Orbax API version mismatch | Checkpoint loading | Inspect checkpoint directory structure before writing loading code |

---

## Sources

- Direct code analysis: `vae/cnn_vae_model.py` — checkpoint key naming contract (comments)
- Direct code analysis: `vae/vae_level_utils.py` — correct `(x=col, y=row)` convention in `tokens_to_level()`
- Direct code analysis: `src/jaxued/environments/maze/level.py` — `is_well_formatted()` checks; `from_str()` confirms `(x, y)` convention
- Direct code analysis: `vae/cnn_vae_losses.py` — wall masking pattern (ground truth vs predicted at inference)
- Direct code analysis: `examples/maze_plr.py` — existing CluttrVAE loading via pickle; CMA-ES integration pattern; JIT context
- Direct code analysis: `vae/cmaes_manager.py` — latent-dim agnostic, no changes needed
- Project context: `.planning/PROJECT.md` — coordinate convention documented explicitly; Orbax format stated for CNN-VAE checkpoint

---
*Pitfalls research for: CNN-VAE grid decoder integration into CMA-ES/JAX maze pipeline*
*Researched: 2026-03-11*
