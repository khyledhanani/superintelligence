# Codebase Concerns

**Analysis Date:** 2026-03-23

## Tech Debt

**CMA-ES evosax API compatibility:**
- Issue: Code supports both old and new evosax API with runtime detection. The import fallback and `_NEW_API` flag branch logic in `vae/cmaes_manager.py` creates dual-maintenance burden.
- Files: `vae/cmaes_manager.py` (lines 16-21), `examples/maze_plr.py` (lines 504-510)
- Impact: Unclear which API version is canonical; if evosax API changes again, both branches must be updated. No error if incompatible version is installed.
- Fix approach: Pin evosax version in requirements, remove legacy API branch, or document minimum version requirement explicitly.

**Loose exception handling with generic Exception:**
- Issue: Code uses bare `Exception` instead of specific error types, making it hard to distinguish recoverable from fatal errors.
- Files: `src/jaxued/level_sampler.py:65`, `src/jaxued/level_sampler.py:323`
- Impact: Callers cannot distinguish between prioritization validation failure and other unexpected errors.
- Fix approach: Replace with `ValueError` for validation failures, document expected exception types at class/function level.

**Hardcoded sys.path manipulations:**
- Issue: Multiple scripts manually insert parent directories into sys.path to resolve imports. This breaks if directory structure changes and makes code non-relocatable.
- Files: `examples/maze_plr.py:30-33`, `llm/test_generator.py:35`, `vae/compare_accel_vs_cmaes.py:18`
- Impact: Scripts cannot be moved or run from different working directories. Reduces reproducibility when installing as package.
- Fix approach: Use proper package structure with `__init__.py` or install as editable package (`pip install -e .`). Avoid `sys.path.insert()`.

**Broad exception catching masking errors:**
- Issue: Code catches `(ImportError, Exception)` which will swallow unexpected errors including logic bugs.
- Files: `examples/maze_plr.py:353`, `examples/maze_plr.py:384`
- Impact: GCS upload failures silently fall back to gcloud CLI; logic bugs in google-cloud-storage will be masked as "client failed".
- Fix approach: Catch only `ImportError` for missing optional dependency; let other exceptions propagate or wrap in custom exception.

**Unsafe checkpoint loading:**
- Issue: VAE checkpoint loading uses `pickle.load()` without validation, trusting arbitrary binary files.
- Files: `examples/maze_plr.py:495-496`
- Impact: Malicious checkpoint files could execute arbitrary code. No validation of checkpoint format/schema.
- Fix approach: Use `orbax.checkpoint` for safe loading, or validate checkpoint format before unpickling.

## Known Bugs

**CMA-ES state initialization shape mismatch potential:**
- Symptoms: evosax initialization may fail silently if latent_dim inference is wrong
- Files: `vae/cmaes_manager.py:43-44`, `examples/maze_plr.py:598-601`
- Trigger: When using new evosax API, if `CMA_ES` constructor doesn't properly set `num_dims` based on the dummy solution shape
- Workaround: Assertion on line 598-601 catches shape mismatch but error message could be clearer

**Hardcoded maze dimensions in VAE utilities:**
- Symptoms: VAE assumes 13x13 grids and 52-token sequences; will fail silently with different grid sizes
- Files: `vae/vae_level_utils.py:17-20` (GRID_SIZE=13, SEQ_LEN=52, MAX_WALLS=50)
- Trigger: If code tries to decode levels from VAE trained on different grid size
- Impact: Produces corrupted levels without clear error message

**Missing agent/goal position validation in level repair:**
- Symptoms: If both agent and goal land on same position after repair, the offset logic wraps incorrectly
- Files: `vae/vae_level_utils.py:31-34`
- Trigger: Rare edge case when `agent == goal` and shifting by 1 lands on goal again
- Workaround: Very unlikely in practice (1/170 base probability, lower after clipping)

**Incomplete Craftax environment support:**
- Symptoms: ValueError raised if pixel-based Craftax environments are requested
- Files: `examples/craftax/craftax_plr.py:544-545`
- Trigger: Attempting to train on `Craftax-*-Pixels-v1` environments
- Impact: Feature gap; symbolic environments only, blocks pixel-based curriculum

## Security Considerations

**Unsafe pickle loading from checkpoint files:**
- Risk: Arbitrary code execution from untrusted checkpoint files
- Files: `examples/maze_plr.py:495`
- Current mitigation: None
- Recommendations: Use `orbax.checkpoint` or implement schema validation before unpickling

**GCS credentials may be exposed in command-line fallback:**
- Risk: `gcloud` command fallback may expose authentication in process logs if subprocess fails
- Files: `examples/maze_plr.py:357`, `examples/maze_plr.py:389`
- Current mitigation: Error messages are truncated (first 200 chars)
- Recommendations: Use google-cloud-storage library exclusively; wrap gcloud in shell with `set +x` to hide env vars

**Unvalidated config from YAML/checkpoint sources:**
- Risk: Config files loaded with `yaml.safe_load()` but checkpoint dict validation is missing
- Files: `examples/maze_plr.py:483-492`
- Current mitigation: Assertions check schema exists but not format
- Recommendations: Use Pydantic or TypedDict to validate all loaded configs before use

## Performance Bottlenecks

**PCA buffer latent analysis batched with fixed chunk size:**
- Problem: Hard-coded batch size of 512 when encoding buffer snapshots through VAE
- Files: `examples/maze_plr.py:1215`
- Cause: Prevent OOM on GPU but may not be optimal for all memory configurations
- Improvement path: Make batch size configurable based on available VRAM; auto-tune based on first mini-batch

**While-loop in MazeSolved shortest-path precomputation:**
- Problem: `jax.lax.while_loop` used for Bellman value iteration; convergence check is element-wise comparison, no early termination
- Files: `src/jaxued/environments/maze/env_solved.py:82`
- Cause: May iterate many times even after convergence on most cells
- Improvement path: Track max-norm of delta and terminate when below threshold, or use fixed iteration count

**Level sampler weight computation happens every sample:**
- Problem: `level_weights()` recomputed on every call to `sample_replay_level`, even in scan loops
- Files: `src/jaxued/level_sampler.py:121`
- Cause: Not JIT-friendly; weights are vmap'd and sorted each call
- Improvement path: Cache weights in sampler state, update only when scores/timestamps change

**Buffer dump encoding to VAE latent space not parallelized:**
- Problem: Sequential for-loop encodes batches through VAE one at a time in post-training
- Files: `examples/maze_plr.py:1215-1218`
- Cause: Prevents JAX batching/pipelining; CPU-bound wait
- Improvement path: Collect all batches, vmap encode in single JAX call

## Fragile Areas

**CMA-ES integration with PLR cycle:**
- Files: `examples/maze_plr.py:664-706`, `vae/cmaes_manager.py`
- Why fragile: CMA-ES state is stored in `train_state.es_state` as arbitrary pytree; periodic resets replace entire state without smoothing. If evosax API changes, initialization chain breaks.
- Safe modification: Test that `es_state` shape matches expected dimensions before and after reset; use explicit state shape assertions
- Test coverage: No unit tests for CMA-ES state transitions; only integration tests in main training loop

**VAE checkpoint compatibility:**
- Files: `examples/maze_plr.py:495-497`
- Why fragile: Checkpoint loading assumes specific pickle format and dict structure; no schema validation
- Safe modification: Always test VAE decoding on a small batch before training; add checkpoint version field
- Test coverage: No checkpoint validation tests

**Maze level repair logic:**
- Files: `vae/vae_level_utils.py:23-43`
- Why fragile: Multiple interdependent clipping and repair operations (wall removal, agent/goal conflict resolution, sorting). Changes to one step can silently corrupt others.
- Safe modification: Add property checks after repair (assert no walls at agent/goal, assert agent != goal, assert walls sorted)
- Test coverage: `tests/test_examples_kinda.py` does minimal validation

**Multi-branch training loop with state tracking:**
- Files: `examples/maze_plr.py:900-916`
- Why fragile: `jax.lax.switch` selects between `on_new`, `on_replay`, `on_mutate` based on `train_state.update_state` enum. Branch logic relies on implicit assumptions about which branch can follow which.
- Safe modification: Document state machine explicitly; add assertions at branch entry points validating preconditions
- Test coverage: Only exercise main training flow; no edge case tests for state transitions

## Scaling Limits

**Level buffer capacity is fixed at initialization:**
- Current capacity: 4000 (default via `--level_buffer_capacity`)
- Limit: Entire buffer is pre-allocated as JAX arrays; scales as O(4000 * level_size) memory
- Scaling path: Switch to dynamic allocation with append/insert semantics, or implement ring buffer with configurable max age

**VAE latent space assumes 64 dimensions:**
- Current capacity: Hard-coded in VAE model and CMA-ES initialization
- Limit: Changing latent_dim requires retraining VAE and updating cmaes config
- Scaling path: Accept `latent_dim` as constructor parameter throughout; test multi-scale VAEs

**Evaluation grid is 13x13 cells:**
- Current capacity: 169 cells max, supports ~50-wall mazes
- Limit: Cannot scale to larger or variable-size levels without retraining VAE
- Scaling path: Implement spatial encoding (e.g., 2D convolution VAE) and variable-length token sequences

**Training environment count is coupled to CMA-ES popsize:**
- Current capacity: `num_train_envs = popsize = 32`
- Limit: Cannot increase parallelism without proportional increase in CMA-ES population (and computational cost)
- Scaling path: Decouple with CMA-ES subsampling or distributed evaluation

## Dependencies at Risk

**evosax library API stability:**
- Risk: Already required compatibility shim for API migration; no clear semantic versioning
- Impact: Future evosax updates may break CMA-ES initialization again
- Migration plan: Pin to specific evosax version; maintain legacy branch or vendor the algorithm

**orbax.checkpoint may replace current checkpoint format:**
- Risk: Currently using Python pickle for VAE checkpoints, which is fragile
- Impact: If VAE format changes, old checkpoints become incompatible
- Migration plan: Implement orbax-based checkpoint saving; script to migrate old pickles

**JAX version pinning (0.5.3):**
- Risk: Code explicitly avoids setting XLA_FLAGS (relies on JAX 0.5.3 auto-discovery)
- Impact: Upgrading JAX may require investigating libdevice issues again
- Migration plan: Test with newer JAX LTS; document any XLA_FLAGS needed

## Missing Critical Features

**No progressive curriculum for VAE latent space:**
- Problem: CMA-ES starts from random uniform latents; no warm-start from domain knowledge
- Blocks: Cannot leverage prior buffer to initialize search region
- Fix approach: Encode buffer levels to latent means; initialize CMA-ES mean to PCA centroid

**No online model learning for VAE:**
- Problem: VAE is pre-trained and frozen; cannot adapt to new level statistics online
- Blocks: Cannot improve level generation quality as agent learns
- Fix approach: Periodic VAE fine-tuning on accumulated buffer (requires separate training loop)

**No diversity tracking in level generation:**
- Problem: CMA-ES tracks only fitness (regret), not behavioral novelty
- Blocks: May converge to narrow subset of interesting behaviors
- Fix approach: Add map-elites-style archive or behavioral diversity metrics to CMA-ES fitness

**No checkpointing of CMA-ES state:**
- Problem: If training is interrupted, CMA-ES state is lost
- Blocks: Cannot resume CMA-ES search from same point
- Fix approach: Serialize es_state to orbax checkpoint; restore on resume

## Test Coverage Gaps

**CMA-ES state transitions not tested:**
- What's not tested: Reset logic, tell/ask consistency, evosax API fallback
- Files: `vae/cmaes_manager.py`, `examples/maze_plr.py:700-706`
- Risk: Silent failures in periodic reset or fitness updates
- Priority: High — CMA-ES is core to VAE experiment pipeline

**VAE decoding robustness not tested:**
- What's not tested: Corrupted tokens, edge cases (agent/goal collision, oversized walls), repair logic
- Files: `vae/vae_level_utils.py:23-43`
- Risk: Invalid levels reach training loop, causing silent NaN values in loss
- Priority: High — impacts all runs using `--use_cmaes`

**Buffer snapshot serialization/deserialization:**
- What's not tested: Dump/load round-trip, data format stability, PCA analysis on corrupted dumps
- Files: `examples/maze_plr.py:1195-1268`
- Risk: Post-training analysis may fail on old checkpoint formats
- Priority: Medium — affects offline analysis only

**Configuration validation:**
- What's not tested: Invalid combinations (e.g., `--use_cmaes` without VAE paths), type coercion from argparse
- Files: `examples/maze_plr.py:478-480`, `examples/maze_plr.py:1272-1357`
- Risk: User error leads to confusing failures deep in training loop
- Priority: Medium — early validation would save debugging time

---

*Concerns audit: 2026-03-23*
