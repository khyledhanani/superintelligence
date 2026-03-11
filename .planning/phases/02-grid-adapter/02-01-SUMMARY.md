---
phase: 02-grid-adapter
plan: 01
subsystem: vae
tags: [jax, vmap, jit, cnn-vae, level-decode, maze, flax]

# Dependency graph
requires:
  - phase: 01-checkpoint
    provides: "CNN-VAE checkpoint downloaded to vae/checkpoints/cnn_vae/; CnnLstmDecoder interface verified"
provides:
  - "vae/cnn_vae_level_utils.py: decode_latent_to_levels_grid(decode_fn, z_batch, rng) -> batched Level"
  - "_decode_single_z(decode_fn, z, rng) -> single Level (vmappable inner function)"
  - "GRID_SIZE = 13 constant"
affects:
  - 03-training-integration
  - scripts/test_grid_adapter.py (Phase 2 Plan 02)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Grid decode: sigmoid(wall_logits) > 0.5 for wall map (GRID-01)"
    - "Wall-masked argmax: jnp.where(wall_mask_flat, -1e9, logits.flatten()) for goal/agent placement (GRID-02)"
    - "Flat-to-coordinate: col=flat%13 (x), row=flat//13 (y); Level.pos=[x,y]=[col,row] (GRID-03/04)"
    - "Collision resolution: jnp.where(goal==agent, (agent+1)%169, agent) — JIT-compatible (GRID-05)"
    - "Wall clear: wall_map.at[y,x].set(False) at goal and agent positions (GRID-06)"
    - "Agent dir: jax.random.randint(rng, (), 0, 4).astype(uint8) (GRID-07)"
    - "Batch decode: jax.vmap(_decode_single_z, in_axes=(None, 0, 0))(decode_fn, z_batch, rngs) (GRID-08)"
    - "JIT usage: jax.jit(decode_latent_to_levels_grid, static_argnums=(0,)) — decode_fn is static"

key-files:
  created:
    - vae/cnn_vae_level_utils.py
  modified: []

key-decisions:
  - "New file vae/cnn_vae_level_utils.py — NOT modifying vae_level_utils.py (INTG-03: CluttrVAE path preserved)"
  - "decode_fn is a static argument: callers must use jax.jit(..., static_argnums=(0,)) or functools.partial"
  - "Argmax (deterministic) for goal/agent placement — not stochastic sampling (CMA-ES fitness requires deterministic decode given z)"
  - "Post-hoc collision fix via jnp.where (+1 wrap) is sufficient — verified 0 collisions in 1000-sample research test"

patterns-established:
  - "Pattern: decode_fn closure must accept unbatched z (64,) -> (13,13) each; caller adds [None] for decoder batch dim"
  - "Pattern: validate batched Level with jax.vmap(lambda l: l.is_well_formatted())(levels) — NOT levels.is_well_formatted() directly"

requirements-completed: [GRID-01, GRID-02, GRID-03, GRID-04, GRID-05, GRID-06, GRID-07, GRID-08]

# Metrics
duration: 6min
completed: 2026-03-11
---

# Phase 2 Plan 01: Grid Adapter Summary

**JIT-compatible CNN-VAE grid-to-Level adapter implementing GRID-01..08: sigmoid wall threshold, wall-masked argmax, x=col/y=row coordinate transform, jnp.where collision fix, and jax.vmap batch decode**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-11T21:12:16Z
- **Completed:** 2026-03-11T21:18:02Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Created `vae/cnn_vae_level_utils.py` with `decode_latent_to_levels_grid(decode_fn, z_batch, rng)` as the public API
- Implemented all GRID-01..08 requirements in a single self-contained file with zero new dependencies
- Verified JIT compilation, batch shapes (N,13,13)/(N,2)/(N,) and dtypes (bool/uint32/uint32/uint8) pass smoke test

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement vae/cnn_vae_level_utils.py** - `b57d5a8` (feat)
2. **Task 2: Smoke-check JIT compiles with mock decode_fn** - verified inline, no separate commit needed (Task 1 commit covers the final clean file)

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `vae/cnn_vae_level_utils.py` — Grid-based decode adapter for CNN-VAE (CnnLstmDecoder) output; exports `decode_latent_to_levels_grid`, `_decode_single_z`, `GRID_SIZE`

## Decisions Made

- **New file, not modification:** Created `vae/cnn_vae_level_utils.py` rather than adding to `vae_level_utils.py`, keeping the CluttrVAE (token-based) and CNN-VAE (grid-based) decode paths strictly separate per INTG-03.
- **JIT static_argnums pattern:** `decode_fn` is a Python callable and must be treated as static in JIT. Callers use `jax.jit(decode_latent_to_levels_grid, static_argnums=(0,))` or `functools.partial(decode_latent_to_levels_grid, decode_fn)` before calling `jax.jit`. This is standard JAX practice for higher-order functions.
- **Deterministic argmax:** Used `jnp.argmax` (deterministic) for goal/agent placement rather than stochastic sampling, ensuring CMA-ES fitness is reproducible given a fixed `z`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] JIT verify command used incorrect pattern for Python callable arguments**

- **Found during:** Task 2 (smoke-check)
- **Issue:** The plan's verify command `jax.jit(decode_latent_to_levels_grid)(mock_fn, ...)` fails because JAX JIT cannot handle Python callables as non-static arguments. This is a fundamental JAX constraint: `jax.jit(fn)` treats all arguments as JAX arrays unless `static_argnums` is specified.
- **Fix:** Ran smoke test with `jax.jit(decode_latent_to_levels_grid, static_argnums=(0,))` — the correct API usage. The implementation itself is correct; only the test invocation pattern differs from the plan's verify command.
- **Files modified:** None (implementation unchanged; test ran inline)
- **Verification:** All 3 test assertions pass: shapes (4,13,13)/(4,2)/(4,2)/(4,), dtypes (bool/uint32/uint32/uint8), JIT compiles and runs
- **Committed in:** b57d5a8 (Task 1 commit — implementation was already correct)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in verify invocation pattern, not in implementation)
**Impact on plan:** Implementation is complete and correct. The deviation was in the smoke test invocation, not the implementation. Phase 2 Plan 02 verification script should use `static_argnums=(0,)` or `functools.partial` when JIT-compiling `decode_latent_to_levels_grid`.

## Issues Encountered

JAX JIT cannot accept Python callables as non-static arguments. The plan's verify command assumed `jax.jit(fn)(python_function, ...)` would work — it does not. The correct pattern is `jax.jit(fn, static_argnums=(0,))(python_function, ...)`. The implementation is correct and the JIT pattern is documented in key-decisions.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `vae/cnn_vae_level_utils.py` is complete and ready for Plan 02 (verification script: `scripts/test_grid_adapter.py`)
- Phase 2 Plan 02 should build `decode_fn` closure using the Pattern 3 from RESEARCH.md (CnnLstmDecoder with loaded params) and run GRID-01..09 checks including `is_well_formatted()` on 1000 samples
- For JIT: use `jax.jit(decode_latent_to_levels_grid, static_argnums=(0,))` or `functools.partial(decode_latent_to_levels_grid, decode_fn)` at call site

---
*Phase: 02-grid-adapter*
*Completed: 2026-03-11*
