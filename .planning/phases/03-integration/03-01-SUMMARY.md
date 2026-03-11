---
phase: 03-integration
plan: 01
subsystem: training
tags: [cnn-vae, cma-es, maze-plr, orbax, flax, jax, decoder]

# Dependency graph
requires:
  - phase: 02-grid-adapter
    provides: "decode_latent_to_levels_grid adapter (vae/cnn_vae_level_utils.py), CnnLstmDecoder (vae/cnn_vae_model.py)"
  - phase: 01-checkpoint
    provides: "Orbax checkpoint at vae/checkpoints/cnn_vae/default/, PyTreeCheckpointer load pattern"
provides:
  - "CNN-VAE as default decode path in examples/maze_plr.py when --use_cmaes is set"
  - "--use_clutr_vae flag to fall back to original CluttrVAE token decoder"
  - "decode_latent_to_levels_grid called in CMA-ES dr_step for CNN-VAE path"
  - "PCA block guarded to only run for CluttrVAE path"
affects: [04-smoke-test, training-loop, cma-es-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Conditional VAE setup block: CNN-VAE default (no YAML config) vs CluttrVAE fallback (--use_clutr_vae)"
    - "_cnn_vae_latent_dim=64 constant replaces vae_cfg['latent_dim'] for CNN-VAE path"
    - "PCA block guarded with config.get('use_clutr_vae') to prevent encode_fn TypeError"

key-files:
  created: []
  modified:
    - examples/maze_plr.py

key-decisions:
  - "CNN-VAE is the default decoder when --use_cmaes is set; no extra flags required"
  - "--use_clutr_vae is a BooleanOptionalAction that falls back to CluttrVAE (requires --vae_checkpoint_path and --vae_config_path)"
  - "PCA block (post-training) only runs for CluttrVAE path — CNN-VAE has no encoder in this context"
  - "vae_cfg['latent_dim'] only referenced inside elif use_clutr_vae branch; ternary guards prevent NameError in CNN-VAE path"

patterns-established:
  - "Conditional VAE dispatch: _needs_vae and not config.get('use_clutr_vae') for CNN-VAE default"
  - "decode dispatch in dr_step: config.get('use_clutr_vae') -> decode_latent_to_levels else decode_latent_to_levels_grid"

requirements-completed: [INTG-01, INTG-02, INTG-03, INTG-04]

# Metrics
duration: 2min
completed: 2026-03-11
---

# Phase 3 Plan 01: CNN-VAE Integration Summary

**CNN-VAE wired as default decoder in maze_plr.py CMA-ES path: Orbax checkpoint load + decode_latent_to_levels_grid dispatch, --use_clutr_vae flag preserves original CluttrVAE fallback**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-11T22:44:36Z
- **Completed:** 2026-03-11T22:46:41Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Added `CnnLstmDecoder` and `decode_latent_to_levels_grid` imports from `vae/` to `maze_plr.py`
- Replaced monolithic CluttrVAE-only `_needs_vae` block with conditional CNN-VAE/CluttrVAE setup using `--use_clutr_vae` gate
- CNN-VAE path loads Orbax checkpoint from `vae/checkpoints/cnn_vae/default/` using absolute path (required by tensorstore)
- CMA-ES `dr_step` decode call now dispatches to `decode_latent_to_levels_grid` (CNN-VAE default) or `decode_latent_to_levels` (CluttrVAE fallback)
- Post-training PCA block guarded with `config.get("use_clutr_vae")` to prevent `vae_encode_fn` TypeError in CNN-VAE path

## Task Commits

Each task was committed atomically:

1. **Task 1+2: Add --use_clutr_vae flag, CNN-VAE setup block, decode dispatch, PCA guard** - `502cfcd` (feat)

**Plan metadata:** (pending final docs commit)

## Files Created/Modified
- `examples/maze_plr.py` - CNN-VAE imports, conditional VAE setup block, decode dispatch in dr_step, PCA guard, --use_clutr_vae argparser entry

## Decisions Made
- Combined Tasks 1 and 2 into a single commit since both modified the same file and were closely coupled; no correctness tradeoff
- `_cnn_vae_latent_dim = 64` constant used instead of YAML config (CNN-VAE has no config file)
- All `vae_cfg` references remain inside the `elif use_clutr_vae` branch; ternary expressions on lines 561 and 569 use Python short-circuit evaluation to avoid NameError in CNN-VAE path

## Deviations from Plan

None - plan executed exactly as written. All four edits (imports, VAE setup block, PCA guard, argparser) applied as specified.

## Issues Encountered
None. AST parse passed immediately after each edit. `orbax.checkpoint as ocp` was already imported at line 17 (plan noted this as a conditional check — confirmed present, no duplicate import added).

## User Setup Required
None - no external service configuration required. CNN-VAE checkpoint is already present at `vae/checkpoints/cnn_vae/default/`.

## Next Phase Readiness
- `--use_cmaes` training with CNN-VAE default is ready to run: `python examples/maze_plr.py --use_cmaes`
- CluttrVAE fallback is preserved: `python examples/maze_plr.py --use_cmaes --use_clutr_vae --vae_checkpoint_path X --vae_config_path Y`
- Phase 3 Plan 02 smoke test (1000-step CMA-ES run + BFS solvability check) is the next step

---
*Phase: 03-integration*
*Completed: 2026-03-11*
