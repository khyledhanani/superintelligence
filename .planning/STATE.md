# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-11)

**Core value:** CMA-ES with CNN-VAE produces valid, solvable maze Levels and runs a complete 20k training experiment comparable to the ACCEL baseline.
**Current focus:** Phase 5 — PCA-Space CMA-ES Search

## Current Position

Phase: 5 of 5 (PCA-Space CMA-ES Search)
Plan: 1 of 3 in current phase — COMPLETE
Status: Phase 5 Plan 01 complete — PCA utilities (5 functions) created in vae/cnn_vae_pca_utils.py
Last activity: 2026-03-13 — Completed Phase 5 Plan 01 (PCA utility library, Stage 1 weight-norm + Stage 2 SVD)

Progress: [########░░] 70%

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 3.75min
- Total execution time: 0.25 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-checkpoint | 1 | 5min | 5min |
| 02-grid-adapter | 2 | 8min | 4min |
| 03-integration | 2 | 25min | 12.5min |
| 05-pca-space-cma-es-search | 1 | 3min | 3min |

**Recent Trend:**
- Last 5 plans: 01-01 (5min), 02-01 (6min), 02-02 (2min), 03-01 (2min), 03-02 (23min), 05-01 (3min)
- Trend: 05-01 fast (pure file creation, no GPU infrastructure work needed)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Setup]: CNN-VAE as default decoder (user wants CNN-VAE as primary path, not opt-in)
- [Setup]: Download checkpoint locally to `vae/checkpoints/cnn_vae/` (simpler than GCS runtime loading)
- [Setup]: Reuse existing launch scripts (minimize new infrastructure)
- [01-01]: Orbax loader is PyTreeCheckpointer on cnn_vae/default/ subdir (StandardCheckpointHandler format, not step-indexed)
- [01-01]: Absolute paths required for ocp.PyTreeCheckpointer (relative paths fail in tensorstore layer)
- [01-01]: GCS project is 'open-endedness-personal'; must pass explicitly to storage.Client()
- [02-01]: New file vae/cnn_vae_level_utils.py — NOT modifying vae_level_utils.py (INTG-03: CluttrVAE path preserved)
- [02-01]: decode_fn is a static argument in JIT — use jax.jit(decode_latent_to_levels_grid, static_argnums=(0,)) or functools.partial
- [02-01]: Deterministic argmax (not stochastic sampling) for goal/agent placement ensures CMA-ES fitness is reproducible
- [02-02]: JIT static_argnums=(0,) required for decode_fn Python callable — confirmed passing all GRID-01..09 on 1000-sample batch
- [03-01]: CNN-VAE is default when --use_cmaes set; --use_clutr_vae flag gates CluttrVAE fallback
- [03-01]: vae_cfg["latent_dim"] only in elif use_clutr_vae branch; ternary guards in lines 561/569 use short-circuit to avoid NameError
- [03-01]: PCA block guarded with config.get("use_clutr_vae") — CNN-VAE has no encoder in maze_plr.py context
- [03-02]: MazeSolved constructor uses max_height/max_width (not height/width)
- [03-02]: VALD-02 validated via 1000-step simulation (is_valid.mean()*100 on popsize=32 batches) — GPU smoke test deferred (sideswipe occupied by NAMM training)
- [03-02]: cmaes/valid_structure_pct = 100.0% over 1000 simulated DR steps; BFS solvability = 100% on 50 levels
- [03-02]: maze_plr.py --use_cmaes confirmed to load CNN-VAE and start training (5-step CPU run successful)
- [05-01]: cumulative norm threshold 0.85 for Stage 1 K selection (not fixed K) — adapts to any checkpoint
- [05-01]: full_matrices=False in np.linalg.svd essential when N >> 64 (avoids large U matrix)
- [05-01]: Closures in make_*_decode_fn capture jnp arrays (not numpy) for JIT/vmap compatibility
- [05-01]: Stage 1 uses zeros as mu_mean baseline by default (VAE prior mean) — dataset not needed at search time
- [05-01]: pc_stds whitening ensures unit-variance CMA-ES dimensions (sigma_init=0.5 becomes meaningful)

### Roadmap Evolution

- Phase 5 added: PCA-Space CMA-ES Search

### Pending Todos

None.

### Blockers/Concerns

None — all Phase 1 blockers resolved:
- google-cloud-storage SDK installed into jax_env
- Checkpoint structure confirmed: PyTreeCheckpointer on cnn_vae/default/
- GCS auth working via legacy credentials

## Session Continuity

Last session: 2026-03-13
Stopped at: Completed 05-01-PLAN.md — PCA utility library created (5 functions in vae/cnn_vae_pca_utils.py + validation script)
Resume file: None
