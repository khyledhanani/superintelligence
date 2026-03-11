# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-11)

**Core value:** CMA-ES with CNN-VAE produces valid, solvable maze Levels and runs a complete 20k training experiment comparable to the ACCEL baseline.
**Current focus:** Phase 3 — Integration

## Current Position

Phase: 3 of 4 (Integration)
Plan: 1 of 2 in current phase — COMPLETE
Status: Phase 3 Plan 01 complete — CNN-VAE wired as default decoder in maze_plr.py
Last activity: 2026-03-11 — Completed Phase 3 Plan 01 (CNN-VAE integration into maze_plr.py)

Progress: [#####░░░░░] 50%

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
| 03-integration | 1 | 2min | 2min |

**Recent Trend:**
- Last 5 plans: 01-01 (5min), 02-01 (6min), 02-02 (2min), 03-01 (2min)
- Trend: Stable

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

### Pending Todos

None.

### Blockers/Concerns

None — all Phase 1 blockers resolved:
- google-cloud-storage SDK installed into jax_env
- Checkpoint structure confirmed: PyTreeCheckpointer on cnn_vae/default/
- GCS auth working via legacy credentials

## Session Continuity

Last session: 2026-03-11
Stopped at: Completed 03-01-PLAN.md — CNN-VAE wired as default decoder in maze_plr.py; Phase 3 Plan 01 complete
Resume file: None
