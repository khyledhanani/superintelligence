# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-11)

**Core value:** CMA-ES with CNN-VAE produces valid, solvable maze Levels and runs a complete 20k training experiment comparable to the ACCEL baseline.
**Current focus:** Phase 2 — Grid Adapter

## Current Position

Phase: 2 of 4 (Grid Adapter)
Plan: 2 of 2 in current phase — COMPLETE
Status: Phase 2 complete, ready for Phase 3
Last activity: 2026-03-11 — Completed Phase 2 Plan 02 (GRID-01..09 verification script)

Progress: [####░░░░░░] 40%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 4.3min
- Total execution time: 0.22 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-checkpoint | 1 | 5min | 5min |
| 02-grid-adapter | 2 | 8min | 4min |

**Recent Trend:**
- Last 5 plans: 01-01 (5min), 02-01 (6min), 02-02 (2min)
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

### Pending Todos

None.

### Blockers/Concerns

None — all Phase 1 blockers resolved:
- google-cloud-storage SDK installed into jax_env
- Checkpoint structure confirmed: PyTreeCheckpointer on cnn_vae/default/
- GCS auth working via legacy credentials

## Session Continuity

Last session: 2026-03-11
Stopped at: Completed 02-02-PLAN.md — scripts/test_grid_adapter.py verified all GRID-01..09, Phase 2 complete
Resume file: None
