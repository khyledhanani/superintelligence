# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-11)

**Core value:** CMA-ES with CNN-VAE produces valid, solvable maze Levels and runs a complete 20k training experiment comparable to the ACCEL baseline.
**Current focus:** Phase 2 — Grid Adapter

## Current Position

Phase: 2 of 4 (Grid Adapter)
Plan: 1 of 2 in current phase
Status: In progress
Last activity: 2026-03-11 — Completed Phase 2 Plan 01 (grid-to-Level adapter implementation)

Progress: [##░░░░░░░░] 20%

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: 5.5min
- Total execution time: 0.19 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-checkpoint | 1 | 5min | 5min |
| 02-grid-adapter | 1 | 6min | 6min |

**Recent Trend:**
- Last 5 plans: 01-01 (5min), 02-01 (6min)
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

### Pending Todos

None.

### Blockers/Concerns

None — all Phase 1 blockers resolved:
- google-cloud-storage SDK installed into jax_env
- Checkpoint structure confirmed: PyTreeCheckpointer on cnn_vae/default/
- GCS auth working via legacy credentials

## Session Continuity

Last session: 2026-03-11
Stopped at: Completed 02-grid-adapter-01-PLAN.md — vae/cnn_vae_level_utils.py implemented and smoke-tested
Resume file: None
