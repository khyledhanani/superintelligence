---
phase: 03-reproducibility-infrastructure
plan: 01
subsystem: llm
tags: [level-cache, wandb, reproducibility, audit-trail, ablation]

# Dependency graph
requires:
  - phase: 02-grid-adapter
    provides: LLMInjectionManager, _do_injection pipeline, LLMInjectionConfig, maze_plr.py argparse integration

provides:
  - LevelCache class (llm/level_cache.py): disk cache for accepted LLM levels with .npy + JSON sidecar
  - WandB hash table logging per injection event (wandb.Table with step/idx/hash columns)
  - --llm_inject_start_step CLI flag replacing --llm_warmup_steps for ablation clarity
  - Backward compat read of old --llm_warmup_steps in from_config_dict()
  - LevelCache wired into maze_plr.py: results/<run_name>/llm_levels/<seed>/ created on startup

affects:
  - 03-02 (launch scripts will use --llm_inject_start_step)
  - Any external config files using llm_warmup_steps (handled by backward compat)

# Tech tracking
tech-stack:
  added: [hashlib SHA-256 for wall_map hashing, wandb.Table for hash batch logging]
  patterns:
    - Accept-only caching (only LLM seeds that passed gate are saved, not mutations)
    - Hash-first audit trail (16-char SHA-256 prefix for space-efficient WandB logging)
    - Backward compat via config.get() chaining (new_key, old_key, default)

key-files:
  created:
    - llm/level_cache.py
  modified:
    - llm/injector.py
    - llm/injection_config.py
    - examples/maze_plr.py

key-decisions:
  - "Only LLM seeds (valid_levels) are cached -- mutations are not part of the audit trail per CONTEXT.md"
  - "Hash computed from wall_map cast to dtype=bool for deterministic hashing across dtype variants"
  - "accepted_hashes collected even when level_cache is None (compute_hash path) so WandB table is always populated"
  - "LevelCache path includes seed: results/<run_name>/llm_levels/<seed>/ for per-seed isolation"
  - "backward compat: from_config_dict reads llm_inject_start_step with llm_warmup_steps as fallback"

patterns-established:
  - "Level cache files: step_{step:05d}_idx_{idx:03d}.npy + .json in results/<run_name>/llm_levels/<seed>/"
  - "JSON sidecar fields: wall_map_hash, injection_step, batch_index, gate_scores, accept, timestamp"
  - "WandB hash table: columns=[step, batch_index, wall_map_hash], one row per accepted seed per event"

requirements-completed: [EXPT-02, EXPT-03]

# Metrics
duration: 12min
completed: 2026-03-24
---

# Phase 3 Plan 01: Reproducibility Infrastructure Summary

**SHA-256 wall_map audit trail via LevelCache (.npy + JSON sidecar) with WandB hash table logging, and --llm_inject_start_step flag replacing --llm_warmup_steps for independent ablation control**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-03-24T10:17:00Z
- **Completed:** 2026-03-24T10:29:45Z
- **Tasks:** 2
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments
- Created `llm/level_cache.py` with `LevelCache` class: `save_accepted()` writes `.npy` + JSON sidecar per accepted level; `compute_hash()` for hash-only logging when cache is disabled
- Wired `LevelCache` into `LLMInjectionManager`: accepted seeds cached after Step 4, hashes collected for WandB; mutations explicitly excluded per CONTEXT.md
- Added `wandb.Table` with `[step, batch_index, wall_map_hash]` rows to `log_payload` at each injection event (guarded with `if accepted_hashes:` to avoid empty tables)
- Renamed `--llm_warmup_steps` to `--llm_inject_start_step` for ablation clarity (EXPT-03); backward compat via `config.get("llm_inject_start_step", config.get("llm_warmup_steps", 5000))`
- Instantiated `LevelCache` in `maze_plr.py` at `results/<run_name>/llm_levels/<seed>/` and passed to `LLMInjectionManager`

## Task Commits

Each task was committed atomically:

1. **Task 1: Create LevelCache class and wire into injector** - `efc30f8` (feat)
2. **Task 2: Rename --llm_warmup_steps to --llm_inject_start_step** - `30daed1` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `llm/level_cache.py` - New: LevelCache class with save_accepted() and compute_hash() methods
- `llm/injector.py` - Added LevelCache import, level_cache param to __init__, Step 4b cache block, WandB Table in log_payload, current_step param to _do_injection()
- `llm/injection_config.py` - Renamed warmup_steps -> inject_start_step; updated from_config_dict() with backward compat
- `examples/maze_plr.py` - Renamed argparse flag; added LevelCache import + instantiation; updated LLMInjectionManager call

## Decisions Made
- Only `valid_levels` (LLM seeds that passed gate) are cached — mutations are derived variants not suitable for the audit trail
- Hash computed from `np.asarray(level.wall_map, dtype=bool).tobytes()` for deterministic cross-dtype result
- `accepted_hashes` populated via `compute_hash()` even when `level_cache is None` — WandB table always shows hashes
- LevelCache path includes seed sub-directory for per-seed isolation when multiple seeds share a run_name directory
- Backward compat: `from_config_dict()` tries `llm_inject_start_step` first, falls back to `llm_warmup_steps`

## Deviations from Plan

None — plan executed exactly as written. The backward compat string `"llm_warmup_steps"` in `from_config_dict()` is explicitly required by the plan spec (not a stale reference).

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- EXPT-02 (audit trail) and EXPT-03 (ablation flags) are satisfied
- Plan 03-02 can proceed: launch scripts should use `--llm_inject_start_step` (not `--llm_warmup_steps`)
- Level cache files will be written to `results/<run_name>/llm_levels/<seed>/` on all LLM runs

---
*Phase: 03-reproducibility-infrastructure*
*Completed: 2026-03-24*
