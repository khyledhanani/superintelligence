# Phase 5: Refactor and Four-Way Comparison - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Clean rewrite of `accel_training/train.py` to a two-mode architecture (replay / es_step), removing MAP-Elites archive and archive warm-up entirely. Then run four comparison experiments at 20k updates (same seed) and produce thesis comparison plots. Fitness weight ablations belong in Phase 6.

</domain>

<decisions>
## Implementation Decisions

### Refactor approach
- Clean rewrite of train.py — not surgical edits, start from scratch with only the two-mode logic
- Two modes only: `replay` (PLR buffer → train agent) and `es_step` (ES ask() → VAE decode → eval → insert into PLR buffer → tell())
- No MAP-Elites archive, no archive warm-up
- Bootstrap strategy: always run `es_step` until buffer has a minimum number of levels (e.g. 50), then switch to the configured ratio
- Replay/es_step ratio controlled via a single config.yml float (e.g. `replay_ratio: 0.8`)
- Full pipeline audit: every file that imports from `accel_training/` is reviewed and updated to match the new two-mode interface (not just train.py + config.yml)

### Experiment execution
- Single launcher script runs all four experiments (sequentially or submits to job scheduler)
- WandB naming: consistent run names (`accel-baseline`, `cma-es`, `ns-es`, `sv-cma-es`) + shared group tag (e.g. `phase5-comparison`) for easy filtering
- Pre-launch validation: SV-CMA-ES smoke run for 1–2k updates confirming buf_score rises above the old ~0.004 ceiling before committing to full runs
- ACCEL baseline (`examples/maze_plr.py`) runs as-is, black-box — no modifications to the original file

### Plot & analysis design
- Primary metric: regret vs update steps (smoothed rolling mean, e.g. window=50) for single seed per method
- Two separate figures:
  - Figure 1: four-method comparison (ACCEL baseline, CMA-ES, NS-ES, SV-CMA-ES)
  - Figure 2: ablation curves (Phase 6 — placeholder in notebook for now)
- Jupyter notebook pulls WandB data via API, smooths, and produces both figures — easy to iterate on and include in thesis appendix

### Claude's Discretion
- Exact bootstrap threshold (number of levels before switching to replay ratio)
- Rolling mean window size for smoothing
- Exact plot styling (colors, line widths, legend placement)
- Temp file handling and run concurrency in the launcher script

</decisions>

<specifics>
## Specific Ideas

- "This should also work cleanly for cma_es and ns_es baselines (same two-mode structure)" — the refactor must generalize across all three ES strategies, not just SV-CMA-ES
- "Eventually I'd want to test the two agents on some validation sets" — not in Phase 5, deferred to Phase 6 or later
- The previous validation run showed buf_score barely reaching 0.004 at iter 720 — the pre-launch smoke test should explicitly check that this ceiling is broken

</specifics>

<deferred>
## Deferred Ideas

- Fitness weight ablation studies (α/β sweep for SV-CMA-ES) — Phase 6
- Validation set evaluation: run saved agent checkpoints on a fixed held-out maze set — Phase 6 or later

</deferred>

---

*Phase: 05-ablations-and-analysis*
*Context gathered: 2026-03-02*
