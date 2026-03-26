# Phase 3: Reproducibility Infrastructure - Context

**Gathered:** 2026-03-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Accepted LLM levels are cached to disk with wall_map hashes logged to WandB, and comparison launch scripts exist for running ACCEL+LLM vs ACCEL-only control with matching seeds and conditions. Analysis tooling produces comparison tables from WandB data.

</domain>

<decisions>
## Implementation Decisions

### Level cache layout
- Cache lives inside each run's output directory (e.g. `results/<run_name>/llm_levels/`)
- Files named by step + batch index: `step_05000_idx_003.npy`
- Only LLM-generated levels are cached — not the full PLR buffer
- Only accepted levels are saved to disk; rejected levels are logged in WandB metrics but not cached
- Each `.npy` file has a JSON sidecar with core audit fields: wall_map SHA-256 hash, injection step, gate scores (td_error_emd, solve_rate), accept/reject decision, timestamp

### Launch script design
- Hardcoded per machine, matching existing `launch_50k_*.sh` conventions
- Two scripts: `launch_llm_injection.sh` (ACCEL+LLM) and `launch_accel_only_control.sh` (ACCEL-only)
- 3 seeds per condition (0, 1, 2)
- LLM injection runs target GPU nodes: albacore, smew, canada (one seed per machine)
- Control runs assigned separately (TPU or sequential on GPUs after LLM runs)
- WandB project: `JAXUED_LLM`
- WandB group names: `accel-llm` / `accel-only`

### Determinism scope
- JAX-side determinism only: same JAX seed produces identical training trajectory until the first LLM injection event
- LLM API non-determinism accepted — different API calls will produce different mazes across reruns
- Reproducibility ensured by logging everything: cached levels on disk + wall_map hashes in WandB enable full audit trail
- No replay mode needed for this phase

### Wall-map hash logging
- SHA-256 of `wall_map.tobytes()`, truncated to first 16 hex characters
- Logged as batch summary per injection event (WandB table row with list of hashes per step)

### Analysis tooling
- Comparison table script (like existing `compare_phase4_results.py`)
- Filters runs by WandB group name (`accel-llm`, `accel-only`) in `JAXUED_LLM` project
- Shows solve rate (mean ± std) plus LLM-specific metrics: acceptance rate, injected count, diversity score mean

### Claude's Discretion
- Exact JSON sidecar field names and structure
- How `--llm_inject_start_step` default is set
- Comparison script output formatting
- Control run machine assignment strategy

</decisions>

<specifics>
## Specific Ideas

- Launch scripts should follow the same SSH + nohup + `LD_LIBRARY_PATH` pattern as existing `examples/launch_50k_*.sh` scripts
- Comparison script should follow the same WandB API pattern as `scripts/compare_phase4_results.py`
- Cache directory should be self-contained per run so archiving/deleting a run removes its levels too

</specifics>

<deferred>
## Deferred Ideas

- Cache + replay mode (`--replay_llm_levels <dir>`) for exact post-injection reproduction — future enhancement if needed
- Solve rate curve plots (matplotlib mean ± std shading) — can add during Phase 4 if needed for thesis figures

</deferred>

---

*Phase: 03-reproducibility-infrastructure*
*Context gathered: 2026-03-24*
