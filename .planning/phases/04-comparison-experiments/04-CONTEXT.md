# Phase 4: Comparison Experiments - Context

**Gathered:** 2026-03-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Run 3 ACCEL+LLM seeds to 50k training steps, compare against existing JAXUED_LEARNABILITY
ACCEL-only baseline runs using SFL learnability as the primary metric, and produce a
comparison table + learning curves for the thesis. The question being answered: does LLM
maze injection improve agent generalisation?

</domain>

<decisions>
## Implementation Decisions

### Control condition
- Reuse existing runs from the **JAXUED_LEARNABILITY** WandB project as the ACCEL-only baseline
- Do NOT run fresh control seeds (saves GPU time; existing runs match parameters)
- Primary comparison metric: **SFL learnability** p*(1-p), NOT MaxMC regret
- Exact WandB metric key for JAXUED_LEARNABILITY runs: to be determined at analysis time
  (auto-detect from run history)

### Execution strategy
- Run seeds **in parallel**: one seed per GPU node (albacore / smew / canada)
- Modify `launch_llm_injection.sh`: add `SEED=0` variable at top of script so user sets it
  per-machine — no for-loop, single seed per invocation
- Log destination: `/tmp/` on each GPU node (WandB is source of truth; tmp logs for debug)

### Analysis scope
- Primary output: **both** comparison table AND learning curves
  - Table: mean ± std SFL learnability at end of training (3 ACCEL+LLM vs JAXUED_LEARNABILITY)
  - Learning curves: SFL learnability over training steps for both conditions (mean + shaded std)
- Also include injection diagnostics as supporting evidence:
  - Gate acceptance rate per injection event
  - Total LLM mazes that entered the buffer across the run
- Output format: table printed to stdout + plots saved as PNG files

### LLM model / provider
- Switch to **OpenRouter or Ollama** immediately — do NOT use `claude-code` CLI for production
  runs (quota risk: ~1200 calls estimated over a 50k run)
- OpenRouter is preferred if a paid API key is available (supports Claude Sonnet/Haiku,
  cheaper per-token than CLI subscription abuse)
- Ollama is the fallback if no API key (self-hosted, free, but model quality varies)
- **Pre-flight task**: set up OpenRouter API key (`OPENROUTER_API_KEY` env var) and update
  `llm/config.yaml` provider to `openrouter` before starting 50k runs
- Model: Claude Sonnet via OpenRouter (or equivalent quality model if using Ollama)
- `llm/config.yaml` already has `openrouter` section configured — just needs the env var and
  provider field switched

### Contingency plan
- **Gate accepts nothing by ~20k steps**: intervene by lowering `difficulty_threshold` from
  0.6 → 0.4 first; if still failing, adjust the generation prompt
- Goal: at least some LLM mazes must enter the buffer — the experiment requires LLM injection
  to be active, not just wired
- **Run crashes**: restart from latest Orbax checkpoint (already saved every 2 steps by the
  training loop); accept minor step-count discrepancy

### Claude's Discretion
- Exact WandB API calls and metric key auto-detection in the comparison script
- Plot styling, axis labels, colour scheme
- Whether to checkpoint-validate before starting (or just trust existing infrastructure)

</decisions>

<specifics>
## Specific Ideas

- The `compare_llm_injection.py` script already exists (Phase 3 artifact) — extend it rather
  than rewriting
- `launch_llm_injection.sh` already exists with ablation variables at top — add SEED variable
  there, remove the for-loop
- Injection diagnostics (gate acceptance rate, injected maze count) are already logged to
  WandB as `llm/acceptance_rate` and `llm/injected_count` — pull these in the analysis

</specifics>

<deferred>
## Deferred Ideas

- Full ablation sweep (varying INJECT_START, INJECT_INTERVAL, BATCH_SIZE) — out of scope,
  would be its own phase
- CMA-ES-only vs ACCEL+LLM comparison — not in this phase
- Prompt engineering to reduce Claude verbosity (~44k token responses) — deferred; not
  blocking thesis claim

</deferred>

---

*Phase: 04-comparison-experiments*
*Context gathered: 2026-03-24*
