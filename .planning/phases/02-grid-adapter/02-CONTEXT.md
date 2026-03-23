# Phase 2: Decision Gate and Tuning - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the existing `DecisionGate` + `AgentEvaluator` into the `LLMInjector` pipeline so every LLM maze candidate is evaluated against the live policy before buffer insertion. Extend AgentEvaluator with direct param passing for current policy access. Feed live buffer entropy stats to PromptBuilder. Empirically tune injection hyperparameters via short smoke test runs before committing to paid 50k experiments.

Phase 1's unconditional injection (format validation only) is replaced by gated injection (difficulty + diversity filtering). The ACCEL training loop itself is NOT modified.

</domain>

<decisions>
## Implementation Decisions

### Gate Calibration
- Enable **both** difficulty gate (regret) AND diversity gate (td_error_emd) — matches the tested standalone behavior in `llm/test_generator.py`
- Default thresholds from `llm/config.yaml`: `difficulty_threshold: 0.6`, `min_diversity: 0.02`, `diversity_metric: td_error_emd`
- On all-rejected batch: use `MazeGenerator.generate_with_feedback()` retry loop — LLM gets rejection reasons and retries (up to 2 retries = 3 total attempts per maze)
- Track batch rejection rate in WandB (`llm/batch_all_rejected_count`) — empirically verify whether full-batch rejection actually occurs in practice; if frequent, revisit thresholds
- 100 rollouts per candidate maze (matching standalone default `n_rollouts: 100`)

### Checkpoint Refresh
- **Direct param passing** — pass `train_state.params` directly from the training loop to AgentEvaluator at each injection event. No file I/O, always up-to-date
- Modify AgentEvaluator to accept params directly instead of reloading from checkpoint file
- Keep default `env_params` — they're fixed at initialization (grid size, max steps, reward structure) and don't change during training
- Prompt context (buffer stats, reference mazes) refreshed at **every injection event** via BufferStatsExtractor — always comprehensive, always current

### Tuning Protocol
- **Start conservative, adjust**: Begin with `injection_interval=50`, `batch_size=4`. Run a 5k-step smoke test, check acceptance rate and solve rate stability
- **"Good enough" signal**: BOTH acceptance rate in 30-70% range AND no solve rate drops >0.05 post-injection
- **Both injection patterns configurable** for experiments:
  - Small frequent: `n_raw=4-8`, `injection_interval=50` (default, safe)
  - Larger infrequent: `n_raw=25`, `injection_interval=500` (experiment condition)
- Mutation amplification: `mutations_per_seed=30` → ~18-24 viable mutations per seed. One injection event with n_raw=25 replaces ~7-12% of buffer; n_raw=4-8 replaces ~1.5-4%

### LLM Provider Strategy
- **Smoke testing / dev**: `claude-code` provider with Sonnet (fast, Max plan, no API cost)
- **Full experiment runs**: OpenRouter (one API key, multi-model access). Try OpenAI or Gemini models via pro subscriptions first. Ollama as free fallback if needed
- At interval=50 over 50k steps: ~20 injection events, ~160-240 API calls per run — very manageable cost via OpenRouter

### Injection Stability
- **Warmup period**: Use `--llm_inject_start_step` to delay first injection (default: 1000 steps). Let agent learn basic navigation before injecting OOD mazes
- **On solve rate drops**: Log to WandB, keep going. Short-term instability is expected and part of what we're measuring. No auto-pause mechanism
- **"Works at 3k → works at 30k" guarantee**: System naturally adapts because policy params are always current, buffer stats are always live, and gate evaluates against current policy. Log trend metrics (acceptance rate over time, difficulty scores over time) to catch degradation early
- **Bulletproof smoke testing before paid runs**: Verify (1) gate actually filters, (2) injected levels appear in buffer and get replayed, (3) WandB metrics log correctly, (4) no crashes or silent failures over 5k steps, (5) solve rate curve looks sane

### Claude's Discretion
- Internal refactoring of AgentEvaluator to accept direct params vs file-based loading
- WandB metric naming beyond specified ones
- Exact smoke test script structure
- How to expose both injection patterns as CLI flags

</decisions>

<specifics>
## Specific Ideas

- Correctness before scale — smoke tests must be bulletproof before committing to any paid 50k runs. "Bad results are fine, wrong results are not"
- Both injection patterns (small/frequent vs larger/infrequent) are genuine experiment conditions to compare, not just tuning knobs
- LLM mazes are structural seeds amplified by ACCEL mutation (30 mutations per seed, ~60-80% survive solvability filter) — the injection is more impactful than raw batch size suggests
- Max-priority insertion means injected levels are instantly replayed by PLR, then mutated through normal ACCEL cycle — no special "immediate play" mechanism needed

</specifics>

<deferred>
## Deferred Ideas

- **Provider ablation**: Compare maze quality across Claude Sonnet, GPT-4o, Gemini, Ollama models — future experiment condition
- **Adaptive thresholds**: Auto-adjust gate thresholds based on rolling acceptance rate — complexity with uncertain payoff
- **Auto-pause on instability**: Skip injection events if solve rate drops too much — decided against for now, log and observe instead

</deferred>

---

*Phase: 02-grid-adapter*
*Context gathered: 2026-03-23*
