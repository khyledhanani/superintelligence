# Phase 1: Integration Scaffolding - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the existing standalone LLM maze generation system (`llm/` module) into the ACCEL/PLR training loop (`maze_plr.py`). The core ACCEL loop (P_D coin flip, PPO updates, native mutation) is NOT modified — LLM injection is purely additive. The behavioral decision gate is wired in Phase 2; this phase uses unconditional injection with format validation only.

Key integration: `LLMInjectionManager.maybe_inject(step, train_state)` as the single call site in the training loop outer for-loop.

</domain>

<decisions>
## Implementation Decisions

### LLM Provider Configuration
- Multi-provider support: Claude API, OpenAI API, Ollama — reuse existing provider implementations in `llm/maze_generator.py`
- All LLM settings via CLI flags on `maze_plr.py` (not config file): `--llm_provider`, `--llm_model`, `--llm_inject_interval`, `--llm_batch_size`, etc.
- No default provider — `--llm_provider` is required when `--use_llm` is set
- Synchronous API calls — training pauses during injection (acceptable at interval=3000 steps)
- On API failure: crash training immediately (no retry, no skip — the LLM must work or training stops)

### Mutation Amplification Pipeline
- LLM mazes are **structural seeds**, not direct buffer entries
- Pipeline: LLM generates N_raw mazes → format validation → diversity gate filters to N_gate → each gated maze gets M wall-flip mutations → solvability check on mutations → survivors enter PLR buffer
- **Two-tier scoring strategy:**
  - Tier 1 (seeds): Full policy rollout for SFL/regret scoring — only ~10-15 mazes, negligible cost
  - Tier 2 (mutations): Solvability check only (BFS) — NO policy rollout; PLR handles scoring when replayed
- Reuse ACCEL's existing JAX-compiled wall-flip mutation function — same mutation, same behavior
- Make scoring strategy configurable: `score_seeds_with_rollout=True`, `score_mutations_with_rollout=False`

### Reference Maze Sourcing
- Sample reference mazes directly from live PLR `train_state.sampler` buffer — no file dumps
- Selection strategy: top-regret (highest regret mazes shown to LLM as context)
- NOTE: reference strategy is an important tuning knob — start with top-regret but may change to diverse/mixed

### Buffer Injection
- Mutations enter PLR buffer with **maximum priority score** to force immediate replay and scoring
- Use existing `level_sampler.insert_batch()` — injected levels follow same eviction rules as native levels
- Levels are indistinguishable from native levels once in buffer (same priority scoring, same eviction)
- Internally tagged with lineage metadata for experiment tracking

### Format Validation
- Reuse existing validation from `llm/maze_generator.py` — already handles solvability, border walls, agent/goal placement, grid dimensions
- No separate lightweight validator — single source of truth

### WandB Logging
- Detailed logging per injection event:
  - `llm/injected_count` — total levels entering buffer this event
  - `llm/acceptance_rate` — gate pass rate for seeds
  - `llm/injection_time_seconds` — wall clock time for the injection event
  - Per-seed regret and diversity scores
  - Mutation survival rate (solvable mutations / total mutations)
  - Lineage tracking: how many LLM-origin levels in buffer, replay frequency, lineage depth
  - Buffer occupancy of LLM-origin levels vs native levels

### Configurable Parameters
```python
@dataclass
class LLMInjectionConfig:
    # Timing
    enabled: bool = True
    injection_interval: int = 3000        # steps between injections
    warmup_steps: int = 5000              # no injection before this step

    # LLM generation
    n_raw: int = 25                       # mazes requested from LLM per injection
    reference_maze_strategy: str = "hardest"  # "hardest", "most_diverse", "random", "mixed"
    n_reference_mazes: int = 5            # buffer mazes shown to LLM as context

    # Diversity gate (Phase 1: disabled; Phase 2: enabled)
    gate_enabled: bool = False
    diversity_threshold: float = 0.1
    min_difficulty: float = 0.1
    max_difficulty: float = 0.9

    # Mutation amplification
    amplification_enabled: bool = True
    mutations_per_seed: int = 30
    use_native_regret_filter: bool = True

    # Scoring strategy
    score_seeds_with_rollout: bool = True
    score_mutations_with_rollout: bool = False
    mutations_solvability_check: bool = True

    # Buffer injection
    max_inject_per_event: int = 200

    # Tracking
    track_lineage: bool = True
```

### Integration Point
- Single method call: `LLMInjectionManager.maybe_inject(step, train_state)` at the top of the training loop
- No modification to ACCEL's d=0/d=1/d=2 branching logic
- Injection happens BETWEEN training steps

### Claude's Discretion
- Internal architecture of `LLMInjectionManager` class
- How to extract reference mazes from `train_state.sampler` JAX arrays
- How to handle the JAX ↔ Python boundary for LLM API calls
- Lineage tracking data structure design
- WandB metric naming conventions beyond the specified ones

</decisions>

<specifics>
## Specific Ideas

- "Injecting ~10 LLM mazes into a 4000-level buffer is homeopathic — too few to matter. Use LLM mazes as structural seeds and amplify them with ACCEL's own wall-flip mutations."
- The existing `llm/` module is already production-quality with multi-provider support, metric injection in prompts, and a comprehensive decision gate — integration should reuse it, not rewrite it
- Parallel API calls may be explored later if injection latency becomes a bottleneck — noted as deferred optimization
- The 4-stage testing priority: (1) raw injection without gate/amplification, (2) add gate, (3) add mutation amplification, (4) add lineage tracking — each toggleable via config for independent ablation

</specifics>

<deferred>
## Deferred Ideas

- **Parallel LLM API calls** — explore concurrent API requests to reduce injection latency (future optimization)
- **Alternative reference strategies** — "most_diverse", "mixed" strategies for reference maze selection (tuning knob for Phase 2+)
- **Async prefetch** — pre-fetch LLM mazes in background thread to hide latency (rejected for now: synchronous is simpler)

</deferred>

---

*Phase: 01-checkpoint*
*Context gathered: 2026-03-23*
