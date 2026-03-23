# Phase 2: Decision Gate and Tuning - Research

**Researched:** 2026-03-23
**Domain:** Decision gate wiring, AgentEvaluator refactor, live buffer entropy stats, WandB metrics, injection hyperparameter tuning
**Confidence:** HIGH — all findings verified directly from the codebase (no external libraries introduced)

## Summary

Phase 2 replaces Phase 1's unconditional injection (format validation only) with gated injection: every LLM maze candidate is evaluated by `AgentEvaluator.evaluate_level_multi_rollout()`, the resulting trajectory is scored by `evaluate_candidate()` in `decision_gate.py`, and only mazes passing both difficulty and diversity thresholds enter the buffer. All the infrastructure already exists; the work is wiring.

The main engineering challenge is that `AgentEvaluator.__init__()` currently loads a checkpoint from disk via `load_agent()`, capturing a frozen `train_state` at construction time. Phase 2 must refactor this to accept live `train_state.params` at call time so the evaluator uses the current policy at each injection event. The `_build_rollout_fn()` method already captures `train_state` by closure — that reference must become a parameter instead. The second challenge is threading live buffer entropy stats into `build_generation_prompt()` as `global_metrics`; `BufferStatsExtractor.extract_buffer_summary()` already computes these, they just need to be converted to `MetricEntry` objects and passed through.

The retry loop (`generate_with_feedback()`) is also fully implemented in `MazeGenerator` and expects an `agent_evaluator` argument and `DiversityThresholds`. `LLMInjectionManager._do_injection()` currently calls `self.generator.generate()` (no gate). Phase 2 switches that call to `self.generator.generate_with_feedback()` for the gate-enabled path, passing the live `AgentEvaluator` and `DiversityThresholds` from config. The existing `injection_config.py` has placeholder gate fields (`gate_enabled`, `diversity_threshold`, `min_difficulty`) that need to be connected to the real `DiversityThresholds` dataclass from `decision_gate.py`.

**Primary recommendation:** Refactor `AgentEvaluator` to accept `params` at evaluate-time rather than from `load_agent()` at construction, then wire it into `LLMInjectionManager` with `generate_with_feedback()` and live buffer stats passed as `global_metrics`.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Gate Calibration
- Enable **both** difficulty gate (regret) AND diversity gate (td_error_emd) — matches the tested standalone behavior in `llm/test_generator.py`
- Default thresholds from `llm/config.yaml`: `difficulty_threshold: 0.6`, `min_diversity: 0.02`, `diversity_metric: td_error_emd`
- On all-rejected batch: use `MazeGenerator.generate_with_feedback()` retry loop — LLM gets rejection reasons and retries (up to 2 retries = 3 total attempts per maze)
- Track batch rejection rate in WandB (`llm/batch_all_rejected_count`) — empirically verify whether full-batch rejection actually occurs in practice; if frequent, revisit thresholds
- 100 rollouts per candidate maze (matching standalone default `n_rollouts: 100`)

#### Checkpoint Refresh
- **Direct param passing** — pass `train_state.params` directly from the training loop to AgentEvaluator at each injection event. No file I/O, always up-to-date
- Modify AgentEvaluator to accept params directly instead of reloading from checkpoint file
- Keep default `env_params` — they're fixed at initialization (grid size, max steps, reward structure) and don't change during training
- Prompt context (buffer stats, reference mazes) refreshed at **every injection event** via BufferStatsExtractor — always comprehensive, always current

#### Tuning Protocol
- **Start conservative, adjust**: Begin with `injection_interval=50`, `batch_size=4`. Run a 5k-step smoke test, check acceptance rate and solve rate stability
- **"Good enough" signal**: BOTH acceptance rate in 30-70% range AND no solve rate drops >0.05 post-injection
- **Both injection patterns configurable** for experiments:
  - Small frequent: `n_raw=4-8`, `injection_interval=50` (default, safe)
  - Larger infrequent: `n_raw=25`, `injection_interval=500` (experiment condition)
- Mutation amplification: `mutations_per_seed=30` → ~18-24 viable mutations per seed. One injection event with n_raw=25 replaces ~7-12% of buffer; n_raw=4-8 replaces ~1.5-4%

#### LLM Provider Strategy
- **Smoke testing / dev**: `claude-code` provider with Sonnet (fast, Max plan, no API cost)
- **Full experiment runs**: OpenRouter (one API key, multi-model access). Try OpenAI or Gemini models via pro subscriptions first. Ollama as free fallback if needed
- At interval=50 over 50k steps: ~20 injection events, ~160-240 API calls per run — very manageable cost via OpenRouter

#### Injection Stability
- **Warmup period**: Use `--llm_inject_start_step` to delay first injection (default: 1000 steps). Let agent learn basic navigation before injecting OOD mazes
- **On solve rate drops**: Log to WandB, keep going. Short-term instability is expected and part of what we're measuring. No auto-pause mechanism
- **"Works at 3k → works at 30k" guarantee**: System naturally adapts because policy params are always current, buffer stats are always live, and gate evaluates against current policy. Log trend metrics (acceptance rate over time, difficulty scores over time) to catch degradation early
- **Bulletproof smoke testing before paid runs**: Verify (1) gate actually filters, (2) injected levels appear in buffer and get replayed, (3) WandB metrics log correctly, (4) no crashes or silent failures over 5k steps, (5) solve rate curve looks sane

### Claude's Discretion
- Internal refactoring of AgentEvaluator to accept direct params vs file-based loading
- WandB metric naming beyond specified ones
- Exact smoke test script structure
- How to expose both injection patterns as CLI flags

### Deferred Ideas (OUT OF SCOPE)
- **Provider ablation**: Compare maze quality across Claude Sonnet, GPT-4o, Gemini, Ollama models — future experiment condition
- **Adaptive thresholds**: Auto-adjust gate thresholds based on rolling acceptance rate — complexity with uncertain payoff
- **Auto-pause on instability**: Skip injection events if solve rate drops too much — decided against for now, log and observe instead
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GATE-01 | Existing `DecisionGate.evaluate_candidate()` (fully implemented) is wired into `LLMInjectionManager` pipeline to filter every LLM maze before buffer insertion | `evaluate_candidate()` in `decision_gate.py` is ready to use. Wire via `generate_with_feedback()` in `_do_injection()`. Pass `DiversityThresholds(difficulty_threshold=0.6, min_diversity=0.02, diversity_metric="td_error_emd")` |
| GATE-02 | Existing `AgentEvaluator` (loads checkpoint once) is extended with a refresh mechanism to re-load current policy params at each injection event | Refactor: remove `load_agent()` from `__init__`, add `update_params(params)` method; `_build_rollout_fn()` must use instance params rather than closure-captured params |
| GATE-03 | WandB logs `llm/injected_count`, `llm/acceptance_rate`, `llm/diversity_score_mean`, `llm/retained_rate` at each injection step | `log_payload` dict in `_do_injection()` already has `llm/injected_count`, `llm/acceptance_rate`, `llm/retained_rate` — add `llm/diversity_score_mean` from `GateResult.summary["mean_diversity"]` |
| GATE-04 | Existing `PromptBuilder.build_generation_prompt()` is fed live buffer entropy stats computed from `train_state.sampler`, not hardcoded or stale values | `BufferStatsExtractor.extract_buffer_summary()` returns `mean_score`, `max_score`, `min_score`, `score_std` — convert these to `MetricEntry` objects with `metric_key="scalar_regret"` and pass as `global_metrics` argument to `build_generation_prompt()` |
</phase_requirements>

---

## Standard Stack

### Core (no new libraries — all already in codebase)

| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| `DecisionGate.evaluate_candidate()` | `llm/decision_gate.py:144` | Score and filter candidate mazes on difficulty+diversity | Fully implemented, no changes needed |
| `DiversityThresholds` | `llm/decision_gate.py:35` | Configuration dataclass for gate thresholds | Fully implemented |
| `AgentEvaluator` | `llm/agent_evaluator.py` | Runs agent rollouts on candidate mazes, returns trajectory dict | Needs param-refresh refactor |
| `MazeGenerator.generate_with_feedback()` | `llm/maze_generator.py:741` | Full feedback loop: generate → evaluate → gate → retry | Fully implemented |
| `BufferStatsExtractor` | `llm/buffer_stats.py` | Extracts summary stats and reference mazes from live sampler | Fully implemented |
| `build_generation_prompt()` | `llm/prompt_builder.py:362` | Builds LLM prompt with references and global_metrics | Accepts `global_metrics: List[MetricEntry]`, needs to receive live buffer stats |
| `LLMInjectionManager._do_injection()` | `llm/injector.py:249` | Orchestrates the injection pipeline | Needs gate wiring |
| `LLMInjectionConfig` | `llm/injection_config.py` | Config dataclass for injection parameters | Has placeholder gate fields — needs real connection |
| `TrainState` | `examples/maze_plr.py:45` | Flax train state with `.params` (network weights) and `.sampler` (PLR buffer) | Already passed to `maybe_inject()` via `runner_state` |

### Supporting

| Component | Location | Purpose | When to Use |
|-----------|----------|---------|-------------|
| `MetricEntry` | `llm/prompt_builder.py:19` | Single metric injected into LLM prompt | Convert buffer summary stats to `MetricEntry` for `global_metrics` |
| `GateResult` | `llm/decision_gate.py:69` | Result of gate evaluation including issues, summary, pair_metrics | Extract `summary["mean_diversity"]` for GATE-03 WandB metric |
| `wandb.log()` | `llm/injector.py:422` | Log metrics to WandB | Already in `_do_injection()` log_payload |

### No New Libraries Required

All needed functionality is implemented. Phase 2 is entirely wiring work within existing files. No pip installs needed.

---

## Architecture Patterns

### Recommended File Structure (changes only)

```
llm/
├── agent_evaluator.py       # MODIFY: add update_params(params), refactor _build_rollout_fn
├── injector.py              # MODIFY: wire gate, switch to generate_with_feedback, add global_metrics
├── injection_config.py      # MODIFY: add gate threshold fields with correct types, wire to DiversityThresholds
└── buffer_stats.py          # MINOR: add extract_global_metrics() helper returning List[MetricEntry]
```

### Pattern 1: AgentEvaluator Direct Param Passing

**What:** Remove `load_agent()` from `__init__`, accept `apply_fn` and initial `params` at construction, add `update_params(params)` to refresh the network weights before each rollout.

**Why:** Current `_build_rollout_fn()` captures `train_state` from `self.train_state` by closure. When `train_state.params` changes (every training step), the closed-over reference is stale. The refactor makes `rollout` read `self.params` dynamically at call time, or rebuilds the closure on each `update_params()` call.

**Key constraint:** `_build_rollout_fn(num_levels)` returns a `@jax.jit`-compiled function. JIT compilation happens on the first call for a given `num_levels`. If params change between calls, two options exist:
  - **Option A (simpler):** Store params as `self.params`; rebuild rollout function on each `update_params()` call (re-JIT compiles once per injection event — acceptable cost given injection is ~20x/50k run)
  - **Option B (faster):** Use `functools.partial` or pass params as argument to the JIT function (avoids re-JIT but requires signature change)

**Recommendation:** Option A — rebuild rollout fn on `update_params()`. Re-JIT cost is ~seconds once per injection event; JAX traces are cached by shape, not value, so this compiles once per num_levels value.

**Example pattern:**
```python
class AgentEvaluator:
    def __init__(self, apply_fn, params, env_params, num_steps=250, seed=42):
        """Accept apply_fn and initial params directly — no checkpoint loading."""
        self.apply_fn = apply_fn
        self.params = params
        self.env_params = env_params
        self.num_steps = num_steps
        self.rng = jax.random.PRNGKey(seed)
        self.eval_env = Maze(max_height=13, max_width=13,
                             agent_view_size=5, normalize_obs=True)
        self._rollout_fn = None  # rebuilt on update_params()
        self._rollout_fn_num_levels = None

    def update_params(self, params) -> None:
        """Refresh policy params. Call at each injection event.
        Forces rollout fn rebuild so next evaluate uses current policy."""
        self.params = params
        self._rollout_fn = None  # invalidate cached jit
        self._rollout_fn_num_levels = None

    def _build_rollout_fn(self, num_levels: int):
        """Rebuild with current self.params."""
        if self._rollout_fn is not None and self._rollout_fn_num_levels == num_levels:
            return self._rollout_fn
        # ... build @jax.jit rollout using self.apply_fn, self.params ...
        self._rollout_fn = rollout
        self._rollout_fn_num_levels = num_levels
        return rollout
```

### Pattern 2: Wiring generate_with_feedback() in _do_injection()

**What:** `LLMInjectionManager._do_injection()` currently calls `self.generator.generate()` for each seed. With gate enabled, switch to `self.generator.generate_with_feedback()` which:
1. Generates a valid maze
2. Runs `agent_evaluator.evaluate_level_multi_rollout()` (100 rollouts)
3. Calls `evaluate_candidate()` with `DiversityThresholds`
4. If rejected: builds `build_diversity_feedback_prompt()` and retries (up to 2 more times)
5. Returns `GenerationResult` with `.gate_result` attribute

**Key insight:** `generate_with_feedback()` expects `reference_trajectories` (list of trajectory dicts from the reference mazes). These must be computed at the start of `_do_injection()` by running `agent_evaluator.evaluate_level_multi_rollout()` on each reference level. This is the most expensive new operation (~100 rollouts × N_references = ~500 rollouts total per injection event).

**Call site pattern:**
```python
def _do_injection(self, runner_state: tuple) -> tuple:
    rng, train_state = runner_state

    # Refresh evaluator with current policy
    self.agent_evaluator.update_params(train_state.params)

    # Extract references (existing)
    references = self.buffer_stats.extract_references(train_state.sampler)
    buffer_summary = self.buffer_stats.extract_buffer_summary(train_state.sampler)

    # Compute reference trajectories for gate
    ref_trajectories = [
        self.agent_evaluator.evaluate_level_multi_rollout(ref_level, n_rollouts=100)
        for ref_level in reference_levels  # Level objects from sampler
    ]
    ref_labels = [ref.label for ref in references]

    # Build global_metrics from buffer_summary for prompt context (GATE-04)
    global_metrics = _build_global_metrics(buffer_summary)

    # Generate with gate feedback loop
    for i in range(seeds_generated):
        result = self.generator.generate_with_feedback(
            agent_evaluator=self.agent_evaluator,
            reference_trajectories=ref_trajectories,
            reference_labels=ref_labels,
            references=references,
            global_metrics=global_metrics,
            diversity_thresholds=self.diversity_thresholds,
            max_diversity_retries=2,
            n_rollouts=100,
        )
        # result.gate_result contains accepted/issues/summary
```

**Note:** Reference Level objects must be extracted from the sampler alongside the `ReferenceMaze` objects. `BufferStatsExtractor.extract_references()` currently returns `ReferenceMaze` (for prompt context) but not the raw `Level` objects. A companion method or extension is needed to also return Level objects for trajectory computation.

### Pattern 3: Live Buffer Stats as global_metrics (GATE-04)

**What:** Convert `BufferStatsExtractor.extract_buffer_summary()` output to `List[MetricEntry]` and pass as `global_metrics` to `build_generation_prompt()` via `generate_with_feedback()`.

**Example:**
```python
def _build_global_metrics(buffer_summary: dict) -> List[MetricEntry]:
    """Convert buffer summary stats to MetricEntry list for prompt context."""
    from llm.prompt_builder import MetricEntry
    return [
        MetricEntry(
            name="Buffer Mean Regret",
            value=buffer_summary["mean_score"],
            description="Mean regret score across all active buffer levels",
            higher_is="more challenging curriculum",
            metric_key="scalar_regret",
        ),
        MetricEntry(
            name="Buffer Max Regret",
            value=buffer_summary["max_score"],
            description="Highest regret score in buffer (hardest level for agent)",
            higher_is="harder top level",
            metric_key="scalar_regret",
        ),
        MetricEntry(
            name="Buffer Size",
            value=buffer_summary["buffer_size"],
            description="Number of active levels in the PLR replay buffer",
            metric_key="",
        ),
    ]
```

### Pattern 4: LLMInjectionConfig Gate Field Wiring

**What:** `injection_config.py` has placeholder gate fields (`gate_enabled: bool = False`, `diversity_threshold: float = 0.1`, `min_difficulty: float = 0.1`). These need to be connected to the real `DiversityThresholds` dataclass fields from `decision_gate.py`.

**Locked gate config:** `difficulty_threshold=0.6`, `min_diversity=0.02`, `diversity_metric="td_error_emd"`. These should be CLI-configurable for the tuning protocol.

**Needed additions to `LLMInjectionConfig`:**
```python
# Gate configuration (Phase 2)
gate_enabled: bool = True              # --llm_gate (default on in Phase 2)
difficulty_threshold: float = 0.6     # --llm_difficulty_threshold
min_diversity: float = 0.02           # --llm_min_diversity
diversity_metric: str = "td_error_emd"  # --llm_diversity_metric
max_diversity_retries: int = 2        # --llm_max_diversity_retries
n_rollouts_gate: int = 100            # --llm_n_rollouts (for gate evaluation)
```

**Corresponding `from_config_dict()` additions:**
```python
gate_enabled=config.get("llm_gate", True),
difficulty_threshold=config.get("llm_difficulty_threshold", 0.6),
min_diversity=config.get("llm_min_diversity", 0.02),
diversity_metric=config.get("llm_diversity_metric", "td_error_emd"),
```

### Pattern 5: WandB Metric Additions (GATE-03)

**What:** Current `log_payload` in `_do_injection()` has `llm/acceptance_rate`, `llm/injected_count`, `llm/retained_rate`. Missing for GATE-03:
- `llm/diversity_score_mean` — mean of `GateResult.summary["mean_diversity"]` across accepted candidates
- `llm/batch_all_rejected_count` — running count of injection events where all candidates were rejected

**Example addition to log_payload:**
```python
log_payload.update({
    "llm/diversity_score_mean": np.mean(diversity_scores) if diversity_scores else 0.0,
    "llm/difficulty_score_mean": np.mean(difficulty_scores) if difficulty_scores else 0.0,
    "llm/gate_rejection_rate": gate_rejected / seeds_generated if seeds_generated > 0 else 0.0,
    "llm/batch_all_rejected_count": self.batch_all_rejected_count,
})
```

### Anti-Patterns to Avoid

- **Rebuilding the JIT function on every call (not just on `update_params()`):** `_build_rollout_fn()` currently rebuilds if `num_levels` changes. That is correct. Do not rebuild on every `evaluate_level()` call — that would re-JIT on every maze evaluation, which is extremely slow.
- **Loading checkpoint from disk at injection time:** The whole point of GATE-02 is to avoid file I/O. Never add `load_agent()` back into the injection path.
- **Computing reference trajectories inside `generate_with_feedback()` loop:** Reference trajectories should be computed ONCE at the start of `_do_injection()`, not re-computed for each of the N_raw candidates. This is 5 references × 100 rollouts = 500 rollouts total per event, regardless of N_raw.
- **Using `gate_enabled: bool` from old Phase 1 placeholder without renaming fields:** The old placeholder fields in `LLMInjectionConfig` have wrong defaults/types. Replace them cleanly rather than patching.
- **Passing stale buffer stats to prompt:** Buffer stats must be extracted at the start of `_do_injection()` from the current `train_state.sampler`, not cached from a previous event.
- **Ignoring `batch_all_rejected_count` WandB metric:** The CONTEXT.md explicitly requests this to empirically verify whether full-batch rejection is a real problem. Do not skip it.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gate logic | Custom threshold checking code | `evaluate_candidate()` in `decision_gate.py` | Already handles difficulty+diversity, CENIE, all metrics, returns structured `GateResult` |
| LLM retry on rejection | Custom retry loop | `generate_with_feedback()` in `maze_generator.py:741` | Already implements conversation history, diversity feedback prompt, up to N retries |
| Diversity feedback prompt | Custom prompt string | `build_diversity_feedback_prompt()` in `prompt_builder.py:477` | Handles path overlays, similarity issues, analysis sections |
| Agent rollout batching | Custom JAX vmap loop | `evaluate_level_multi_rollout()` in `agent_evaluator.py:153` | Already computes solve_rate, best_return, all_returns across N rollouts |
| Buffer stats extraction | Direct sampler dict access | `BufferStatsExtractor.extract_buffer_summary()` | Already handles edge cases (empty buffer, size=0) |
| Reference trajectory computation | File-based `.npz` loading | `AgentEvaluator.evaluate_levels()` on Level objects from live sampler | Already batched, handles Level.stack() properly |

**Key insight:** The friend's codebase has already solved every hard problem in this phase. The task is integration, not invention.

---

## Common Pitfalls

### Pitfall 1: JIT Recompilation on Param Change

**What goes wrong:** After `update_params()`, if `_build_rollout_fn()` captures `params` by value inside `@jax.jit`, the old compiled function still uses old params. The new maze gets evaluated against a stale policy.

**Why it happens:** JAX JIT compiles closures by tracing; the compiled function captures the array value at trace time, not a reference.

**How to avoid:** After `update_params()`, invalidate the cached rollout fn (`self._rollout_fn = None`). On next `evaluate_level()` call, `_build_rollout_fn()` rebuilds using `self.params` at that time. The JIT traces again (once), then subsequent calls for the same `num_levels` use the newly traced version.

**Warning signs:** Acceptance rate never changes despite policy improving significantly (gate always sees same difficulty estimate).

### Pitfall 2: Reference Trajectory Computation is Expensive

**What goes wrong:** Computing 5 reference trajectories × 100 rollouts each = 500 rollouts at the start of `_do_injection()`. At injection_interval=50, this happens every 50 steps. This is a ~2-5 second overhead per injection event.

**Why it happens:** `evaluate_level_multi_rollout()` is not parallelized across different levels — it does batched rollouts for one level at a time.

**How to avoid:** Use `AgentEvaluator.evaluate_levels()` if all reference levels can be batched in one call. Check whether reference trajectories can be computed in a single `_evaluate_batch()` call for all N_references simultaneously (this would reduce it to ~100 rollouts total rather than 100 × N_references).

**Warning signs:** Injection events taking >30 seconds. Check with logging the time per step in `_do_injection()`.

### Pitfall 3: Stale Reference Levels When Extracting Trajectories

**What goes wrong:** `BufferStatsExtractor.extract_references()` returns `ReferenceMaze` objects (for prompt building) but NOT the raw `Level` objects needed to run agent rollouts. If you try to reconstruct Level objects from the ASCII grid strings inside `ReferenceMaze`, you lose information (agent_dir, exact integer positions).

**Why it happens:** `extract_references()` was designed for prompt building, not for trajectory computation.

**How to avoid:** Extend `BufferStatsExtractor` (or add a companion method) to also return the raw `Level` pytree objects alongside `ReferenceMaze` objects. Alternatively, return both in a tuple: `(references, ref_levels)`. Both are extracted from the same sampler in one pass.

**Warning signs:** Gate diversity scores are all near zero (reference trajectories built from reconstructed levels that differ from original).

### Pitfall 4: AgentEvaluator __init__ Signature Break

**What goes wrong:** `AgentEvaluator.__init__()` currently takes `checkpoint_dir` and `checkpoint_step`. Phase 2 changes this to accept `apply_fn` and `params`. Any existing code (smoke tests, `test_generator.py`) that creates `AgentEvaluator(checkpoint_dir=...)` will break.

**Why it happens:** The refactor changes the public API of `AgentEvaluator`.

**How to avoid:** Two options:
  - Keep old constructor as a `@classmethod AgentEvaluator.from_checkpoint(checkpoint_dir, ...)` — preserves backward compat for `test_generator.py`
  - Or add `params` as an optional arg; if provided, skip `load_agent()`

Recommendation: Keep `from_checkpoint()` classmethod for test_generator.py compatibility, new primary constructor takes `apply_fn, params, env_params` directly.

**Warning signs:** Import errors in test_generator.py or any existing standalone usage.

### Pitfall 5: Gate Threshold Calibration — 0.6 May Be Too Aggressive

**What goes wrong:** `difficulty_threshold=0.6` was calibrated against offline buffer samples. At step 1k-5k, the live agent may produce low regret on all mazes (agent is bad, solves nothing, regret is always 0 because value estimates are near 0). The gate rejects everything.

**Why it happens:** Regret formula is `max_return - V(s_t)`. If `max_return=0` (agent never solves) and `V(s_t)=0` (random init), regret is 0. Gate sees difficulty=0 < 0.6, rejects.

**How to avoid:** The smoke test at 5k steps will reveal this. The tuning protocol is exactly designed to catch it. Note in STATE.md: "DiversityThresholds(difficulty_threshold=0.3, min_diversity=0.04) defaults were not tuned for live training" — this is a known concern. Start with threshold=0.6 per locked decision, but the smoke test will show whether to adjust.

**Warning signs:** `llm/acceptance_rate` is 0% in all smoke test events. Check `llm/difficulty_score_mean` — if near 0, the threshold is too high for current training stage.

### Pitfall 6: generate_with_feedback() Already Handles Feedback Loop — Don't Duplicate

**What goes wrong:** Developer reads `_do_injection()` and its inner loop, and adds a custom retry loop around `generate()` to handle gate rejections, duplicating what `generate_with_feedback()` already does.

**Why it happens:** The Phase 1 code calls `generate()` in a loop. It's tempting to add gate logic into that same loop.

**How to avoid:** Replace the per-seed `generate()` call with a single `generate_with_feedback()` call. The method internally handles: initial generation, agent rollout, gate evaluation, diversity feedback prompt, and retries.

---

## Code Examples

### Example 1: AgentEvaluator from_checkpoint classmethod (backward compat)

Source: Analysis of existing `agent_evaluator.py:37-82` — refactor pattern

```python
@classmethod
def from_checkpoint(cls, checkpoint_dir: str, checkpoint_step: int = -1,
                    num_steps: int = 250, seed: int = 42) -> "AgentEvaluator":
    """Load agent from checkpoint directory (backward compat for test_generator.py)."""
    from cross_evaluate import load_agent
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    train_state, config, env, env_params = load_agent(checkpoint_dir, checkpoint_step)
    if train_state is None:
        raise RuntimeError(f"Failed to load agent from {checkpoint_dir}")
    return cls(
        apply_fn=train_state.apply_fn,
        params=train_state.params,
        env_params=env_params,
        num_steps=num_steps,
        seed=seed,
    )
```

### Example 2: update_params() and cached rollout fn invalidation

Source: Analysis of existing `_build_rollout_fn` in `agent_evaluator.py:84-135`

```python
def update_params(self, params) -> None:
    """Refresh policy params at each injection event.

    Invalidates the JIT-compiled rollout function so the next
    evaluate_level() call retraces with current params.
    """
    self.params = params
    self._rollout_fn = None
    self._rollout_fn_num_levels = None

def _build_rollout_fn(self, num_levels: int):
    """Build JIT-compiled rollout for current self.params."""
    if (self._rollout_fn is not None and
            self._rollout_fn_num_levels == num_levels):
        return self._rollout_fn

    apply_fn = self.apply_fn
    params = self.params  # snapshot current params into closure
    eval_env = self.eval_env
    env_params = self.env_params
    num_steps = self.num_steps

    @jax.jit
    def rollout(rng, levels):
        # ... same body as current, using params from closure ...
        pass

    self._rollout_fn = rollout
    self._rollout_fn_num_levels = num_levels
    return rollout
```

### Example 3: BufferStatsExtractor extension to return Level objects

Source: Analysis of `buffer_stats.py:35-95`

```python
def extract_references_with_levels(
    self, sampler: dict
) -> Tuple[List[ReferenceMaze], List]:
    """Return both ReferenceMaze objects (for prompt) and Level objects (for rollouts).

    Returns:
        (references, levels) where levels is a list of Level pytree objects
        matching the indices selected for references.
    """
    size = int(np.asarray(sampler["size"]))
    if size == 0:
        return [], []

    scores = np.asarray(sampler["scores"])[:size]
    levels_pytree = sampler["levels"]

    # ... same selection logic as extract_references() ...
    selected_indices = ...

    references = []
    ref_levels = []
    for i, idx in enumerate(selected_indices):
        level = jax.tree_util.tree_map(lambda x: x[idx], levels_pytree)
        ascii_grid = level.to_str()
        # ... build ReferenceMaze as before ...
        references.append(ref)
        ref_levels.append(level)  # raw Level object for trajectory computation

    return references, ref_levels
```

### Example 4: Wired _do_injection() skeleton (Phase 2)

Source: Synthesized from `llm/injector.py:249`, `generate_with_feedback()` signature at `maze_generator.py:741`

```python
def _do_injection(self, runner_state: tuple) -> tuple:
    rng, train_state = runner_state

    # Refresh evaluator with current policy params (GATE-02)
    self.agent_evaluator.update_params(train_state.params)

    # Extract references AND Level objects in one pass (GATE-04 + GATE-01)
    references, ref_levels = self.buffer_stats.extract_references_with_levels(
        train_state.sampler
    )
    buffer_summary = self.buffer_stats.extract_buffer_summary(train_state.sampler)

    # Compute reference trajectories for gate (expensive: N_refs × 100 rollouts)
    ref_trajectories = []
    ref_labels = []
    for ref, level in zip(references, ref_levels):
        traj = self.agent_evaluator.evaluate_level_multi_rollout(level, n_rollouts=100)
        ref_trajectories.append(traj)
        ref_labels.append(ref.label)

    # Build global_metrics from live buffer summary (GATE-04)
    global_metrics = _build_global_metrics(buffer_summary)

    # Build DiversityThresholds from config
    from llm.decision_gate import DiversityThresholds
    thresholds = DiversityThresholds(
        difficulty_threshold=self.config.difficulty_threshold,
        min_diversity=self.config.min_diversity,
        diversity_metric=self.config.diversity_metric,
    )

    # Generate N_raw candidates with gate feedback loop (GATE-01)
    valid_levels = []
    diversity_scores = []
    gate_accepted = 0
    gate_rejected = 0

    for i in range(self.config.n_raw):
        result = self.generator.generate_with_feedback(
            agent_evaluator=self.agent_evaluator,
            reference_trajectories=ref_trajectories,
            reference_labels=ref_labels,
            references=references,
            global_metrics=global_metrics,
            diversity_thresholds=thresholds,
            max_diversity_retries=self.config.max_diversity_retries,
            n_rollouts=self.config.n_rollouts_gate,
        )
        if result.success:
            valid_levels.append(result.level)
            gate_accepted += 1
            if hasattr(result, 'gate_result') and result.gate_result:
                d = result.gate_result.summary.get("mean_diversity", 0.0)
                diversity_scores.append(d)
        else:
            gate_rejected += 1

    # Track batch_all_rejected events
    if gate_accepted == 0 and self.config.n_raw > 0:
        self.batch_all_rejected_count += 1

    # ... rest of injection (mutation amplification, buffer insert) unchanged from Phase 1 ...

    # WandB logging (GATE-03)
    log_payload.update({
        "llm/diversity_score_mean": float(np.mean(diversity_scores)) if diversity_scores else 0.0,
        "llm/gate_rejection_rate": gate_rejected / self.config.n_raw if self.config.n_raw > 0 else 0.0,
        "llm/batch_all_rejected_count": self.batch_all_rejected_count,
    })
```

---

## State of the Art

| Old Approach (Phase 1) | Current Approach (Phase 2) | Impact |
|------------------------|---------------------------|--------|
| Unconditional injection (format-valid → insert) | Gated injection (difficulty + diversity filter before insert) | Prevents OOD-but-trivial mazes from polluting buffer |
| AgentEvaluator unused in training loop | AgentEvaluator with live params at each injection event | Gate always evaluates against current policy capability |
| `generate()` — single attempt per seed | `generate_with_feedback()` — up to 3 LLM attempts with rejection reasons | LLM can correct its output when first attempt fails diversity check |
| Buffer stats not in prompt | `global_metrics` with live regret mean/max/std | LLM understands current curriculum difficulty before generating |
| `gate_enabled: False` in `LLMInjectionConfig` | `gate_enabled: True`, wired to `DiversityThresholds` | Config drives real gate behavior |

---

## Open Questions

1. **Reference trajectory batching for performance**
   - What we know: `evaluate_levels()` in `agent_evaluator.py:208` does a single batched `_evaluate_batch()` call for multiple levels. `evaluate_level_multi_rollout()` runs N_rollouts for one level at a time.
   - What's unclear: Can we run multi-rollout evaluation for all N_references simultaneously in one `_evaluate_batch()` call? (e.g., `N_references × N_rollouts` levels in one vmap). This would be faster but requires building `Level.stack([level] * n_rollouts for each ref)`.
   - Recommendation: Start with sequential per-reference evaluation (simpler). If injection events take >20s, batch all reference rollouts in one call.

2. **AgentEvaluator `env_params` initialization**
   - What we know: Current `__init__` rebuilds `eval_env = Maze(...)` and reassigns `self.env_params = self.eval_env.default_params` — ignoring the env_params from `load_agent()`. This is already a bug in Phase 1 code.
   - What's unclear: Whether the maze constructor params (max_height=13, agent_view_size=5, normalize_obs=True) are always correct for all training configurations.
   - Recommendation: In the new `__init__`, accept `env_params` explicitly from caller. `LLMInjectionManager` should pass `env_params` from the training loop's `env_params` variable (already available at injection setup time).

3. **`generate_with_feedback()` expects `reference_trajectories` with specific keys**
   - What we know: The method calls `evaluate_candidate(candidate_traj, reference_trajectories, ...)`. `evaluate_candidate()` uses `ref["dones"]`, `ref["values"]`, `ref["positions"]`. The `td_error_divergence()` function needs specific keys including `dones`.
   - What's unclear: Does `evaluate_level_multi_rollout()` return all required keys? From `agent_evaluator.py:194-205`: yes — returns `dones`, `values`, `positions`, `rewards`, `entropy`, `hstates`, `best_return`, `solve_rate`.
   - Recommendation: No action needed; keys are already compatible.

4. **Smoke test script structure**
   - What we know: CONTEXT.md says "exact smoke test script structure" is Claude's discretion.
   - What's unclear: Whether a standalone script or an existing example script modification is preferred.
   - Recommendation: A standalone `scripts/smoke_test_llm_gate.sh` that calls `maze_plr.py` with `--use_llm --num_env_steps 5000 --llm_inject_interval 50 --llm_batch_size 4 --llm_provider claude-code --llm_gate`. Output is inspected manually. Simple and requires no new Python code.

---

## Sources

### Primary (HIGH confidence)

All findings are directly verified from the project codebase:

- `llm/decision_gate.py` — `DiversityThresholds`, `GateResult`, `evaluate_candidate()` full implementation
- `llm/agent_evaluator.py` — `AgentEvaluator.__init__()`, `_build_rollout_fn()`, `evaluate_level_multi_rollout()` full implementation
- `llm/maze_generator.py:741-1086` — `generate_with_feedback()` full implementation including conversation history, gate retry loop, `build_diversity_feedback_prompt()`
- `llm/injector.py` — `LLMInjectionManager._do_injection()` Phase 1 implementation to be extended
- `llm/buffer_stats.py` — `BufferStatsExtractor.extract_references()` and `extract_buffer_summary()`
- `llm/prompt_builder.py` — `MetricEntry`, `ReferenceMaze`, `build_generation_prompt()` signatures
- `llm/injection_config.py` — `LLMInjectionConfig` with existing placeholder gate fields
- `llm/config.yaml` — calibrated gate thresholds: `difficulty_threshold: 0.6`, `min_diversity: 0.02`
- `examples/maze_plr.py:45-56` — `TrainState` dataclass structure confirming `.params` and `.sampler` fields
- `.planning/STATE.md` — "Gate threshold calibration needed — DiversityThresholds defaults were not tuned for live training"

### Secondary (MEDIUM confidence)

- CONTEXT.md decisions (user-defined) — gate thresholds, retry count, smoke test protocol, WandB metric names

### Tertiary (LOW confidence)

None — no external sources required; all claims verified from codebase.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — everything is already in the codebase, verified by reading each file
- Architecture: HIGH — refactor patterns derive directly from existing class structure
- Pitfalls: HIGH — derived from concrete code analysis (JIT closure semantics, API signature breaks, regret=0 at training start)
- Gate thresholds: MEDIUM — user decisions from CONTEXT.md, empirical validation required in smoke test

**Research date:** 2026-03-23
**Valid until:** Phase 2 complete (thresholds may need adjustment after smoke test)
