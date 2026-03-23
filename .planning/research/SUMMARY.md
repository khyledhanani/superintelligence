# Project Research Summary

**Project:** LLM Injection into JAX/Flax UED Training Pipeline
**Domain:** LLM-augmented Unsupervised Environment Design (UED) — periodic maze injection into ACCEL/PLR curriculum learning
**Researched:** 2026-03-23
**Confidence:** HIGH

## Executive Summary

This project integrates a pre-built LLM maze generation subsystem (`llm/`) into a mature JAX/Flax reinforcement learning training pipeline (`examples/maze_plr.py`). The LLM component (MazeGenerator, PromptBuilder, AgentEvaluator, DecisionGate) is already fully implemented and tested standalone. The training pipeline is proven across 50k-step experiments. The work is a targeted bridge between these two complete systems, not a greenfield build. The core thesis claim is that LLM-generated seed levels — filtered by a behavioral diversity gate — improve generalization beyond what ACCEL mutation alone achieves.

The recommended approach is a synchronous Python hook injected between JAX eval steps (outside `jax.lax.scan`), calling the existing `LLMInjector` orchestration class. Accepted levels enter the buffer via `LevelSampler.insert_batch()` with regret-derived scores, making them indistinguishable from buffer entries, and naturally subject to ACCEL mutation on subsequent replay. This requires no new dependencies — the entire integration runs on the installed `jax_env` conda environment using `requests`, `subprocess`, `threading`, and existing local modules. The only new files are `llm/injector.py` (orchestration) and `llm/buffer_stats.py` (buffer-to-prompt conversion).

The dominant risk is not technical complexity but experimental rigor: three pitfalls can silently corrupt results without visible errors — incorrect score initialization causing immediate buffer eviction of LLM levels, a stale AgentEvaluator evaluating candidates against an outdated policy, and lack of level caching making comparison experiments irreproducible. All three must be addressed before running comparison experiments. A secondary risk is training instability from injecting structurally novel OOD levels too aggressively; the mitigation is a warmup period and small batch sizes (4-8 levels per injection event).

---

## Key Findings

### Recommended Stack

The integration requires zero new package installations. The `jax_env` conda environment already contains every needed library: `requests 2.32.5` for OpenRouter HTTP calls, `subprocess` stdlib for claude CLI invocation, `threading` stdlib for background generation, `jax 0.5.3` and `numpy 2.2.6` for buffer operations, `wandb 0.24.2` for logging, and `python-dotenv 1.2.1` for API key management. The LLM-to-buffer pipeline reuses `Level.from_str()` (text-to-pytree conversion), `Level.stack()` (batching), and `LevelSampler.insert_batch()` (eviction-aware insertion) — all already verified working.

**Core technologies:**
- `subprocess` + claude CLI: LLM invocation via existing `_call_claude_code()` — no API key management, no new dependency
- `requests`: OpenRouter HTTP calls via existing `_call_openai_compatible()` — provider switchable without code changes
- `threading.Thread`: Background LLM generation — LLM latency (~5-60s) far exceeds eval step time (~500ms/interval), making non-blocking generation essential
- `LevelSampler.insert_batch()`: Buffer injection with eviction — accepts batched Level pytree + scores, handles slot eviction automatically
- `AgentEvaluator`: Compute regret/learnability score for injected levels — reuse as-is, refresh checkpoint at each injection event

Do NOT add: `anthropic` SDK (not installed, CLI sufficient), `asyncio` (synchronous injection at eval boundary is simpler and correct), or any separate LLM microservice (adds IPC complexity for no gain).

### Expected Features

**Must have (P1 — required for thesis comparison experiment):**
- Periodic injection hook at `eval_step` boundary in `maze_plr.py` — the entire experiment depends on this
- Configurable `--llm_inject_interval N` and `--llm_batch_size M` CLI args — ablation requires these as first-class parameters
- Buffer statistics extraction: top-K mazes by regret formatted as `ReferenceMaze[]` for LLM prompt context
- Agent checkpoint access mid-training — gate evaluation requires the live policy, not a startup checkpoint
- Decision gate integration (td_error_emd + difficulty filter) — the behavioral diversity gate is the thesis differentiator
- `insert_batch()` for accepted mazes with regret-derived scores — incorrect score initialization silently breaks retention
- WandB logging: `llm/injected_count`, `llm/acceptance_rate`, `llm/diversity_score_mean`, `llm/retained_rate` — without these, the experiment produces no evidence
- Two comparison launch scripts: ACCEL+LLM and ACCEL-only control

**Should have (P2 — add after v1 validates concept):**
- Metric-informed prompts using live buffer entropy + path overlay — adds LLM context quality when acceptance rate is too low
- Diversity feedback loop (LLM retries with specific rejection reasons) — add when WandB shows high rejection rate
- Injection timing ablation via `--llm_inject_start_step` — add when v1 shows timing effects
- Reference maze selection strategy ablation (`diverse` vs `top_regret`)
- Level caching to disk for reproducibility — required before reporting comparison results

**Defer (v2+):**
- CENIE novelty gate — sophisticated but adds ~5-10s per injection; justified only if td_error_emd gate misses behavioral redundancy
- Direct `train_state.params` passing to evaluator (eliminates file I/O) — performance optimization, not correctness
- Online prompt optimization — meta-learning for prompts, out of scope for thesis

### Architecture Approach

The architecture is a clean Python hook between JIT boundaries. The outer Python `for eval_step` loop in `maze_plr.py` is the only valid injection point — `jax.lax.scan` compiles its body to XLA and cannot accommodate Python-side HTTP calls. The injection pipeline flows: extract buffer stats (D2H transfer of top-K levels) → evaluate reference trajectories via `AgentEvaluator` → call LLM via `MazeGenerator` → filter via `DecisionGate` → insert accepted `Level` objects into sampler via `insert_batch()` → replace `runner_state` with updated `train_state`. `Level` is the universal currency across all components — LLM text, VAE latents, and ACCEL mutations all produce `Level` objects, making them interchangeable from the buffer's perspective.

**Major components:**
1. `LLMInjector` (NEW: `llm/injector.py`) — orchestrates injection scheduling, calls to MazeGenerator/AgentEvaluator/DecisionGate, and writes back to the buffer; keeps `maze_plr.py` clean (one method call)
2. `BufferStatsExtractor` (NEW: `llm/buffer_stats.py`) — converts JAX sampler dict to Python metrics and `ReferenceMaze[]` objects for prompt building
3. `maze_plr.py` (MODIFY) — add `--use_llm`/`--llm_interval` args, instantiate `LLMInjector` before the for-loop, call `injector.maybe_inject(runner_state, eval_step)` inside the loop
4. `MazeGenerator` (existing, no changes) — LLM API calls, text-to-Level parsing, feedback retry loop
5. `DecisionGate` + `AgentEvaluator` (existing, no changes) — behavioral diversity filtering with refreshed checkpoint at each injection event

### Critical Pitfalls

1. **JAX JIT boundary violation** — calling LLM generation inside `lax.scan` or `train_step` causes silent no-ops or trace errors; LLM code must live in the Python `for` loop, not the compiled inner loop. Verify with WandB injection counter.

2. **Buffer score initialization trap** — inserting LLM levels with `score=0.0` at a full buffer causes immediate eviction; use the gate's computed regret as insertion score. Verify with `llm_retained_rate > 50%` after 1000 post-injection steps.

3. **Maze format validation gaps** — `Level.is_well_formatted()` does not catch missing border walls, trivially solvable mazes (goal adjacent to agent), or dtype mismatches causing XLA retracing; add `validate_llm_level()` wrapper with border wall check + BFS path length > 5 + dtype assertions.

4. **Stale AgentEvaluator** — constructing `AgentEvaluator` once at training start and reusing it means the gate evaluates all candidates against the step-0 policy; refresh by loading the latest checkpoint at each injection event (add ~1-2s overhead, acceptable).

5. **Irreproducible experiments from stochastic LLM** — without caching, two runs with the same JAX seed diverge after the first injection event and cannot be compared; cache accepted levels as `.npy` + metadata JSON at injection time and log wall_map hashes to WandB.

---

## Implications for Roadmap

Based on the combined research, the component dependency graph from ARCHITECTURE.md maps directly to a 4-phase roadmap. Each phase produces independently verifiable deliverables and avoids accumulating undetected failures.

### Phase 1: Integration Scaffolding and Format Validation

**Rationale:** `Level` format correctness is the critical path — it blocks every downstream component (gate evaluation, buffer insertion, training stability). Establishing the Python/JAX boundary explicitly prevents the most catastrophic pitfall (silent no-ops from JIT violations). This phase has no external dependencies and delivers testable outputs in isolation.

**Delivers:** Working `LLMInjector` skeleton with disabled gate (inject every N steps unconditionally), `validate_llm_level()` with border wall + path length + dtype checks, `BufferStatsExtractor` tested against a live buffer, WandB logging skeleton. Verified: injection counter increments, `llm_retained_rate > 50%` with fake levels.

**Addresses features:** Periodic injection hook, configurable `--llm_inject_interval`/`--llm_batch_size`, maze format conversion, buffer insert with correct score initialization, WandB logging skeleton.

**Avoids pitfalls:** JAX JIT boundary violation (Phase 1 explicitly establishes the boundary), buffer eviction trap (correct score initialization from day one), maze format crashes (validate before any gate or buffer call), AgentEvaluator sys.path fragility (CI test from non-root directory).

**Research flag:** Standard patterns — no additional research needed. All APIs verified in existing codebase.

---

### Phase 2: Decision Gate Integration and Injection Frequency Tuning

**Rationale:** Once format validation is solid, wire the live `AgentEvaluator` + `DecisionGate` into the injection pipeline. This phase requires care around checkpoint staleness and OOD instability — both must be verified before running reportable experiments. Gate thresholds and injection frequency are first-class experimental parameters that need empirical tuning.

**Delivers:** Full gate-filtered injection: `AgentEvaluator` refreshed per injection event, `DecisionGate.evaluate_candidate()` called on each LLM candidate, accepted levels inserted with gate-computed regret score. Verified: acceptance rate < 100% (gate is active), solve rate does not drop > 0.1 within 500 post-injection steps, gate evaluation timestamp delta < 1 injection interval.

**Addresses features:** Decision gate integration (td_error_emd + difficulty), agent checkpoint access mid-training, injection frequency tuning (`--llm_batch_size` ablation in [4, 8, 16]), warmup period via `--llm_inject_start_step`.

**Avoids pitfalls:** Stale AgentEvaluator (refresh checkpoint per injection), OOD injection instability (warmup period + batch size limit of 4-8), per-level WandB logging overhead (one `wandb.log` dict per injection event).

**Research flag:** Needs empirical tuning. The optimal `--llm_inject_interval` and `--llm_batch_size` cannot be determined analytically — run a 3×3 grid (interval in [50, 100, 200] eval steps × batch size in [4, 8, 16]) at 10k steps before committing to experiment hyperparameters.

---

### Phase 3: Experiment Infrastructure and Reproducibility

**Rationale:** Comparison experiments require reproducibility before any results can be reported. Level caching and hash logging must be in place before running the ACCEL vs ACCEL+LLM comparison. This phase also produces the launch scripts and comparison analysis tooling that make the thesis experiment executable.

**Delivers:** Level cache (accepted levels serialized as `.npy` + metadata JSON), wall_map hash logging to WandB, two comparison launch scripts (`launch_llm_injection.sh`, `launch_accel_only_control.sh`), updated `scripts/compare_llm_injection.py`. Verified: re-run with same seed produces identical WandB curves up to first injection event; injected level hashes appear in WandB.

**Addresses features:** Seed-controlled reproducibility, level caching, comparison experiment launch scripts.

**Avoids pitfalls:** Irreproducible experiments (caching before any reportable run), comparison validity (identical seeds, buffer sizes, eval sets across conditions).

**Research flag:** Standard patterns — caching and launch script patterns are well-established in existing codebase.

---

### Phase 4: Comparison Experiments and Thesis Results

**Rationale:** With infrastructure verified (Phase 1-3), run the actual thesis comparison: ACCEL-only control vs ACCEL+LLM injection. Analyse results for evidence of the core claim (LLM injection improves generalization beyond ACCEL alone). Add P2 features (metric-informed prompts, diversity feedback) if v1 shows subthreshold effect.

**Delivers:** 3-seed ACCEL+LLM runs vs 3-seed ACCEL-only control (matching existing 50k experiment structure), WandB comparison table, statistical analysis of solve rate differences, ablation on injection frequency if needed.

**Addresses features:** Comparison experiments, ACCEL mutation of LLM-injected mazes (emergent, no code change), injection timing analysis (add `--llm_inject_start_step` if Phase 3 data suggests timing effect).

**Uses:** All Phase 1-3 infrastructure; `wandb` for multi-run comparison; existing `scripts/compare_phase4_results.py` pattern.

**Research flag:** May need phase-specific research if results are mixed. If acceptance rate is consistently low, investigate metric-informed prompts (P2). If injected levels cluster structurally, investigate structural diversity in batch generation.

---

### Phase Ordering Rationale

- **Phase 1 before Phase 2:** Format validation must be verified before any gate evaluation — invalid levels passed to `AgentEvaluator` cause JAX crashes, not graceful errors.
- **Phase 2 before Phase 3:** Injection hyperparameters (interval, batch size, warmup) must be empirically tuned before locking them into comparison launch scripts.
- **Phase 3 before Phase 4:** Level caching and hash logging are prerequisites for reportable results — running comparison experiments without them produces irreproducible data.
- **Phases 1-2 independent of 50k GPU experiments:** Phases 1-2 can run on a single GPU with 5k-step validation runs. No need to wait for ongoing 50k experiments to complete.

### Research Flags

Needs deeper research during planning:
- **Phase 2:** Injection hyperparameter tuning — empirical grid search required; no analytical answer exists
- **Phase 4:** If LLM acceptance rate < 20% consistently — investigate whether metric-informed prompts (P2 features) are needed for thesis viability

Standard patterns (skip research-phase):
- **Phase 1:** All APIs verified from source; Level format conversion fully implemented
- **Phase 3:** Level caching + launch script patterns match existing `examples/launch_*.sh` and `scripts/` conventions

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All recommendations derived from direct source code inspection. No speculative claims. Zero new dependencies identified. |
| Features | HIGH | Feature analysis grounded in existing codebase + PROJECT.md. MVP scope is clear and tightly bounded. |
| Architecture | HIGH | Component boundaries and data flow verified by reading all relevant source files. No ambiguous integration points. |
| Pitfalls | HIGH | All pitfalls derived from actual code inspection of eviction logic, JIT tracing semantics, and evaluator construction patterns — not speculation. |

**Overall confidence:** HIGH

### Gaps to Address

- **Threading vs synchronous injection:** The STACK.md identifies this as the only genuine open question. LLM latency (~5-60s) almost certainly exceeds eval interval (~500ms), making background threading advisable. Measure actual latency on albacore/smew setup in Phase 1 before deciding.
- **Gate threshold calibration:** `DiversityThresholds(difficulty_threshold=0.3, min_diversity=0.04)` values are defaults from the standalone `llm/` module. They were not tuned for the live training regime. May need adjustment in Phase 2 to achieve a useful acceptance rate (target: 30-70%).
- **Buffer contamination monitoring:** No existing metric tracks LLM level retention post-injection. `llm_retained_rate` must be explicitly computed and logged as a new WandB metric in Phase 1 — it is not a free metric.
- **Baseline for comparison:** The ACCEL-only control condition uses the same launch scripts as existing `accel-baseline` 50k experiments. Confirm whether the 50k `accel-baseline` results (already running) are sufficient as the control condition, or whether fresh control runs alongside LLM injection runs are needed for identical experimental conditions.

---

## Sources

### Primary (HIGH confidence — direct codebase inspection)

- `llm/maze_generator.py` — MazeGenerator, provider implementations, GenerationConfig, `_parse_level`, `_bfs_solvable`
- `llm/decision_gate.py` — DecisionGate, DiversityThresholds, GateResult, `evaluate_candidate`
- `llm/agent_evaluator.py` — AgentEvaluator, rollout JIT, checkpoint loading, sys.path manipulation
- `llm/prompt_builder.py` — ReferenceMaze, MetricEntry, `build_generation_prompt`, `build_diversity_feedback_prompt`
- `llm/config.yaml` — provider config, timeout=600s, injection parameters
- `examples/maze_plr.py` — outer Python for-loop (injection point), TrainState fields, `insert_batch` call sites (lines 708, 854)
- `src/jaxued/level_sampler.py` — `insert()`, `insert_batch()`, `_insert_new` eviction logic (lines 145-193)
- `src/jaxued/environments/maze/level.py` — `Level.from_str()`, `is_well_formatted()`, dtype contracts
- `.planning/codebase/STACK.md` — confirmed installed versions
- `.planning/codebase/ARCHITECTURE.md` — existing system layers and training cycle
- `.planning/PROJECT.md` — project requirements, constraints, out-of-scope items

### Secondary (MEDIUM confidence — training data reference)

- Voyager (Wang et al. 2023) — comparable LLM+RL integration pattern (used only for comparable system analysis in FEATURES.md)
- General PCG-LLM literature — context for behavioral vs structural diversity gates

---
*Research completed: 2026-03-23*
*Ready for roadmap: yes*
