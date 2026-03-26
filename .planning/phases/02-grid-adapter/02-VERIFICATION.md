---
phase: 02-grid-adapter
verified: 2026-03-23T19:30:00Z
status: human_needed
score: 6/6 automated must-haves verified
re_verification: false
human_verification:
  - test: "Run 10k-step smoke test with --use_llm --llm_gate and check WandB llm/acceptance_rate"
    expected: "llm/acceptance_rate is between 0.2 and 0.8 (gate is active, not rejecting/passing everything)"
    why_human: "Acceptance rate depends on live LLM generation quality and gate threshold calibration — cannot verify without running LLM API calls against real policy"
  - test: "Check agent solve rate after each injection event in a 10k-step run"
    expected: "Solve rate does not drop more than 0.1 within 500 steps after any injection event"
    why_human: "Training stability under OOD maze injection requires an actual training run to observe"
  - test: "Confirm AgentEvaluator checkpoint timestamp vs injection time"
    expected: "AgentEvaluator uses params from train_state at injection time (timestamp delta < one injection interval)"
    why_human: "The code path calls update_params(train_state.params) at each injection event — mechanically correct, but live timing confirmation requires running the training loop"
---

# Phase 2: Decision Gate and Tuning — Verification Report

**Phase Goal:** Every LLM maze candidate is evaluated against the live policy via AgentEvaluator and filtered by DecisionGate before buffer insertion, with checkpoint refresh per injection event and empirically-tuned hyperparameters
**Verified:** 2026-03-23T19:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

All automated structural checks passed. Three success criteria require human validation with a live training run.

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | WandB `llm/acceptance_rate` is between 20%-80% across a 10k-step run | ? HUMAN | `llm/injector.py:495` logs `"llm/acceptance_rate": acceptance_rate` — metric exists and will log; actual rate depends on live LLM generation |
| 2 | Agent solve rate does not drop more than 0.1 within 500 steps after injection | ? HUMAN | No mechanism to verify statically — requires live run |
| 3 | AgentEvaluator uses current policy checkpoint at each injection event | VERIFIED | `injector.py:288` calls `self.agent_evaluator.update_params(train_state.params)` at top of `_do_injection()`, before any generation |
| 4 | `PromptBuilder.build_generation_prompt()` receives live buffer entropy stats from `train_state.sampler` | VERIFIED | `injector.py:299-300` computes `buffer_summary = self.buffer_stats.extract_buffer_summary(sampler)` and `global_metrics = BufferStatsExtractor.extract_global_metrics(buffer_summary)`, passed to both code paths (gated: `generate_with_feedback()` line 342; ungated: `generate()` line 366) |

**Score:** 2/4 truths verified statically, 2/4 require human validation, 0/4 failed

---

### Plan-level Must-Haves (Plan 02-01 and 02-02 combined)

#### Plan 02-01 Must-Haves

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | AgentEvaluator constructed with apply_fn + params directly (no checkpoint file) | VERIFIED | `agent_evaluator.py:64-98` — `__init__(self, apply_fn, params, env_params, ...)`, no `checkpoint_dir` param |
| 2 | `AgentEvaluator.update_params(params)` refreshes policy weights and invalidates cached JIT | VERIFIED | `agent_evaluator.py:100-112` — sets `self.params = params`, `self._rollout_fn = None`, `self._rollout_fn_num_levels = None` |
| 3 | `AgentEvaluator.from_checkpoint()` classmethod preserves backward compatibility | VERIFIED | `agent_evaluator.py:114-153` — classmethod calls `load_agent()` internally; `test_generator.py:1108,1238` — both call sites updated to `AgentEvaluator.from_checkpoint()` |
| 4 | `LLMInjectionConfig` has gate_enabled, difficulty_threshold, min_diversity, diversity_metric, max_diversity_retries, n_rollouts_gate with correct defaults | VERIFIED | `injection_config.py:39-44` — all six fields present; Python check confirmed gate_enabled=True, difficulty_threshold=0.6, min_diversity=0.02, diversity_metric="td_error_emd", max_diversity_retries=2, n_rollouts_gate=100 |
| 5 | `BufferStatsExtractor.extract_references_with_levels()` returns both ReferenceMaze list and Level list | VERIFIED | `buffer_stats.py:35-99` — returns `(references, level_objects)` tuple; `extract_references()` delegates to it (DRY) |
| 6 | `extract_global_metrics()` converts buffer summary to `List[MetricEntry]` | VERIFIED | `buffer_stats.py:158-193` — static method returning 3 MetricEntry objects; Python check confirmed output |

**Score:** 6/6 truths verified

#### Plan 02-02 Must-Haves

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Every LLM maze candidate evaluated by AgentEvaluator and filtered by DecisionGate before buffer insertion | VERIFIED | `injector.py:335-363` — gated path calls `generate_with_feedback()` which internally handles rollout+gate; only appends to `valid_levels` when `not result.diversity_issues` |
| 2 | AgentEvaluator uses current train_state.params at each injection event | VERIFIED | `injector.py:282-288` — `gate_active` guard and `self.agent_evaluator.update_params(train_state.params)` at top of `_do_injection()` |
| 3 | WandB logs llm/acceptance_rate, llm/diversity_score_mean, llm/batch_all_rejected_count at each injection step | VERIFIED | `injector.py:495,501,504` — all three keys present in `log_payload` dict passed to `wandb.log()` |
| 4 | `generate_with_feedback()` retry loop used instead of `generate()` | VERIFIED | `injector.py:337` — gated path calls `self.generator.generate_with_feedback(...)` not `generate()` |
| 5 | Live buffer entropy stats passed as global_metrics to LLM prompt | VERIFIED | `injector.py:299-300,342,366` — computed from live `train_state.sampler` and passed to both gated and ungated code paths |
| 6 | Smoke test script runs 5k-step training with gate enabled | VERIFIED | `scripts/smoke_test_llm_gate.sh` exists, is executable, uses `--llm_gate`, `--llm_difficulty_threshold 0.6`, `--llm_n_rollouts 100`, `--num_env_steps 5000` |

**Score:** 6/6 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `llm/agent_evaluator.py` | Direct-param AgentEvaluator with update_params() and from_checkpoint() | VERIFIED | 351 lines, substantive implementation — full JIT rollout, batched evaluation, multi-rollout |
| `llm/injection_config.py` | Gate configuration fields with Phase 2 defaults | VERIFIED | All 6 gate fields present with correct defaults; from_config_dict() maps all CLI flags |
| `llm/buffer_stats.py` | extract_references_with_levels() and extract_global_metrics() | VERIFIED | Both methods implemented; extract_references() delegates to extract_references_with_levels() |
| `llm/injector.py` | Gated injection pipeline using generate_with_feedback() with AgentEvaluator and DecisionGate | VERIFIED | 520 lines; dual-path _do_injection(); imports DiversityThresholds from decision_gate |
| `examples/maze_plr.py` | Gate CLI flags and AgentEvaluator construction from live train_state | VERIFIED | 6 gate CLI flags added; AgentEvaluator constructed at `llm_config.gate_enabled` check using `train_state_init.apply_fn`, `train_state_init.params`, `env_params` |
| `scripts/smoke_test_llm_gate.sh` | Smoke test script for validating gate integration at 5k steps | VERIFIED | Exists, executable, correct flags |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `llm/agent_evaluator.py` | `llm/injector.py` | `update_params()` called at each injection event | WIRED | `injector.py:288` — `self.agent_evaluator.update_params(train_state.params)` |
| `llm/buffer_stats.py` | `llm/injector.py` | `extract_references_with_levels()` and `extract_global_metrics()` in `_do_injection()` | WIRED | `injector.py:294,300` — both called; Level objects used for ref_trajectories loop |
| `llm/injection_config.py` | `llm/injector.py` | `DiversityThresholds` constructed from config fields | WIRED | `injector.py:318-322` — `DiversityThresholds(difficulty_threshold=self.config.difficulty_threshold, min_diversity=self.config.min_diversity, diversity_metric=self.config.diversity_metric)` |
| `examples/maze_plr.py` | `llm/injector.py` | `train_state.apply_fn` and `train_state.params` passed for AgentEvaluator construction | WIRED | `maze_plr.py:1061-1065` — `AgentEvaluator(apply_fn=train_state_init.apply_fn, params=train_state_init.params, env_params=env_params)`, then `LLMInjectionManager(..., agent_evaluator=agent_evaluator)` |
| `llm/injector.py` | `llm/maze_generator.py` | Calls `generate_with_feedback()` instead of `generate()` | WIRED | `injector.py:337` — gated path, `injector.py:366` — ungated fallback |
| `llm/injector.py` | `llm/decision_gate.py` | `DiversityThresholds` constructed from config and passed to `generate_with_feedback()` | WIRED | `injector.py:30` import, `injector.py:318-322` construction, `injector.py:343` passed as `diversity_thresholds=thresholds` |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| GATE-01 | 02-02-PLAN.md | DecisionGate.evaluate_candidate() wired into LLMInjector pipeline to filter every LLM maze before buffer insertion | SATISFIED | `injector.py:337-363` — `generate_with_feedback()` calls gate internally; only levels with empty `result.diversity_issues` are appended to `valid_levels` |
| GATE-02 | 02-01-PLAN.md | AgentEvaluator extended with refresh mechanism for current policy params at each injection event | SATISFIED | `agent_evaluator.py:100-112` — `update_params()` method; `injector.py:288` — called at start of every `_do_injection()` |
| GATE-03 | 02-02-PLAN.md | WandB logs llm/injected_count, llm/acceptance_rate, llm/diversity_score_mean, llm/retained_rate at each injection step | SATISFIED | `injector.py:489-504` — all four REQUIREMENTS.md keys present: `llm/injected_count:490`, `llm/acceptance_rate:495`, `llm/diversity_score_mean:501`, `llm/retained_rate:496`; plus additional gate metrics `llm/difficulty_score_mean`, `llm/gate_rejection_rate`, `llm/batch_all_rejected_count` |
| GATE-04 | 02-01-PLAN.md | PromptBuilder.build_generation_prompt() fed live buffer entropy stats from train_state.sampler | SATISFIED | `injector.py:299-300` — computes `buffer_summary` from `train_state.sampler`; `extract_global_metrics()` converts to `List[MetricEntry]`; passed to both `generate_with_feedback()` and `generate()` code paths |

All 4 GATE requirements satisfied. No orphaned requirements — REQUIREMENTS.md traceability table shows GATE-01 through GATE-04 assigned to Phase 2 with status Complete.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | — |

No TODO/FIXME/PLACEHOLDER comments found in any of the four modified files. No empty implementations or stub return patterns. No console.log-only handlers.

### Human Verification Required

#### 1. Gate Acceptance Rate (Success Criterion 1)

**Test:** Run `bash scripts/smoke_test_llm_gate.sh` on a GPU node (smew or canada), then check WandB project JAXUED_SMOKE group llm-gate-smoke.
**Expected:** `llm/acceptance_rate` stays between 0.2 and 0.8 over the 5k-step smoke test. Values near 0 or 1 indicate the gate thresholds are miscalibrated.
**Why human:** Cannot determine acceptance rate without live LLM API calls and actual policy rollouts. The metric key and logging code are present and correct; the threshold behavior requires a real run.

#### 2. Training Stability After Injection (Success Criterion 2)

**Test:** During the same smoke test (or a 10k-step run), observe the solve rate curve in WandB around each injection event (logged at `llm/injected_count > 0` steps).
**Expected:** Solve rate does not drop more than 0.1 within 500 steps after any injection event.
**Why human:** Training stability is emergent behavior that cannot be verified statically. Requires observing the actual solve rate curve.

#### 3. Live Policy Params Confirmation (Success Criterion 3)

**Test:** Add a quick print or WandB log of `jnp.sum(train_state.params)` vs `jnp.sum(evaluator.params)` inside `_do_injection()` for one run, or simply trust the code path: `update_params(train_state.params)` is called unconditionally when `gate_active`.
**Expected:** The params used for rollouts match the training step's params (not a stale checkpoint from initialization).
**Why human:** Code inspection confirms `update_params(train_state.params)` is called at line 288 before any rollout, but confirming the values are actually different from initialization requires a live run.

---

## Gaps Summary

No structural gaps found. All artifacts exist, are substantive, and are fully wired. The three human verification items are runtime behaviors that cannot be confirmed statically.

The phase goal statement "with checkpoint refresh per injection event" is mechanically satisfied by the `update_params(train_state.params)` call at the top of `_do_injection()`. The "empirically-tuned hyperparameters" goal is partially satisfied — the locked decisions from CONTEXT.md (difficulty_threshold=0.6, min_diversity=0.02, diversity_metric=td_error_emd) are encoded as defaults; empirical validation of these choices requires a live run (Success Criterion 1).

---

_Verified: 2026-03-23T19:30:00Z_
_Verifier: Claude (gsd-verifier)_
