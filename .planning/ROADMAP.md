# Roadmap: LLM Diversity Injection for UED Maze Training

## Overview

This roadmap bridges two complete systems: the existing JAX/Flax ACCEL/PLR training pipeline and the pre-built LLM maze generation subsystem. Phase 1 establishes the core integration scaffolding and validates maze format correctness — the critical path that unblocks everything downstream. Phase 2 wires in the behavioral diversity gate with live policy evaluation and tunes injection hyperparameters empirically. Phase 3 adds level caching and reproducibility infrastructure required before any results can be reported. Phase 4 runs the thesis comparison experiments (ACCEL-only vs ACCEL+LLM injection) and analyses the evidence for the core claim.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Integration Scaffolding** - LLMInjector pipeline with format validation, buffer insertion, and WandB logging — gate disabled, injection unconditional
- [ ] **Phase 2: Decision Gate and Tuning** - Wire live AgentEvaluator + DecisionGate, refresh checkpoint per injection, tune interval/batch-size hyperparameters empirically
- [ ] **Phase 3: Reproducibility Infrastructure** - Level caching to disk, wall_map hash logging, comparison launch scripts, analysis tooling
- [ ] **Phase 4: Comparison Experiments** - Run ACCEL+LLM vs ACCEL-only control at 50k steps, analyse solve rate differences, ablate injection frequency if needed

## Phase Details

### Phase 1: Integration Scaffolding
**Goal**: The training loop can inject LLM-generated mazes into the buffer at configurable intervals with correct format validation, score initialization, and WandB logging — without the behavioral gate active
**Depends on**: Nothing (first phase)
**Requirements**: INTG-01, INTG-02, INTG-03, INTG-04, INTG-05, INTG-06
**Success Criteria** (what must be TRUE):
  1. Running `maze_plr.py --use_llm --llm_inject_interval 50 --llm_batch_size 8` starts training and the WandB `llm/injected_count` counter increments at every 50th eval step
  2. All LLM-generated mazes pass `validate_llm_level()` before buffer insertion — invalid mazes (missing border walls, path length <= 5, dtype mismatch) are rejected and logged, never crash training
  3. Accepted LLM levels are inserted into the PLR buffer via `insert_batch()` with regret-derived scores, and `llm/retained_rate` remains above 50% after 1000 post-injection steps
  4. `BufferStatsExtractor` converts live `train_state.sampler` state to `ReferenceMaze[]` objects formatted for LLM prompt context without requiring `.npz` file dumps
  5. `LLMInjector.maybe_inject()` is the only call site in `maze_plr.py` — the training loop outer for-loop is not polluted with injection logic
**Plans**: 2 plans in 2 waves

Plans:
- [ ] 01-01-PLAN.md — Config dataclass, CLI flags, and BufferStatsExtractor (Wave 1)
- [ ] 01-02-PLAN.md — LLMInjectionManager orchestrator and training loop hook (Wave 2)

### Phase 2: Decision Gate and Tuning
**Goal**: Every LLM maze candidate is evaluated against the live policy via `AgentEvaluator` and filtered by `DecisionGate` before buffer insertion, with checkpoint refresh per injection event and empirically-tuned hyperparameters
**Depends on**: Phase 1
**Requirements**: GATE-01, GATE-02, GATE-03, GATE-04
**Success Criteria** (what must be TRUE):
  1. WandB `llm/acceptance_rate` is between 20% and 80% across a 10k-step validation run — the gate is active and neither rejecting everything nor passing everything
  2. Agent solve rate does not drop more than 0.1 within 500 steps after any injection event — injected OOD mazes do not destabilize training
  3. `AgentEvaluator` uses the current policy checkpoint at each injection event — the timestamp delta between checkpoint mtime and injection time is less than one injection interval
  4. `PromptBuilder.build_generation_prompt()` receives live buffer entropy stats computed from `train_state.sampler`, not hardcoded or stale values
**Plans**: TBD

Plans:
- [ ] 02-01: TBD

### Phase 3: Reproducibility Infrastructure
**Goal**: Accepted LLM levels are cached to disk with wall_map hashes logged to WandB, and comparison launch scripts exist for running ACCEL+LLM vs ACCEL-only control with matching seeds and conditions
**Depends on**: Phase 2
**Requirements**: EXPT-02, EXPT-03
**Success Criteria** (what must be TRUE):
  1. Re-running a completed training session with the same JAX seed produces identical WandB solve rate curves up to and including the first injection event
  2. Every accepted LLM level is written to disk as `.npy` + metadata JSON, and its `wall_map` hash appears in WandB — an auditor can reconstruct which levels were injected into which run
  3. `launch_llm_injection.sh` and `launch_accel_only_control.sh` exist, use matching seeds and buffer sizes, and both successfully start training on the GPU nodes
  4. `--llm_inject_start_step` and `--llm_inject_interval` are independently configurable and an ablation can be run by changing only those flags
**Plans**: TBD

Plans:
- [ ] 03-01: TBD

### Phase 4: Comparison Experiments
**Goal**: At least 3 ACCEL+LLM seeds and 3 ACCEL-only control seeds complete 50k-step training runs, producing a WandB comparison table with statistical evidence for or against the core claim
**Depends on**: Phase 3
**Requirements**: EXPT-01
**Success Criteria** (what must be TRUE):
  1. WandB `JAXUED_LLM` project contains 6+ completed runs (3 ACCEL+LLM, 3 ACCEL-only) with matched seeds and identical non-injection hyperparameters
  2. `scripts/compare_llm_injection.py` produces a comparison table showing mean and std of final solve rate for each condition
  3. The core claim is evaluable from the data: ACCEL+LLM mean solve rate is either higher than ACCEL-only (supporting the thesis) or the gap is quantified and explained
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Integration Scaffolding | 0/2 | Not started | - |
| 2. Decision Gate and Tuning | 0/? | Not started | - |
| 3. Reproducibility Infrastructure | 0/? | Not started | - |
| 4. Comparison Experiments | 0/? | Not started | - |
