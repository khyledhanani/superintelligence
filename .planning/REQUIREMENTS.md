# Requirements: LLM Diversity Injection for UED Maze Training

**Defined:** 2026-03-23
**Core Value:** LLM-generated mazes must measurably improve agent generalization (solve rate on held-out benchmarks) compared to ACCEL-only and CMA-ES-only baselines.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Integration Core

- [ ] **INTG-01**: Training loop injects LLM-generated mazes at configurable interval (`--llm_inject_interval N` steps)
- [ ] **INTG-02**: Existing `MazeGenerator` validation is extended with border wall check and BFS path length >5 (text→Level via `Level.from_str()` and `_bfs_solvable()` already exist)
- [ ] **INTG-03**: Accepted LLM levels are inserted into the PLR buffer via `insert_batch()` with regret-derived scores
- [x] **INTG-04**: CLI flags `--use_llm`, `--llm_batch_size`, `--llm_config` control injection behavior
- [ ] **INTG-05**: `LLMInjector` orchestration class in `llm/injector.py` encapsulates the full injection pipeline
- [x] **INTG-06**: Existing `test_generator.py` buffer-to-prompt functions (`select_references`, `build_references_with_metrics`) are adapted to work with live `train_state.sampler` instead of `.npz` dumps

### Decision Gate

- [ ] **GATE-01**: Existing `DecisionGate.evaluate_candidate()` (already fully implemented) is wired into the `LLMInjector` pipeline to filter every LLM maze before buffer insertion
- [ ] **GATE-02**: Existing `AgentEvaluator` (loads checkpoint once) is extended with a refresh mechanism to re-load current policy params at each injection event
- [ ] **GATE-03**: WandB logs `llm/injected_count`, `llm/acceptance_rate`, `llm/diversity_score_mean`, `llm/retained_rate` at each injection step
- [ ] **GATE-04**: Existing `PromptBuilder.build_generation_prompt()` (already accepts `global_metrics` and `references` with `path_overlay`) is fed live buffer entropy stats computed from `train_state.sampler`

### Experiments

- [ ] **EXPT-01**: Comparison launch scripts for ACCEL+LLM injection vs ACCEL-only control with matching seeds
- [ ] **EXPT-02**: Accepted levels are cached to disk (`.npy` + metadata JSON) with wall_map hashes logged to WandB for reproducibility
- [ ] **EXPT-03**: Injection frequency is ablatable via `--llm_inject_start_step` and `--llm_inject_interval` parameters

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Enhanced Generation

- **EGEN-01**: Diversity feedback loop — LLM retries with specific rejection reasons when gate rejects candidates
- **EGEN-02**: Reference maze selection strategy ablation (`diverse` vs `top_regret` in prompt context)

### Advanced Metrics

- **AMET-01**: CENIE novelty gate integration alongside td_error_emd for behavioral redundancy filtering
- **AMET-02**: Direct `train_state.params` passing to evaluator (eliminates file I/O overhead)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| LLM-based mutation | Too costly per API call — mutation handled by existing ACCEL/CMA-ES |
| Async/concurrent LLM generation | Periodic synchronous/threaded injection is sufficient for research |
| Multi-provider LLM ablation | Claude API via existing config is enough for thesis |
| Fine-tuning or training the LLM | Use off-the-shelf Claude — no prompt optimization meta-learning |
| New RL algorithms | PPO + ACCEL/PLR stays as-is; only buffer injection changes |
| Online prompt optimization | Meta-learning for prompts adds complexity with uncertain payoff |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INTG-01 | Phase 1 | Pending |
| INTG-02 | Phase 1 | Pending |
| INTG-03 | Phase 1 | Pending |
| INTG-04 | Phase 1 | Complete (01-01) |
| INTG-05 | Phase 1 | Pending |
| INTG-06 | Phase 1 | Complete (01-01) |
| GATE-01 | Phase 2 | Pending |
| GATE-02 | Phase 2 | Pending |
| GATE-03 | Phase 2 | Pending |
| GATE-04 | Phase 2 | Pending |
| EXPT-01 | Phase 4 | Pending |
| EXPT-02 | Phase 3 | Pending |
| EXPT-03 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0

---
*Requirements defined: 2026-03-23*
*Last updated: 2026-03-23 after 01-01 completion — INTG-04, INTG-06 complete*
