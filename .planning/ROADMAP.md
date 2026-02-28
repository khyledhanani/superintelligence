# Roadmap: ES-ACCEL Integration

## Overview

This roadmap extends the existing ACCEL UED codebase with behavioral diversity mechanisms, transforming environment generation from regret-maximization into open-ended curriculum discovery. The work proceeds through a strict dependency chain: behavior signature extraction unblocks everything else, so it comes first. Infrastructure follows, then NS-ES validates the approach as an MVP, then Behavioral SV-CMA-ES delivers the primary thesis contribution. The final phase runs ablations and produces the comparison results needed for the thesis.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Verified baseline agent and JAX-native behavior signature extraction
- [ ] **Phase 2: Buffer and Fitness Infrastructure** - Enhanced buffer, k-NN novelty scoring, composite fitness, modular ES interface
- [ ] **Phase 3: NS-ES Integration** - First end-to-end ES with composite fitness, archive warm-up, two-bucket sampling wired in (MVP)
- [ ] **Phase 4: Behavioral SV-CMA-ES** - N-particle CMA-ES with Stein repulsion in behavior space (primary thesis contribution)
- [ ] **Phase 5: Ablations and Analysis** - Fitness weight sweeps, regret curve comparisons across all three methods

## Phase Details

### Phase 1: Foundation
**Goal**: The agent baseline is verified correct and behavior signatures can be extracted from any rollout
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02
**Success Criteria** (what must be TRUE):
  1. A training run of `maze_plr.py` with ACCEL produces regret curves that match the DCD repo reference (same score_function, same hyperparameters) — the baseline is valid
  2. Given any maze level and a loaded agent, `extract_behavior_signature()` returns a fixed-length JAX array representing the agent's visit-count histogram over grid cells
  3. The behavior extractor passes `jax.jit(f).lower(args).compile()` without error — it is JIT-compatible and will not silently fall back to eager mode
  4. Behavior signatures are visually distinct for qualitatively different levels (sparse maze vs dense maze produces different histograms, confirmed by inspection)
**Plans**: 2 plans
Plans:
- [x] 01-PLAN-01.md — Agent verification: PPO/ACCEL code comparison vs DCD + smoke test
- [x] 01-PLAN-02.md — Behavior signature: implement extract_behavior_signature() in es/regret_fitness.py

### Phase 2: Buffer and Fitness Infrastructure
**Goal**: The shared components that all ES strategies depend on are built, tested, and stable before any strategy is implemented
**Depends on**: Phase 1
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04
**Success Criteria** (what must be TRUE):
  1. The replay buffer stores a `behavior_sig` field per level via `level_extra`; inserting a level with a behavior signature and retrieving it returns the same vector
  2. k-NN novelty scoring over the full buffer (up to 4000 entries) runs inside `jax.jit` without ConcretizationTypeError and returns a scalar novelty score for any candidate behavior signature
  3. Composite fitness F = α·Regret + β·Novelty is computed correctly: given known regret and novelty values, the output matches α*regret + β*novelty with configurable weights
  4. The modular ES interface defines `ask(state, rng) -> (candidates, state)` and `tell(state, candidates, fitness) -> state`; the existing CMA-ES wraps behind this interface and runs without behavioral changes
**Plans**: 3 plans
Plans:
- [ ] 02-01-PLAN.md — ES interface: ESStrategy Protocol + CMAESStrategy wrapper (INFRA-01)
- [ ] 02-02-PLAN.md — k-NN novelty + composite fitness pure functions (INFRA-03, INFRA-04)
- [ ] 02-03-PLAN.md — Buffer behavior_sig integration + integration tests (INFRA-02)

### Phase 3: NS-ES Integration
**Goal**: A complete end-to-end training run executes with NS-ES providing environments via composite fitness, the archive initializes from warm-up before training, and metrics appear in WandB
**Depends on**: Phase 2
**Requirements**: ALGO-01, INTEG-01, INTEG-02, INTEG-03
**Success Criteria** (what must be TRUE):
  1. Running `maze_plr.py --use_es_mutation --es_strategy ns_es` completes a full training run without crashing — the end-to-end pipeline is wired
  2. The archive warm-up phase runs before the main training loop: 256 random latents are evaluated, behavior signatures extracted, and inserted into the buffer before step 0
  3. WandB logs show both `regret` and `novelty_score` per generation, plus `replay_buffer_size` and `buffer_occupied` — the composite fitness signal is observable
  4. Two-bucket sampling is active: at each mutation step, environments come from the replay buffer with probability p and from the ES frontier with probability 1-p; both p values are configurable
  5. The NS-ES training run produces a regret curve that is comparable (plottable side-by-side) with a vanilla CMA-ES baseline run using the same random seed
**Plans**: TBD

### Phase 4: Behavioral SV-CMA-ES
**Goal**: The primary thesis contribution is implemented and runs end-to-end: N independent CMA-ES particles apply Stein repulsion in behavior space to maintain diversity across the particle population
**Depends on**: Phase 3
**Requirements**: ALGO-02
**Success Criteria** (what must be TRUE):
  1. Running `maze_plr.py --use_es_mutation --es_strategy sv_cma_es --n_particles N` completes a full training run — the multi-particle strategy is integrated end-to-end
  2. Behavior-space repulsion is active: after ask(), Stein gradient is computed between particles using behavior signatures, and the candidate latents are adjusted before tell() — this is observable by logging per-particle behavior distances before and after repulsion
  3. Particle diversity is maintained throughout training: the mean pairwise behavior distance across particles does not collapse to near-zero within the first 500 steps (logged to WandB)
  4. The SV-CMA-ES run produces a regret curve plottable alongside NS-ES and vanilla CMA-ES for direct comparison
**Plans**: TBD

### Phase 5: Ablations and Analysis
**Goal**: The thesis comparison is complete: regret curves across all three methods are plotted, fitness weight ablations are run, and the results are reproducible
**Depends on**: Phase 4
**Requirements**: COMP-01
**Success Criteria** (what must be TRUE):
  1. A single script (or notebook) produces side-by-side regret curve plots for vanilla ACCEL, NS-ES, and Behavioral SV-CMA-ES from WandB run data — the comparison is presentable
  2. Fitness weight ablations are run with at least three α/β configurations (e.g., 1.0/0.0, 0.8/0.2, 0.5/0.5); results are logged and the plot shows which configuration performs best
  3. All three strategies run from the same initial seed and produce the same result when re-run — reproducibility is confirmed
  4. The LLM injection interface is designed and documented as a hook point in the ES strategy: a comment block or stub function exists where external guidance would be injected, with notes on the expected interface
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 2/2 | Complete | 2026-02-28 |
| 2. Buffer and Fitness Infrastructure | 1/3 | In Progress|  |
| 3. NS-ES Integration | 0/TBD | Not started | - |
| 4. Behavioral SV-CMA-ES | 0/TBD | Not started | - |
| 5. Ablations and Analysis | 0/TBD | Not started | - |
