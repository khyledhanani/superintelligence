# Roadmap: ES-ACCEL Integration

## Overview

This roadmap extends the existing ACCEL UED codebase with behavioral diversity mechanisms, transforming environment generation from regret-maximization into open-ended curriculum discovery. The work proceeds through a strict dependency chain: behavior signature extraction unblocks everything else, so it comes first. Infrastructure follows, then NS-ES validates the approach as an MVP, then Behavioral SV-CMA-ES delivers the primary thesis contribution. Phase 5 refactors the training architecture to a clean two-mode design and runs four comparison experiments. Phase 6 runs fitness weight ablations and validation analysis.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation** - Verified baseline agent and JAX-native behavior signature extraction
- [x] **Phase 2: Buffer and Fitness Infrastructure** - Enhanced buffer, k-NN novelty scoring, composite fitness, modular ES interface
- [x] **Phase 3: NS-ES Integration** - First end-to-end ES with composite fitness, archive warm-up, two-bucket sampling wired in (MVP)
- [x] **Phase 4: Behavioral SV-CMA-ES** - N-particle CMA-ES with Stein repulsion in behavior space (primary thesis contribution)
- [ ] **Phase 5: Refactor and Four-Way Comparison** - Clean two-mode train.py rewrite, four comparison experiments (ACCEL, CMA-ES, NS-ES, SV-CMA-ES) at 20k updates, thesis comparison plots
- [ ] **Phase 6: Ablation Studies** - Fitness weight sweeps for SV-CMA-ES (α/β ablations), validation set evaluation

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
- [x] 02-01-PLAN.md — ES interface: ESStrategy Protocol + CMAESStrategy wrapper (INFRA-01)
- [x] 02-02-PLAN.md — k-NN novelty + composite fitness pure functions (INFRA-03, INFRA-04)
- [x] 02-03-PLAN.md — Buffer behavior_sig integration + integration tests (INFRA-02)

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
**Plans**: 3 plans
Plans:
- [x] 03-01-PLAN.md — NSESStrategy implementation (ALGO-01)
- [x] 03-02-PLAN.md — behavior_sig extraction, archive warm-up, WandB, ES routing in train.py (INTEG-01, INTEG-02, INTEG-03)
- [x] 03-03-PLAN.md — Phase 3 tests: NSESStrategy, behavior_sig, two-bucket guard, end-to-end smoke (ALGO-01, INTEG-01, INTEG-02, INTEG-03)

### Phase 4: Behavioral SV-CMA-ES
**Goal**: The primary thesis contribution is implemented and runs end-to-end: N independent CMA-ES particles apply Stein repulsion in behavior space to maintain diversity across the particle population
**Depends on**: Phase 3
**Requirements**: ALGO-02
**Success Criteria** (what must be TRUE):
  1. Running `maze_plr.py --use_es_mutation --es_strategy sv_cma_es --n_particles N` completes a full training run — the multi-particle strategy is integrated end-to-end
  2. Behavior-space repulsion is active: after ask(), Stein gradient is computed between particles using behavior signatures, and the candidate latents are adjusted before tell() — this is observable by logging per-particle behavior distances before and after repulsion
  3. Particle diversity is maintained throughout training: the mean pairwise behavior distance across particles does not collapse to near-zero within the first 500 steps (logged to WandB)
  4. The SV-CMA-ES run produces a regret curve plottable alongside NS-ES and vanilla CMA-ES for direct comparison
**Plans**: 3 plans
Plans:
- [x] 04-01-PLAN.md -- stein.py + SVCMAESStrategy core algorithm implementation
- [x] 04-02-PLAN.md -- train.py sv_cma_es routing, __init__.py export, --n_particles argparse
- [x] 04-03-PLAN.md -- Phase 4 tests: init, ask, tell, repulsion, N=1 degradation, smoke test

### Phase 5: Refactor and Four-Way Comparison
**Goal**: accel_training/train.py is rewritten as a clean two-mode pipeline (replay / es_step), the full codebase is audited for compatibility, and four comparison experiments produce thesis-quality comparison plots
**Depends on**: Phase 4
**Requirements**: COMP-01
**Success Criteria** (what must be TRUE):
  1. train.py is rewritten with only two modes — `replay` (PLR buffer → agent training) and `es_step` (ES ask() → VAE decode → eval → PLR buffer insert → tell()); no MAP-Elites archive, no archive warm-up, replay/es_step ratio configurable via config.yml
  2. Full pipeline audit complete: every file importing from accel_training/ is reviewed and updated; all three ES strategies (cma_es, ns_es, sv_cma_es) run without error under the new architecture
  3. Pre-launch validation passes: SV-CMA-ES runs 1–2k updates and buf_score rises clearly above the previous ~0.004 ceiling — architecture is confirmed working before committing full runs
  4. All four experiments complete at 20k updates (same seed): ACCEL baseline (examples/maze_plr.py as-is), CMA-ES, NS-ES, SV-CMA-ES — runs are named and grouped in WandB for easy comparison
  5. A Jupyter notebook produces two thesis-quality figures from WandB data: (1) side-by-side regret-vs-updates comparison for all four methods, smoothed single-seed curves; (2) placeholder panel for ablations (Phase 6)
**Plans**: 2 plans
Plans:
- [ ] 05-01-PLAN.md — Rewrite train.py as two-mode pipeline + update config.yml and tests
- [ ] 05-02-PLAN.md — Launcher script for four experiments + Jupyter notebook for thesis figures

### Phase 6: Ablation Studies
**Goal**: Fitness weight ablations quantify the contribution of novelty in SV-CMA-ES, and agents are evaluated on held-out validation mazes
**Depends on**: Phase 5
**Requirements**: COMP-01
**Success Criteria** (what must be TRUE):
  1. SV-CMA-ES is run with at least three α/β configurations (1.0/0.0, 0.8/0.2, 0.5/0.5); all runs logged to WandB and plotted in the ablation figure
  2. The ablation plot shows which fitness weighting performs best on regret-per-update efficiency
  3. Trained agent checkpoints from Phase 5 are evaluated on a fixed held-out validation maze set; solved rate and regret are reported per method
  4. All ablation runs use the same seed as Phase 5 comparison runs for a fair comparison
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 2/2 | Complete | 2026-02-28 |
| 2. Buffer and Fitness Infrastructure | 3/3 | Complete | 2026-02-28 |
| 3. NS-ES Integration | 3/3 | Complete | 2026-03-02 |
| 4. Behavioral SV-CMA-ES | 3/3 | Complete | 2026-03-02 |
| 5. Refactor and Four-Way Comparison | 1/2 | In Progress|  |
| 6. Ablation Studies | 0/TBD | Not started | - |
