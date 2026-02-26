# Requirements: ES-ACCEL Integration

**Defined:** 2026-02-26
**Core Value:** The ES module must find diverse, high-regret environments that continuously challenge the agent — without collapsing to a single mode — so the agent develops generalizable skills through open-ended curriculum learning.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Foundation

- [ ] **FOUND-01**: Agent PPO/ACCEL training verified to match DCD repo implementation
- [ ] **FOUND-02**: Behavior signature vector extracted from agent rollout on any level (visit-count histogram over grid cells, JAX-compatible)

### ES Infrastructure

- [ ] **INFRA-01**: Modular ES strategy interface with ask/tell API supporting swappable algorithms
- [ ] **INFRA-02**: Replay buffer extended with `behavior_sig` field per level via `level_extra`
- [ ] **INFRA-03**: Composite fitness function F = α·Regret + β·Novelty with configurable weights
- [ ] **INFRA-04**: k-NN novelty scoring against buffer behavior signatures (pure JAX, JIT-compatible)

### Algorithms

- [ ] **ALGO-01**: NS-ES strategy implementation with composite fitness and buffer-as-novelty-archive
- [ ] **ALGO-02**: Behavioral SV-CMA-ES with N CMA-ES particles and Stein repulsion in behavior space

### Integration

- [ ] **INTEG-01**: Two-bucket sampling wired into ACCEL training loop (replay prob p + ES frontier 1-p)
- [ ] **INTEG-02**: Archive warm-up phase (init_pop evaluation before training starts)
- [ ] **INTEG-03**: End-to-end training pipeline with ES-generated curriculum, WandB logging, checkpointing

### Comparison

- [ ] **COMP-01**: Regret curve comparison across methods (vanilla ACCEL vs NS-ES vs SV-CMA-ES)

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Seed Control

- **SEED-01**: Reproducible seed control — audit RNG seeding across training pipeline

### Open-Ended Search

- **OPEN-01**: Convergence detection with restart — detect sigma collapse, restart from diverse archive samples
- **OPEN-02**: LLM injection interface — hook point in ES strategy for external guidance on search direction

### Advanced Diversity

- **DIV-01**: AURORA-style dynamic behavioral space — secondary autoencoder learns behavior axes from rollouts

## Out of Scope

| Feature | Reason |
|---------|--------|
| MAP-Elites with fixed descriptors as primary method | Contradicts thesis claim of open-ended behavioral diversity |
| PyTorch reimplementation of JAX components | Wastes time; existing JAX code works |
| Non-maze environments (lava, keys/doors) | Wall-only for controlled comparison |
| Web UI / dashboard | Research code; WandB suffices |
| Retraining the VAE | Treated as fixed pretrained component |
| Full AURORA implementation | Deferred — too complex for this milestone |
| QDax migration | Cost exceeds benefit for current scope |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | TBD | Pending |
| FOUND-02 | TBD | Pending |
| INFRA-01 | TBD | Pending |
| INFRA-02 | TBD | Pending |
| INFRA-03 | TBD | Pending |
| INFRA-04 | TBD | Pending |
| ALGO-01 | TBD | Pending |
| ALGO-02 | TBD | Pending |
| INTEG-01 | TBD | Pending |
| INTEG-02 | TBD | Pending |
| INTEG-03 | TBD | Pending |
| COMP-01 | TBD | Pending |

**Coverage:**
- v1 requirements: 12 total
- Mapped to phases: 0
- Unmapped: 12 ⚠️

---
*Requirements defined: 2026-02-26*
*Last updated: 2026-02-26 after initial definition*
