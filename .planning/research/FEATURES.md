# Features Research: ES-ACCEL Integration

**Research Date:** 2026-02-26
**Dimension:** Features — What capabilities are needed and how to categorize them

## Table Stakes (Must Have for Valid Comparison)

These are required for the ES-enhanced system to be a legitimate research contribution.

### 1. Modular ES Strategy Interface
- **Description:** Swappable ES algorithms (CMA-ES, NS-ES, Behavioral SV-CMA-ES) behind a common API
- **Complexity:** Medium
- **Dependencies:** None — foundational component
- **Why table stakes:** Can't compare approaches without a clean interface

### 2. Behavior Signature Extraction
- **Description:** Visit-count histogram over grid cells, accumulated during agent rollout (JAX-compatible)
- **Complexity:** Medium
- **Dependencies:** Requires extending `regret_fitness.py` rollout to return trajectory data
- **Why table stakes:** Both NS-ES and Behavioral SV-CMA-ES depend on this
- **CRITICAL BLOCKING DEPENDENCY:** Must be built first

### 3. Enhanced Replay Buffer
- **Description:** Add `behavior_sig` field to buffer (currently stores only `max_return, latent`)
- **Complexity:** Low — LevelSampler's `level_extra` dict already supports arbitrary fields
- **Dependencies:** Behavior signature extraction
- **Why table stakes:** Buffer needs behavior data to serve as novelty archive

### 4. Novelty Score in Fitness
- **Description:** k-NN distance in behavior-signature space against buffer contents
- **Complexity:** Medium
- **Dependencies:** Behavior signatures, enhanced buffer
- **Why table stakes:** Core mechanism for preventing mode collapse

### 5. Two-Bucket Sampling (Replay vs ES Frontier)
- **Description:** Probability p for replay buffer, 1-p for ES-generated environments
- **Complexity:** Low — mechanism already exists in ACCEL loop, needs verification
- **Dependencies:** ES strategy interface
- **Why table stakes:** Prevents catastrophic forgetting during ES-driven training

### 6. Regret Curve Comparison Tooling
- **Description:** Side-by-side plots of regret, solve rate, diversity across methods
- **Complexity:** Low
- **Dependencies:** End-to-end training working
- **Why table stakes:** Results need to be presentable for thesis/paper

### 7. Agent Verification Against DCD
- **Description:** Confirm `maze_plr.py` agent matches Facebook Research DCD implementation
- **Complexity:** Medium (code comparison + testing)
- **Dependencies:** None
- **Why table stakes:** Invalid baseline invalidates all comparisons

### 8. Reproducible Seed Control
- **Description:** Audit `rng_np` seeding in training pipeline
- **Complexity:** Low
- **Dependencies:** None
- **Why table stakes:** Reproducibility required for research

## Differentiators (Novel Contributions)

These are the novel research contributions that distinguish this work.

### 1. Behavioral SV-CMA-ES
- **Description:** Repulsion kernel in agent-behavior space, not latent space
- **Complexity:** Very High — primary thesis contribution
- **Dependencies:** Behavior signatures, modular ES interface
- **Research novelty:** Applying Stein Variational Gradient Descent to UED with behavioral diversity

### 2. NS-ES (Novelty Search ES)
- **Description:** F = α·Regret + β·Novelty with unified buffer-as-archive
- **Complexity:** High — MVP contribution
- **Dependencies:** Behavior signatures, enhanced buffer, k-NN novelty
- **Research novelty:** Novelty search integrated directly with ACCEL replay buffer

### 3. Fitness Composition API
- **Description:** Composable fitness with configurable weights (regret, novelty, diversity terms)
- **Complexity:** Medium
- **Dependencies:** Individual fitness components
- **Research novelty:** Enables ablation studies on fitness signal composition

### 4. LLM Injection Interface (Design Only)
- **Description:** Hook point in ES strategy for external guidance on search direction
- **Complexity:** Low (interface design) → High (actual LLM integration, deferred)
- **Dependencies:** Modular ES interface
- **Research novelty:** Bridge between ES and foundation model guidance

### 5. Convergence Detection with Restart
- **Description:** Detect sigma collapse in ES, restart from diverse archive samples
- **Complexity:** Medium
- **Dependencies:** ES strategy interface, archive with diverse entries
- **Research novelty:** Automated open-ended search without manual intervention

## Anti-Features (Deliberately NOT Building)

| Feature | Reason |
|---------|--------|
| MAP-Elites with fixed descriptors as primary method | Contradicts thesis claim of open-ended behavioral diversity |
| PyTorch reimplementation of JAX components | Wastes time; existing JAX code works |
| Non-maze environments (lava, keys/doors) | Out of scope — wall-only for controlled comparison |
| Web UI / dashboard | Research code; WandB suffices |
| Retraining the VAE | Treated as fixed pretrained component |
| Full AURORA dynamic behavioral space | Defer to future work — too complex for this milestone |

## Feature Dependencies

```
Behavior Signature Extraction ──┬──→ Enhanced Replay Buffer
                                ├──→ NS-ES (novelty fitness)
                                └──→ Behavioral SV-CMA-ES (repulsion kernel)

Modular ES Interface ───────────┬──→ NS-ES
                                ├──→ Behavioral SV-CMA-ES
                                ├──→ Convergence Detection
                                └──→ LLM Injection Interface

Agent Verification ─────────────→ All comparison experiments
```

## MVP Path

**For one valid thesis claim:** Build NS-ES (regret + novelty fitness), verify it outperforms vanilla CMA-ES on regret curves and solve rates. Behavioral SV-CMA-ES is the stretch contribution after NS-ES validates the approach.

---
*Features research: 2026-02-26*
