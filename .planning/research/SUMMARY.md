# Project Research Summary

**Project:** ES-ACCEL Integration — Behavioral Diversity in Unsupervised Environment Design
**Domain:** Evolutionary Strategies + Unsupervised Environment Design (UED) / Reinforcement Learning
**Researched:** 2026-02-26
**Confidence:** HIGH

## Executive Summary

This project integrates behavioral diversity mechanisms into an existing ACCEL (Adversarially Compelled Curriculum Learning) training loop backed by a JAX/Flax/evosax stack. The core thesis challenge is that vanilla CMA-ES, already integrated in the codebase, collapses to a single high-regret environment topology — documented in `es/REGRET_PIPELINE_README.md`. The solution is to augment the evolutionary search with behavior-aware fitness signals and multi-modal ES strategies (NS-ES and Behavioral SV-CMA-ES), transforming environment generation from regret-maximization into open-ended curriculum discovery.

The recommended approach is strictly additive: no library installs are needed, no existing training loop components change, and the ACCEL replay buffer doubles as the novelty archive — the single most elegant finding of the research. The implementation proceeds through a clear dependency chain: behavior signature extraction enables novelty scoring, which enables NS-ES, which then validates the approach before the more complex Behavioral SV-CMA-ES is attempted. The MVP is a working NS-ES (composite fitness F = α·Regret + β·Novelty) that demonstrably outperforms vanilla CMA-ES on regret curves.

The two dominant risks are: (1) JAX JIT incompatibility from dynamic control flow in the new behavioral code, and (2) novelty reward hacking where ES maximizes a superficial diversity signal while regret stays flat. Both are mitigated by concrete design rules: all new paths must be JAX-native with fixed-shape arrays, and novelty weight β must remain subordinate to regret weight α throughout early training.

## Key Findings

### Recommended Stack

The full stack is already installed. JAX + Flax + Optax + evosax + jaxued covers every algorithm in scope. The only gap is SV-CMA-ES, which is not in evosax and must be implemented as a custom wrapper around `evosax.algorithms.CMA_ES` — approximately 80-100 lines of code that injects a Stein repulsion gradient between the ask and tell steps. For k-NN novelty scoring over the 4000-entry replay buffer, `jnp.linalg.norm` broadcasting is sufficient; no external library is appropriate here (scikit-learn and FAISS both break JAX JIT).

See `.planning/research/STACK.md` for full detail.

**Core technologies:**
- **JAX:** Tensor computation and JIT compilation — everything must be JAX-native
- **evosax CMA_ES:** Base ES algorithm; writable with ask/tell API for NS-ES and SV-CMA-ES
- **Flax:** Neural network modules including the AURORA autoencoder (deferred to future work)
- **Optax:** Optimizer for online autoencoder training if AURORA pursued
- **jaxued LevelSampler:** Replay buffer with `level_extra` dict — doubles as novelty archive
- **WandB:** Logging and experiment tracking — no dashboard needed

### Expected Features

The research identifies a strict dependency chain. Behavior signature extraction is the foundational blocker: NS-ES and Behavioral SV-CMA-ES both depend on it and cannot be built without it.

See `.planning/research/FEATURES.md` for full detail.

**Must have (table stakes):**
- **Behavior Signature Extraction** — foundational; all other novel features depend on this; must be built first
- **Modular ES Strategy Interface** — enables clean swapping between CMA-ES, NS-ES, SV-CMA-ES for comparison
- **Enhanced Replay Buffer** — add `behavior_sig` field to `level_extra`; required for archive-based novelty
- **Novelty Score in Fitness** — k-NN distance in behavior space; core mechanism against mode collapse
- **Two-Bucket Sampling** — replay vs ES-generated environments; already partially present, needs verification
- **Agent Verification vs DCD** — invalid baseline invalidates all comparisons
- **Reproducible Seed Control** — required for research reproducibility
- **Regret Curve Comparison Tooling** — required to present results

**Should have (differentiators — novel contributions):**
- **NS-ES** — composite fitness F = α·Regret + β·Novelty; MVP thesis contribution
- **Behavioral SV-CMA-ES** — Stein repulsion in behavior space; primary thesis contribution; build after NS-ES validates
- **Fitness Composition API** — configurable weights enabling ablation studies
- **Convergence Detection with Restart** — automated open-ended search

**Defer to future work:**
- Full AURORA dynamic behavioral space — too complex for this milestone
- LLM injection interface — interface design only, no actual integration now
- QDax migration — quality-diversity patterns, useful for later milestones
- Non-maze environments — out of scope for controlled comparison

### Architecture Approach

The architecture is deliberately non-invasive. The ES plugs into the existing `jax.lax.switch` inside `maze_plr.py` as a new `elif config["use_es_mutation"]` branch, alongside the existing MAP-Elites and PLWM branches. The outer training loop, PPO update, and level sampling are untouched. The only structural additions to `maze_plr.py` are: (1) a new `es_state` field in `TrainState`, and (2) the new ES branch in `on_mutate_levels`. The `LevelSampler` needs no code changes — its `level_extra` dict already supports arbitrary per-level fields.

See `.planning/research/ARCHITECTURE.md` for full detail.

**Major components:**
1. **Behavior Extractor** — extends `regret_fitness.py` rollout to return trajectory data; outputs visit-count histogram over grid cells
2. **Fitness Evaluator** — computes composite fitness from regret + k-NN novelty; pure JAX vectorized distances
3. **ES Strategy (Swappable)** — `CMAESStrategy`, `NSESStrategy`, `BehavioralSVCMAESStrategy` behind a common ask/tell interface
4. **Enhanced Buffer Manager** — thin wrapper around existing `LevelSampler`; adds `behavior_sig` to `level_extra`; the replay buffer IS the novelty archive
5. **Convergence Detector** — deferred; interface ready when ES strategy module is built

**Data flow per training step (ES branch):**
Buffer latents → ES.ask() → candidate latents → VAE decode → levels → PPO rollout → trajectories + returns → BehaviorExtractor → behavior signatures; compute_regret → regret scores → FitnessEvaluator (k-NN novelty) → composite fitness → ES.tell() → updated ES state → high-fitness entries → LevelSampler.insert()

### Critical Pitfalls

See `.planning/research/PITFALLS.md` for full detail.

1. **CMA-ES mode collapse** — Vanilla CMA-ES collapses to a single environment topology (documented in codebase). Mitigation: use NS-ES or Behavioral SV-CMA-ES from the start; never use pure CMA-ES without a diversity mechanism.

2. **Novelty reward hacking** — ES maximizes a superficial diversity signal while regret stays flat. Mitigation: keep α >> β (e.g., 0.8/0.2); design behavior signatures to capture task-relevant behavior; enforce minimum regret threshold for archive insertion.

3. **JAX JIT incompatibility** — Dynamic control flow in new behavioral code silently falls back to eager mode or raises `ConcretizationTypeError`. Mitigation: all new code must use `jax.lax.cond/scan/switch`; fixed-size arrays with masking; test with `jax.jit(f).lower(args).compile()` before training runs.

4. **Empty archive bootstrap failure** — ES mutation called before archive has entries degrades to domain randomization. Mitigation: explicit `init_pop` warm-up phase (256 random latents evaluated before main loop); log `occupied_cells` from step 0.

5. **VAE latent OOD at high sigma** — Mutation sigma above ~0.8 produces degenerate decoded sequences. Mitigation: keep sigma in [0.3, 0.8]; clip mutated latents to [-3, 3]; monitor `solvability_rate` per generation.

## Implications for Roadmap

The dependency chain from research maps directly to a phase structure. Behavior signatures are the universal blocker, so they come first. The buffer extension is cheap and comes immediately after. NS-ES is the MVP and must be validated before the more complex SV-CMA-ES is attempted. Integration and comparison experiments come last.

### Phase 1: Foundation — Baseline Verification and Behavior Extraction

**Rationale:** Agent verification and behavior signature extraction are the two independent prerequisites for all subsequent work. Neither can be skipped. Seed control is cheap and blocks reproducibility.
**Delivers:** Verified baseline agent; behavior signature vectors extracted from rollouts; reproducible training configuration
**Addresses:** Agent Verification vs DCD, Behavior Signature Extraction, Reproducible Seed Control (FEATURES.md table stakes)
**Avoids:** Invalid baseline invalidating all comparisons (Pitfall #10 metric mismatch); JIT incompatibility from day one (Pitfall #3)
**Research flag:** SKIP research-phase — agent verification is code comparison work; behavior extraction is a well-understood extension of existing rollout code

### Phase 2: Buffer and Fitness Infrastructure

**Rationale:** With behavior signatures available, the buffer can be extended and the fitness evaluator built. These are low-complexity, high-leverage changes that unlock all ES strategies.
**Delivers:** Enhanced replay buffer with `behavior_sig` field; k-NN novelty scoring; composite fitness F = α·Regret + β·Novelty; modular ES strategy interface
**Addresses:** Enhanced Replay Buffer, Novelty Score in Fitness, Modular ES Strategy Interface, Fitness Composition API (FEATURES.md)
**Uses:** Pure JAX vectorized k-NN (no new libraries); `LevelSampler.level_extra` extension (STACK.md, ARCHITECTURE.md)
**Avoids:** Novelty reward hacking — design behavior signatures carefully here (Pitfall #5); k-NN scalability — use vectorized JAX with subsampling (Pitfall #6)
**Research flag:** SKIP research-phase — patterns are well-documented; k-NN in JAX is standard

### Phase 3: NS-ES Integration (MVP)

**Rationale:** NS-ES is the MVP contribution. It is the simplest novel algorithm, uses everything built in Phases 1-2, and must be validated before SV-CMA-ES. A working NS-ES is a publishable result on its own.
**Delivers:** Functional NS-ES strategy; end-to-end training with composite fitness; archive warm-up (`init_pop`); two-bucket sampling verified; WandB metrics for comparison
**Addresses:** NS-ES, Two-Bucket Sampling, Regret Curve Comparison Tooling (FEATURES.md)
**Implements:** ES Strategy (NSESStrategy), Enhanced Buffer Manager wired into training loop (ARCHITECTURE.md)
**Avoids:** Empty archive bootstrap (Pitfall #8 — explicit init_pop); two-bucket miscalibration (Pitfall #12 — start at replay_prob=0.7); regret metric mismatch (Pitfall #10 — set --score_function MaxMC)
**Research flag:** NEEDS research-phase — first end-to-end ES integration; wiring ask/tell into `on_mutate_levels`, TrainState extension, and checkpointing together has integration complexity

### Phase 4: Behavioral SV-CMA-ES

**Rationale:** The primary thesis contribution. Built after NS-ES validates the behavior-signature approach. SV-CMA-ES adds N independent CMA-ES particles with Stein repulsion in behavior space — significantly higher complexity than NS-ES.
**Delivers:** Custom SV-CMA-ES implementation; behavior-space repulsion kernel with median bandwidth heuristic; multi-particle diversity enforcement; comparison experiments across CMA-ES / NS-ES / SV-CMA-ES
**Addresses:** Behavioral SV-CMA-ES (FEATURES.md differentiator)
**Uses:** Custom Stein gradient injected between evosax CMA_ES ask/tell steps (STACK.md)
**Avoids:** Kernel bandwidth mismatch (Pitfall #9 — median heuristic, normalized behavior space); CMA-ES mode collapse (Pitfall #1 — addressed by design); archive staleness in long runs (Pitfall #4 — staleness decay enabled)
**Research flag:** NEEDS research-phase — Stein Variational Gradient Descent kernel implementation details; behavior-space normalization choices; multi-particle evosax state management

### Phase 5: Ablations, Analysis, and Future Hooks

**Rationale:** With all three strategies working, ablation studies validate individual contributions. Convergence detection and the LLM injection interface design close out the thesis.
**Delivers:** Ablation study results (fitness weight sweeps); convergence detection with restart; LLM injection interface design; final comparison plots
**Addresses:** Convergence Detection with Restart, LLM Injection Interface (FEATURES.md); Fitness Composition API ablations
**Avoids:** Scope creep into AURORA or non-maze environments (FEATURES.md anti-features)
**Research flag:** SKIP research-phase — standard ablation methodology; convergence detection is well-understood; LLM interface is design-only

### Phase Ordering Rationale

- **Behavior extraction first** because it blocks NS-ES, SV-CMA-ES, and the buffer extension simultaneously. No other phase is possible without it.
- **NS-ES before SV-CMA-ES** because NS-ES validates the behavior-signature approach at lower implementation cost. If NS-ES shows no improvement, SV-CMA-ES assumptions need revisiting.
- **Buffer and fitness infrastructure as a separate phase** because these are shared components. Building them once cleanly, before any strategy implementation, avoids duplication and rework.
- **Integration complexity concentrated in Phase 3** rather than spread across phases, so the first full end-to-end run surfaces all wiring issues before SV-CMA-ES complexity is added.

### Research Flags

Phases likely needing `/gsd:research-phase` during planning:
- **Phase 3 (NS-ES Integration):** First end-to-end wiring of ES ask/tell into JAX training loop with TrainState extensions and Orbax checkpointing — integration surface is broad
- **Phase 4 (Behavioral SV-CMA-ES):** Stein kernel implementation, behavior-space normalization, multi-particle evosax state management — novel algorithm with sparse documentation

Phases with standard patterns (skip research-phase):
- **Phase 1 (Foundation):** Code comparison and rollout extension — no novel patterns
- **Phase 2 (Buffer and Fitness):** JAX vectorized k-NN, LevelSampler extension — well-documented patterns
- **Phase 5 (Ablations):** Standard ML ablation methodology

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All findings grounded in direct codebase inspection; evosax API confirmed; no external library uncertainty |
| Features | HIGH | Dependency graph derived from code; table stakes vs differentiators are unambiguous given thesis goals |
| Architecture | HIGH | Integration points identified from direct code reading; `level_extra` mechanism confirmed; data flow matches existing patterns |
| Pitfalls | HIGH | Pitfalls 1, 2, 4, 7, 8, 10 all have direct evidence in codebase (existing code, comments, parameter names); Pitfalls 3, 5, 6, 9 are standard JAX/ES/novelty-search failure modes |

**Overall confidence:** HIGH

### Gaps to Address

- **Behavior signature dimensionality:** Research establishes visit-count histogram as the design, but grid size and histogram resolution need empirical tuning. Handle during Phase 1 implementation.
- **Fitness weight calibration (α/β):** Recommendation is α=0.8, β=0.2 as starting point. Actual values need ablation (Phase 5). Do not block Phase 3 on perfect weights.
- **CLUTTR repair and latent re-encoding:** Re-encoding repaired sequences via `cluttr_encoder.py` is flagged (Pitfall #11) but not yet in the feature list. Add as a Phase 2 or Phase 3 task to avoid corrupting behavior signature archive with misaligned latents.
- **SV-CMA-ES population size N:** Not determined by research. Literature suggests N=5-20 CMA-ES particles. Needs empirical testing in Phase 4.
- **AURORA:** Deferred to future work. The Flax/Optax stack can support it, but the implementation complexity is out of scope for this milestone.

## Sources

### Primary (HIGH confidence — direct codebase inspection)
- `es/evolve_envs.py` — evosax CMA_ES import, current ES integration point
- `es/regret_fitness.py` — rollout function structure, pure-regret fitness, trajectory data availability
- `es/metrics.py` — existing diversity metrics (latent L2, Hamming); not yet used in fitness
- `es/REGRET_PIPELINE_README.md` — documented CMA-ES mode collapse and regret ceiling
- `map_elites_mutation_service.py` — staleness decay parameter, sigma recommendations, init_pop pattern
- `map_elites.py` — explicit init_pop phase (256 random latents before main loop)
- `jaxued/level_sampler.py` — `level_extra` dict mechanism; insert/update API
- `maze_plr.py` — TrainState definition, `on_mutate_levels` branch structure, score_function flag
- `vae_decoder.py` / `env_bridge.py` / `cluttr_encoder.py` — VAE decode/encode pipeline

### Secondary (MEDIUM confidence — established research literature)
- Stein Variational Gradient Descent (Liu & Wang 2016) — repulsion kernel design for SV-CMA-ES
- Novelty Search with ES (Conti et al. 2018) — NS-ES composite fitness design
- AURORA (Grillotti & Cully 2022) — deferred feature, Flax AE design patterns
- ACCEL (Parker-Holder et al. 2022) — original ACCEL algorithm; DCD implementation reference

### Tertiary (LOW confidence — inference, needs validation)
- Optimal behavior signature dimensionality — inferred from grid sizes; needs empirical tuning
- Optimal α/β fitness weights — community consensus starting point (0.8/0.2); ablation required
- SV-CMA-ES population size N — literature range (5-20); empirical testing required

---
*Research completed: 2026-02-26*
*Ready for roadmap: yes*
