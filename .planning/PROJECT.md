# LLM Diversity Injection for UED Maze Training

## What This Is

Integrating LLM-based maze generation into the JAXUED unsupervised environment design training pipeline. The LLM periodically generates structurally novel, out-of-distribution mazes informed by buffer statistics, injects them into the training buffer via a diversity+learnability gate, and lets existing ACCEL/CMA-ES mutation proliferate them — improving agent generalization beyond what random generation and mutation alone achieve.

## Core Value

LLM-generated mazes must measurably improve agent generalization (solve rate on held-out benchmarks) compared to ACCEL-only and CMA-ES-only baselines.

## Requirements

### Validated

<!-- Existing capabilities from codebase -->

- ✓ JAXUED training loop with DR/PLR/ACCEL/PAIRED — existing
- ✓ Level buffer with prioritized replay and scoring (LevelSampler) — existing
- ✓ ACCEL mutation (wall-flip, 1-3 walls per mutation) — existing
- ✓ CMA-ES search in VAE latent space with PCA dimensionality reduction — existing
- ✓ 3-way training cycle: CMA-ES gen → replay → ACCEL mutate — existing
- ✓ WandB experiment tracking and visualization — existing
- ✓ Checkpoint save/restore via Orbax — existing
- ✓ LLM maze generator (standalone, Claude API via `llm/`) — existing (friend's work)
- ✓ Prompt builder with buffer metric context injection — existing (friend's work)
- ✓ Decision gate for difficulty/diversity filtering — existing (friend's work)
- ✓ Agent evaluator for LLM-generated mazes — existing (friend's work)
- ✓ Diversity metrics: CENIE, DTW, SFL, value error, regret, entropy — existing

### Active

<!-- New work to integrate LLM injection into training -->

- [ ] Connect LLM generator to training loop with periodic batch injection
- [ ] Maze format conversion: LLM text output → JAX Level objects in the buffer
- [ ] Buffer statistics extraction for LLM prompt context (hard mazes, structural patterns)
- [ ] Decision gate integration: filter LLM mazes before buffer insertion
- [ ] Injection frequency as configurable parameter (every N training steps)
- [ ] LLM-injected mazes get mutated by existing ACCEL/CMA-ES mechanisms
- [ ] WandB logging of LLM injection events (count, acceptance rate, diversity delta)
- [ ] Comparison experiments: ACCEL-only vs ACCEL+LLM injection
- [ ] Diversity analysis: buffer metrics before/after LLM injection
- [ ] Ablation experiments on injection frequency and gating thresholds
- [ ] Launch scripts for all experimental conditions (~10-15 runs, 2-3 conditions × 3-5 seeds)

### Out of Scope

- LLM-based mutation — too costly per call; mutation handled by ACCEL/CMA-ES
- Real-time/async LLM generation — periodic batch is sufficient for research
- Multi-provider LLM support — Claude API via existing config is enough
- Training the LLM or fine-tuning it — use off-the-shelf Claude
- New RL algorithms — PPO + ACCEL/PLR stays as-is

## Context

- **Branch**: `llm-injection` — friend's work on LLM generation is here
- **Existing experiments**: 50k-step runs completed across ACCEL baseline, CMA-ES pruned, PCA-CMA-ES with refit, PCA-CMA-ES + ACCEL (see MEMORY.md)
- **Prior results**: PCA-CMA-ES achieved 0.74 solve rate (best seed); vanilla ACCEL mean 0.39; full CMA-ES mean 0.46
- **LLM code status**: Working standalone (`llm/` directory), tested by friend, not yet integrated into training loop
- **Metrics code status**: Working (`metrics/` directory), used for post-training analysis and LLM prompt context
- **Maze format**: 13x13 grid with wall_map (bool), goal_pos, agent_pos, agent_dir — LLM generates text tokens that need conversion to this format
- **Compute**: GPU nodes (albacore 4070 Ti, smew 3090 Ti, canada 3090 Ti), TPU (cma-es-v4)

## Constraints

- **Tech stack**: JAX/Flax training pipeline must not change — LLM injection wraps around existing loop
- **Compute budget**: LLM API calls cost money — injection must be periodic, not per-step
- **Training speed**: LLM calls (seconds) must not block JAX training (milliseconds per step)
- **Maze validity**: All injected mazes must pass `is_well_formatted()` validation
- **Reproducibility**: Experiments must be reproducible with seed control, logged to WandB
- **CUDA 13.1**: GPU scripts need LD_LIBRARY_PATH fix for cuSOLVER compatibility

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Periodic batch injection (not async) | Simpler to implement, sufficient for research, avoids concurrency complexity | — Pending |
| LLM for seed generation only, not mutation | Cost constraint — mutation is cheap via ACCEL/CMA-ES | — Pending |
| Reuse existing decision gate from friend's code | Already implements diversity+learnability filtering | — Pending |
| Use buffer statistics to inform LLM prompts | LLM needs context about what's already in training to generate novel mazes | — Pending |

---
*Last updated: 2026-03-23 after initialization*
