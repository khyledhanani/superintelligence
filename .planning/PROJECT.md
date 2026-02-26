# ES-ACCEL: Evolutionary Strategy Integration for ACCEL UED

## What This Is

An extension of the ACCEL (Adaptive Curriculum via Constrained Evolution in Latent space) framework for Unsupervised Environment Design (UED). It replaces ACCEL's simple random mutation mechanism with intelligent evolutionary strategy (ES) search in VAE latent space, using behaviorally-aware diversity to prevent mode collapse and drive open-ended agent learning on wall-based minigrid environments.

## Core Value

The ES module must find diverse, high-regret environments that continuously challenge the agent — without collapsing to a single mode — so the agent develops generalizable skills through open-ended curriculum learning.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. Existing in codebase. -->

- ✓ JAX-based UED library (jaxued) with environment abstractions — existing
- ✓ Maze environment (wall-based minigrids) with Level/EnvState types — existing
- ✓ PPO agent training loop with GAE advantage estimation — existing
- ✓ PLR/ACCEL level sampler with replay buffer, prioritized sampling, staleness — existing
- ✓ VAE encoder/decoder for minigrid ↔ latent space mapping — existing (checkpoint available)
- ✓ Conv-VAE variant (experimental) — existing
- ✓ CMA-ES evolution in latent space — existing
- ✓ MAP-Elites archive with behavioral descriptors — existing
- ✓ WandB experiment tracking and Orbax checkpointing — existing
- ✓ Environment validation and CLUTTR sequence repair — existing

### Active

<!-- Current scope. Building toward these. -->

- [ ] Modular ES strategy interface — swappable algorithms behind a common API
- [ ] Behavioral SV-CMA-ES implementation — repulsion based on agent trajectory similarity, not latent distance
- [ ] Novelty Search ES implementation — novelty score via k-NN on behavior signatures against replay buffer
- [ ] Modular fitness function — composable fitness beyond scalar regret (regret + novelty + diversity terms)
- [ ] Two-bucket environment generation — replay buffer (consolidation, prob p) + ES frontier (exploration, prob 1-p)
- [ ] Behavior signature extraction — compressed representation of agent's interaction with a level
- [ ] Enhanced replay buffer — stores (latent z, regret, behavior signature) per level, acts as active archive
- [ ] Integration with ACCEL training loop — ES replaces random mutation as the environment generation mechanism
- [ ] Agent verification against DCD repo — confirm PPO/ACCEL agent matches original Facebook Research implementation
- [ ] End-to-end training pipeline — agent trains with ES-generated curriculum, logs metrics, checkpoints
- [ ] Comparison tooling — regret curve comparison between original ACCEL and ES-enhanced variants

### Out of Scope

- LLM-guided search direction injection — deferred to later phase, but ES interface designed to accommodate it
- Non-wall environment types (lava, keys/doors) — wall-only for now
- Rewriting existing PyTorch components in JAX — use as-is, bridge if needed
- Mobile/web UI — research code, CLI/script execution only
- MAP-Elites with fixed behavioral descriptors — redundant with ACCEL's buffer; replaced by behaviorally-aware ES

## Context

**Research lineage:** Built on Dr. Parker Holder's ACCEL paper for UED. The core insight is that ACCEL's random mutations are a crude search — an ES with behavioral diversity awareness should find better curricula.

**Collaborative project:** Multiple contributors working on different components. Friend(s) built the VAE, conv-VAE, and adapted the agent training. User originally built CMA-ES and MAP-Elites integration. This branch (`feat/es-accel-integration`) consolidates and extends.

**Key problem:** Standard CMA-ES collapses to one mode. MAP-Elites with fixed behavioral descriptors becomes obsolete as the agent learns. The ES must define diversity based on the agent's current behavior, not static environment features.

**ES algorithm candidates (shortlist for testing):**

1. **Behavioral SV-CMA-ES** — Stein Variational CMA-ES with repulsive kernel on agent behavior space. Buffer acts as "repulsive anchor" — prevents generating levels that fail the agent in ways already captured.

2. **Novelty Search ES (NS-ES)** — Combined fitness F = α·Regret + β·Novelty. The replay buffer doubles as the novelty archive. k-NN on behavior signatures determines novelty. Most elegant integration since buffer and archive are unified.

3. **AURORA-style ES** — Secondary autoencoder learns behavioral latent space from agent rollouts. Provides dynamic diversity axes that evolve with the agent.

**Original ACCEL repo:** https://github.com/facebookresearch/dcd — agent training code should match this.

## Constraints

- **Framework**: JAX preferred for new code (evosax is JAX-native). Existing PyTorch code kept as-is.
- **ES Library**: evosax — JAX-native evolutionary strategies, already in stack
- **Environment**: Wall-based minigrids only (Maze environment in jaxued)
- **VAE**: Use existing trained checkpoint(s); don't retrain
- **Modularity**: ES strategy must be swappable — will test multiple algorithms and later inject LLM guidance
- **Thesis**: Results need to be reproducible, documented, and comparable between methods

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Behavioral diversity over latent-space diversity | Fixed axes become obsolete as agent learns; behavior-based diversity stays relevant | — Pending |
| Unified buffer as replay + novelty archive | Eliminates redundant data structures; elegant integration with ACCEL loop | — Pending |
| Modular ES interface with shortlist of algorithms | Need to test multiple approaches; unclear which works best for this domain | — Pending |
| JAX-first, bridge PyTorch if needed | evosax and jaxued are JAX; avoid rewriting working code | — Pending |
| Scalar regret insufficient as sole fitness signal | Risk of mode collapse with pure regret; need diversity term in fitness | — Pending |

---
*Last updated: 2026-02-26 after initialization*
