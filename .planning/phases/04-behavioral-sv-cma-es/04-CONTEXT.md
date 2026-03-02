# Phase 4: Behavioral SV-CMA-ES - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement SVCMAESStrategy: N independent CMA-ES particles that maintain behavioral diversity via Stein repulsion in behavior space. Each step all N particles generate candidates, evaluated environments are used to compute Stein gradients over behavior signatures, and particle means are pushed apart. The strategy integrates into the existing train.py ES routing alongside CMAESStrategy and NSESStrategy. Fitness ablations and plotting are Phase 5.

</domain>

<decisions>
## Implementation Decisions

### Multi-particle training loop flow
- All N particles run every step (not round-robin): each step all particles call ask(), evaluate, apply repulsion, then tell()
- Concatenate all N*pop_size candidate latents into a single batch → single eval_fn call → split results back by particle after evaluation
- Order of operations per step:
  1. All N particles call ask() → N*pop_size candidate latents
  2. Evaluate all N*pop_size candidates (decode → rollout → extract behavior sigs + regrets)
  3. Compute Stein gradient using pre-repulsion behavior sigs across particles
  4. Apply Stein gradient to candidate latents (nudge by epsilon)
  5. Re-evaluate repelled latents (second eval pass) → final behavior sigs + regrets
  6. Each particle calls tell() with its own repelled candidates and regrets
  7. After tell(), update each particle's CMA mean using the Stein gradient
  8. PLR buffer receives post-repulsion candidates and their re-evaluated regrets

### Stein kernel and repulsion mechanics
- Kernel: RBF (Gaussian) with median heuristic bandwidth — `h = median(pairwise_sq_dists)^2 / log(N)`, computed fresh each step from the current particle behavior signatures
- Repulsion target: CMA means are adjusted AFTER tell(), not candidate latents before tell()
  - Note: roadmap success criterion 2 wording "candidate latents adjusted before tell()" is imprecise — the actual intent is means adjusted after tell()
  - The Stein gradient is computed during the candidate evaluation phase (using behavior sigs) and applied to means post-tell()
- Repulsion step size epsilon: fixed value from config, default 0.01
- Fitness for each particle's tell(): pure regret only (no composite fitness) — Stein repulsion replaces the novelty bonus from NS-ES

### Particle initialization
- N particles start from random means: each particle's mean initialized from N(0, sigma_init) with a different RNG key — not zeros like CMAESStrategy
- All N particles share the same sigma_init (from config), independently seeded
- Internal state structure: list of N CMAESStrategy instances, each with its own state dict — no JAX vmap batching
- N=1 degrades gracefully to plain CMA-ES: repulsion step is skipped when N=1, behavior is identical to CMAESStrategy baseline

### WandB observability
- Log aggregate only: `mean_pairwise_behavior_dist` (scalar) — mean over all particle-pair distances in behavior space, logged every wandb_log_freq steps
- Log before-repulsion and after-repulsion mean values as two separate metrics per step: `sv_behavior_dist_pre` and `sv_behavior_dist_post`
- No automatic collapse detection or early stopping — user inspects WandB curves manually
- Per-particle individual metrics are NOT logged (too noisy for thesis plots)

### Claude's Discretion
- Exact implementation of median heuristic bandwidth (numerical stability, epsilon floor to avoid divide-by-zero)
- How to handle the case where all particles happen to have identical behavior sigs (zero-gradient edge case)
- Whether to clip or normalize the Stein gradient before applying it to means
- Config key names for new hyperparameters (epsilon, n_particles)
- File location for SVCMAESStrategy (alongside nses_strategy.py in es_components/)
- Test structure for the new strategy

</decisions>

<specifics>
## Specific Ideas

- The train.py ES routing block currently branches on `es_strategy_name in {"ns_es", "cma_es"}` — `sv_cma_es` needs its own branch
- The `--n_particles` CLI flag needs wiring through argparse → config dict → SVCMAESStrategy constructor
- Success criterion 2 in the roadmap needs wording update: "CMA means are adjusted after tell() using the Stein gradient computed from particle behavior signatures"
- The two eval passes per step (pre-repulsion and post-repulsion) will roughly double the compute cost per new step compared to NS-ES — this is expected and acceptable for the thesis

</specifics>

<deferred>
## Deferred Ideas

- None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-behavioral-sv-cma-es*
*Context gathered: 2026-03-02*
