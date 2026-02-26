# Pitfalls Research: ES-ACCEL Integration

**Research Date:** 2026-02-26
**Dimension:** Pitfalls — What commonly goes wrong and how to prevent it

## Critical Pitfalls

### Pitfall 1: CMA-ES Mode Collapse (Single Maximum)

**What goes wrong:** Standard CMA-ES converges to a single high-regret environment topology and produces variations of the same level. The agent overfits to one type of challenge.

**Evidence in codebase:** `es/REGRET_PIPELINE_README.md` documents empirical CMA-ES collapse and regret ceiling.

**Prevention:**
- Never use vanilla CMA-ES as the sole ES without a diversity mechanism
- Use NS-ES or Behavioral SV-CMA-ES from the start
- Monitor population diversity in behavior space, not just latent space

**Phase:** ES Strategy implementation — design for multi-modality from day one.

---

### Pitfall 2: Regret as Sole Fitness Signal → Mode Collapse

**What goes wrong:** Pure regret maximization finds environments that exploit a single agent weakness. The ES "mode collapses" to generating variations that trigger the same failure mode.

**Evidence in codebase:** `es/regret_fitness.py` uses pure regret with no novelty term. `es/metrics.py` tracks latent L2 diversity and Hamming diversity but doesn't use them in fitness.

**Prevention:**
- Composite fitness: F = α·Regret + β·Novelty from the start
- Keep α >> β initially (e.g., 0.8/0.2) — novelty is a tiebreaker
- Monitor `mean_regret` separately from `novelty_score` in WandB
- Include minimum regret threshold for archive insertion

**Phase:** Fitness Evaluator implementation.

---

### Pitfall 3: JAX JIT Incompatibility with Dynamic Control Flow

**What goes wrong:** Behavior signature extraction, k-NN computation, and ES repulsion kernels involve dynamic shapes or Python-side logic that breaks `jax.jit` compilation.

**Prevention:**
- All new code must use `jax.lax.cond`, `jax.lax.scan`, `jax.lax.switch` — no Python if/for inside JIT regions
- Use fixed-size arrays with masking (e.g., buffer has 4000 slots, mask by `size` counter)
- Use JAX vectorized operations, not Python loops
- Test with `jax.jit(f).lower(args).compile()` to catch tracing errors before training runs
- Use `chex.assert_is_jax_array()` at JIT boundaries during development

**Detection:**
- `ConcretizationTypeError` at compile time (explicit — easy to catch)
- Per-iteration time much slower than expected (silent fallback to eager)

**Phase:** Any phase adding new JAX-compiled paths (behavior extraction, k-NN, SV-CMA repulsion).

---

### Pitfall 4: Behavioral Descriptor Staleness (Static Axes vs Dynamic Agent)

**What goes wrong:** MAP-Elites uses fixed behavioral descriptors (obstacle count, BFS path length). These capture *environment structure*, not *agent experience*. As the agent learns, environments that were once hard become easy but remain in the archive with stale high-regret scores.

**Evidence in codebase:** `map_elites_mutation_service.py` implements `staleness_decay_rate` — the parameter exists precisely because this pitfall is anticipated. Default `me_staleness_decay_rate=2e-5`.

**Prevention:**
- Use staleness decay (don't set to 0 in long runs)
- Periodically re-evaluate archive entries against current agent
- For behavioral ES: behavior signature archive must be tied to current agent policy, not frozen checkpoint
- Log `mean_staleness` throughout training

**Phase:** Integration phase — address before long training runs.

---

### Pitfall 5: Novelty Reward Hacking

**What goes wrong:** ES maximizes novelty alone — generating environments where agent behavior is superficially different (e.g., different trajectory length) while regret is low. High novelty metric but no actual learning.

**Prevention:**
- Design behavior signatures to capture *task-relevant* behavior: proportion of steps in each grid region, success/failure, wall collisions. Avoid: raw step count, total reward
- Keep α >> β initially (regret dominant, novelty as tiebreaker)
- Monitor `mean_regret` separately — if novelty grows while regret is flat, reduce β
- Minimum regret threshold for archive insertion

**Detection:**
- `novelty_score` trending up while `mean_regret` flat or declining
- Generated environments are all trivially easy

**Phase:** NS-ES fitness function implementation.

---

### Pitfall 6: k-NN Scalability as Buffer Grows

**What goes wrong:** k-NN over 4000 buffer entries becomes bottleneck if done naively in Python/numpy.

**Prevention:**
- Implement k-NN entirely in JAX (vectorized pairwise distances with masking)
- Use approximate k-NN: subsample 256 entries from buffer for novelty estimation
- Cap novelty archive separately (256-512 entries via reservoir sampling) if full buffer is too expensive
- Profile with `jax.profiler.trace()`

**Detection:** Training step time increases linearly with buffer growth.

**Phase:** NS-ES implementation with buffer-as-archive.

---

### Pitfall 7: VAE Latent Space Out-of-Distribution at High Sigma

**What goes wrong:** When mutation sigma > 1.5, mutated latents leave VAE training distribution. Decoder outputs degenerate sequences (repeated tokens, overlapping positions).

**Evidence in codebase:** `repair_cluttr_sequence` catches some issues but can't recover structurally impossible environments. `map_elites_mutation_service.py` recommends `sigma ≈ 1.5 × bin_width ≈ 0.3-0.5`.

**Prevention:**
- Keep mutation sigma in [0.3, 0.8]
- Monitor `solvability_rate` per generation — if below 40%, reduce sigma
- Clip mutated latents to [-3, 3]
- Add repair-quality check: verify `len(unique(obstacles)) >= min_obstacles`

**Phase:** Any phase involving latent-space mutation.

---

### Pitfall 8: Empty Archive Bootstrap Problem

**What goes wrong:** ES mutation called before archive has entries. Falls back to random latents, which bypass fitness evaluation. Archive stays empty indefinitely — system degrades to domain randomization.

**Evidence in codebase:** `map_elites.py` has explicit `init_pop` phase (256 random latents evaluated before main loop). The integrated `map_elites_mutation_service.py` does NOT have this warm-up.

**Prevention:**
- Run explicit archive initialization: sample and evaluate `init_pop` random latents before training starts
- Log `occupied_cells` from step 0 — if 0 after 100 mutation steps, warm-up failed
- Reduce `replay_prob` during first N training steps to give mutation branch more opportunities

**Phase:** Integration phase (wiring ES into ACCEL loop).

---

## Moderate Pitfalls

### Pitfall 9: SV-CMA-ES Kernel Bandwidth Mismatch

**What goes wrong:** Repulsive kernel bandwidth (h) too small → no diversity enforcement. Too large → search loses coherence.

**Prevention:** Use median heuristic: `h = median(pairwise_distances)^2 / log(n)`. Normalize behavior signatures before computing kernel. Start with low repulsion, increase as distribution stabilizes.

**Phase:** SV-CMA-ES implementation.

---

### Pitfall 10: Regret Metric Mismatch (MaxMC vs PVL)

**What goes wrong:** ES uses MaxMC regret but ACCEL sampler uses PVL. Environments generated by ES get deprioritized by sampler because scores don't align.

**Evidence in codebase:** `regret_fitness.py` notes "this is an ACCEL-inspired regret proxy via max_mc, not the exact ACCEL metric." `maze_plr.py` exposes `--score_function MaxMC|pvl`.

**Prevention:**
- Set `--score_function MaxMC` in ACCEL loop to match ES metric
- Document metric choice explicitly in experiments
- If PVL needed, compute inside training loop using live agent gradients

**Phase:** Integration — resolve before comparison experiments.

---

### Pitfall 11: CLUTTR Repair Masking Latent Space Geometry

**What goes wrong:** Repaired sequences don't correspond to their stored latent z. Parent sampling uses wrong latents, mutations explore wrong neighborhood.

**Prevention:**
- Re-encode repaired sequences back to latent space using VAE encoder (`es/cluttr_encoder.py`)
- Minimize repair by keeping sigma low
- Penalize fitness for environments requiring heavy repair

**Phase:** Any phase storing latents alongside decoded sequences.

---

### Pitfall 12: Two-Bucket Probability Miscalibration

**What goes wrong:** replay_prob too high (>0.9) → ES has no impact. Too low (<0.5) → agent trains on bad ES environments early.

**Prevention:**
- Start with `replay_prob=0.7`
- Log fraction from each bucket separately
- Verify `mutation_updates / total_updates > 0.2`

**Phase:** Integration phase parameter tuning.

---

## Phase-Specific Warning Summary

| Phase Topic | Key Pitfall | Mitigation |
|-------------|------------|------------|
| Behavior signature extraction | JIT incompatibility (#3) | Use `jax.lax.*`, test with `jit.lower()` |
| NS-ES fitness function | Novelty reward hacking (#5) | α >> β; monitor regret separately |
| SV-CMA-ES repulsion | Kernel bandwidth (#9) | Median heuristic; normalize behavior space |
| k-NN novelty scoring | Scalability (#6) | Vectorized JAX; subsample archive |
| First end-to-end run | Empty archive (#8) | Explicit init_pop phase |
| Long training runs | Archive staleness (#4) | Staleness decay; periodic re-evaluation |
| Comparison experiments | Metric mismatch (#10) | Align score_function between ES and ACCEL |
| Latent mutation | OOD sigma (#7) | Cap sigma at 0.8; monitor solvability |

---
*Pitfalls research: 2026-02-26*
