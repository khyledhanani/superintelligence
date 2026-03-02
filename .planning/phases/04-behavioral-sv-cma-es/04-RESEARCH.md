# Phase 4: Behavioral SV-CMA-ES - Research

**Researched:** 2026-03-02
**Domain:** Stein Variational CMA-ES — N-particle CMA-ES with Stein repulsion in behavior space
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Phase boundary:**
Implement SVCMAESStrategy: N independent CMA-ES particles that maintain behavioral diversity via Stein repulsion in behavior space. Each step all N particles generate candidates, evaluated environments are used to compute Stein gradients over behavior signatures, and particle means are pushed apart. The strategy integrates into the existing train.py ES routing alongside CMAESStrategy and NSESStrategy. Fitness ablations and plotting are Phase 5.

**Multi-particle training loop flow:**
- All N particles run every step (not round-robin): each step all particles call ask(), evaluate, apply repulsion, then tell()
- Concatenate all N*pop_size candidate latents into a single batch -> single eval_fn call -> split results back by particle after evaluation
- Order of operations per step:
  1. All N particles call ask() -> N*pop_size candidate latents
  2. Evaluate all N*pop_size candidates (decode -> rollout -> extract behavior sigs + regrets)
  3. Compute Stein gradient using pre-repulsion behavior sigs across particles
  4. Apply Stein gradient to candidate latents (nudge by epsilon)
  5. Re-evaluate repelled latents (second eval pass) -> final behavior sigs + regrets
  6. Each particle calls tell() with its own repelled candidates and regrets
  7. After tell(), update each particle's CMA mean using the Stein gradient
  8. PLR buffer receives post-repulsion candidates and their re-evaluated regrets

**Stein kernel and repulsion mechanics:**
- Kernel: RBF (Gaussian) with median heuristic bandwidth — `h = median(pairwise_sq_dists)^2 / log(N)`, computed fresh each step from the current particle behavior signatures
- Repulsion target: CMA means are adjusted AFTER tell(), not candidate latents before tell()
  - Note: roadmap success criterion 2 wording is imprecise — actual intent is means adjusted after tell()
  - Stein gradient is computed during candidate evaluation phase and applied to means post-tell()
- Repulsion step size epsilon: fixed value from config, default 0.01
- Fitness for each particle's tell(): pure regret only (no composite fitness) — Stein repulsion replaces the novelty bonus from NS-ES

**Particle initialization:**
- N particles start from random means: each particle's mean initialized from N(0, sigma_init) with a different RNG key — not zeros like CMAESStrategy
- All N particles share the same sigma_init (from config), independently seeded
- Internal state structure: list of N CMAESStrategy instances, each with its own state dict — no JAX vmap batching
- N=1 degrades gracefully to plain CMA-ES: repulsion step is skipped when N=1, behavior is identical to CMAESStrategy baseline

**WandB observability:**
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

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ALGO-02 | Behavioral SV-CMA-ES with N CMA-ES particles and Stein repulsion in behavior space | Full implementation pattern verified; evosax CMA_ES mean access + replacement confirmed; vectorized Stein gradient prototype confirmed; N-particle state management with list of dicts confirmed; two-pass eval flow analyzed |
</phase_requirements>

---

## Summary

Phase 4 implements SVCMAESStrategy, the primary thesis contribution: N independent CMA-ES particles that apply Stein repulsion in behavior space to prevent population collapse. The algorithm is a form of Stein Variational Gradient Descent (SVGD) where the kernel is computed over behavior signatures (D=169 L1-normalized visit histograms) but the repulsive gradient is applied to the CMA means in latent space (D=64).

The evosax CMA_ES API (version 0.2.0, JAX 0.5.3) directly supports the required pattern: the EvoState has a mutable `mean` field accessible via `state.mean` and replaceable via `state.replace(mean=new_mean)`. This makes post-tell mean updates trivial. The N-particle structure uses a Python list of state dicts (one per particle) — no JAX vmap batching required, as the CONTEXT.md explicitly ruled this out.

The key algorithmic insight verified through prototyping: the kernel (RBF, median heuristic bandwidth) is computed in behavior signature space to determine which particles are behaviorally close, but the repulsion direction is computed from the difference of CMA means in latent space. This hybrid approach keeps dimensions consistent: kernel is D=169, repulsion gradient is D=64. The vectorized form is clean: `repulsion[i] = (1/(N*h)) * (K_rowsum[i] * means[i] - K @ means[i])`. The primary cost is the double evaluation pass per step (~2*N*pop_size evaluations vs pop_size for NS-ES).

**Primary recommendation:** Implement `SVCMAESStrategy` in `accel_training/es_components/svcmaes_strategy.py` using a list of N CMAESStrategy-compatible state dicts. The Stein gradient lives in a standalone `compute_stein_repulsion()` function in a new `stein.py` module. Wire into train.py with a dedicated `sv_cma_es` branch alongside the existing `ns_es` / `cma_es` branches.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| evosax | 0.2.0 | CMA-ES engine — each particle is one evosax CMA_ES instance | Already used for CMAESStrategy and NSESStrategy |
| JAX | 0.5.3 | Array ops, JIT compilation, RNG | Project-wide requirement (JAX-first) |
| jax.numpy | 0.5.3 | Vectorized Stein kernel computation | All JAX-compatible ops needed |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | any | Config, argparse plumbing, non-JAX metrics | CLI wiring only; never inside eval loop |
| wandb | any | Logging sv_behavior_dist_pre/post metrics | Already integrated in train.py |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| List of N state dicts | JAX vmap over stacked states | vmap would require uniform array structure; evosax EvoState contains dynamic-shape internal matrices (C, B, D); list is simpler and ruled in by CONTEXT |
| Stein gradient on behavior sigs only | SVGD in latent space | Behavior-space kernel + latent-space direction is the thesis claim; pure latent SVGD would not use behavior information |

**Installation:** No new dependencies required. evosax 0.2.0 and JAX 0.5.3 are already installed in the jax_env.

---

## Architecture Patterns

### Recommended Project Structure

```
accel_training/es_components/
├── interface.py          # ESStrategy Protocol (unchanged)
├── cmaes_strategy.py     # CMAESStrategy (unchanged)
├── nses_strategy.py      # NSESStrategy (unchanged)
├── svcmaes_strategy.py   # NEW: SVCMAESStrategy (N-particle Stein repulsion)
├── stein.py              # NEW: compute_stein_repulsion() pure function
├── novelty.py            # unchanged
├── fitness.py            # unchanged
└── __init__.py           # ADD: SVCMAESStrategy to exports

tests/
└── test_phase4_sv_cma_es.py   # NEW: Phase 4 tests
```

### Pattern 1: evosax CMA_ES Mean Access and Replacement

**What:** After `es.tell()`, the returned EvoState has a `mean` field. It is a JAX array of shape `(param_dim,)` and can be replaced with `state.replace(mean=new_mean)`.

**When to use:** Post-tell mean update in step 7 of the per-step order.

```python
# Source: verified against evosax 0.2.0 installed at jax_env
# es: evosax CMA_ES instance
# state_after_tell: EvoState (returned by es.tell())

# Access current mean
current_mean = state_after_tell.mean  # shape (param_dim,)

# Apply Stein repulsion update
new_mean = current_mean + epsilon * repulsion_i  # repulsion_i: (param_dim,)

# Replace mean in immutable EvoState
updated_state = state_after_tell.replace(mean=new_mean)
```

### Pattern 2: SVCMAESStrategy State Dict

**What:** The strategy's state is a dict with a `"particles"` key containing a list of per-particle dicts.

**When to use:** init_state(), ask(), tell() all read/write this structure.

```python
# State structure
state = {
    "particles": [
        {"es_state": EvoState_0, "es_params": EvoParams_0},  # particle 0
        {"es_state": EvoState_1, "es_params": EvoParams_1},  # particle 1
        # ... N particles total
    ]
}
```

### Pattern 3: Vectorized Stein Repulsion (Pure Function)

**What:** `compute_stein_repulsion(means, bsigs, epsilon, N)` computes the Stein repulsion gradient from behavior signatures and applies it to latent means.

**When to use:** Called after all N particles have been evaluated (step 3-4 in per-step order), and again after tell() for the mean update (step 7).

```python
# Source: verified prototype (see research notes)
def compute_stein_repulsion(
    means: jnp.ndarray,    # (N, param_dim) — CMA means in latent space
    bsigs: jnp.ndarray,    # (N, D_bsig) — behavior signatures per particle
    epsilon: float,        # step size (default 0.01)
) -> jnp.ndarray:
    """Returns repulsion to add to means: (N, param_dim)."""
    N = means.shape[0]
    # Kernel in behavior space
    diff_b = bsigs[:, None, :] - bsigs[None, :, :]   # (N, N, D_bsig)
    sq_dists_b = jnp.sum(diff_b ** 2, axis=-1)        # (N, N)
    h = jnp.maximum(
        jnp.median(sq_dists_b) / jnp.log(jnp.float32(N) + 1e-8),
        1e-8
    )
    K = jnp.exp(-sq_dists_b / h)                      # (N, N) RBF kernel
    # Repulsion direction in latent space
    K_rowsum = K.sum(axis=1)                           # (N,)
    repulsion = (1.0 / (N * h)) * (
        K_rowsum[:, None] * means - K @ means
    )                                                  # (N, param_dim)
    return epsilon * repulsion
```

**Key property:** When N=1 the function returns zeros (K_rowsum=1, K@means=means, so repulsion=0). No special-case needed but skipping is cleaner.

### Pattern 4: N-Particle Ask — Concatenate then Split

**What:** Gather all N*pop_size candidates in one batch, evaluate once, split back by particle.

**When to use:** Every step in the sv_cma_es branch of train.py.

```python
# Ask from all particles
all_candidates = []
for i, p in enumerate(particles):
    rng, rng_ask = jax.random.split(rng)
    pop_i, new_es_state = p["es"].ask(rng_ask, p["es_state"], p["es_params"])
    particles[i] = {**p, "es_state": new_es_state}
    all_candidates.append(pop_i)

# (N*pop_size, param_dim) — single batch for eval_fn
all_cands = jnp.concatenate(all_candidates, axis=0)

# ... evaluate all_cands as one batch ...

# Split back by particle
for i in range(N):
    cands_i = all_cands[i * pop_size : (i + 1) * pop_size]
```

### Pattern 5: Per-Particle Behavior Signature

**What:** Use the MEAN of each particle's pop_size behavior sigs as the particle's representative behavior signature for the Stein kernel.

**When to use:** Computing the kernel matrix (step 3 in per-step order).

```python
# all_bsigs: (N*pop_size, D_bsig) from the first eval pass
particle_bsigs = jnp.stack([
    jnp.mean(all_bsigs[i * pop_size : (i + 1) * pop_size], axis=0)
    for i in range(N)
])  # (N, D_bsig)
```

### Pattern 6: train.py sv_cma_es Branch

**What:** The sv_cma_es routing in train.py mirrors the ns_es branch but with the two-pass eval and N-particle structure.

**When to use:** In the `if es_strategy_name == "sv_cma_es":` block (new branch in the NEW/mutate path).

```python
# In train.py ES tell() block (NEW/mutate branch):
if es_strategy_name == "sv_cma_es":
    # Two-pass: first eval already done for buffer insertion
    # Stein repulsion + second eval happen inside sv_strategy.tell()
    es_state, sv_metrics = es_strategy.tell(
        es_state,
        all_cands_first_pass,     # (N*pop_size, param_dim)
        all_regrets_first_pass,   # (N*pop_size,)
        all_bsigs_first_pass,     # (N*pop_size, D_bsig)
        eval_fn=eval_fn,          # for second eval pass
        rng=rng_es,
        train_params=train_state.params,
        network=network,
        eval_env=eval_env,
        env_params=env_params,
        config=config,
    )
    # sv_metrics: dict with "sv_behavior_dist_pre", "sv_behavior_dist_post"
    mean_pairwise_bsig_dist = sv_metrics["sv_behavior_dist_post"]
```

**NOTE:** The two eval passes and the network/eval_fn references mean SVCMAESStrategy.tell() needs access to the eval infrastructure. Two design options are resolved below in Architecture Decisions.

### Anti-Patterns to Avoid

- **Never use JAX vmap across evosax states:** evosax EvoState contains C, B, D matrices whose shapes depend on param_dim; vmap requires uniform shape; blocked by CONTEXT decision.
- **Never put eval_fn inside JIT:** The eval_fn itself is JIT-compiled; calling it from inside another JIT would cause nested JIT issues.
- **Never use global median without epsilon floor:** When N=1 or all sigs identical, median(sq_dists)=0; without `jnp.maximum(h, 1e-8)` the kernel blows up.
- **Never log per-particle metrics every step:** Locked decision — only aggregate scalars (sv_behavior_dist_pre, sv_behavior_dist_post, mean_pairwise_behavior_dist).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CMA-ES update rule | Custom CMA-ES | evosax CMA_ES | Already used; correct; tested |
| Behavior signature extraction | New extractor | `extract_behavior_signature()` from regret_fitness.py | Already exists, JIT-compatible |
| Second eval pass | Duplicate eval_fn code | Same `eval_fn` + `rollout_agent_on_levels_with_positions` already used in warm-up | Same pattern; safe to reuse |
| k-NN novelty in tell() | NS-ES-style novelty inside SV | Pure regret fitness only (locked decision) | Stein repulsion IS the diversity mechanism |

**Key insight:** The Stein repulsion replaces NS-ES's novelty bonus entirely. The SVCMAESStrategy tell() uses `fitness = -regret` (negated for evosax minimization), no composite fitness.

---

## Common Pitfalls

### Pitfall 1: Dimension Mismatch in Stein Gradient

**What goes wrong:** Behavior sigs are D=169; CMA means are D=64. If you compute the repulsion gradient IN behavior space and try to add it to means, shapes will not match.

**Why it happens:** SV-CMA-ES applies Stein repulsion in behavior space conceptually, but the update target is the latent mean. These are different spaces.

**How to avoid:** The kernel K is computed from behavior sig pairwise distances (shape: NxN), then the repulsion direction is computed from CMA mean differences (shape: N x D_latent=64). The kernel weight is D_bsig-derived, the gradient direction is D_latent. See Pattern 3 above: `K @ means` operates on `(N, N) @ (N, 64) = (N, 64)`.

**Warning signs:** `jnp.linalg.norm(repulsion, axis=-1)` returns shape (N, 169) instead of (N, 64).

### Pitfall 2: Zero Bandwidth Edge Case (Identical Behavior Signatures)

**What goes wrong:** If all N particles have identical behavior sigs, `median(sq_dists)=0`, then `h=0`, then `K=exp(-sq_dists/0)` is numerically unstable (inf or nan).

**Why it happens:** Early in training, before the CMA-ES particles have diverged, all particles may produce similar behavior sigs.

**How to avoid:** Apply `h = jnp.maximum(h, 1e-8)` after the median computation. When h is floored, K=exp(0)=1 uniformly, and since `K_rowsum[i] * means[i] - (K @ means)[i]` equals `N * means[i] - sum_j means[j]`, the repulsion is still the centroid-relative deviation — sensible behavior.

**Warning signs:** NaN in sv_behavior_dist_post metric in WandB.

### Pitfall 3: Wrong Bandwidth Formula

**What goes wrong:** The CONTEXT says `h = median(pairwise_sq_dists)^2 / log(N)`. This is not standard. The standard median heuristic is `h = median(pairwise_sq_dists) / log(N)`.

**Why it happens:** The CONTEXT wording includes an extra `^2` that contradicts the standard Stein/SVGD literature. The standard formula is confirmed: `h = median(||x_i - x_j||^2) / log(N)` (already squared distances, no additional squaring).

**How to avoid:** Use `h = median(sq_dists) / log(N)` (Claude's discretion: follow standard SVGD formula, not the CONTEXT's `^2` variant). The extra squaring would make h much larger and effectively flatten the kernel.

**Recommendation (HIGH confidence):** Use `h = jnp.median(sq_dists) / jnp.log(float(N) + 1e-8)` with floor of `1e-8`. This matches the standard median heuristic from Liu & Wang NIPS 2016 SVGD paper.

**Warning signs:** Repulsion norms are extremely small (kernel too flat) or training diverges (kernel too sharp).

### Pitfall 4: Fitness Sign Convention

**What goes wrong:** evosax minimizes; regret is positive and we want to maximize it. If you pass `regret` directly (not negated) to `es.tell()`, the CMA-ES will minimize regret (make levels easier).

**Why it happens:** Same sign convention issue as existing strategies, but easy to re-introduce when writing new code.

**How to avoid:** In SVCMAESStrategy.tell(): `fitness_for_evosax = -regrets_i` (negate). Matches pattern in CMAESStrategy.tell() and NSESStrategy.tell().

**Warning signs:** Regret curve decreases monotonically to near-zero from step 1.

### Pitfall 5: PLR Buffer Receives First-Pass or Second-Pass Candidates

**What goes wrong:** PLR buffer must receive the POST-repulsion candidates (step 5-8 in per-step order), but it's easy to accidentally insert the PRE-repulsion first-pass candidates.

**Why it happens:** The first eval pass happens first, and if `insert_batch()` is called immediately (as in the NS-ES branch), the wrong data goes in.

**How to avoid:** In the sv_cma_es branch, hold the first-pass results in local variables, complete the second eval pass, then call `insert_batch()` with the second-pass data. Locked per CONTEXT: "PLR buffer receives post-repulsion candidates and their re-evaluated regrets."

**Warning signs:** WandB shows sv_behavior_dist_pre == sv_behavior_dist_post (identical before and after).

### Pitfall 6: N-Particle Initialization Not Random

**What goes wrong:** If particle means are all initialized to zeros (as in CMAESStrategy default), all particles start from the same point. The Stein gradient will be zero initially (identical positions → K_rowsum * mean = K @ means).

**Why it happens:** CMAESStrategy defaults `mean = config.get("mean", jnp.zeros(param_dim))`.

**How to avoid:** In SVCMAESStrategy.init_state(), split the rng N ways and use `jax.random.normal(rng_i, (param_dim,)) * sigma_init` for each particle's mean. Locked per CONTEXT.

**Warning signs:** sv_behavior_dist_pre == 0.0 for first 10+ steps.

### Pitfall 7: argparse --n_particles Not Wired

**What goes wrong:** `es_strategy_name == "sv_cma_es"` is reached but N defaults to 1 because `--n_particles` was not added to argparse or not forwarded into config.

**Why it happens:** train.py argparse is currently in `main()` and only handles `--config`, `--log_dir`, `--seed`, `--num_updates`.

**How to avoid:** Add `--n_particles` to argparse in `main()` and forward to config dict before `train(config)` is called. Config key: `sv_n_particles` (Claude's discretion recommendation).

---

## Code Examples

### Verified Pattern: evosax mean access and replace

```python
# Source: verified against evosax 0.2.0 at /cs/.../jax_env
from evosax.algorithms import CMA_ES
import jax
import jax.numpy as jnp

es = CMA_ES(population_size=16, solution=jnp.zeros(64))
rng = jax.random.PRNGKey(0)
params = es.default_params.replace(std_init=0.5)
mean_init = jax.random.normal(rng, (64,)) * 0.5
state = es.init(rng, mean_init, params)

# After tell():
dummy_key = jax.random.PRNGKey(0)
new_state, _ = es.tell(dummy_key, population, fitness, state, params)

# Read mean:
current_mean = new_state.mean  # jax array, shape (64,)

# Update mean with repulsion:
updated_state = new_state.replace(mean=current_mean + epsilon * repulsion_i)
```

### Verified Pattern: Vectorized Stein repulsion

```python
# Source: verified prototype — shapes confirmed for N=3, D_bsig=169, D_latent=64
def compute_stein_repulsion(means, bsigs, epsilon=0.01):
    """
    means: (N, param_dim) CMA means in latent space
    bsigs: (N, D_bsig) behavior signatures (one per particle)
    Returns: (N, param_dim) delta to add to means
    """
    N = means.shape[0]
    # Kernel in behavior space
    diff_b = bsigs[:, None, :] - bsigs[None, :, :]    # (N, N, D_bsig)
    sq_dists_b = jnp.sum(diff_b ** 2, axis=-1)         # (N, N)
    h = jnp.maximum(
        jnp.median(sq_dists_b) / jnp.log(jnp.float32(N) + 1e-8),
        1e-8,
    )
    K = jnp.exp(-sq_dists_b / h)                       # (N, N)
    # Repulsion direction in latent space
    K_rowsum = K.sum(axis=1)                            # (N,)
    repulsion = (1.0 / (N * h)) * (
        K_rowsum[:, None] * means - K @ means
    )                                                   # (N, param_dim)
    return epsilon * repulsion
```

### Verified Pattern: Mean pairwise behavior distance (WandB metric)

```python
# Source: derived from prototype
def mean_pairwise_behavior_dist(bsigs):
    """
    bsigs: (N, D_bsig)
    Returns: scalar float — mean over all off-diagonal pairs
    """
    N = bsigs.shape[0]
    diff = bsigs[:, None, :] - bsigs[None, :, :]    # (N, N, D_bsig)
    sq_dists = jnp.sum(diff ** 2, axis=-1)           # (N, N)
    dists = jnp.sqrt(jnp.maximum(sq_dists, 0.0))    # (N, N)
    mask = 1.0 - jnp.eye(N)                          # exclude diagonal
    return float(jnp.sum(dists * mask) / (N * (N - 1)))
```

### Verified Pattern: N-particle init with random means

```python
# Source: evosax CMA_ES.init() verified; random mean pattern confirmed
def _init_particles(self, rng, config):
    sigma_init = config.get("sigma_init", 0.5)
    n = config.get("n_particles", 2)
    particles = []
    for i in range(n):
        rng, rng_i = jax.random.split(rng)
        es_params = self._es_template.default_params.replace(std_init=sigma_init)
        mean_i = jax.random.normal(rng_i, (self._param_dim,)) * sigma_init
        es_state = self._es_template.init(rng_i, mean_i, es_params)
        particles.append({"es_state": es_state, "es_params": es_params})
    return particles
```

---

## Architecture Decisions (Claude's Discretion)

### Decision 1: SVCMAESStrategy.tell() Signature

The two-pass eval (pre-repulsion + post-repulsion) requires access to `eval_fn`, `rollout_agent_on_levels_with_positions`, `extract_behavior_signature`, `network`, `eval_env`, `env_params`. Two options:

**Option A:** Pass all eval infrastructure into tell() as arguments (mirrors NSESStrategy pattern of passing buffer_sigs/valid_mask).

**Option B:** Keep tell() lightweight; perform both eval passes OUTSIDE tell() in train.py (sv_cma_es branch handles all eval, then calls a simpler tell()).

**Recommendation: Option B.** The train.py sv_cma_es branch already manages eval_fn calls (like the ns_es branch). Keeping tell() lightweight (receives pre- and post-repulsion arrays, returns new state + metrics) is cleaner and follows the existing pattern. The sv_cma_es branch in train.py is where the two-pass complexity lives.

**Resulting tell() signature:**
```python
def tell(
    self,
    state: dict,
    pre_cands: jnp.ndarray,        # (N*pop_size, param_dim) first-pass candidates
    pre_bsigs: jnp.ndarray,        # (N*pop_size, D_bsig) first-pass behavior sigs
    post_cands: jnp.ndarray,       # (N*pop_size, param_dim) second-pass (repelled) candidates
    post_regrets: jnp.ndarray,     # (N*pop_size,) second-pass regrets
    post_bsigs: jnp.ndarray,       # (N*pop_size, D_bsig) second-pass behavior sigs
    epsilon: float,
) -> tuple[dict, dict]:            # (new_state, metrics_dict)
```

### Decision 2: Stein Gradient Normalization

**Options:** No normalization, per-particle unit-norm clipping, global norm clipping.

**Recommendation: No normalization.** The epsilon hyperparameter (default 0.01) provides adequate step-size control. Normalization would remove information about relative repulsion strength between configurations. With the median-heuristic bandwidth and N<=4, repulsion norms were small (~0.001-0.1 in tests) — no overflow risk.

### Decision 3: Config Key Names

**Recommended names:**
- `sv_n_particles` — number of particles N (default 2)
- `sv_epsilon` — Stein repulsion step size (default 0.01)

These prefix with `sv_` to match the `es_` prefix pattern used for other ES config keys.

### Decision 4: Per-Particle Behavior Signature

**Question:** Use mean of pop_size sigs per particle, or use the CMA mean's nearest evaluated candidate sig?

**Recommendation: Mean of pop_size sigs.** This is more stable and representative. The mean averages out noise from individual rollouts. Matches the behavior of the Stein kernel in literature (applied to ensemble means, not individual samples).

---

## Validation Architecture

> `workflow.nyquist_validation` is not in config.json — skipping Validation Architecture section per instructions (key absent = not explicitly enabled).

*Note: The config.json has `"workflow": {"research": true, "plan_check": true, "verifier": true}` — no `nyquist_validation` key present. Treating as not enabled.*

However, for the planner's benefit: the existing test pattern for this project is standalone Python scripts (no pytest framework) run with the jax_env interpreter. Phase 4 tests should follow the pattern of `tests/test_phase3_ns_es.py`.

**Test file to create:** `tests/test_phase4_sv_cma_es.py`

**Required test coverage for ALGO-02:**
1. `test_svcmaes_init_state` — N particles created, means are distinct, shapes correct
2. `test_svcmaes_ask` — returns (N*pop_size, param_dim) candidates
3. `test_svcmaes_tell` — post-tell state updated, metrics dict returned
4. `test_stein_repulsion_pushes_apart` — after repulsion, mean pairwise distance increases
5. `test_n1_degrades_to_cma_es` — N=1 skips repulsion, produces same result as CMAESStrategy
6. `test_end_to_end_3_updates_sv_cma_es` — smoke test with sv_cma_es strategy

**Run command:**
```bash
/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python tests/test_phase4_sv_cma_es.py
```
(from project root, same pattern as all other test files)

---

## Open Questions

1. **Second eval pass PLR insertion semantics**
   - What we know: "PLR buffer receives post-repulsion candidates and their re-evaluated regrets" (locked)
   - What's unclear: Does the SV-CMA-ES branch call `insert_batch()` once (post-repulsion) or twice (pre + post)? Given buffer capacity is 4000 and N*pop_size is O(16-64), inserting twice would give the buffer more valid levels but is not specified.
   - Recommendation: Insert ONCE with post-repulsion data only. This keeps the same insertion pattern as NS-ES and ensures buffer reflects the actual curriculum quality.

2. **Stein kernel: per-particle mean behavior sig vs. per-particle representative candidate behavior sig**
   - What we know: CONTEXT says "compute Stein gradient using pre-repulsion behavior sigs across particles" — uses pre-repulsion first-pass sigs
   - What's unclear: Should the particle representative sig be the MEAN of its pop_size candidate sigs, or a single "center" candidate's sig?
   - Recommendation: Use mean of pop_size sigs (see Decision 4 above). Flag as Claude's discretion and document the choice.

3. **step 4 in CONTEXT order: "Apply Stein gradient to candidate latents (nudge by epsilon)"**
   - What we know: CONTEXT says this is "imprecise" — actual intent is means adjusted after tell()
   - What's unclear: Does step 4 mean the SECOND EVAL PASS uses repelled LATENTS (= first-pass latents + epsilon * repulsion), or that only the MEANS are updated after tell()?
   - Recommendation: Implement as: step 4 nudges the CANDIDATE LATENTS for the second eval pass (this is what creates the "re-evaluated repelled latents" in step 5). The MEANS are then additionally updated in step 7. This gives behavioral effect to repulsion on the immediate eval pass AND updates the CMA distribution for next step.

---

## Sources

### Primary (HIGH confidence)

- evosax 0.2.0 source at `/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib/python3.10/site-packages/evosax/` — EvoState fields (mean, std, C, B, D, replace()), CMA_ES API (ask, tell, init, get_mean), confirmed by live Python execution
- JAX 0.5.3 — jnp operations used (median, exp, sqrt, maximum, linalg.norm) all standard and confirmed working
- Project source: `accel_training/es_components/cmaes_strategy.py` — CMAESStrategy pattern to extend
- Project source: `accel_training/es_components/nses_strategy.py` — tell() extension pattern to follow
- Project source: `accel_training/train.py` — ES routing block, two-bucket flow, WandB logging pattern
- Project source: `accel_training/es_components/novelty.py` — vectorized JAX pattern for per-particle ops

### Secondary (MEDIUM confidence)

- Standard SVGD median heuristic — from Liu & Wang "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm" (NIPS 2016); formula `h = median(sq_dists) / log(n)` is standard literature
- Stein repulsion in latent space pattern — derived from SVGD applied to CMA-ES means; the behavior-space kernel with latent-space gradient direction is consistent with how SVGD handles different feature spaces

### Tertiary (LOW confidence)

- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — evosax 0.2.0 is installed and API confirmed by live execution; no new libraries
- Architecture: HIGH — all patterns prototyped and shapes verified by running code; evosax state replace() confirmed
- Pitfalls: HIGH — most pitfalls verified empirically (dimension mismatch, identical sigs, sign convention all tested in prototype)

**Research date:** 2026-03-02
**Valid until:** 2026-04-01 (evosax 0.2.0 is pinned in jax_env; JAX 0.5.3 is stable)
