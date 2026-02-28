# Phase 2: Buffer and Fitness Infrastructure - Context

**Gathered:** 2026-02-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the shared JAX components that all ES strategies depend on: (1) extend the replay buffer to store `behavior_sig` per level via `level_extra`, (2) implement k-NN novelty scoring jit-compatible over the full buffer, (3) composite fitness function F = α·Regret + β·Novelty, and (4) a modular ES interface (ask/tell/init_state) with CMA-ES wrapped behind it. Creating new ES strategies is out of scope.

</domain>

<decisions>
## Implementation Decisions

### behavior_sig storage
- `behavior_sig` is a first-class field inside `level_extra` (not a parallel array)
- New insertions into the replay buffer MUST include a `behavior_sig` — error at insertion time if missing
- Fixed dimensionality is guaranteed by `extract_behavior_signature` — no runtime shape enforcement needed beyond what the function provides

### k-NN novelty pool
- Old levels already in the buffer that lack a `behavior_sig` are EXCLUDED from the k-NN candidate pool
- No zero-filling of legacy levels — the pool only contains levels with real signatures

### Composite fitness function
- Pure function: `compute_fitness(regret, novelty, alpha, beta) -> scalar`
- α and β are static for the duration of a run (no annealing)
- α and β live in the ES config / hyperparams dict alongside other ES hyperparameters
- Novelty combined raw: F = α·regret + β·novelty (no normalization step)

### ES interface contract
- Enforced as a `typing.Protocol` with three required methods: `init_state(rng, config) -> state`, `ask(state, rng) -> (candidates, state)`, `tell(state, candidates, fitness) -> state`
- `candidates` type: `jnp.ndarray` of shape `(pop_size, param_dim)` — directly usable with vmap
- CMA-ES wrapped as a thin `CMAESStrategy` class that delegates to existing CMA-ES code — zero behavioral change, no refactor of working code

### Claude's Discretion
- k-NN implementation strategy (exact vs approximate; how to avoid ConcretizationTypeError inside `jax.jit` — brute-force distance matrix is likely fine at 4000 entries)
- Exact file/module structure for the new components
- Internal representation of ES state (PyTree structure)

</decisions>

<specifics>
## Specific Ideas

- No specific references — open to standard JAX patterns for jit-compatible nearest-neighbor search

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-buffer-and-fitness-infrastructure*
*Context gathered: 2026-02-28*
