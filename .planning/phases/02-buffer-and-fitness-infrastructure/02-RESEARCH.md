# Phase 2: Buffer and Fitness Infrastructure - Research

**Researched:** 2026-02-28
**Domain:** JAX-pure data structures, k-NN novelty scoring, fitness composition, typing.Protocol ES interface
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### behavior_sig storage
- `behavior_sig` is a first-class field inside `level_extra` (not a parallel array)
- New insertions into the replay buffer MUST include a `behavior_sig` — error at insertion time if missing
- Fixed dimensionality is guaranteed by `extract_behavior_signature` — no runtime shape enforcement needed beyond what the function provides

#### k-NN novelty pool
- Old levels already in the buffer that lack a `behavior_sig` are EXCLUDED from the k-NN candidate pool
- No zero-filling of legacy levels — the pool only contains levels with real signatures

#### Composite fitness function
- Pure function: `compute_fitness(regret, novelty, alpha, beta) -> scalar`
- α and β are static for the duration of a run (no annealing)
- α and β live in the ES config / hyperparams dict alongside other ES hyperparameters
- Novelty combined raw: F = α·regret + β·novelty (no normalization step)

#### ES interface contract
- Enforced as a `typing.Protocol` with three required methods: `init_state(rng, config) -> state`, `ask(state, rng) -> (candidates, state)`, `tell(state, candidates, fitness) -> state`
- `candidates` type: `jnp.ndarray` of shape `(pop_size, param_dim)` — directly usable with vmap
- CMA-ES wrapped as a thin `CMAESStrategy` class that delegates to existing CMA-ES code — zero behavioral change, no refactor of working code

### Claude's Discretion
- k-NN implementation strategy (exact vs approximate; how to avoid ConcretizationTypeError inside `jax.jit` — brute-force distance matrix is likely fine at 4000 entries)
- Exact file/module structure for the new components
- Internal representation of ES state (PyTree structure)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFRA-01 | Modular ES strategy interface with ask/tell API supporting swappable algorithms | ES interface contract section; evosax CMA_ES API analysis; typing.Protocol pattern |
| INFRA-02 | Replay buffer extended with `behavior_sig` field per level via `level_extra` | Buffer extension section; existing `levels_extra` key analysis; insertion validation pattern |
| INFRA-03 | Composite fitness function F = α·Regret + β·Novelty with configurable weights | Fitness composition section; pure-function pattern; weight placement in ES config |
| INFRA-04 | k-NN novelty scoring against buffer behavior signatures (pure JAX, JIT-compatible) | k-NN section; brute-force distance matrix; ConcretizationTypeError avoidance patterns |
</phase_requirements>

---

## Summary

Phase 2 builds four tightly coupled components that all subsequent ES strategies depend on. All are pure JAX, all must be JIT-compatible, and none require new third-party libraries — the full stack (JAX, evosax, flax, chex) is already installed. The key research finding is that there is no "magic library" for any of these components: they are all well-defined 10-30 line functions that follow patterns already established in the codebase.

The most technically subtle component is the k-NN novelty scorer. The ConcretizationTypeError risk is real in JAX JIT: any Python control flow that branches on the runtime value of a traced array (such as `if buffer_size > k`) will fail at compile time. The solution is to avoid all such branches inside jitted code — use masked distance matrices with `jnp.where` to exclude unoccupied buffer slots, and compute exact brute-force pairwise distances over the fixed capacity-4000 array. At 169 dimensions and 4000 entries, the distance matrix is 4000×169 float32 = ~2.7 MB, well within GPU SRAM.

The ES interface gap is important: the existing evosax `CMA_ES` API is `ask(key, state, params)` and `tell(key, population, fitness, state, params)` — five arguments with key and params separate. The new Protocol interface is `ask(state, rng)` and `tell(state, candidates, fitness)` — three arguments with state folding in the evosax params. The `CMAESStrategy` wrapper must absorb the evosax `params` object into its own state dict so callers never touch it.

**Primary recommendation:** Implement all four components as standalone pure functions / thin classes in a new `accel_training/es_components/` module. Never modify `src/jaxued/level_sampler.py` or `es/regret_fitness.py` directly — extend via wrapper patterns.

---

## Standard Stack

### Core (already installed — no new installs needed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| JAX | installed (jax[cpu]) | Array ops, JIT, vmap, lax.scan | All existing code is JAX; required for JIT-compatible k-NN |
| jax.numpy | (same) | Distance matrices, topk, argsort | Pure array ops, JIT-safe |
| flax.struct | (same) | PyTree-registered state containers | evosax state uses flax struct; ES state should match |
| evosax.algorithms.CMA_ES | installed | CMA-ES ask/tell | Already used in es/evolve_envs.py; wrapping it, not replacing it |
| typing.Protocol | stdlib | Structural subtyping for ES interface | No runtime overhead; checked by mypy/pyright; clean contract |
| chex | installed | Shape assertions in dev/test | Already used across codebase |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| jax.lax.top_k | (same) | k smallest distances for k-NN | Use instead of full argsort when k << N |
| jax.lax.scan | (same) | JIT-compatible loops (if any) | Only if iterating over fixed-length sequences inside jit |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Brute-force distance matrix | FAISS / approximate NN | FAISS is not JIT-compatible and not installed; brute-force is correct and fast at N=4000, d=169 |
| typing.Protocol | ABC with abstractmethod | Protocol is lighter, requires no inheritance from callers, and is more idiomatic for structural typing |
| flax.struct.dataclass for ES state | plain Python dict | flax struct is a registered JAX PyTree — safe to pass through JIT/vmap; plain dict is also fine but less explicit |

**Installation:** No new packages needed — all dependencies already in `environment.yml`.

---

## Architecture Patterns

### Recommended Project Structure

```
accel_training/
├── es_components/           # NEW: all Phase 2 components live here
│   ├── __init__.py          # exports: ESStrategy, CMAESStrategy, compute_novelty, compute_fitness
│   ├── interface.py         # typing.Protocol: ESStrategy
│   ├── cmaes_strategy.py    # CMAESStrategy: thin wrapper around evosax CMA_ES
│   ├── novelty.py           # compute_novelty_knn(): jit-compatible brute-force k-NN
│   └── fitness.py           # compute_fitness(): α·regret + β·novelty
es/
├── regret_fitness.py        # UNCHANGED (Phase 1 output; behavior_sig already here)
src/jaxued/
├── level_sampler.py         # UNCHANGED (behavior_sig flows through existing level_extra)
accel_training/
├── train.py                 # Will be modified in Phase 3 to use ESStrategy — not touched here
```

The `level_extra` dict already exists in `LevelSampler`. The buffer extension (INFRA-02) is purely a *usage convention*: callers must pass `level_extra={"behavior_sig": sig, "latent": latent, "max_return": max_return}` at insertion time. The sampler code already handles arbitrary dict keys in `levels_extra`.

### Pattern 1: level_extra with behavior_sig (INFRA-02)

**What:** Add `behavior_sig` as a key inside the `level_extra` dict passed to `insert` / `insert_batch`. The buffer stores it as `sampler["levels_extra"]["behavior_sig"]` shaped `(capacity, 169)`.

**When to use:** Every insertion call into the PLR buffer must include `behavior_sig`.

**Validation approach:** Since the user decision says "error at insertion time if missing", the validation should be at the Python boundary, not inside jitted code. A simple `assert "behavior_sig" in level_extra` before calling `insert_batch` is correct — this is Python-level enforcement.

**Example:**
```python
# Source: inferred from src/jaxued/level_sampler.py LevelSampler.initialize()
# Placeholder at buffer init time:
pholder_level_extra = {
    "behavior_sig": jnp.zeros(169, dtype=jnp.float32),   # 13*13
    "latent":       jnp.zeros(64,  dtype=jnp.float32),
    "max_return":   jnp.array(-jnp.inf),
}
sampler = level_sampler.initialize(pholder_level, pholder_level_extra)

# Insertion with behavior_sig (assert before jitted call):
assert "behavior_sig" in level_extra, "behavior_sig must be provided for every insertion"
sampler, _ = level_sampler.insert_batch(sampler, levels, scores, level_extra)
```

**Retrieval:**
```python
# All stored signatures, shape (capacity, 169):
all_sigs = sampler["levels_extra"]["behavior_sig"]
# Mask to only valid (filled) slots:
valid_mask = jnp.arange(level_sampler.capacity) < sampler["size"]
```

### Pattern 2: JIT-compatible Brute-Force k-NN (INFRA-04)

**What:** Compute novelty of a candidate behavior signature against the buffer pool using full pairwise L2 distance, masking out unfilled slots.

**When to use:** Called during novelty computation for each candidate in `ask`/`tell` cycle.

**The ConcretizationTypeError trap:** Any `if sampler["size"] > 0:` or dynamic slice based on `size` inside a `@jax.jit` function will fail because `size` is a traced value. Use `jnp.where(valid_mask, distances, jnp.inf)` to mask instead. The k-NN then picks the k smallest values from a fixed-size array where invalid slots are infinity.

**Memory budget:** 4000 entries × 169 dims × 4 bytes = 2.7 MB for the distance matrix. One candidate vs. 4000 buffer entries = one 4000-element distance vector. For a batch of pop_size candidates: `pop_size × 4000` distances. For pop_size=32: 128K floats = 0.5 MB. Acceptable.

**Example:**
```python
# Source: JAX official docs — jnp.lax patterns for masked reduction
# Inside @jax.jit:
def compute_novelty_knn(candidate_sig, buffer_sigs, valid_mask, k=5):
    """
    Args:
        candidate_sig: (D,) float32 — the query behavior signature
        buffer_sigs:   (capacity, D) float32 — all stored signatures (including empty slots)
        valid_mask:    (capacity,) bool — True for filled slots
        k:             int (static) — number of nearest neighbors
    Returns:
        novelty: scalar float32 — mean distance to k nearest neighbors
    """
    # Squared L2 distances: (capacity,)
    diffs = buffer_sigs - candidate_sig[None, :]   # (capacity, D)
    sq_dists = jnp.sum(diffs ** 2, axis=-1)        # (capacity,)
    # Mask out empty slots (set to inf so they're never selected)
    masked = jnp.where(valid_mask, sq_dists, jnp.inf)
    # k smallest distances — jax.lax.top_k returns LARGEST; negate trick:
    neg_masked = -masked
    top_neg, _ = jax.lax.top_k(neg_masked, k)
    top_sq_dists = -top_neg                        # k smallest squared distances
    # Mean sqrt-distance = novelty score
    novelty = jnp.mean(jnp.sqrt(jnp.maximum(top_sq_dists, 0.0)))
    return novelty
```

**Batched version for pop_size candidates:**
```python
# vmap over candidates — each gets its own novelty score
compute_novelty_batch = jax.vmap(compute_novelty_knn, in_axes=(0, None, None, None))
# usage:
novelty_scores = compute_novelty_batch(candidates_sigs, buffer_sigs, valid_mask, k=5)
# shape: (pop_size,)
```

### Pattern 3: Composite Fitness (INFRA-03)

**What:** Pure scalar function. Regret is already positive (MaxMC regret from jaxued). Higher regret = harder level. Higher novelty = more diverse. Both maximized via positive weighting.

**Note on sign convention:** The existing `regret_fitness()` in `es/regret_fitness.py` returns **negated** regret for evosax (which minimizes). For the composite fitness function, use raw (positive) regret. The caller is responsible for negating the composite score if feeding to evosax.

**Example:**
```python
# Source: user decision from 02-CONTEXT.md
def compute_fitness(regret: float, novelty: float, alpha: float, beta: float) -> float:
    """Composite fitness: F = alpha * regret + beta * novelty.

    Args:
        regret:  scalar — MaxMC regret (positive, higher = harder)
        novelty: scalar — k-NN novelty score (positive, higher = more diverse)
        alpha:   weight for regret component (from ES config)
        beta:    weight for novelty component (from ES config)
    Returns:
        F: scalar composite fitness (higher = better for curriculum quality)
    """
    return alpha * regret + beta * novelty
```

**Batched form:**
```python
# For a batch of candidates:
composite = alpha * regrets + beta * novelty_scores  # shape (pop_size,)
# Negate before passing to evosax (which minimizes):
fitness_for_evosax = -composite
```

### Pattern 4: typing.Protocol ES Interface (INFRA-01)

**What:** A structural protocol defining the ask/tell/init_state contract. Any class implementing these three methods satisfies the protocol without explicit inheritance.

**Evosax API mismatch:** evosax `CMA_ES.ask(key, state, params)` has `key` and `params` separate. The Protocol's `ask(state, rng)` folds params into state. The `CMAESStrategy` wrapper must store `es_params` inside its own state container.

**Example — Protocol definition:**
```python
# Source: typing.Protocol (stdlib Python 3.8+)
from typing import Protocol
import jax
import jax.numpy as jnp

class ESStrategy(Protocol):
    def init_state(self, rng: jax.Array, config: dict) -> dict:
        """Initialize ES state from config. Returns a state dict."""
        ...

    def ask(self, state: dict, rng: jax.Array) -> tuple[jnp.ndarray, dict]:
        """Generate pop_size candidate parameter vectors.
        Returns: (candidates shape (pop_size, param_dim), new_state)
        """
        ...

    def tell(self, state: dict, candidates: jnp.ndarray, fitness: jnp.ndarray) -> dict:
        """Update ES state given evaluated candidates and fitness.
        Returns: new_state
        """
        ...
```

**Example — CMAESStrategy wrapper:**
```python
from evosax.algorithms import CMA_ES
import jax
import jax.numpy as jnp

class CMAESStrategy:
    """Thin wrapper around evosax CMA_ES satisfying ESStrategy Protocol.

    evosax API: ask(key, state, params), tell(key, population, fitness, state, params)
    Protocol:   ask(state, rng),         tell(state, candidates, fitness)

    Stores evosax params inside the state dict to satisfy the Protocol's 2-arg ask/tell.
    Zero behavioral change to the underlying CMA-ES algorithm.
    """

    def __init__(self, param_dim: int, pop_size: int):
        dummy_solution = jnp.zeros(param_dim)
        self._es = CMA_ES(population_size=pop_size, solution=dummy_solution)

    def init_state(self, rng: jax.Array, config: dict) -> dict:
        es_params = self._es.default_params
        # Optionally override std_init from config:
        if "sigma_init" in config:
            es_params = es_params.replace(std_init=config["sigma_init"])
        es_state = self._es.init(rng, es_params)
        return {"es_state": es_state, "es_params": es_params}

    def ask(self, state: dict, rng: jax.Array) -> tuple[jnp.ndarray, dict]:
        population, new_es_state = self._es.ask(rng, state["es_state"], state["es_params"])
        new_state = {**state, "es_state": new_es_state}
        return population, new_state

    def tell(self, state: dict, candidates: jnp.ndarray, fitness: jnp.ndarray) -> dict:
        # evosax tell requires a key (used for tie-breaking); create a fixed one
        dummy_key = jax.random.PRNGKey(0)
        new_es_state, _ = self._es.tell(
            dummy_key, candidates, fitness, state["es_state"], state["es_params"]
        )
        return {**state, "es_state": new_es_state}
```

**Important:** `CMAESStrategy` is NOT decorated with `@jax.jit` — its methods are plain Python. JIT happens at the call site in the training loop where `ask` and `tell` are traced. The state dict (containing the evosax `flax.struct` state) IS a valid JAX PyTree because evosax State is a `flax.struct.dataclass`.

### Anti-Patterns to Avoid

- **Branching on traced array inside jit:** `if sampler["size"] > k:` inside `@jax.jit` raises `ConcretizationTypeError`. Use `jnp.where(mask, ...)` instead.
- **jnp.bincount inside jit:** Dynamic output shape — not JIT-compatible. The behavior_sig is already computed in Phase 1 using `jax.nn.one_hot(...).sum(axis=0)` which is safe.
- **jnp.unique inside jit:** Dynamic output shape. Do not use for filtering the novelty pool.
- **Modifying level_sampler.py:** The existing code works. Extend via usage convention (pass `behavior_sig` in level_extra dict), not by modifying the LevelSampler class.
- **Storing evosax params outside state:** If `es_params` is not in the returned state dict, `CMAESStrategy.tell` cannot access it, breaking the Protocol.
- **Using `typing.Protocol` with `runtime_checkable`:** Not needed; structural checking at type-check time is sufficient. Avoid the runtime overhead of `isinstance` checks.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CMA-ES update rule | Custom covariance adaptation | `evosax.algorithms.CMA_ES` | Already installed, tested, JIT-compatible; wrapping is 20 lines |
| k-NN with dynamic index into buffer | Python loop over filled slots | Fixed-capacity masked distance matrix with `jnp.where` | Python loops break JIT tracing |
| PyTree-compatible ES state | Custom pytree registration | `flax.struct.dataclass` (already used by evosax) | Automatic pytree registration, immutable updates via `.replace()` |
| Top-k selection | `jnp.argsort()[:k]` with dynamic slice | `jax.lax.top_k` | `top_k` is O(n log k) and static-shape output; argsort on full buffer is O(n log n) but also works |

**Key insight:** The buffer extension (INFRA-02) requires zero new code in `level_sampler.py`. The existing `levels_extra` mechanism handles arbitrary dict keys. The only work is on the caller side: passing `behavior_sig` in every `level_extra` dict.

---

## Common Pitfalls

### Pitfall 1: ConcretizationTypeError in k-NN JIT

**What goes wrong:** `jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected` when trying to slice the buffer to only filled slots inside a `@jax.jit` function.

**Why it happens:** `sampler["size"]` is a traced JAX scalar at JIT compile time. Any Python operation that branches on it (slicing, `if`, dynamic range) requires a concrete value.

**How to avoid:** Never slice the buffer array based on `size` inside jitted code. Instead:
1. Compute `valid_mask = jnp.arange(capacity) < sampler["size"]` — this is a traced boolean array, which is fine.
2. Pass the full fixed-capacity array to the distance function.
3. Use `jnp.where(valid_mask, distances, jnp.inf)` to exclude invalid slots.

**Warning signs:** Any `sampler["size"]` appearing in a Python conditional inside `@jax.jit`.

### Pitfall 2: evosax tell() Requires a Key Argument

**What goes wrong:** Calling `self._es.tell(population, fitness, state, params)` (4 args) when evosax 0.x expects `tell(key, population, fitness, state, params)` (5 args).

**Why it happens:** The installed evosax version (from `environment.yml`) requires a PRNGKey as the first argument to `tell`. This is confirmed by the `base.py` source at `/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib/python3.10/site-packages/evosax/algorithms/base.py` line 122: `def tell(self, key, population, fitness, state, params)`.

**How to avoid:** In `CMAESStrategy.tell`, always pass a `dummy_key = jax.random.PRNGKey(0)` as the first arg to `self._es.tell()`. The key is used only for tie-breaking inside evosax; a fixed key is fine.

**Warning signs:** `TypeError: tell() missing 1 required positional argument` or wrong number of arguments.

### Pitfall 3: behavior_sig placeholder must match runtime shape

**What goes wrong:** If the placeholder `behavior_sig` passed to `level_sampler.initialize()` has shape `(169,)` but runtime insertions provide `(pop_size, 169)` batched signatures, the `jax.tree_util.tree_map(...).at[idx].set(y)` inside `_insert_new` will broadcast incorrectly.

**Why it happens:** `LevelSampler.initialize()` uses `jnp.array([x]).repeat(capacity, axis=0)` to tile the placeholder. The stored array shape is `(capacity, *placeholder.shape)`. So the placeholder must be `(169,)` — the per-level shape, not the batch shape.

**How to avoid:** Always pass `jnp.zeros(169, dtype=jnp.float32)` as the placeholder for `behavior_sig`, not `jnp.zeros((pop_size, 169))`.

**Warning signs:** Shape mismatch errors during `insert_batch`; `levels_extra["behavior_sig"]` having unexpected shape.

### Pitfall 4: evosax CMA_ES init() API Change

**What goes wrong:** Calling `es.init(key, mean_vector, params)` (3 args) when installed evosax expects `es.init(key, params)` (2 args), or vice versa.

**Why it happens:** The evolve_envs.py in the codebase calls `es.init(init_key, init_mean, es_params)` — 3 args with an explicit mean. The installed evosax `base.py` shows `def init(self, key, params)` — 2 args. This inconsistency exists in the codebase.

**Verified from source:** The installed evosax at `/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib/python3.10/site-packages/evosax/algorithms/base.py` line 96-103: `def init(self, key, params) -> State`. There is NO `mean` argument in the base `init`. The mean is set via `es_params` (which has a `std_init` but not mean — the mean is set internally in `_init` from the solution provided at construction).

**How to avoid:** In `CMAESStrategy.init_state()`, call `self._es.init(rng, es_params)` — 2 args, no explicit mean. If a warm-start mean is needed, it should be encoded in the `solution` arg passed to `CMA_ES(population_size=N, solution=initial_mean)` at construction time.

**Warning signs:** `TypeError: init() takes 3 positional arguments but 4 were given`.

### Pitfall 5: State dict is not a registered PyTree if it contains plain Python objects

**What goes wrong:** If `state["es_state"]` is an evosax `flax.struct.dataclass` (registered pytree) but `state` is a plain Python dict, passing `state` through `jax.jit` boundary works for dicts (JAX treats dict as a pytree). But if any value inside is a non-pytree Python object (e.g., a Python int), JAX will treat it as a static leaf.

**How to avoid:** Keep ES state dicts containing only JAX arrays and evosax struct instances. Scalars like `alpha` and `beta` belong in the config dict (Python-side), not in the JAX-traced state.

---

## Code Examples

Verified patterns from official sources and codebase inspection:

### Existing level_extra usage (from accel_training/train.py lines 348-354)
```python
# Source: accel_training/train.py (existing working code)
# Current level_extra dict (Phase 1):
sampler, _ = level_sampler.insert_batch(
    train_state.sampler,
    levels_pad,
    regrets_pad,
    {"max_return": max_returns_pad, "latent": latents_jax_pad},
)

# Phase 2 extension — add behavior_sig to the same dict:
assert "behavior_sig" in level_extra, "All insertions must include behavior_sig"
sampler, _ = level_sampler.insert_batch(
    train_state.sampler,
    levels_pad,
    regrets_pad,
    {
        "max_return":    max_returns_pad,           # shape (num_envs,)
        "latent":        latents_jax_pad,           # shape (num_envs, 64)
        "behavior_sig":  behavior_sigs,             # shape (num_envs, 169)
    },
)
```

### Initialize buffer with behavior_sig placeholder
```python
# Source: src/jaxued/level_sampler.py LevelSampler.initialize() analysis
pholder_level_extra = {
    "max_return":   jnp.array(-jnp.inf),
    "latent":       jnp.zeros(64, dtype=jnp.float32),
    "behavior_sig": jnp.zeros(169, dtype=jnp.float32),  # per-level, not batched
}
sampler = level_sampler.initialize(pholder_level, pholder_level_extra)
# Result: sampler["levels_extra"]["behavior_sig"].shape == (capacity, 169)
```

### Retrieve behavior signatures for novelty computation
```python
# Source: src/jaxued/level_sampler.py Sampler TypedDict analysis
# Get all stored sigs (fixed shape, capacity entries):
buffer_sigs = sampler["levels_extra"]["behavior_sig"]  # (capacity, 169)
# Build valid mask (only filled slots):
valid_mask = jnp.arange(level_sampler.capacity) < sampler["size"]  # (capacity,) bool
# Pass to k-NN (inside or outside jit — both work):
novelty = compute_novelty_knn(candidate_sig, buffer_sigs, valid_mask, k=5)
```

### evosax CMA_ES ask/tell cycle (from es/evolve_envs.py lines 228-234, confirmed against installed source)
```python
# Source: es/evolve_envs.py (existing working code) + evosax base.py
# Full evosax API (what CMAESStrategy must delegate to):
population, es_state = es.ask(ask_key, es_state, es_params)   # returns (pop, state)
# ... evaluate population to get fitness ...
es_state, metrics = es.tell(tell_key, population, fitness, es_state, es_params)
```

### Valid mask k-NN (JIT-compatible)
```python
# Source: JAX docs — masked reductions pattern
import jax
import jax.numpy as jnp

@jax.jit
def compute_novelty_knn(candidate_sig, buffer_sigs, valid_mask, k):
    # candidate_sig: (D,)
    # buffer_sigs:   (capacity, D)
    # valid_mask:    (capacity,) bool
    diffs = buffer_sigs - candidate_sig[None, :]       # (capacity, D)
    sq_dists = jnp.sum(diffs ** 2, axis=-1)            # (capacity,)
    masked = jnp.where(valid_mask, sq_dists, jnp.inf)  # mask empty slots
    # top_k returns LARGEST, so negate to get smallest:
    neg_top, _ = jax.lax.top_k(-masked, k)
    top_sq_dists = -neg_top                            # k smallest sq distances
    return jnp.mean(jnp.sqrt(jnp.maximum(top_sq_dists, 0.0)))
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Archive-separate novelty pool | Unified replay buffer as novelty archive | Pre-phase design decision | No redundant data structure; buffer_size is the novelty pool size |
| scikit-learn NearestNeighbors | Brute-force JAX distance matrix | JAX-first decision | JIT-compatible; no scikit-learn dependency inside jit |
| evosax API exposed directly | Protocol-wrapped with absorbed params | Phase 2 decision | Swappable algorithms (NS-ES, SV-CMA-ES) without caller changes |

**Deprecated/outdated:**
- **Direct evosax API in training loop:** `es.ask(key, state, params)` with key and params separate is the evosax API. Phase 2 wraps this behind the Protocol so future strategy swaps don't require touching the training loop.

---

## Open Questions

1. **k value for k-NN novelty**
   - What we know: NS-ES paper uses k=10 (Lehman & Stanley 2011); common range is 5-25
   - What's unclear: optimal k for 169-dim behavior space; smaller k = more sensitive to local clusters
   - Recommendation: Default to k=5 in config, expose as a hyperparameter in the ES config dict. Mark as revisable after Phase 3 validation.

2. **Novelty for first N insertions when buffer has fewer than k entries**
   - What we know: The masked distance matrix with `jnp.inf` handles this — top_k will return `inf` for empty slots, sqrt(inf) = inf novelty
   - What's unclear: Whether infinite novelty for early candidates distorts the composite fitness
   - Recommendation: Add a `min_buffer_for_novelty` threshold in config (e.g., 10); return `novelty=0.0` if `sampler["size"] < min_buffer_for_novelty`. Implement with `jnp.where(sampler["size"] >= min_k, computed_novelty, 0.0)` — but this comparison is a Python-side check (outside jit), not inside jit.

3. **Whether CMAESStrategy.tell needs the actual candidates array**
   - What we know: evosax `tell` takes `population` — the actual parameter vectors, not just their indices. The Protocol's `tell(state, candidates, fitness)` passes candidates through.
   - What's unclear: Nothing — this is confirmed by evosax source.
   - Recommendation: Pass candidates through; this is already in the Protocol signature.

---

## Sources

### Primary (HIGH confidence)

- `/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib/python3.10/site-packages/evosax/algorithms/base.py` — confirmed `ask(key, state, params)`, `tell(key, population, fitness, state, params)`, `init(key, params)` signatures
- `/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib/python3.10/site-packages/evosax/algorithms/distribution_based/cma_es.py` — CMA_ES State/Params struct fields
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/src/jaxued/level_sampler.py` — full LevelSampler implementation; `levels_extra` key, `initialize()`, `insert_batch()`, `update_batch()` behavior
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/es/regret_fitness.py` — Phase 1 output; `extract_behavior_signature()` and `rollout_agent_on_levels_with_positions()` already implemented
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/accel_training/train.py` — existing `level_extra` usage pattern (`{"max_return": ..., "latent": ...}`)
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/es/evolve_envs.py` — existing evosax `es.ask()` / `es.tell()` call pattern

### Secondary (MEDIUM confidence)

- Python stdlib `typing.Protocol` documentation — structural subtyping, no inheritance required, available Python 3.8+
- JAX official patterns for masked reductions (`jnp.where` + fixed-shape arrays for JIT-safe conditionals)

### Tertiary (LOW confidence)

- NS-ES paper (Lehman & Stanley 2011) — k=10 for novelty k-NN; not directly verified against this codebase's dimensionality

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified by direct file inspection of installed packages and existing codebase
- Architecture: HIGH — LevelSampler source fully read; `levels_extra` mechanism confirmed; evosax API confirmed from source
- Pitfalls: HIGH — ConcretizationTypeError and evosax API confirmed by source inspection; `init()` arity mismatch confirmed from base.py
- k-NN k-value: LOW — NS-ES paper reference only; no project-specific validation yet

**Research date:** 2026-02-28
**Valid until:** 2026-03-30 (stable libraries; evosax API confirmed from installed source)
