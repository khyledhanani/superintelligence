# Stack Research: ES-ACCEL Integration

**Research Date:** 2026-02-26
**Dimension:** Stack — What libraries and tools are needed for behavioral ES integration

## Current Stack (Already In Place)

| Component | Library | Status |
|-----------|---------|--------|
| Core ML | JAX | ✓ Installed |
| Neural Networks | Flax | ✓ Installed |
| Optimization | Optax | ✓ Installed |
| ES Algorithms | evosax | ✓ Installed |
| RL Environments | jaxued (custom) | ✓ Installed |
| Checkpointing | Orbax 0.5.3 | ✓ Installed |
| Logging | WandB | ✓ Installed |
| Distributions | Distrax | ✓ Installed |

## Key Finding 1: evosax Does NOT Have SV-CMA-ES

**Confidence:** HIGH

The only evosax import in the codebase is `from evosax.algorithms import CMA_ES` (in `es/evolve_envs.py`). SV-CMA-ES is not a standard evosax algorithm. It must be custom-implemented as ~80-100 lines wrapping N independent `CMA_ES` instances with a Stein repulsion gradient term injected after each ask/tell step.

**Recommendation:** Implement SV-CMA-ES as a custom wrapper around evosax's `CMA_ES`. The ask/tell API is compatible — just add the repulsion gradient between the ask and tell steps.

## Key Finding 2: k-NN for Novelty Search Needs No External Library

**Confidence:** HIGH

The replay buffer holds 4000 levels. k-NN over 4000 entries is trivial with `jnp.linalg.norm` broadcasting:
```python
dists = jnp.linalg.norm(archive_sigs - candidate[None,:], axis=-1)
```

**DO NOT add:**
- scikit-learn (breaks JIT)
- FAISS (C++ dependency, overkill for 4000 entries)
- scipy.spatial (NumPy-only, not JIT-compatible)

JAX built-in distance computation suffices. For approximate k-NN at scale, subsample 256 entries from the buffer.

## Key Finding 3: Behavior Signatures Need a Trajectory Extension to `regret_fitness.py`

**Confidence:** HIGH

Current `rollout_agent_on_levels` discards positions and actions inside `step_fn`. The data is computed but not returned. Extending return values to include action sequences and position trajectories gives all raw material for behavior signatures.

**No new library needed** — pure JAX ops:
- `jnp.bincount` for action histogram
- Custom histogram2d for visitation map over grid cells

## Key Finding 4: AURORA Needs Flax + Optax (Already In Stack)

**Confidence:** MEDIUM (AURORA is deferred but architecture should accommodate it)

AURORA's secondary autoencoder is a small Flax MLP/conv AE trained online. Both Flax and Optax are already in the stack. Only new component: an online training loop that retrains the AE every N generations.

## Key Finding 5: No New Pip Installs Required

**Confidence:** HIGH

All three target algorithms (Behavioral SV-CMA-ES, NS-ES, AURORA) can be built from the existing JAX + evosax + Flax + Optax stack.

**QDax** is a potential future optimization (it has MAP-Elites and quality-diversity implementations in JAX) but its migration cost exceeds benefit for this milestone. Note for later.

## Stack Gaps

| Gap | Solution | New Install? |
|-----|----------|-------------|
| SV-CMA-ES algorithm | Custom wrapper around evosax CMA_ES | No |
| k-NN novelty scoring | Pure JAX vectorized distances | No |
| Behavior signature extraction | Extend regret_fitness.py rollout | No |
| AURORA autoencoder | Small Flax MLP (deferred) | No |
| Behavior-aware repulsion kernel | Custom Stein gradient in JAX | No |

## Recommendations

1. **Do not add dependencies** — the current stack is sufficient
2. **Implement SV-CMA-ES as custom code** wrapping evosax's CMA_ES ask/tell API
3. **All k-NN and novelty computation must be pure JAX** — JIT-compatible, no Python-side loops
4. **Consider QDax** for future milestones if quality-diversity becomes central

---
*Stack research: 2026-02-26*
