# Architecture Research: ES-ACCEL Integration

**Research Date:** 2026-02-26
**Dimension:** Architecture — How ES modules integrate with existing ACCEL training loop

## Key Finding 1: No Structural Changes to Outer ACCEL Loop

The ES integrates as the internals of the existing `on_mutate_levels` branch (branch 2 in `jax.lax.switch` inside `maze_plr.py`). The current code already has:

```python
if config["use_map_elites_mutation"]:
    # MAP-Elites mutation path
elif config["use_plwm_mutation"]:
    # PLWM mutation path
```

Add `elif config["use_es_mutation"]:` alongside these. The outer training loop, PPO update, and level sampling logic remain untouched.

## Key Finding 2: No Changes to LevelSampler

The `level_extra` dict mechanism in jaxued's `LevelSampler` already supports storing arbitrary per-level data. Just pass an extended `pholder_level_extra` at initialization:

```python
pholder_level_extra = {
    "max_return": ...,
    "latent_z": jnp.zeros(latent_dim),
    "behavior_sig": jnp.zeros(sig_dim),
}
```

All `insert()` and `update()` methods already propagate `level_extra` correctly.

## Key Finding 3: One TrainState Field to Add

Add `es_state: Any = struct.field(pytree_node=True)` to `TrainState` for the evosax `ESState` pytree. This enables checkpointing the ES state alongside the agent.

## Component Architecture

### Component 1: Behavior Extractor
- **Purpose:** Extract behavior signature from agent rollout on a level
- **Input:** Agent params, environment level, RNG key
- **Output:** Behavior signature vector (e.g., visit-count histogram over grid cells)
- **Location:** New module, called within rollout loop
- **Integration:** Extends `regret_fitness.py` rollout to return trajectory data alongside regret

### Component 2: Fitness Evaluator
- **Purpose:** Compute composite fitness from regret + novelty + optional diversity terms
- **Input:** Regret score, behavior signature, buffer behavior signatures, weights (α, β)
- **Output:** Scalar fitness value
- **Location:** New module
- **Integration:** Called after rollout, before ES tell step
- **k-NN novelty:** `dists = jnp.linalg.norm(buffer_sigs - candidate_sig[None,:], axis=-1)`, take mean of k smallest

### Component 3: ES Strategy (Swappable)
- **Purpose:** Generate candidate latent vectors via evolutionary search
- **Interface:** `ask(state, rng) → (candidates, state)`, `tell(state, candidates, fitness) → state`
- **Implementations:**
  - `CMAESStrategy` — Wrapper around evosax CMA_ES (baseline)
  - `NSESStrategy` — CMA-ES with novelty-augmented fitness
  - `BehavioralSVCMAESStrategy` — N CMA-ES particles with Stein repulsion in behavior space
- **Location:** New module with strategy pattern
- **Integration:** Called within `on_mutate_levels` branch

### Component 4: Enhanced Buffer Manager
- **Purpose:** Manage buffer entries with behavior signatures, handle insertion/eviction
- **Input:** New (latent, regret, behavior_sig) tuples from ES
- **Output:** Updated buffer state
- **Location:** Thin wrapper around existing LevelSampler
- **Integration:** Uses `level_extra` dict — no LevelSampler changes needed

### Component 5: Convergence Detector (Deferred)
- **Purpose:** Detect when ES has converged (sigma collapse) and trigger restart
- **Input:** ES state (sigma history)
- **Output:** Boolean restart signal
- **Location:** Optional module, checked after ES tell step
- **Integration:** Resets ES state from diverse archive samples when triggered

## Data Flow

```
Training Step (on_mutate_levels branch):

  1. Buffer latents z ──→ ES.ask() ──→ candidate z vectors
                                           │
  2. candidate z ──→ VAE.decode() ──→ environment levels
                                           │
  3. levels ──→ PPO rollout ──→ trajectories + returns
                                           │
  4. trajectories ──→ BehaviorExtractor ──→ behavior signatures
     returns ──→ compute_regret() ──→ regret scores
                                           │
  5. (regret, behavior_sig, buffer_sigs) ──→ FitnessEvaluator
                                           │
  6. composite fitness ──→ ES.tell() ──→ updated ES state
                                           │
  7. high-fitness (z, regret, sig) ──→ LevelSampler.insert() ──→ updated buffer
```

## Integration Points with Existing Code

| Existing Component | Integration Point | Change Required |
|-------------------|-------------------|-----------------|
| `maze_plr.py` TrainState | Add `es_state` field | Minimal |
| `maze_plr.py` `on_mutate_levels` | New ES branch alongside MAP-Elites/PLWM | New branch |
| `LevelSampler` | Extended `level_extra` at init | None — already supports it |
| `regret_fitness.py` rollout | Return trajectory data alongside regret | Extend return values |
| `vae_decoder.py` | Decode latent → level (unchanged) | None |
| `env_bridge.py` | `cluttr_sequence_to_level` (unchanged) | None |
| Config / argparse | New ES-related flags | Additive |

## Suggested Build Order

1. **Behavior Extractor** (Phase 1) — foundational; everything else depends on it
2. **Enhanced Buffer init** (Phase 2) — extend `level_extra` with behavior_sig field
3. **Fitness Evaluator with novelty** (Phase 3) — k-NN novelty computation
4. **ES Strategy implementations** (Phase 4) — CMA-ES baseline, then NS-ES, then SV-CMA-ES
5. **Integration into `on_mutate_levels`** (Phase 5) — wire everything into the training loop
6. **Convergence Detector** (Phase 6) — deferred, but interface ready

## Unified Buffer as Novelty Archive

The replay buffer's `sampler["levels_extra"]["behavior_sig"]` serves as the k-NN archive. **No separate data structure needed.** This is the "elegant" NS-ES integration described in the project vision — the ACCEL replay buffer and the novelty archive are the same thing.

---
*Architecture research: 2026-02-26*
