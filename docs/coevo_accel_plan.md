# Co-Evolutionary ACCEL Implementation Plan

## Context

The user has designed a co-evolutionary extension to ACCEL where a VAE/level model is trained **online** on the evolving replay buffer contents, making mutations in latent space increasingly meaningful as the buffer fills with structured levels. This addresses the failure mode of using a fixed offline-trained VAE (trained on random data), where latent perturbations are no different from random tile edits.

Three key components must co-evolve:
1. Agent (PPO updates, unchanged from ACCEL)
2. Replay buffer (levels evolving toward higher regret)
3. Level model (MazeTaskAwareVAE retrained periodically on buffer contents)

The user explicitly asks to: (1) save the plan as an instructions file, (2) execute it (implement the system).

---

## Files to Create

### 1. `docs/coevo_accel_plan.md`
Verbatim copy of user's plan document. No modifications.

### 2. `es/online_level_model.py` (~200 lines)
Utilities for online level model management — training, encoding, decoding, and mutations. These functions are called from **Python-level** (outside JIT) for retraining, and the trained params are later used **inside JIT** for mutations via `encode_maze_levels` / `decode_maze_latents`.

Key exports:
```python
def init_level_model_state(rng, cfg: dict, height: int, width: int) -> Tuple[dict, optax.OptState]
def retrain_level_model(params, opt_state, dataset: dict, rng, cfg: dict, n_steps: int) -> Tuple[dict, optax.OptState, dict]
def extract_buffer_grids(sampler: dict, height: int, width: int) -> np.ndarray  # (N, H, W, 3)
def encode_batch_np(params, grids: np.ndarray, latent_dim: int, height: int, width: int) -> np.ndarray  # (N, D)
def compute_pca_from_latents(latents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]  # eigvecs (D,D), eigvals (D,)
def build_reservoir_dataset(sample_random_level_fn, rng, n_levels: int) -> dict
```

**Training uses the existing `compute_loss` from `vae/train_maze_ae.py`** (stage1 only, to stay task-agnostic). Static targets computed on-the-fly via `compute_structural_targets_from_grids`. Dynamic targets (p_ema, success_obs_count) come from `sampler["levels_extra"]`.

### 3. `examples/maze_coevo_accel.py` (~600-700 lines)
Main training script. Imports all module-level utilities from `maze_plr.py` (`compute_gae`, `sample_trajectories_rnn`, `update_actor_critic_rnn`, `ActorCritic`, `compute_score`, `rollout_success_from_rewards`, etc.) — these are all defined at module scope and importable.

---

## Architecture

### TrainState Extension

```python
class TrainState(BaseTrainState):
    # All existing fields from maze_plr.TrainState (sampler, update_state, etc.)
    # NEW co-evolutionary fields:
    level_model_params: core.FrozenDict      # MazeTaskAwareVAE params (always valid - random init)
    level_model_initialized: chex.Array      # bool scalar: True once first retrain completes
    level_model_pca_eigvecs: chex.Array      # (latent_dim, latent_dim) - identity until first PCA
    level_model_pca_eigvals: chex.Array      # (latent_dim,) - ones until first PCA
```

`level_model_params` are always valid JAX arrays (random-initialized at startup). The `level_model_initialized` flag gates whether latent mutations or fallback random mutations are used.

### Inside-JIT Mutation Branch (key innovation)

The mutation branch uses `jax.lax.cond` to switch between fallback and co-evolutionary mutation:

```python
def on_mutate_levels(rng, train_state):
    parent_levels = train_state.replay_last_level_batch

    def coevo_mutate(rng):
        # Encode parents → z
        parent_grids = jax.vmap(maze_level_to_grid)(
            parent_levels.wall_map, parent_levels.goal_pos, parent_levels.agent_pos)
        encoder_params = extract_maze_encoder_params(train_state.level_model_params)
        z = encode_maze_levels(encoder_params, parent_grids)  # (B, latent_dim)

        # Perturb: gaussian or pca-scaled depending on config
        rng_mut, rng_dec = jax.random.split(rng)
        noise = jax.random.normal(rng_mut, z.shape)
        if config["coevo_mutation_mode"] == "gaussian":
            z_child = z + config["coevo_sigma"] * noise
        else:  # pca-scaled
            scaled = noise * jnp.sqrt(train_state.level_model_pca_eigvals) * config["coevo_sigma"]
            z_child = z + (train_state.level_model_pca_eigvecs @ scaled[..., None])[..., 0]

        # Decode z_child → Level
        decoder_params = extract_maze_decoder_params(train_state.level_model_params)
        child_levels = decode_maze_latents(decoder_params, z_child,
            jax.random.split(rng_dec, num_train_envs), temperature=0.25)
        return child_levels

    def random_mutate(rng):
        return jax.vmap(mutate_level, (0, 0, None))(
            jax.random.split(rng, num_train_envs), parent_levels, config["num_edits"])

    child_levels = jax.lax.cond(
        train_state.level_model_initialized,
        coevo_mutate, random_mutate, rng_mutate)

    # Standard rollout + score + insert into buffer (same as ACCEL)
    ...
```

Both `encode_maze_levels` and `decode_maze_latents` from `es/maze_ae.py` are pure JAX functions — fully JIT-compatible.

For **mixed mutation** (configurable): sample a bernoulli variable inside jit and use `jax.lax.cond` to branch between coevo and random with probability `coevo_mutation_prob`.

### Outside-JIT Retraining Loop

```python
for eval_step in range(num_eval_steps):
    runner_state, metrics = train_and_eval_step(runner_state, None)  # JIT-compiled inner loop

    total_updates = int(train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates)
    past_burn_in = total_updates >= config["coevo_burn_in"]
    should_retrain = past_burn_in and (eval_step % config["coevo_retrain_every"] == 0)

    if should_retrain and int(train_state.sampler["size"]) >= config["coevo_min_buffer_size"]:
        # Extract buffer contents as numpy
        buffer_grids = extract_buffer_grids(train_state.sampler, height, width)

        # Compute static targets on-the-fly (BFS/graph features)
        static_targets = np.array(jax.jit(compute_structural_targets_from_grids)(jnp.array(buffer_grids)))
        p_ema = np.array(train_state.sampler["levels_extra"]["success_ema"][:buffer_size])
        obs_count = np.array(train_state.sampler["levels_extra"]["success_obs_count"][:buffer_size])

        # Mix in reservoir (random mazes to prevent latent collapse)
        dataset = build_dataset_with_reservoir(buffer_grids, static_targets, p_ema, obs_count,
                                               reservoir_grids, reservoir_statics, config)

        # Retrain level model
        level_model_params, level_model_opt_state, retrain_metrics = retrain_level_model(
            level_model_params, level_model_opt_state, dataset, rng_retrain, cfg, n_steps)

        # Update PCA directions (for pca-scaled mutation)
        latents = encode_batch_np(level_model_params, buffer_grids, latent_dim, height, width)
        eigvecs, eigvals = compute_pca_from_latents(latents)

        # Push updated params back into TrainState (JAX pytree update)
        train_state = runner_state[1].replace(
            level_model_params=core.freeze(level_model_params),
            level_model_initialized=jnp.array(True),
            level_model_pca_eigvecs=jnp.array(eigvecs),
            level_model_pca_eigvals=jnp.array(eigvals),
        )
        runner_state = (runner_state[0], train_state)
```

---

## New Config Keys

```python
# Co-evolutionary ACCEL additions
"use_coevo": True,
"coevo_burn_in": 500,             # ACCEL updates before first retrain
"coevo_retrain_every": 1,         # Retrain every N eval cycles (each = eval_freq updates)
"coevo_retrain_steps": 1000,      # Gradient steps per retrain
"coevo_min_buffer_size": 50,      # Min buffer entries before first retrain
"coevo_mutation_prob": 0.8,       # Prob of latent vs random mutation (after burn-in)
"coevo_sigma": 0.5,               # Gaussian noise sigma
"coevo_mutation_mode": "gaussian", # "gaussian" | "pca" | "interp" | "mixed"
"coevo_reservoir_size": 100,      # Random mazes mixed into training data
"level_model_latent_dim": 32,     # Smaller than 64 for speed (ablate 16/32/64)
"level_model_lr": 3e-4,
"level_model_batch_size": 64,
"level_model_beta": 0.05,
"level_model_lambda_static": 1.0,
"level_model_lambda_metric": 0.2,
"level_model_lambda_valid": 0.5,
```

---

## Critical Files to Read/Import

- `examples/maze_plr.py` (lines 1-566): All module-level utilities to import
- `es/maze_ae.py`: `MazeTaskAwareVAE`, `maze_level_to_grid`, `encode_maze_levels`, `decode_maze_latents`, `extract_maze_encoder_params`, `extract_maze_decoder_params`, `compute_structural_targets_from_grids`
- `vae/train_maze_ae.py`: `compute_loss`, `weighted_bce`, `dice_loss`, `huber_mean`, `pairwise_metric_loss` (reuse for online training)
- `jaxued/environments/maze/util.py`: `make_level_mutator_minimax`, `make_level_generator`
- `jaxued/level_sampler.py`: `LevelSampler`

---

## Loop Structure Decision

Use same `scan + eval_freq` structure as `maze_plr.py`:
- `train_and_eval_step` = `jax.lax.scan(train_step, runner_state, None, eval_freq)` (JIT-compiled)
- Level model retraining happens between these outer Python calls
- Retraining cadence: every `coevo_retrain_every` eval cycles = every `coevo_retrain_every × eval_freq` ACCEL updates
- This matches the existing logging/checkpointing infrastructure exactly

---

## Verification

```bash
# Smoke test (fast, 50 updates)
python examples/maze_coevo_accel.py --seed 0 --num_updates 50 --coevo_burn_in 20 --coevo_retrain_every 1 --coevo_retrain_steps 100 --num_train_envs 8

# Full ablation runs (compare with ACCEL baseline):
python examples/maze_plr.py --use_accel --seed 0       # Baseline
python examples/maze_coevo_accel.py --seed 0            # Co-evolutionary ACCEL
```

Log metrics to track: `coevo_retrain_loss`, `coevo_model_initialized`, `buffer_latent_diversity` (pairwise distances of buffer latents), `mutation_acceptance_rate` (fraction of mutants that enter buffer).
