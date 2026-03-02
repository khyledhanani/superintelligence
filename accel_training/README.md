# ACCEL Training (VAE + MAP-Elites) Summary

This folder implements a custom ACCEL-style trainer where level generation comes from a VAE latent space plus MAP-Elites, while replay and scoring are handled by `jaxued`'s `LevelSampler`.

## Files

- `accel_training/config.yml`
  - Centralized hyperparameters.
  - Key defaults:
    - `n_candidates: 16`
    - `replay_prob: 0.8`
    - `use_accel: true`
    - `level_buffer_capacity: 4000`

- `accel_training/ppo_utils.py`
  - Pure JAX PPO utilities adapted from `examples/maze_plr.py`:
    - `compute_gae`
    - `sample_trajectories_rnn` (LSTM rollout via `jax.lax.scan`)
    - `update_actor_critic_rnn` (PPO update over epochs/minibatches)

- `accel_training/ued_interface.py`
  - Custom UED interface replacing random level generation:
    - `load_vae()` loads decoder params from VAE checkpoint.
    - `generate_candidates()` samples random + archive-mutated latents.
    - `mutate_latents()` Gaussian mutation of replay latents (ACCEL step).
    - `update_archive()` inserts valid candidates into MAP-Elites archive.
    - `build_eval_fn()` returns `@jax.jit` eval function:
      decode latents -> build levels -> rollout -> MaxMC regret.

- `accel_training/train.py`
  - Main ACCEL training loop with three branches:
    - `new`: generate candidates, evaluate, insert into replay buffer.
    - `replay`: sample from buffer, run PPO update, refresh scores.
    - `mutate`: mutate last replay latents, evaluate, insert (ACCEL self-pacing).
  - Stores latent vectors in `level_extra["latent"]` so replay levels can be mutated next.
  - Logs to `metrics.csv` and saves checkpoints (no WandB dependency).

## Run

```bash
cd /path/to/superintelligence
python accel_training/train.py
python accel_training/train.py --log_dir runs/exp1/
python accel_training/train.py --num_updates 100
```

## Main Differences vs DCD-Style Setup

- Random/wall-flip level generation is replaced by VAE latent generation and mutation.
- MAP-Elites archive is maintained separately (NumPy/Python side) to enforce diversity over behavior-descriptor cells.
- Replay buffer (`LevelSampler`) carries both level data and latent vectors (`level_extra`) for ACCEL mutation chaining.
- Logging is lightweight (`metrics.csv`, local checkpoints) instead of WandB-first workflows.
