# ES Regret Pipeline Quickstart

This folder supports two evolution modes in `evolve_envs.py`:

- `placeholder`: structural fitness only (no RL agent).
- `regret`: ACCEL-inspired regret proxy fitness (`max_mc`) using a frozen agent checkpoint.

## Requirements

- Activate `jax_env` (or equivalent env with `jax`, `evosax`, `flax`, `orbax`, `jaxued`).
- Run from `es/`:

```bash
cd /cs/student/msc/csml/2025/gmaralla/superintelligence/es
```


## Run

```bash
python evolve_envs.py \
  --fitness_mode regret \
  --agent_checkpoint_dir agent_folder \
  --num_generations 50 \
  --pop_size 16 \
  --rollout_steps 256 \
  --no_warm_start \
  --eval_policy_mode deterministic \
  --log_freq 1 \
  --output_subdir _regret_g50_p16
```

## Key parameters

- `--rollout_steps`:
  - Number of environment steps used to evaluate each candidate level.
  - Higher values give a more reliable regret estimate (agent has more time/episodes), but increase runtime per generation.
  - Typical start: `64` for smoke tests, `256` for real runs.

- `--num_generations`:
  - Number of CMA-ES optimization iterations.
  - More generations means longer search and typically better best-found regret, but more total compute.

- `--eval_policy_mode`:
  - Controls how agent actions are chosen during fitness evaluation.
  - Affects fitness noise and therefore ES stability.

- `deterministic` vs `stochastic`:
  - `deterministic`: uses `argmax(logits)` at each step.
  - `stochastic`: samples from the policy distribution.
  - `deterministic` is usually better for optimization stability (lower variance fitness signal).
  - `stochastic` can better reflect exploration behavior, but produces noisier fitness estimates.
  - Practical workflow: optimize with `deterministic`, then optionally compare against `stochastic` in an ablation.

## Outputs

Each run writes to `es/<output_subdir>/`, including:

- `evolved_envs.npy`: final population sequences.
- `best_env.npy`: best sequence by fitness.
- `fitness_history.npy`: `[best_fitness, mean_fitness]` per generation.
- `best_latent.npy`: best latent vector.
- `best_regret.npy`, `mean_regret.npy`, `solvability_rate.npy`, `latent_diversity.npy`, `sequence_diversity.npy`, `cma_sigma.npy`, `metrics_summary.json`.

## Quick checks

```bash
ls -lh _regret_g10_p8
python - <<'PY'
import numpy as np
d = "_regret_g10_p8"
print("best_regret:", np.load(f"{d}/best_regret.npy"))
print("mean_regret:", np.load(f"{d}/mean_regret.npy"))
print("solvability:", np.load(f"{d}/solvability_rate.npy"))
PY
```

## Visualize environments

Preview final population:

```bash
python visualize_envs.py --evolved _regret_g10_p8/evolved_envs.npy --num_envs 8 --output _regret_g10_p8/evolved_preview.png
```

## Notes

- Regret mode uses a frozen agent from `agent_folder` (no retraining during ES).
- `--no_warm_start` is recommended in regret mode.
- `deterministic` policy mode (`argmax`) gives a more stable ES fitness signal than stochastic sampling.
