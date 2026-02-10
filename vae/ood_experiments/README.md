# Maze VAE OOD Experiments

This folder contains a full experiment pipeline for your checklist:

1. Train a VAE on maze-token sequences.
2. Validate decoded mazes (`valid`, `path_len`, `branching`, `loops`).
3. Compute training feature stats.
4. Generate mazes from OOD latent regimes.
5. Score samples with feature-based and latent-based OOD.
6. Plot validity vs OOD and summarize best tradeoff groups.

## Files

- `modeling.py`: VAE model and config.
- `checkpointing.py`: Orbax + pickle checkpoint load/save helpers.
- `maze_metrics.py`: maze decoder, validator, feature extraction.
- `train_vae_orbax.py`: VAE training script with Orbax checkpointing.
- `run_ood_experiments.py`: end-to-end OOD experiment runner.

## Install (example)

```bash
pip install jax flax optax orbax-checkpoint matplotlib pyyaml
```

## Train VAE

```bash
python /Users/khyledhanani/Documents/superintelligence/vae/ood_experiments/train_vae_orbax.py \
  --data_path /Users/khyledhanani/Documents/superintelligence/vae/datasets/cluttr_envs_20k.npy \
  --output_dir /Users/khyledhanani/Documents/superintelligence/vae/ood_experiments/runs/train_run_01 \
  --num_steps 200000 \
  --save_every 5000
```

## Run OOD Experiments (using saved checkpoint)

```bash
python /Users/khyledhanani/Documents/superintelligence/vae/ood_experiments/run_ood_experiments.py \
  --data_path /Users/khyledhanani/Documents/superintelligence/vae/datasets/cluttr_envs_20k.npy \
  --checkpoint_path /path/to/your/orbax_or_pickle_checkpoint \
  --output_dir /Users/khyledhanani/Documents/superintelligence/vae/ood_experiments/runs/ood_eval_01 \
  --alphas 1.5,2,3,4 \
  --betas 0.5,1.0,1.5,2.0 \
  --decode_mode sample
```

## Outputs

`run_ood_experiments.py` writes:

- `sample_scores.csv`: per-sample validity, maze metrics, feature OOD, latent OOD.
- `group_summary.csv`: per-regime validity rates and OOD means.
- `validity_vs_ood.png`: validity-vs-feature OOD and validity-vs-latent OOD plots.
- `feature_stats.npz`, `latent_stats.npz`: reference statistics.
- `run_metadata.json`: checkpoint + run metadata.

## Notes

- The VAE architecture here matches `vae/train_vae.py` defaults.
- Orbax checkpoints are loaded automatically when a checkpoint directory is passed.
- Pickle checkpoints (`.pkl`) are also supported for compatibility.
