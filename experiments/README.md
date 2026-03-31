# Injection Experiments

This directory contains the offline injection experiment pipeline: take LLM-generated maze seeds, mutate them, score them, inject them into a PLR buffer at various percentages, and resume training.

## End-to-end pipeline

The full experiment has 3 stages:

```
[Step 0] Prepare LLM seeds (.txt -> .npz)
    |
[Step 1] Mutate seeds + score by SFL + create merged buffers at 5/10/15/20/25%
    |
[Step 2] Resume training from 10k checkpoint with each injected buffer
```

## Prerequisites

You need:
- A pre-trained agent checkpoint (default: `gcs_artifacts/agent/39`)
- A PLR buffer snapshot (default: `gcs_artifacts/buffer/buffer_dump_final.npz`)
- LLM-generated maze files in `llm/generated_levels/` (maze_*.txt)
- GPU access for training (Steps 1-2)
- WandB account for logging

## Quick start

The simplest way to run everything for one seed:

```bash
bash experiments/gpu_scripts/run_llm_injection_seed0.sh
```

This runs all 3 steps automatically. It skips steps that already completed (checks for output files).

## Scripts reference

### GPU launch scripts (`gpu_scripts/`)

These are the main entry points. Each runs the full 3-step pipeline:

| Script | What it does |
|--------|-------------|
| `run_llm_injection_seed0.sh` | LLM injection pipeline, seed 0 |
| `run_llm_injection_seed1.sh` | LLM injection pipeline, seed 1 |
| `run_llm_injection_seed2.sh` | LLM injection pipeline, seed 2 |
| `run_remaining_seeds.sh` | Runs seeds 1 and 2 sequentially |
| `run_wallflip.sh` | Wall-flip mutation baseline (no LLM, ACCEL-style). Takes args: `[target_eligible] [inject_pcts] [num_edits] [seeds...]` |
| `run_vae_noise.sh` | VAE noise mutation strategy |
| `run_vae_interp.sh` | VAE interpolation mutation strategy |

### Python scripts

| Script | Purpose |
|--------|---------|
| `prepare_llm_seeds.py` | Converts `llm/generated_levels/maze_*.txt` files into `seeds.npz` |
| `run_injection_experiment.py` | **Core pipeline**: loads seeds, mutates, scores by SFL, creates merged buffers at each injection % |
| `mutation_strategies.py` | Pluggable mutation strategies: `WallFlipMutator`, `VAENoiseMutator`, `VAEInterpolationMutator` |
| `eval_all_injection_runs.py` | Re-evaluates trained checkpoints with 100 eval attempts |
| `eval_all_injection_runs.sh` | Shell wrapper for batch evaluation |
| `analyze_mutations.py` | Analyze mutation quality (SFL distribution, survival rates) |
| `analyze_buffer_provenance.py` | Track which injected levels survived training |
| `evaluate_seeds.py` | Evaluate raw LLM seeds before mutation |
| `render_seeds.py` | Render seed mazes as images |
| `harvest_llm_seeds.py` | Extract LLM seeds from training run artifacts |
| `fetch_gcs_data.py` | Download checkpoints and buffers from GCS |
| `fix_wandb_offset.py` | Fix WandB step offset for resumed runs |
| `visualize_latent_space.py` | Visualize level embeddings in latent space |

### Training ablation scripts

These train with specific injection configs for ablation studies:

| Script | Config |
|--------|--------|
| `train_div020_5pct.sh` | 5% injection, diversity threshold 0.020 |
| `train_div020_10pct.sh` | 10% injection, diversity threshold 0.020 |
| `train_nodiv_5pct.sh` | 5% injection, no diversity gate |
| `train_nodiv_10pct.sh` | 10% injection, no diversity gate |

## Mutation strategies

All strategies share the same interface and produce solvable mazes:

| Strategy | Flag | How it works |
|----------|------|-------------|
| `wall_flip` | `--mutation_strategy wall_flip` | Random wall flips + BFS solvability filter (ACCEL-style) |
| `vae_noise` | `--mutation_strategy vae_noise` | Encode seed in VAE latent space, add Gaussian noise, decode |
| `vae_interpolation` | `--mutation_strategy vae_interpolation` | Interpolate between seed pairs in VAE latent space |

## Provenance tracking

Every merged buffer `.npz` includes an `origins` array:

| Value | Meaning |
|-------|---------|
| 0 | Organic (original buffer level) |
| 1 | LLM seed (direct from LLM) |
| 2 | LLM mutation (wall-flip or VAE descendant of a seed) |

Plus `origin_ids`: a unique hash per injected level so you can track whether it survived through training.

## Output structure

```
output_dir/
  experiment_log.json         # Metadata, timing, counts
  merged_buffer_5pct.npz      # Buffer with 5% LLM levels injected
  merged_buffer_10pct.npz     # Buffer with 10% LLM levels
  merged_buffer_15pct.npz
  merged_buffer_20pct.npz
  merged_buffer_25pct.npz
  training_5pct/              # Training output for 5% run
    buffer_dumps/             # Periodic buffer snapshots
    models/                   # Policy checkpoints
  training_10pct/
  ...
```

## WandB

- **Project**: `JAXUED_LLM_INJECTION`
- **Groups**: `llm_inject_5pct`, `llm_inject_10pct`, ..., `wallflip_e5_5pct`, etc.
- **Eval group**: `eval100` (re-evaluation with 100 attempts)
