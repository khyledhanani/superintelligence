# Training & Evaluation Scripts

These are the main entry points for training RL agents on maze environments using JaxUED.

## Core scripts

### `maze_plr.py` -- Main training script

Implements PLR (Prioritized Level Replay) + ACCEL with optional LLM injection. This is the script you'll use for both training and evaluation.

**Basic training (ACCEL, no LLM):**
```bash
python examples/maze_plr.py --use_accel --num_updates 10000 --seed 0
```

**Training with a pre-loaded buffer (injection experiment):**
```bash
python examples/maze_plr.py \
  --use_accel \
  --num_updates 10000 \
  --preload_buffer_npz /path/to/merged_buffer_10pct.npz \
  --resume_checkpoint_dir /path/to/checkpoint \
  --seed 0 \
  --run_name inject_llm_10pct_seed0 \
  --project JAXUED_LLM_INJECTION \
  --score_function sfl --num_sfl_rollouts 10
```

**Training with live LLM injection:**
```bash
python examples/maze_plr.py \
  --use_llm \
  --llm_provider openrouter \
  --llm_model gpt-5.4 \
  --llm_config llm/config.yaml \
  --llm_inject_interval 3000 \
  --use_accel \
  --num_updates 50000
```

**Evaluation:**
```bash
python examples/maze_plr.py \
  --mode eval \
  --checkpoint_directory ./checkpoints/<run_name>/<seed> \
  --checkpoint_to_eval <update_step> \
  --eval_num_attempts 100
```

### `evaluate_buffer.py` -- Evaluate agent on buffer levels

Loads an agent checkpoint and evaluates it on all levels in a buffer `.npz` file. Useful for measuring how well the agent solves the levels it trained on.

```bash
python examples/evaluate_buffer.py \
  --agent_checkpoint_dir /path/to/checkpoint \
  --buffer_npz /path/to/buffer.npz \
  --num_attempts 10
```

### `cross_evaluate.py` -- Cross-evaluation

Evaluates an agent from one run on the buffer from a different run. This measures transfer: can an agent trained with LLM-injected levels solve the baseline levels, and vice versa?

```bash
python examples/cross_evaluate.py \
  --agent_checkpoint_dir checkpoints/accel_only/0 \
  --buffer_npz /path/to/llm_inject_buffer.npz \
  --num_attempts 10
```

### `maze_dr.py` -- Domain Randomization baseline

```bash
python examples/maze_dr.py
```

### `maze_paired.py` -- PAIRED baseline

```bash
python examples/maze_paired.py
```

## Shell launch scripts

| Script | What it runs |
|--------|-------------|
| `launch_llm_injection.sh` | Seed 0, GPT-5.4 via OpenRouter, live LLM injection |
| `launch_llm_injection_seed1_cli.sh` | Seed 1, Claude Sonnet via OpenRouter, live LLM injection |
| `launch_accel_only_control.sh` | ACCEL baseline (no LLM) |
| `launch_comparison.sh` | Side-by-side comparison runs |
| `launch_cross_eval.sh` | Cross-evaluation between runs |
| `launch_vae_comparison.sh` | VAE-based mutation comparison |
| `run_kl_cmaes_pop64.sh` | CMA-ES with KL-based objective |
| `run_pca_beta2.sh` | PCA-based VAE with beta=2 |
| `run_weighted_pca.sh` | Weighted PCA VAE variant |

## Key CLI flags for `maze_plr.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--use_accel` | false | Enable ACCEL mutations |
| `--use_llm` | false | Enable live LLM injection |
| `--num_updates` | 30000 | Total PPO update steps |
| `--num_train_envs` | 512 | Number of parallel training environments |
| `--seed` | 0 | Random seed |
| `--score_function` | `MaxMC` | Level scoring: `MaxMC`, `sfl`, `regret` |
| `--num_sfl_rollouts` | 10 | Rollouts per level for SFL scoring |
| `--preload_buffer_npz` | None | Load a pre-built buffer instead of starting empty |
| `--resume_checkpoint_dir` | None | Resume training from a checkpoint |
| `--buffer_dump_interval` | 250 | Steps between buffer snapshots |
| `--project` | `JAXUED` | WandB project name |
| `--run_name` | auto | WandB run name |
| `--wandb_group` | None | WandB group for organizing runs |
| `--gcs_bucket` | None | GCS bucket for uploading artifacts |

## Output

```
checkpoints/<run_name>/<seed>/
  config.json                 # Full training config
  models/<step>/              # Policy + value network params
  buffer_dumps/<step>.npz     # PLR buffer snapshots
  orbax/                      # Full Orbax training state
```

Eval results go to `results/<run_name>/`.
