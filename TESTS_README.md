# Experiment Commands

Environment setup (required before all commands):

```bash
export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p /tmp/jax_cache /tmp/wandb
```

`--output_dir` auto-increments a seed subdirectory (e.g. `buffer_dumps/accel_baseline` → `buffer_dumps/accel_baseline/seed0`, then `seed1`, etc.). The directory is created immediately on launch to avoid collisions between parallel runs.

---

## 1. ACCEL Baseline (no LLM)

Plain ACCEL training with SFL scoring, 30k updates. No LLM injection.

```bash
python examples/maze_plr.py \
  --project JAXUED_LLM_INJECTION \
  --score_function sfl \
  --num_updates 30000 \
  --eval_freq 250 \
  --eval_num_attempts 10 \
  --buffer_dump_interval 250 \
  --use_accel \
  --run_name accel_baseline \
  --wandb_group accel_baseline \
  --output_dir buffer_dumps/accel_baseline
```

---

## 2. LLM Injection (continuous)

ACCEL + LLM injection every 5000 steps starting at step 2500, targeting 5% LLM levels in buffer. Uses claude-code provider with config from `llm/config.yaml`.

```bash
python examples/maze_plr.py \
  --project JAXUED_LLM_INJECTION \
  --score_function sfl \
  --num_updates 30000 \
  --eval_freq 250 \
  --eval_num_attempts 10 \
  --buffer_dump_interval 250 \
  --use_accel \
  --use_llm \
  --llm_provider claude-code \
  --llm_config llm/config.yaml \
  --llm_inject_interval 5000 \
  --llm_inject_start_step 2500 \
  --llm_batch_size 10 \
  --llm_target_buffer_pct 0.05 \
  --llm_amplification \
  --llm_mutation_retries 5 \
  --run_name llm_injection \
  --wandb_group llm_injection_fresh \
  --output_dir buffer_dumps/llm_injection_fresh
```

See `examples/launch_reseeds.sh` for the tmux multi-seed launcher.

---

## 3. Test Generator (standalone level generation)

Generate LLM levels from a saved buffer + agent checkpoint without training. Useful for testing prompts, gate thresholds, and visualizing generated levels.

```bash
python -m llm.test_generator \
  --buffer-path buffer_dumps/accel_baseline/seed0/buffer_dumps/buffer_dump_2500.npz \
  --agent-dir buffer_dumps/accel_baseline/seed0/checkpoints \
  --checkpoint-step 10 \
  --n 8 \
  --provider claude-code \
  --model sonnet \
  --feedback \
  --n-rollouts 50 \
  --strategy hybrid-kmedoid \
  --num-refs 5 \
  --inject-metrics
```

Most flags default from `llm/config.yaml`. Key overrides:

- `--dry-run` -- build prompts only, skip LLM calls
- `--no-inject-metrics` -- skip agent rollouts, use buffer scores only
- `--difficulty-threshold 0.1` -- min SFL to accept
- `--min-diversity 1.0` -- min L2 distance from references
- `--max-diversity-retries 5` -- feedback loop retry limit

---

## 4. Inject-Once (one-shot LLM injection then train)

Load a base ACCEL checkpoint + buffer, generate LLM levels once (with gate, mutations, etc.), inject them into the buffer, then continue ACCEL training with no further injection. Useful for measuring the effect of a single injection event.

```bash
python examples/maze_plr.py \
  --project JAXUED_LLM_INJECTION \
  --score_function sfl \
  --num_updates 10000 \
  --eval_freq 250 \
  --eval_num_attempts 10 \
  --buffer_dump_interval 250 \
  --use_accel \
  --use_llm \
  --inject_once \
  --llm_provider claude-code \
  --llm_config llm/config.yaml \
  --llm_batch_size 10 \
  --llm_target_buffer_pct 0.05 \
  --llm_amplification \
  --llm_mutation_retries 5 \
  --resume_checkpoint_dir buffer_dumps/accel_baseline/seed0/checkpoints \
  --preload_buffer_npz buffer_dumps/accel_baseline/seed0/buffer_dumps/buffer_dump_2500.npz \
  --run_name inject_once \
  --wandb_group inject_once \
  --output_dir buffer_dumps/inject_once
```

The `--inject_once` flag:
1. Loads agent params from `--resume_checkpoint_dir`
2. Loads buffer from `--preload_buffer_npz`
3. Runs the full LLM injection pipeline once (generate, gate, mutate, insert)
4. Dumps the post-injection buffer as step 0
5. Disables further injection and trains normally
