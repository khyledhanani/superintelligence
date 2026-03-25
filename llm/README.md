# LLM Injection Pipeline

Periodically injects LLM-generated mazes into the ACCEL PLR training buffer.
The LLM receives reference mazes from the current buffer (with agent trajectory overlays)
and generates novel mazes that must pass difficulty and diversity gates before insertion.

## How it works

1. **Every N training steps** (default 3000), the injector pulls top-regret reference mazes from the PLR buffer
2. **LLM generates a batch** of mazes (default 10) using reference mazes + agent feedback as context
3. **Difficulty gate** filters mazes by SFL learnability (dynamic threshold from buffer mean)
4. **Diversity gate** filters mazes by TD-error EMD distance from references (dynamic threshold from buffer median)
5. **Accepted seeds** are inserted into the PLR buffer with max priority (instant replay)
6. **Mutations** — 30 guaranteed solvable per seed, evaluated with 10-rollout SFL scoring, compete for buffer slots like standard ACCEL mutations
7. **Fault tolerant** — failed LLM calls or unsolvable mazes are skipped, never crash the run

## Key files

| File | Description |
|------|-------------|
| `injector.py` | Main orchestrator — scheduling, gate logic, buffer insertion |
| `injection_config.py` | Config dataclass with all injection parameters |
| `config.yaml` | Default config values (thresholds, metrics, LLM settings) |
| `maze_generator.py` | LLM API calls (OpenRouter / Claude Code CLI), prompt formatting, retry logic |
| `prompt_builder.py` | Builds the maze generation prompt with references and feedback |
| `decision_gate.py` | Difficulty and diversity metric computation |
| `agent_evaluator.py` | Runs agent rollouts on candidate mazes to get trajectories/scores |
| `buffer_stats.py` | Extracts reference mazes and stats from the PLR buffer |
| `level_cache.py` | Saves accepted/rejected mazes to disk |
| `models.md` | Notes on LLM model selection |

## Files outside `llm/`

| File | What changed |
|------|-------------|
| `examples/maze_plr.py` | CLI args for `--use_llm`, gate modes, WandB metric definitions, logging setup |
| `examples/launch_llm_injection.sh` | Seed 0 launch script (GPT-5.4 via OpenRouter) |
| `examples/launch_llm_injection_seed1_cli.sh` | Seed 1 launch script (Claude Sonnet via OpenRouter) |

## Gate modes

### Difficulty (`--llm_difficulty_gate_mode`)
- `fixed` — static threshold from config (default 0.1)
- `buffer_mean` — must beat mean SFL score of the entire buffer
- `reference_mean` — must beat mean SFL of the reference mazes sent to the LLM
- `competitive` — no gate; seeds inserted with actual SFL score (same as ACCEL)

### Diversity (`--llm_diversity_gate_mode`)
- `fixed` — static threshold from config (default 0.015)
- `buffer_median` — threshold = median pairwise TD-error EMD among reference mazes
- `disabled` — no diversity filtering

## Launching

Both seeds use OpenRouter. Set `OPENROUTER_API_KEY` in your environment.

```bash
# Seed 0 — GPT-5.4
ssh smew
cd /cs/student/project_msc/2025/csml/gmaralla/superintelligence
bash examples/launch_llm_injection.sh

# Seed 1 — Claude Sonnet
ssh barnacle-l
cd /cs/student/project_msc/2025/csml/gmaralla/superintelligence
bash examples/launch_llm_injection_seed1_cli.sh
```

Injection fires at steps: 3000, 6000, 9000, ..., 48000.

## WandB

Project: `JAXUED_LLM`, group: `accel-llm-v2`. LLM metrics under `llm/*`:
- `llm/retained_seeds`, `llm/retained_mutations` — levels inserted per injection
- `llm/acceptance_rate` — fraction of LLM mazes passing the gate
- `llm/effective_difficulty_threshold`, `llm/effective_diversity_threshold` — dynamic gate values
- `llm/injection_time_seconds` — wall time per injection event
- `llm/mutation_survival_rate` — fraction of mutations that are solvable

Saved levels: `results/<run_name>/llm_levels/<seed>/step_XXXXX_idx_NNN.{json,npy}`
