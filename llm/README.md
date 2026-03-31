# LLM Maze Generation Pipeline

Generates novel maze levels using an LLM, then filters them through difficulty and diversity gates before injection into the PLR training buffer.

## How it works

1. **Reference selection** -- Top-regret mazes are pulled from the current PLR buffer
2. **Prompt building** -- Each reference maze is rendered as an ASCII grid with agent trajectory overlays (positions, actions, entropy, value estimates) plus per-maze metrics (regret, learnability, action sequences) and pairwise diversity metrics (TD-error EMD)
3. **LLM generation** -- The prompt is sent to the LLM, which returns a batch of candidate mazes
4. **Validation** -- Each candidate is checked for correct format, border walls, and BFS solvability
5. **Agent evaluation** -- The trained agent runs rollouts on each candidate to compute SFL learnability score
6. **Difficulty gate** -- Candidates must exceed an SFL threshold to be accepted
7. **Diversity gate** -- Candidates must be sufficiently different from references (TD-error EMD distance)
8. **Buffer insertion** -- Accepted seeds are inserted with max priority; mutations are generated and compete for slots

## Key files

| File | Description |
|------|-------------|
| `injector.py` | Main orchestrator: scheduling, gate logic, mutation amplification, buffer insertion |
| `injection_config.py` | `InjectionConfig` dataclass with all tunable parameters |
| `config.yaml` | Default config values (thresholds, metrics, LLM provider settings) |
| `maze_generator.py` | LLM API calls, prompt formatting, response parsing, retry logic |
| `prompt_builder.py` | Builds the generation prompt with reference mazes, trajectories, and metrics |
| `decision_gate.py` | Computes difficulty (SFL/regret) and diversity (TD-error EMD) metrics |
| `agent_evaluator.py` | Runs agent rollouts on candidate mazes to get trajectories and scores |
| `buffer_stats.py` | Extracts reference mazes and buffer-wide statistics |
| `level_cache.py` | Saves accepted/rejected mazes to disk for analysis |
| `test_generator.py` | CLI tool for testing maze generation without a full training run |

## Configuration

All injection parameters live in `config.yaml`. Key settings:

```yaml
provider: claude-code        # LLM provider (claude-code | anthropic | openrouter | ollama)
model: sonnet                # Model alias
temperature: 0.8
n: 8                         # Mazes per generation batch
max_retries: 5               # Retries on invalid LLM output
num_refs: 6                  # Number of reference mazes sent to LLM

gate:
  difficulty_threshold: 0.1  # Min SFL score to accept
  difficulty_metric: sfl     # or "regret"
  min_diversity: null        # Min TD-error EMD (null = disabled)
```

### Supported LLM providers

| Provider | Env var needed | Notes |
|----------|---------------|-------|
| `claude-code` | None | Uses Claude subscription via CLI |
| `anthropic` | `ANTHROPIC_API_KEY` | Direct Anthropic SDK |
| `openrouter` | `OPENROUTER_API_KEY` | Proxy supporting many models |
| `ollama` | `OLLAMA_API_KEY` | Local or cloud Ollama instance |

## Gate modes

### Difficulty (`--llm_difficulty_gate_mode`)

| Mode | Behaviour |
|------|-----------|
| `fixed` | Static threshold from config (default 0.1) |
| `buffer_mean` | Must beat mean SFL of entire buffer |
| `reference_mean` | Must beat mean SFL of the reference mazes |
| `competitive` | No gate; seeds inserted with actual SFL score (ACCEL-style) |

### Diversity (`--llm_diversity_gate_mode`)

| Mode | Behaviour |
|------|-----------|
| `fixed` | Static threshold from config (default 0.015) |
| `buffer_median` | Threshold = median pairwise TD-error EMD among references |
| `disabled` | No diversity filtering |

## Testing maze generation (standalone)

```bash
# Dry run -- builds prompts but skips LLM calls
python -m llm.test_generator --dry-run

# Full test with LLM
OPENROUTER_API_KEY=... python -m llm.test_generator
```

## Integration with training

The pipeline is called from `examples/maze_plr.py` via these CLI flags:

```bash
python examples/maze_plr.py \
  --use_llm \
  --llm_provider openrouter \
  --llm_model gpt-5.4 \
  --llm_config llm/config.yaml \
  --llm_inject_interval 3000 \
  --llm_inject_start_step 5000 \
  --llm_batch_size 25 \
  --llm_n_references 5 \
  --llm_difficulty_gate_mode fixed \
  --llm_diversity_gate_mode fixed \
  --llm_difficulty_threshold 0.6 \
  --llm_min_diversity 0.02 \
  --use_accel
```

## WandB metrics

All LLM injection metrics are logged under the `llm/*` namespace:

| Metric | Description |
|--------|-------------|
| `llm/retained_seeds` | Number of LLM seeds passing both gates |
| `llm/retained_mutations` | Number of wall-flip mutations inserted |
| `llm/acceptance_rate` | Fraction of raw candidates accepted |
| `llm/effective_difficulty_threshold` | Dynamic threshold value used |
| `llm/effective_diversity_threshold` | Dynamic threshold value used |
| `llm/injection_time_seconds` | Wall time per injection event |
| `llm/mutation_survival_rate` | Fraction of mutations that are BFS-solvable |

## Output files

Accepted and rejected mazes are saved to:
```
results/<run_name>/llm_levels/<seed>/step_XXXXX_idx_NNN.{json,npy}
```
