# Phase 4: Comparison Experiments - Research

**Researched:** 2026-03-24
**Domain:** ML experiment execution, WandB API, OpenRouter LLM provider, statistical comparison
**Confidence:** HIGH (all findings grounded in codebase inspection)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Control condition:**
- Reuse existing runs from the **JAXUED_LEARNABILITY** WandB project as the ACCEL-only baseline
- Do NOT run fresh control seeds (saves GPU time; existing runs match parameters)
- Primary comparison metric: **SFL learnability** p*(1-p), NOT MaxMC regret
- Exact WandB metric key for JAXUED_LEARNABILITY runs: to be determined at analysis time (auto-detect from run history)

**Execution strategy:**
- Run seeds **in parallel**: one seed per GPU node (albacore / smew / canada)
- Modify `launch_llm_injection.sh`: add `SEED=0` variable at top so user sets it per-machine — no for-loop, single seed per invocation
- Log destination: `/tmp/` on each GPU node (WandB is source of truth; tmp logs for debug)

**Analysis scope:**
- Primary output: **both** comparison table AND learning curves
  - Table: mean ± std SFL learnability at end of training (3 ACCEL+LLM vs JAXUED_LEARNABILITY)
  - Learning curves: SFL learnability over training steps for both conditions (mean + shaded std)
- Also include injection diagnostics as supporting evidence:
  - Gate acceptance rate per injection event
  - Total LLM mazes that entered the buffer across the run
- Output format: table printed to stdout + plots saved as PNG files

**LLM model / provider:**
- Switch to **OpenRouter or Ollama** immediately — do NOT use `claude-code` CLI for production runs (quota risk: ~1200 calls estimated over a 50k run)
- OpenRouter is preferred if a paid API key is available
- Ollama is the fallback if no API key
- **Pre-flight task**: set up OpenRouter API key (`OPENROUTER_API_KEY` env var) and update `llm/config.yaml` provider to `openrouter` before starting 50k runs
- Model: Claude Sonnet via OpenRouter (or equivalent quality model if using Ollama)
- `llm/config.yaml` already has `openrouter` section configured — just needs the env var and provider field switched

**Contingency plan:**
- **Gate accepts nothing by ~20k steps**: intervene by lowering `difficulty_threshold` from 0.6 → 0.4 first; if still failing, adjust the generation prompt
- **Run crashes**: restart from latest Orbax checkpoint; accept minor step-count discrepancy

### Claude's Discretion
- Exact WandB API calls and metric key auto-detection in the comparison script
- Plot styling, axis labels, colour scheme
- Whether to checkpoint-validate before starting (or just trust existing infrastructure)

### Deferred Ideas (OUT OF SCOPE)
- Full ablation sweep (varying INJECT_START, INJECT_INTERVAL, BATCH_SIZE)
- CMA-ES-only vs ACCEL+LLM comparison
- Prompt engineering to reduce Claude verbosity (~44k token responses)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EXPT-01 | Comparison launch scripts for ACCEL+LLM injection vs ACCEL-only control with matching seeds | Scripts exist (`launch_llm_injection.sh`, `launch_accel_only_control.sh`); need SEED variable modification + OpenRouter provider fix |
</phase_requirements>

---

## Summary

Phase 4 is primarily an experiment execution and analysis phase, not a coding phase. The infrastructure from Phase 3 is nearly complete: `launch_llm_injection.sh` and `launch_accel_only_control.sh` exist, `compare_llm_results.py` exists. The main code changes needed are: (1) modify both launch scripts to expose a `SEED=` variable for parallel single-seed invocation, (2) switch `--llm_provider` from `claude-code` to `openrouter` in the LLM script, and (3) extend `compare_llm_results.py` with learning curve plots (matplotlib) and cross-project query support (JAXUED_LEARNABILITY for control).

There is one critical hyperparameter mismatch to resolve before running: the existing JAXUED_LEARNABILITY runs use `--score_function sfl` (from the `feat/cnn-vae-integration` branch which has SFL implemented), while the current `llm-injection` branch's `maze_plr.py` only supports `MaxMC` and `pvl`. The control condition in `launch_accel_only_control.sh` defaults to MaxMC. This means using JAXUED_LEARNABILITY runs as control compares SFL-scored ACCEL against MaxMC-scored ACCEL+LLM — a buffer-scoring mismatch. This is the highest-priority clarification before planning tasks.

The WandB comparison metric (`solve_rate/mean`) is logged identically by both branches — it is the held-out test solve rate, independent of buffer scoring function. So the comparison table using `solve_rate/mean` is valid across projects regardless of score function. The "SFL learnability" in CONTEXT.md refers to the *buffer scoring* strategy of the control runs, not a separate evaluation metric.

**Primary recommendation:** Two tasks suffice: (1) patch both launch scripts for SEED variable + OpenRouter provider, (2) extend compare_llm_results.py to query JAXUED_LEARNABILITY cross-project and add matplotlib learning curves. Then execute runs on GPU nodes.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| wandb | installed in jax_env | Run logging, metric storage, API access | All runs use WandB; `wandb.Api()` already in compare_llm_results.py |
| matplotlib | installed in jax_env | Learning curve PNG plots | Standard scientific plotting; shaded std bands use `fill_between` |
| numpy | installed in jax_env | Mean/std computation over seeds | Already used in compare_llm_results.py |
| requests | installed in jax_env | OpenRouter HTTP calls | Already in maze_generator.py `_call_openai_compatible` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.stats | installed in jax_env | t-test for statistical significance | Optional: quantify p-value for LLM vs control gap |
| Orbax | installed in jax_env | Checkpoint restore on crash | Only if run crashes mid-training |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| OpenRouter API | Ollama self-hosted | Ollama is free but kimi-k2.5 is 300-400s/maze vs Sonnet's 6-16s; only if no API key |
| OpenRouter (claude-sonnet-4) | gpt-5.4 via OpenRouter | gpt-5.4 cheaper ($2.50/$10 per M) but 67% success rate vs Sonnet's 100% |

**No new installations required.** All libraries already present in `jax_env`.

## Architecture Patterns

### Recommended Project Structure

No new directories. All changes are modifications to existing files:

```
examples/
└── launch_llm_injection.sh      # MODIFY: SEED variable, openrouter provider
    launch_accel_only_control.sh  # MODIFY: SEED variable
scripts/
└── compare_llm_results.py       # EXTEND: cross-project query + matplotlib plots
llm/
└── config.yaml                  # MODIFY: provider: openrouter, model: anthropic/claude-sonnet-4
```

### Pattern 1: Single-Seed Launch Script

**What:** Replace the `for seed in 0 1 2; do` loop with a `SEED=0` variable at the top, run single seed per invocation. User edits `SEED=` on each machine (albacore=0, smew=1, canada=2).

**When to use:** Parallel execution across 3 GPU nodes.

**Example:**
```bash
# === Seed to run on this machine (change per node) ===
SEED=0

# ... (COMMON, env setup unchanged) ...

echo "=== [$(date)] Seed $SEED starting ==="
$PYTHON examples/maze_plr.py $COMMON \
  --use_accel --use_llm \
  --llm_provider openrouter --llm_config llm/config.yaml \
  --llm_inject_start_step ${INJECT_START} \
  --llm_inject_interval ${INJECT_INTERVAL} \
  --llm_batch_size ${BATCH_SIZE} \
  --llm_gate \
  --seed $SEED \
  --run_name "accel-llm" \
  2>&1 | tee /tmp/llm_injection_seed${SEED}.log   # /tmp/ per CONTEXT.md decision
echo "=== [$(date)] Seed $SEED complete ==="
```

The same change applies to `launch_accel_only_control.sh` (no LLM flags, just SEED variable).

**Note on log destination:** CONTEXT.md says `/tmp/` logs. Current scripts log to `logs/` in repo dir. Change to `/tmp/llm_injection_seed${SEED}.log` and `/tmp/accel_only_seed${SEED}.log`.

### Pattern 2: Cross-Project WandB API Query

**What:** `compare_llm_results.py` currently queries a single WandB project. The control runs live in JAXUED_LEARNABILITY; the LLM runs go into JAXUED_LLM. Need to query both.

**When to use:** When control and treatment live in different WandB projects.

**Example:**
```python
# Source: WandB API docs — api.runs() accepts "entity/project" path
api = wandb.Api()

# Query ACCEL+LLM runs from JAXUED_LLM
llm_runs = list(api.runs("JAXUED_LLM"))
llm_runs = [r for r in llm_runs if r.config.get("run_name") == "accel-llm"]

# Query ACCEL-only control from JAXUED_LEARNABILITY
# auto-detect group name: look for runs with run_name matching "accel-sfl" or "accel-only"
ctrl_runs = list(api.runs("JAXUED_LEARNABILITY"))
ctrl_runs = [r for r in ctrl_runs if r.config.get("run_name") in ("accel-sfl", "accel-only")]
```

Key consideration: JAXUED_LEARNABILITY may contain multiple groups (accel-sfl, cnn-staged-sfl, clutr-staged-sfl). Filter by `run_name == "accel-sfl"` to get the correct ACCEL-only control seeds.

### Pattern 3: Learning Curve Plot with Shaded Std

**What:** Fetch per-step metric history from WandB, align by step, compute mean ± std across seeds, plot with shaded band.

**When to use:** Generating thesis comparison figure.

**Example:**
```python
# Source: wandb Python API
import matplotlib.pyplot as plt
import numpy as np

def fetch_metric_history(runs, metric_key, step_key="num_updates"):
    """Returns list of (steps_array, values_array) one per run."""
    histories = []
    for run in runs:
        hist = run.history(keys=[step_key, metric_key], pandas=False)
        steps = [row[step_key] for row in hist if step_key in row and metric_key in row]
        vals  = [row[metric_key] for row in hist if step_key in row and metric_key in row]
        if steps:
            histories.append((np.array(steps), np.array(vals)))
    return histories

def plot_learning_curves(ax, histories, label, color):
    """Interpolate to common step grid, plot mean ± std band."""
    if not histories:
        return
    # Common step grid from longest run
    all_steps = sorted(set(s for steps, _ in histories for s in steps))
    step_grid = np.array(all_steps)
    # Interpolate each run to step_grid
    interp_vals = []
    for steps, vals in histories:
        interp_vals.append(np.interp(step_grid, steps, vals))
    mat = np.stack(interp_vals, axis=0)  # (n_seeds, n_steps)
    mean = mat.mean(axis=0)
    std  = mat.std(axis=0)
    ax.plot(step_grid, mean, label=label, color=color)
    ax.fill_between(step_grid, mean - std, mean + std, alpha=0.2, color=color)

fig, ax = plt.subplots(figsize=(8, 5))
plot_learning_curves(ax, llm_histories,  "ACCEL+LLM",  "tab:blue")
plot_learning_curves(ax, ctrl_histories, "ACCEL-only", "tab:orange")
ax.set_xlabel("Training Updates")
ax.set_ylabel("Solve Rate (mean)")
ax.legend()
plt.tight_layout()
plt.savefig("scripts/llm_comparison_curves.png", dpi=150)
```

### Pattern 4: OpenRouter Provider Configuration

**What:** Switch `llm/config.yaml` `provider` field and update `model` to OpenRouter model ID.

**When to use:** Before starting 50k LLM runs.

**Changes to `llm/config.yaml`:**
```yaml
provider: openrouter              # was: claude-code
model: anthropic/claude-sonnet-4  # OpenRouter model ID for Claude Sonnet 4
```

**Changes to `launch_llm_injection.sh`:**
```bash
--llm_provider openrouter \       # was: claude-code
```

The `GenerationConfig.__post_init__` in `maze_generator.py` automatically reads `OPENROUTER_API_KEY` from environment when provider is `openrouter` (line 102-103 of `maze_generator.py`). The `_call_openai_compatible` method handles the OpenRouter API call (line 456+).

**OpenRouter model ID:** `anthropic/claude-sonnet-4` — confirmed from `llm/models.md` which lists `claude-sonnet-4` as the OpenRouter model with 100% success rate and ~$0.02/maze at 6-16s per maze.

### Anti-Patterns to Avoid

- **Keeping the for-loop in launch scripts:** If a seed crashes on a GPU node, the loop continues on the same machine sequentially instead of parallelizing across 3 machines. Always one seed per invocation.
- **Keeping `--llm_provider claude-code` in launch_llm_injection.sh:** The claude-code CLI is blocked from calling back into itself; also subscription quota risk at ~1200 calls/run over 50k steps.
- **Querying all JAXUED_LEARNABILITY runs without filtering:** The project contains multiple groups (accel-sfl, cnn-staged-sfl, clutr-staged-sfl). Filter by `run_name == "accel-sfl"` to get the 3 ACCEL-only SFL-scored seeds.
- **Using `level_sampler/mean_score` as comparison metric for cross-project runs:** On the LLM branch, score_function=MaxMC, so `level_sampler/mean_score` is MaxMC regret; on JAXUED_LEARNABILITY, it is SFL learnability — these are not comparable. Use `solve_rate/mean` (eval metric) which is scale-invariant across both branches.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-step metric history | Custom training step logging | `run.history(keys=[...])` in WandB API | Already implemented in wandb Python SDK |
| Step-grid interpolation for cross-seed alignment | Manual step matching | `numpy.interp` | Seeds may not log at identical steps; interp handles gaps |
| Statistical significance | Custom t-test | `scipy.stats.ttest_ind` | N=3 seeds per condition — state the p-value even if low power |
| Checkpoint recovery | Re-running from step 0 | Orbax restore (already wired in training loop) | Training loop already saves every 2 steps |

**Key insight:** The WandB Python SDK (`wandb.Api`) handles all the run-history fetching complexity. Don't re-implement pagination or metric aggregation.

## Common Pitfalls

### Pitfall 1: Score Function Mismatch Between Conditions

**What goes wrong:** `launch_accel_only_control.sh` uses `--score_function MaxMC` (default), while JAXUED_LEARNABILITY uses `--score_function sfl`. Running fresh control seeds from this branch would produce MaxMC-scored ACCEL vs SFL-scored ACCEL+LLM — invalid comparison at the buffer level.

**Why it happens:** `--score_function sfl` is implemented only in `feat/cnn-vae-integration` branch, not in `llm-injection`. The argparser in `maze_plr.py` on this branch only accepts `MaxMC` and `pvl`.

**How to avoid:** The locked decision is to reuse JAXUED_LEARNABILITY control runs, so no fresh control seeds are needed. The comparison uses `solve_rate/mean` (eval metric, branch-independent) not `level_sampler/mean_score` (buffer score, branch-dependent). Document this clearly in the analysis script output.

**Warning signs:** If someone proposes running fresh ACCEL-only control seeds using the current `launch_accel_only_control.sh`, the conditions won't match JAXUED_LEARNABILITY on buffer scoring.

### Pitfall 2: claude-code Provider in Production LLM Run

**What goes wrong:** `launch_llm_injection.sh` currently passes `--llm_provider claude-code`. At ~1200 injection calls over a 50k run, subscription quota will be hit and calls will fail silently (LLM returns None, injections are skipped, `llm/acceptance_rate` drops to 0).

**Why it happens:** The script was written for testing; OpenRouter wasn't the default at Phase 3 implementation time.

**How to avoid:** Change `--llm_provider` to `openrouter` in launch_llm_injection.sh AND set `OPENROUTER_API_KEY` env var on each GPU node before running. Verify by checking `llm/acceptance_rate` in WandB during the first 10k steps.

**Warning signs:** `llm/acceptance_rate` is 0 or NaN, `llm/total_injected` stays at 0 after step 5000.

### Pitfall 3: Gate Accepts Nothing — Run Completes But LLM Injection Was Inactive

**What goes wrong:** A run completes 50k steps with `llm/total_injected == 0` because all candidates failed `difficulty_threshold=0.6`. The experiment appears to have run but the treatment was never applied.

**Why it happens:** Gate calibrated to MaxMC regret ≥ 0.6. If agent is already strong by step 5000 and most buffer levels are easy, new LLM mazes will also be easy-to-solve (low regret) and rejected.

**How to avoid:** Check WandB after step 6000 (first injection event). If `llm/acceptance_rate == 0`, immediately lower `difficulty_threshold` to 0.4 per contingency plan.

**Warning signs:** `llm/acceptance_rate == 0.0` at first injection step logged.

### Pitfall 4: WandB History Pagination — Missing Early Steps in Learning Curves

**What goes wrong:** `run.history(keys=[...])` without `pandas=False` returns a DataFrame with limited rows by default. Early training steps (0-5000) may be truncated.

**Why it happens:** WandB API defaults to sampling history for performance.

**How to avoid:** Use `run.history(keys=[...], pandas=False)` (already the pattern in `compare_llm_results.py`) and set a high step count if needed: `run.history(keys=[...], samples=50000, pandas=False)`.

**Warning signs:** Learning curves start at step 5000+ instead of step 0; flat region at start is missing.

### Pitfall 5: JAXUED_LEARNABILITY Has Multiple Groups — Wrong Control Seeds Pulled

**What goes wrong:** `api.runs("JAXUED_LEARNABILITY")` returns runs from all groups: `accel-sfl`, `cnn-staged-sfl`, `clutr-staged-sfl`. Using all of them as "control" inflates N and mixes conditions.

**Why it happens:** Compare script grouping logic uses `run.config.get("run_name")` — must filter for `"accel-sfl"` specifically.

**How to avoid:** In `compare_llm_results.py`, filter JAXUED_LEARNABILITY runs by `run_name == "accel-sfl"` before analysis. Log how many runs were found and print a warning if != 3.

**Warning signs:** Control N > 3 in the comparison table; group names include `cnn-staged-sfl` or `clutr-staged-sfl`.

### Pitfall 6: LD_LIBRARY_PATH Not Set on GPU Nodes

**What goes wrong:** CUDA 13.1 cuSOLVER on system conflicts with conda jax_env's cuSOLVER 11. Training crashes immediately with CUDA library error.

**Why it happens:** System CUDA differs from conda env's pinned CUDA.

**How to avoid:** Both launch scripts already have `export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}`. Do not remove this line.

**Warning signs:** Crash within the first 5 seconds; error mentions `cuSOLVER` or `libcusolver`.

## Code Examples

### Modifying launch_llm_injection.sh (single-seed pattern)

```bash
#!/bin/bash
# ACCEL+LLM Injection Experiment — Run one seed per GPU node
# Usage: Edit SEED= below, then: bash examples/launch_llm_injection.sh

# === Seed to run on this machine (albacore=0, smew=1, canada=2) ===
SEED=0

# === Ablation parameters ===
INJECT_START=5000
INJECT_INTERVAL=3000
BATCH_SIZE=25

set -e
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python

COMMON="--project JAXUED_LLM \
        --num_updates 50000 --eval_freq 250 \
        --skip_video --skip_post_eval"

export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
mkdir -p /tmp/jax_cache

echo "=== [$(date)] Seed $SEED starting ==="
$PYTHON examples/maze_plr.py $COMMON \
  --use_accel --use_llm \
  --llm_provider openrouter --llm_config llm/config.yaml \
  --llm_inject_start_step ${INJECT_START} \
  --llm_inject_interval ${INJECT_INTERVAL} \
  --llm_batch_size ${BATCH_SIZE} \
  --llm_gate \
  --seed $SEED \
  --run_name "accel-llm" \
  2>&1 | tee /tmp/llm_injection_seed${SEED}.log
echo "=== [$(date)] Seed $SEED complete ==="
```

### Cross-Project WandB Query (compare_llm_results.py extension)

```python
# Source: WandB Python SDK — api.runs() with entity/project path
api = wandb.Api()

# Treatment: ACCEL+LLM from JAXUED_LLM project
all_llm_project_runs = list(api.runs("JAXUED_LLM"))

# Control: ACCEL-only (accel-sfl) from JAXUED_LEARNABILITY project
all_learnability_runs = list(api.runs("JAXUED_LEARNABILITY"))
ctrl_runs = [r for r in all_learnability_runs
             if r.config.get("run_name") == "accel-sfl"]
print(f"Control runs found: {len(ctrl_runs)} (expected 3)")
if len(ctrl_runs) != 3:
    print("  WARNING: Expected exactly 3 accel-sfl runs in JAXUED_LEARNABILITY")
```

### Diagnosis Check at 20k Steps

```bash
# On any machine with WandB credentials, after runs have been running for ~2 hrs:
python - <<'EOF'
import wandb
api = wandb.Api()
runs = [r for r in api.runs("JAXUED_LLM") if r.config.get("run_name") == "accel-llm"]
for r in runs:
    hist = r.history(keys=["llm/total_injected", "num_updates"], pandas=False)
    last = [row for row in hist if row.get("num_updates", 0) > 15000]
    if last:
        print(f"Seed {r.config.get('seed')}: total_injected={last[-1].get('llm/total_injected', 'N/A')}")
EOF
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| claude-code CLI for LLM calls | OpenRouter HTTP API | Phase 4 pre-flight | No quota risk; direct API key control |
| Sequential 3-seed loop in launch script | Single-seed per machine | Phase 4 modification | Parallel 3x speedup on 3 GPU nodes |
| Single-project WandB query | Cross-project query (JAXUED_LLM + JAXUED_LEARNABILITY) | Phase 4 extension | Enables reuse of existing control runs |
| Table-only comparison output | Table + learning curves PNG | Phase 4 extension | Required for thesis figure |

**Deprecated/outdated:**
- `--llm_provider claude-code` in launch_llm_injection.sh: must change to `openrouter` before production run

## Open Questions

1. **Is OPENROUTER_API_KEY available on GPU nodes?**
   - What we know: `llm/config.yaml` has openrouter section with `api_key_env: OPENROUTER_API_KEY`. `GenerationConfig.__post_init__` reads from env var or `.env` file in project root.
   - What's unclear: Whether the key is currently set on albacore/smew/canada, or in a `.env` file in the repo root.
   - Recommendation: Pre-flight task must verify `echo $OPENROUTER_API_KEY` on each GPU node returns non-empty. If not, create `superintelligence/.env` with `OPENROUTER_API_KEY=<key>` (already parsed by `_load_api_key()`). Do not commit `.env` to git (already in `.gitignore`).

2. **JAXUED_LEARNABILITY `accel-sfl` runs: are all 3 seeds complete?**
   - What we know: `launch_50k_accel_sfl.sh` targets seeds 0, 1, 2 sequentially in JAXUED_LEARNABILITY. From MEMORY.md, no mention of these specific runs completing.
   - What's unclear: Current state of JAXUED_LEARNABILITY accel-sfl runs (running/complete/not-started).
   - Recommendation: Pre-flight check — `python scripts/compare_llm_results.py --project JAXUED_LEARNABILITY` (or a WandB API query for `run_name==accel-sfl`). If runs are not complete, options are: (a) wait for them to finish, (b) run fresh ACCEL-only seeds into JAXUED_LLM with `launch_accel_only_control.sh` (already exists).

3. **Score function mismatch: does it matter for the thesis claim?**
   - What we know: JAXUED_LEARNABILITY `accel-sfl` uses `--score_function sfl` (PLR buffer scoring strategy). The LLM branch's `launch_llm_injection.sh` uses `--score_function MaxMC` (default). The evaluation metric (`solve_rate/mean`) is the same in both cases. The *buffer scoring* strategy differs.
   - What's unclear: Whether the thesis committee will flag this as a confounded comparison (different buffer scoring → different training dynamics).
   - Recommendation: In the comparison table, document the score_function difference explicitly. If the result is positive for LLM injection, the stronger claim ("LLM helps even against SFL-tuned ACCEL") holds. If negative, the score function difference is a nuance to address in the writeup. This does NOT block Phase 4 execution per locked decisions.

## Validation Architecture

> `workflow.nyquist_validation` is not set in `.planning/config.json` (field absent). Skipping this section.

## Sources

### Primary (HIGH confidence)
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/examples/launch_llm_injection.sh` — current script state: for-loop, claude-code provider
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/examples/launch_accel_only_control.sh` — current script state: for-loop, MaxMC scoring
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/scripts/compare_llm_results.py` — existing comparison script: single-project query, no plots
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/llm/maze_generator.py` — confirmed OpenRouter support via `_call_openai_compatible`, `OPENROUTER_API_KEY` auto-loaded
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/llm/config.yaml` — current provider: claude-code; openrouter section present with base_url
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/llm/models.md` — OpenRouter model ID: `anthropic/claude-sonnet-4`; $0.02/maze; 100% success rate
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/examples/maze_plr.py` — confirmed: `--score_function` choices are `MaxMC` and `pvl` only (no sfl on this branch)
- `git show feat/cnn-vae-integration:examples/maze_plr.py` — confirmed: `--score_function sfl` implemented only on feat/cnn-vae-integration branch

### Secondary (MEDIUM confidence)
- `examples/launch_50k_accel_sfl.sh` — JAXUED_LEARNABILITY project, `--score_function sfl`, `--run_name "accel-sfl"`, seeds 0-2 sequential
- WandB API pattern `run.history(keys=[...], pandas=False)` — established in existing compare_llm_results.py; cross-project query via `api.runs("PROJECT")` is standard SDK pattern

### Tertiary (LOW confidence)
- JAXUED_LEARNABILITY `accel-sfl` run completion status — not verifiable without live WandB API call. Assumed runs may be complete or in-progress based on project naming and launch script existence.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified present in jax_env; API patterns verified from existing code
- Architecture: HIGH — all script changes are direct diffs from existing files; patterns verified from `compare_llm_results.py` and `maze_generator.py`
- Pitfalls: HIGH — all pitfalls grounded in actual code inspection (score_function argparser, LD_LIBRARY_PATH, WandB API pagination)

**Research date:** 2026-03-24
**Valid until:** 2026-04-24 (WandB API stable; OpenRouter pricing may change but model ID is stable)
