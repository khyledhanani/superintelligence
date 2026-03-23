# Stack Research

**Domain:** LLM-based periodic injection into a JAX/Flax RL training loop
**Researched:** 2026-03-23
**Confidence:** HIGH — all recommendations grounded in existing codebase, no speculative claims

---

## Context

This is a **subsequent milestone** research document. The existing stack (JAX 0.5.3, Flax 0.10.7, Orbax, WandB, etc.) is already established in `.planning/codebase/STACK.md`. This document covers only the **new** technology decisions required for LLM injection: API client libraries, threading/injection patterns, maze text-to-array conversion, and buffer management for injected levels.

All recommendations are grounded in reading the existing code (`llm/maze_generator.py`, `examples/maze_plr.py`, `src/jaxued/level_sampler.py`, `src/jaxued/environments/maze/level.py`) and the existing dependency environment.

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python `requests` | 2.32.5 (installed) | HTTP calls to OpenRouter API | Already installed, already used by `MazeGenerator._call_openai_compatible()`. No new dependency. Synchronous call fits periodic injection pattern. |
| `subprocess` (stdlib) | stdlib | claude CLI invocation | Already used by `MazeGenerator._call_claude_code()`. Calls `claude -p -` with JSON output. No API key needed — uses CLI subscription. |
| Python `threading.Thread` | stdlib | Background LLM generation during JAX training | Non-blocking generation: start thread at step N, collect results at step N+K. Zero new dependencies. |
| `jax.tree_util.tree_map` + `Level.stack()` | JAX 0.5.3 (installed) | Batch Level objects for buffer insertion | Already used everywhere in the codebase. Converts a list of `Level` objects into a batched JAX pytree for `insert_batch()`. |
| `LevelSampler.insert_batch()` | local src/ | Insert LLM-generated levels into replay buffer | Existing API in `src/jaxued/level_sampler.py`. Accepts a batched `Level` pytree + scores array. Replaces lowest-scoring levels when buffer is full. |

### Supporting Libraries (already installed, no new installs)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `numpy` | 2.2.6 | Score initialization for injected levels | LLM levels enter with an initial score (e.g. regret from `AgentEvaluator`). Use `np.full(n, initial_score, dtype=np.float32)` before JAX conversion. |
| `jax.numpy` | 0.5.3 | Convert numpy scores to JAX for `insert_batch` | `jnp.array(scores)` converts the initial score array to a JAX device array. |
| `Level.from_str()` | local src/ | Text-to-Level conversion | LLM outputs 13x13 ASCII grid. `Level.from_str(grid_str)` parses it to a `Level` pytree in one call. Already used in `llm/maze_generator.py` line 636. |
| `AgentEvaluator` | local llm/ | Compute regret/learnability for LLM levels | Use to compute initial score for injected levels. Provides trajectory dict needed by `DiversityThresholds` gate. |
| `evaluate_candidate()` | local llm/ | Diversity/difficulty gate | Already implemented. Filters LLM mazes before buffer insertion. |
| `wandb` | 0.24.2 | Log injection events | Log `llm/injected_count`, `llm/accepted_rate`, `llm/latency_ms` at each injection step. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `python-dotenv` (1.2.1, installed) | Load `OPENROUTER_API_KEY` from `.env` | Already used by `llm/maze_generator.py` `_load_api_key()`. Add `.env` to `.gitignore`. |
| `logging` (stdlib) | Debug injection events | Already used throughout `llm/`. Use `logging.getLogger("llm.injector")` for a dedicated injection logger. |

---

## Installation

No new packages required. The entire LLM injection stack runs on what is already in the `jax_env` conda environment.

```bash
# Verify existing dependencies are present
python -c "import requests; print(requests.__version__)"   # 2.32.5
python -c "import jax; print(jax.__version__)"             # 0.5.3
python -c "import numpy; print(numpy.__version__)"         # 2.2.6

# Only NEW requirement if switching from claude-code to direct Anthropic SDK:
# pip install anthropic>=0.40.0
# NOT recommended — see below.
```

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| LLM API client | `subprocess` + `claude` CLI (existing) | `anthropic` Python SDK | `anthropic` SDK is NOT installed in `jax_env`. Installing it adds ~20 transitive deps. The existing `_call_claude_code()` in `maze_generator.py` already works with the CLI. No migration needed. |
| LLM API client | `requests` for OpenRouter (existing) | `openai` SDK | OpenRouter's API is OpenAI-compatible. `requests` with manual JSON payload is simpler and already working. `openai` SDK adds a dependency. |
| Injection timing | Synchronous call in Python `for` loop | `asyncio` event loop | `asyncio` requires rewriting the outer training loop, which uses synchronous JAX calls and WandB. Adds concurrency complexity for no benefit — LLM calls are infrequent (every N=500-2000 steps). |
| Injection timing | `threading.Thread` for background generation | `concurrent.futures.ThreadPoolExecutor` | Both work; `Thread` is simpler for single-shot periodic injection. Use `ThreadPoolExecutor` only if generating multiple batches in parallel (unnecessary for this use case). |
| Buffer insertion | `LevelSampler.insert_batch()` (existing) | Custom buffer slot | `insert_batch()` already handles eviction (replaces lowest-scoring levels), deduplication (if `duplicate_check=True`), and JAX pytree batching. No reason to bypass it. |
| Injection scoring | `AgentEvaluator` regret (existing) | Fixed initial score | Using measured regret gives LLM levels a correct score for PLR/ACCEL replay priority. Fixed score would bias replay sampling incorrectly. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `anthropic` Python SDK for API calls | Not installed. Adding it is unnecessary — `claude` CLI via `subprocess` already works and requires no API key management in code. | Existing `MazeGenerator._call_claude_code()` via `subprocess` |
| `asyncio` / `async def` in the training loop | The training loop (`for eval_step in range(...)`) is synchronous and calls `jax.lax.scan` internally. Introducing `asyncio` would require converting all WandB logging, checkpoint saving, and buffer operations to async — massive refactor for no gain. | `threading.Thread` for background generation; synchronous injection at the Python loop boundary |
| Injecting inside `jax.lax.scan` | `jax.lax.scan` compiles its body at trace time. Python function calls (LLM, subprocess) cannot happen inside a traced function. | Inject at the Python `for eval_step` boundary between scan calls |
| `Level` objects with Python-side buffers (lists) | The sampler state is a JAX pytree (FrozenDict of JAX arrays). Python-side level lists are not visible to the JAX training loop. | `Level.stack()` to create batched JAX pytree, then `insert_batch()` into the sampler |
| `requests.Session` with connection pooling for OpenRouter | LLM calls are infrequent (minutes between calls). Session overhead is negligible. Sessions also require explicit cleanup. | Plain `requests.post()` per call, as in existing `_call_openai_compatible()` |
| Separate LLM generation process / microservice | Adds IPC complexity, process management, networking. Periodic synchronous injection at the Python boundary is simpler, more reproducible, and easier to debug. | Call `MazeGenerator.generate_batch_with_feedback()` directly from the training script |

---

## Injection Pattern (Prescriptive)

### Where to Inject

The outer training loop in `examples/maze_plr.py` (lines 1050-1063):

```python
for eval_step in range(config["num_updates"] // config["eval_freq"]):
    runner_state, metrics = train_and_eval_step(runner_state, None)
    # <-- LLM injection happens HERE between scan calls
    log_eval(metrics, ...)
```

`runner_state` is `(rng, train_state)`. `train_state.sampler` is the buffer. Injection modifies the sampler and replaces `runner_state`:

```python
# Pseudocode for injection at the Python loop boundary
updates_so_far = (eval_step + 1) * config["eval_freq"]
if should_inject(updates_so_far, config["llm_injection_interval"]):
    levels, scores = generate_llm_batch(generator, evaluator, gate, runner_state[1])
    if len(levels) > 0:
        batched = Level.stack(levels)
        jax_scores = jnp.array(scores)
        new_sampler, _ = level_sampler.insert_batch(
            runner_state[1].sampler, batched, jax_scores,
            {"max_return": jnp.full(len(levels), -jnp.inf)}
        )
        new_train_state = runner_state[1].replace(sampler=new_sampler)
        runner_state = (runner_state[0], new_train_state)
```

### Threading Pattern (for Background Generation)

LLM generation takes ~30-600 seconds per call. To avoid blocking training, start generation in a background thread and collect results at the next injection window:

```python
import threading

class LLMInjectionWorker:
    """Runs LLM generation in a background thread."""

    def __init__(self, generator, evaluator, gate):
        self.generator = generator
        self.evaluator = evaluator
        self.gate = gate
        self._thread = None
        self._result = None  # (levels, scores) when done
        self._lock = threading.Lock()

    def start(self, buffer_state):
        """Start background generation. Call at step N."""
        assert self._thread is None or not self._thread.is_alive()
        self._result = None
        self._thread = threading.Thread(
            target=self._run, args=(buffer_state,), daemon=True
        )
        self._thread.start()

    def collect(self):
        """Collect results if ready. Returns (levels, scores) or None."""
        if self._thread is None or self._thread.is_alive():
            return None
        with self._lock:
            result = self._result
            self._result = None
            self._thread = None
        return result

    def _run(self, buffer_state):
        result = generate_llm_batch(
            self.generator, self.evaluator, self.gate, buffer_state
        )
        with self._lock:
            self._result = result
```

This pattern is safe because:
- The background thread only reads `buffer_state` (JAX arrays) at call time — it doesn't hold a live reference to the mutable `runner_state`.
- The main thread collects results and modifies `runner_state` atomically at the Python loop boundary (single-threaded with respect to JAX).
- No shared mutable state between thread and main loop.

**When to use background threading:** Only when LLM latency exceeds `eval_freq * step_time` (i.e., LLM takes longer than one eval interval). For `eval_freq=100` steps at ~5ms/step = 500ms per interval vs LLM latency of ~30s, threading is essential. For very fast models or small `eval_freq`, synchronous is simpler.

### Text-to-Array Conversion (already solved)

`Level.from_str(grid_str)` in `src/jaxued/environments/maze/level.py` handles the full conversion:

```python
# LLM output (13x13 ASCII) -> Level JAX pytree
# Already implemented. Called in llm/maze_generator.py line 636.
level = Level.from_str(grid_str)
# level.wall_map: jnp.array (13,13) bool
# level.goal_pos: jnp.array (2,) uint32
# level.agent_pos: jnp.array (2,) uint32
# level.agent_dir: jnp.array () uint8
# level.width, level.height: int
```

**No new conversion code needed.** The only integration work is:
1. Calling `Level.from_str()` on LLM output (done by `MazeGenerator._parse_level()`)
2. Calling `Level.stack(levels_list)` to batch for `insert_batch()`

### Initial Score for Injected Levels

LLM levels must enter the buffer with a score that reflects their true difficulty. Two options:

| Option | How | When to Use |
|--------|-----|-------------|
| Measured regret via `AgentEvaluator` | Run agent on LLM level, compute MaxMC regret | Preferred — gives correct PLR priority. Adds ~2s per level (100 rollouts). |
| Buffer mean score | `float(sampler["scores"][:sampler["size"]].mean())` | Fallback if evaluator is unavailable. Biases new levels to median priority. |

Use measured regret by default. The `AgentEvaluator` + `evaluate_candidate()` pipeline already computes this as a side effect of the diversity gate.

---

## Stack Patterns by Variant

**If provider is `claude-code` (default in `llm/config.yaml`):**
- Use `subprocess` via existing `_call_claude_code()`. Requires claude CLI in PATH.
- No API key management. Works anywhere claude CLI is installed.
- Latency: 5-60s depending on model.

**If provider is `openrouter`:**
- Use `requests` via existing `_call_openai_compatible()`. Requires `OPENROUTER_API_KEY` in `.env`.
- Enables model switching without CLI dependency.
- Latency: 2-30s. Supports extended thinking models.

**If injection is synchronous (simpler, for fast models or slow eval_freq):**
- Call `MazeGenerator.generate_batch_with_feedback()` directly in the Python loop.
- Blocks training for LLM duration. Acceptable when `eval_freq * step_time > LLM_latency`.

**If injection is background (for slow models or fast eval_freq):**
- Use `LLMInjectionWorker` with `threading.Thread`.
- Training continues during generation. Collect results at next injection window.
- Adds ~10 lines of threading code. No new dependencies.

---

## Version Compatibility

| Package | Version | Compatible With | Notes |
|---------|---------|-----------------|-------|
| `requests` | 2.32.5 | Python 3.10, JAX 0.5.3 | No conflicts. Used by WandB and Google Cloud too. |
| `threading` | stdlib | All Python 3.9+ | JAX releases the GIL during compilation/dispatch. Background LLM thread does not block JAX. |
| `Level.stack()` | local src/ | JAX 0.5.3 | Uses `jax.tree_util.tree_map` + `jnp.stack`. Confirmed working in `llm/agent_evaluator.py`. |
| `LevelSampler.insert_batch()` | local src/ | JAX 0.5.3 | Uses `jax.lax.scan`. JIT-compiled. Confirmed working in `examples/maze_plr.py` lines 708, 854. |

---

## What NOT to Research Further

The following are **settled** — do not revisit in phase-specific research:

- **Maze format conversion:** `Level.from_str()` is verified working. `maze_generator.py` already calls it. No new code.
- **Buffer insertion API:** `insert_batch()` is the correct call. Signature verified in `src/jaxued/level_sampler.py` line 180.
- **LLM API client:** Both `claude-code` (subprocess) and `openrouter` (requests) are implemented and tested in `llm/maze_generator.py`. Do not add `anthropic` SDK.
- **Diversity gate:** `evaluate_candidate()` in `llm/decision_gate.py` is complete. Reuse as-is.
- **Agent evaluation:** `AgentEvaluator` in `llm/agent_evaluator.py` is complete. Reuse as-is.

The only genuine open question is **threading vs synchronous injection** — this depends on measured LLM latency in the actual training setup and can be determined empirically in phase implementation.

---

## Sources

All findings are HIGH confidence — derived directly from source code, not web search.

- `llm/maze_generator.py` — `MazeGenerator` class, provider implementations (`_call_claude_code`, `_call_openai_compatible`, `_call_ollama`), `GenerationConfig`
- `llm/config.yaml` — provider configuration, timeout (600s), injection parameters
- `llm/agent_evaluator.py` — `AgentEvaluator` class, `evaluate_level_multi_rollout()`
- `llm/decision_gate.py` — `evaluate_candidate()`, `DiversityThresholds`
- `examples/maze_plr.py` lines 1047-1063 — outer Python `for eval_step` loop (injection point)
- `examples/maze_plr.py` lines 42-52 — `TrainState` fields including `sampler`
- `examples/maze_plr.py` line 708, 854 — `insert_batch()` call sites
- `src/jaxued/level_sampler.py` lines 145-193 — `insert()`, `insert_batch()` signatures
- `src/jaxued/environments/maze/level.py` lines 31-69 — `Level.from_str()`, field types
- `.planning/codebase/STACK.md` — confirmed installed versions: `requests 2.32.5`, `jax 0.5.3`, `numpy 2.2.6`, `wandb 0.24.2`
- Local pip show — confirmed `anthropic` SDK is NOT installed; `requests`, `httpx`, `aiohttp` ARE installed

---

*Stack research for: LLM maze injection into JAX/Flax JAXUED training pipeline*
*Researched: 2026-03-23*
