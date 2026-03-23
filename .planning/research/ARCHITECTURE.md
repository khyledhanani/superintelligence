# Architecture Research

**Domain:** LLM maze injection into JAX UED training pipeline
**Researched:** 2026-03-23
**Confidence:** HIGH — based on direct codebase inspection of all relevant files

## Standard Architecture

### System Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                     Python Training Loop (maze_plr.py)                │
│                                                                       │
│   for eval_step in range(num_updates // eval_freq):                   │
│   ┌─────────────────────────────────────────────────────────┐        │
│   │              train_and_eval_step (jit compiled)         │        │
│   │   ┌────────────────────────────────────────────────┐    │        │
│   │   │  lax.scan(train_step, runner_state, eval_freq) │    │        │
│   │   │                                                │    │        │
│   │   │  Each train_step: lax.switch on branch 0/1/2  │    │        │
│   │   │    Branch 0: on_new_levels (CMA-ES or random) │    │        │
│   │   │    Branch 1: on_replay_levels                 │    │        │
│   │   │    Branch 2: on_mutate_levels (ACCEL)         │    │        │
│   │   └────────────────────────────────────────────────┘    │        │
│   │   ┌────────────────────────────────────────────────┐    │        │
│   │   │  ** LLM INJECTION HOOK **                      │    │        │
│   │   │  (Python-side, between eval steps)             │    │        │
│   │   │  if step % llm_interval == 0:                  │    │        │
│   │   │    levels = LLMInjector.generate_batch()       │    │        │
│   │   │    train_state = inject_into_buffer(levels)    │    │        │
│   │   └────────────────────────────────────────────────┘    │        │
│   └─────────────────────────────────────────────────────────┘        │
└───────────────────────────────────────────────────────────────────────┘
         │                          │                    │
         ▼                          ▼                    ▼
┌─────────────────┐   ┌─────────────────────┐  ┌──────────────────────┐
│   JAX Buffer    │   │   LLM Subsystem      │  │  Metrics Subsystem   │
│  LevelSampler   │   │                      │  │                      │
│  (pure JAX,     │   │  MazeGenerator       │  │  AgentEvaluator      │
│   jit-compiled) │   │  PromptBuilder       │  │  DecisionGate        │
│                 │   │  DecisionGate        │  │  (regret, SFL,       │
│  insert_batch() │   │                      │  │   CENIE, DTW)        │
│  update_batch() │   │  Ollama/OpenRouter   │  │                      │
│  sample_replay()│   │  API (HTTP)          │  │  (pure Python/numpy) │
└─────────────────┘   └─────────────────────┘  └──────────────────────┘
```

### The Injection Hook: Where LLM Connects to the Training Cycle

The critical architectural insight is that LLM injection cannot live inside `lax.scan` (the jit-compiled inner loop) because LLM calls are Python-side HTTP requests. The hook must live in the Python `for` loop that wraps `train_and_eval_step`:

```
Python for-loop (eval_step):            # ~every eval_freq=10 steps
  ├── train_and_eval_step (jit)         # eval_freq JAX train steps
  ├── log_eval(metrics)                 # WandB logging
  └── ** LLM injection (Python) **      # every N eval_steps
      ├── extract_buffer_stats()        # numpy from sampler dict
      ├── llm_injector.generate_batch() # HTTP call to LLM API
      ├── gate: filter by difficulty+diversity
      └── inject_levels_into_buffer()   # back into JAX sampler
```

**Why this hook location is correct:**
- `train_and_eval_step` is `@jax.jit` decorated — no Python side effects inside it
- `lax.scan` compiles the loop away — cannot conditionally call Python code inside
- The Python `for` loop runs after each `eval_freq` block — this is the natural injection point
- Buffer state (`train_state.sampler`) is a JAX pytree accessible as numpy after the jit call

### Component Boundaries

| Component | Responsibility | Location | Communicates With |
|-----------|---------------|----------|-------------------|
| `TrainLoop` | Outer Python loop, orchestrates injection scheduling | `examples/maze_plr.py` main loop | LLMInjector, TrainState |
| `train_step` (jit) | JAX branch selection and policy update | `examples/maze_plr.py` inner | LevelSampler, env, PPO |
| `on_new_levels` | Branch 0: CMA-ES/random level generation + buffer insert | Inside train_step | CMAESManager, LevelSampler |
| `on_replay_levels` | Branch 1: PLR replay + score update | Inside train_step | LevelSampler, PPO |
| `on_mutate_levels` | Branch 2: ACCEL wall mutation + buffer insert | Inside train_step | LevelSampler, mutate_level |
| `LevelSampler` | Prioritized buffer (capacity=4000), score+staleness weights | `src/jaxued/level_sampler.py` | TrainState.sampler |
| `LLMInjector` | Orchestrates periodic LLM generation + gating + injection | NEW: `llm/injector.py` | MazeGenerator, AgentEvaluator, DecisionGate, LevelSampler |
| `MazeGenerator` | LLM API calls (Ollama/OpenRouter), text→Level parsing | `llm/maze_generator.py` | PromptBuilder, Level |
| `PromptBuilder` | Builds prompts from buffer stats + reference mazes | `llm/prompt_builder.py` | MazeGenerator |
| `AgentEvaluator` | Runs trained agent on Level objects for trajectories | `llm/agent_evaluator.py` | Maze env, ActorCritic checkpoint |
| `DecisionGate` | Filters candidates by regret/SFL/diversity thresholds | `llm/decision_gate.py` | AgentEvaluator output |
| `BufferStatsExtractor` | Converts JAX sampler dict → Python metrics for prompt | NEW: `llm/buffer_stats.py` | sampler dict, metrics/ |
| `Maze.Level` | Data structure: wall_map (13×13 bool), goal_pos, agent_pos, agent_dir | `src/jaxued/environments/maze/level.py` | All components |

## Data Flow

### Training Cycle (existing, unmodified)

```
train_step (jax.jit + lax.scan):
  rng → lax.switch(branch):
    Branch 0 (DR/CMA-ES):
      CMA-ES.ask() → z_population
      decode_latent_to_levels(vae, z) → new_levels: Level[32]
        OR sample_random_level() → new_levels: Level[32]
      env.reset_to_level(levels) → (obs, env_state)
      sample_trajectories_rnn() → (obs, actions, rewards, dones, values)
      compute_gae() → advantages
      compute_score(MaxMC/PVL) → scores: float[32]
      CMA-ES.tell(-scores)
      level_sampler.insert_batch(sampler, levels, scores) → sampler
      update_actor_critic_rnn() → params

    Branch 1 (Replay):
      level_sampler.sample_replay_levels(sampler, 32) → (inds, levels)
      [rollout + PPO update]
      level_sampler.update_batch(sampler, inds, scores) → sampler

    Branch 2 (ACCEL Mutate):
      mutate_level(replay_last_level_batch) → child_levels: Level[32]
      [rollout + score]
      level_sampler.insert_batch(sampler, child_levels, scores) → sampler
```

### LLM Injection Flow (new)

```
Python for-loop (every llm_interval eval steps):

  1. EXTRACT BUFFER STATS (Python-side):
     sampler = runner_state[1].sampler
     buffer_levels = numpy(sampler["levels"][:sampler["size"]])
     buffer_scores = numpy(sampler["scores"][:sampler["size"]])
     → Select top-K hard mazes (by score) as reference mazes
     → Format as ReferenceMaze[] with ASCII grids + metrics

  2. AGENT EVALUATE REFERENCES (Python/JAX):
     AgentEvaluator.evaluate_levels(reference_levels)
     → reference_trajectories: List[dict{positions, values, dones, rewards}]

  3. LLM GENERATION (Python HTTP):
     PromptBuilder.build_generation_prompt(
         references=reference_mazes,
         global_metrics=buffer_stats,
         pairwise_metrics=diversity_between_refs,
     ) → prompt: str
     MazeGenerator.generate(prompt) → raw_text: str
     parse_grid(raw_text) → candidate_level: Level

  4. DECISION GATE (Python/numpy):
     AgentEvaluator.evaluate_level(candidate_level)
     → candidate_trajectory: dict
     DecisionGate.evaluate_candidate(
         candidate_trajectory,
         reference_trajectories,
         thresholds=DiversityThresholds(
             difficulty_threshold=0.3,
             min_diversity=0.04
         )
     ) → GateResult{accepted: bool, issues: List[str]}

     if not accepted:
       PromptBuilder.build_feedback_prompt(issues) → feedback
       MazeGenerator.retry_with_feedback(feedback) → Level
       [repeat up to max_retries]

  5. BUFFER INJECTION (Python → JAX):
     accepted_levels: List[Level]  (1 per successful generation)
     scores = [evaluator.compute_score(l) for l in accepted_levels]

     # Back into JAX: rebuild sampler
     sampler = train_state.sampler
     for level, score in zip(accepted_levels, scores):
         sampler, _ = level_sampler.insert(sampler, level, score)
     train_state = train_state.replace(sampler=sampler)
     runner_state = (rng, train_state)

  6. LOGGING:
     wandb.log({
         "llm/injection_count": len(accepted_levels),
         "llm/acceptance_rate": accepted / attempted,
         "llm/latency_ms": total_latency,
         "llm/gate_diversity_min": min gate score,
     })
```

### State Management: TrainState and the Buffer

```
TrainState (Flax struct — JAX pytree):
  ├── params: PolicyNetwork weights
  ├── opt_state: Adam optimizer state
  ├── sampler: LevelSampler dict
  │   ├── levels: Level[capacity=4000]  ← injected levels go here
  │   ├── scores: float[4000]           ← score determines replay priority
  │   ├── timestamps: int[4000]         ← staleness tracking
  │   ├── size: int                     ← current fill level
  │   └── episode_count: int
  ├── update_state: DR=0 | REPLAY=1    ← controls ACCEL 3-way cycle
  ├── es_state: CMA-ES internal state
  └── [logging fields]

Key constraint: TrainState is a JAX pytree. Modifying it from Python
requires using .replace() to create a new TrainState — no in-place mutation.
This is safe between eval steps (outside the jit boundary).
```

### Buffer Interaction: LLM-Injected Levels and ACCEL/CMA-ES

Once a level is in the buffer via `level_sampler.insert()`, it is indistinguishable from any other buffered level. The 3-way cycle handles it automatically:

```
LLM-injected Level enters buffer at score S
       ↓
Branch 1 (Replay): sampled with P ∝ score+staleness
  → Agent replays it, score updated
       ↓
Branch 2 (ACCEL Mutate): if train_state.update_state == REPLAY
  → child_levels = mutate_level(replay_last_level_batch)
  → children inserted into buffer (inherit lineage)
       ↓
CMA-ES (Branch 0): NOT directly influenced by injected levels
  → CMA-ES searches its own latent space independently
  → No mechanism to seed CMA-ES from LLM outputs (and none needed)
```

**ACCEL will naturally mutate LLM-injected mazes** because after a replay step
(`update_state = REPLAY`), the next `train_step` always calls `on_mutate_levels`,
which mutates `replay_last_level_batch` — which can contain LLM-injected levels
if they were sampled in the preceding replay branch.

## Recommended Project Structure

```
examples/
└── maze_plr.py           # MODIFY: add LLM injection hook in Python for-loop

llm/                      # existing friend's code — no structural changes
├── maze_generator.py     # existing
├── prompt_builder.py     # existing
├── decision_gate.py      # existing
├── agent_evaluator.py    # existing
├── injector.py           # NEW: LLMInjector class (orchestration)
├── buffer_stats.py       # NEW: extract buffer statistics for prompts
└── config.yaml           # existing (or add llm-specific fields)

scripts/
└── compare_llm_injection.py  # NEW: comparison analysis script
```

### Structure Rationale

- **`llm/injector.py` (new):** Single orchestrator class handles injection scheduling, calls to MazeGenerator, AgentEvaluator, and DecisionGate, and writes back to the buffer. Keeps `maze_plr.py` clean — injection is one method call.
- **`llm/buffer_stats.py` (new):** Isolates the logic of reading from a JAX sampler dict and formatting it as prompt context. Avoids polluting `maze_plr.py` with metric computation.
- **`examples/maze_plr.py` (modify):** Add `--use_llm` flag, `--llm_interval N`, instantiate `LLMInjector` before the for-loop, call `injector.maybe_inject(runner_state, eval_step)` inside the for-loop.

## Architectural Patterns

### Pattern 1: Python Hook Between JIT Boundaries

**What:** LLM injection lives in the Python `for` loop between calls to `train_and_eval_step`, not inside the jit-compiled scan. The JAX pytree (TrainState) is modified via `.replace()` before the next jit call.

**When to use:** Any time you need Python-side effects (API calls, file I/O) that should interact with a JAX training loop. The eval_freq loop is the natural boundary.

**Trade-offs:**
- Pro: No JAX tracing issues; full Python flexibility
- Pro: LLM latency doesn't block JAX training (runs synchronously between steps)
- Con: Injection granularity is coarse (every N eval_freq blocks, not every step)
- Con: Buffer injection causes one recompile of insert() — acceptable

**Example:**
```python
# In maze_plr.py main loop
injector = LLMInjector(config, level_sampler) if config["use_llm"] else None

for eval_step in range(config["num_updates"] // config["eval_freq"]):
    runner_state, metrics = train_and_eval_step(runner_state, None)  # jit
    log_eval(metrics, ...)

    if injector and eval_step % config["llm_interval"] == 0:
        runner_state = injector.inject(runner_state, eval_step)       # Python
```

### Pattern 2: Level as a Universal Currency

**What:** The `Maze.Level` dataclass (wall_map: bool[13,13], goal_pos, agent_pos, agent_dir) is the sole interface between all components. LLM text is parsed to Level; VAE latents are decoded to Level; ACCEL mutates Level; LevelSampler stores Level.

**When to use:** Any new generation source (LLM, VAE, random, or future sources) must produce valid `Level` objects. Do not create intermediate representations.

**Trade-offs:**
- Pro: All generation sources are interchangeable from the buffer's perspective
- Pro: `is_well_formatted()` is the single validity contract
- Con: Text-to-Level parsing is brittle (LLM may produce malformed grids)

**Example:**
```python
# Text from LLM → Level (in maze_generator.py)
def parse_grid(text: str) -> Optional[Level]:
    grid = extract_grid(text)           # regex/parsing
    wall_map = grid_to_wall_map(grid)   # bool[13,13]
    goal_pos = find_goal(grid)
    agent_pos, agent_dir = find_agent(grid)
    level = Level(wall_map, goal_pos, agent_pos, agent_dir)
    return level if level.is_well_formatted() else None
```

### Pattern 3: Functional Buffer Mutation

**What:** `LevelSampler` is purely functional — all methods take and return a sampler dict. No in-place mutation. This is required by JAX's functional style.

**When to use:** All buffer interactions. Even from Python-side injection, use `level_sampler.insert()` (returns new sampler) and assign to `train_state.replace(sampler=new_sampler)`.

**Trade-offs:**
- Pro: Correct JAX semantics; no tracing issues
- Pro: Explicit data flow — easy to inspect state before/after injection
- Con: Slightly more verbose than imperative style

**Example:**
```python
# Correct: functional buffer insertion from Python
def inject_into_buffer(train_state, accepted_levels, scores):
    sampler = train_state.sampler
    for level, score in zip(accepted_levels, scores):
        sampler, inserted_idx = level_sampler.insert(sampler, level, float(score))
    return train_state.replace(sampler=sampler)
```

## Data Flow: Maze Format Conversion Pipeline

```
LLM text output:
  "###########\n#.........#\n..."  (ASCII, 13×13)
       ↓
parse_grid() — regex extraction + validation
       ↓
  char grid: np.ndarray[13,13] of {#, ., >, v, <, ^, G}
       ↓
grid_to_wall_map() — '#' → True, others → False
find_goal()        — 'G' → (row, col)
find_agent()       — {>, v, <, ^} → (row, col) + direction int
       ↓
  Level(wall_map=bool[13,13],
        goal_pos=(row,col),
        agent_pos=(row,col),
        agent_dir=int)
       ↓
Level.is_well_formatted()  — validates no overlap, correct dims
       ↓
  JAX array fields: jnp.array for all fields (for vmap compatibility)
       ↓
level_sampler.insert(sampler, level, score)  — into buffer
```

**Critical note:** `Level` fields must be JAX arrays (or Python ints that JAX can trace). When constructing from LLM output (numpy/Python), ensure conversion before `insert()`. The `Level.stack()` class method handles this for batching.

## Anti-Patterns

### Anti-Pattern 1: LLM Call Inside lax.scan or jit

**What people do:** Try to call `MazeGenerator.generate()` inside `train_step` or `train_and_eval_step`.

**Why it's wrong:** `lax.scan` compiles the loop body into XLA. Any Python side effects (HTTP calls, print statements, etc.) only execute during tracing, not during actual execution. The LLM call would run once during compilation and never again, silently producing wrong behavior.

**Do this instead:** Keep LLM generation in the Python for-loop. Use `eval_step % llm_interval == 0` to control frequency.

### Anti-Pattern 2: In-Place Buffer Mutation

**What people do:** Directly index into `train_state.sampler["levels"]` and assign.

**Why it's wrong:** `TrainState` is a Flax struct (frozen dict). Python-side mutation bypasses JAX's pytree semantics and will cause issues when the state is later traced by jit.

**Do this instead:** Use `level_sampler.insert()` which returns a new sampler dict, then `train_state.replace(sampler=new_sampler)`.

### Anti-Pattern 3: Blocking Training on LLM Latency

**What people do:** Inject on every train step (or every eval_freq step unconditionally with no cap on generation count).

**Why it's wrong:** LLM API calls take 2-30 seconds. JAX training takes ~2ms per step. Even injecting every 10 eval_freq blocks (100 steps) adds ~30 seconds per 100 steps, roughly doubling training time.

**Do this instead:** Set `--llm_interval 50` (inject every 50 eval cycles = every 500 steps). Generate a small batch (3-5 mazes) per injection. Consider async generation in background thread if latency is prohibitive.

### Anti-Pattern 4: Injecting Unvalidated Levels

**What people do:** Skip `is_well_formatted()` check and insert LLM output directly.

**Why it's wrong:** LLMs produce malformed grids (wall at agent position, no path to goal, wrong dimensions). Invalid levels cause JAX environment resets to NaN states or silent crashes.

**Do this instead:** Always check `level.is_well_formatted()` before `level_sampler.insert()`. Track and log the invalid percentage via WandB.

### Anti-Pattern 5: Seeding CMA-ES from LLM Output

**What people do:** Try to feed LLM-generated Level objects back into CMA-ES by encoding them through the VAE.

**Why it's wrong:** CMA-ES is optimizing in a specific latent space; injecting externally-derived latents disrupts the Gaussian model and causes degenerate CMA-ES behavior (covariance explosion or collapse).

**Do this instead:** LLM and CMA-ES are parallel, independent generation sources. Both insert into the same buffer. ACCEL mutation is the mechanism that hybridizes them — an LLM-injected maze gets mutated by ACCEL naturally when replayed.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Ollama/OpenRouter API | HTTP POST in `MazeGenerator`, synchronous | API key in `.env` or `OLLAMA_API_KEY` env var; timeout=60s default |
| WandB | `wandb.log()` with LLM-specific keys | Log injection count, acceptance rate, latency per injection event |
| Orbax Checkpoints | No change — checkpoint saves `train_state` including injected levels | Injected levels are indistinguishable in saved buffer |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `maze_plr.py` ↔ `LLMInjector` | Direct Python call: `injector.inject(runner_state, step)` returns `runner_state` | Injector owns no state; stateless between calls except `AgentEvaluator` (loaded checkpoint) |
| `LLMInjector` ↔ `LevelSampler` | Functional: `level_sampler.insert(sampler, level, score)` | Must pass level_sampler instance (has capacity/replay_prob config) |
| `AgentEvaluator` ↔ `Maze.Level` | `evaluate_level(level: Level) → dict` | Builds jit-compiled rollout function on first call; re-JITs on batch size change |
| `BufferStatsExtractor` ↔ `sampler` | numpy conversion: `np.asarray(sampler["levels"].wall_map[:size])` | Must slice by `sampler["size"]` to avoid uninitialized buffer slots |
| `DecisionGate` ↔ `MazeGenerator` | `GateResult.issues` → `build_feedback_prompt(issues)` → retry | Issues are LLM-readable text; feedback is injected as conversation history |

## Build Order Implications

The component dependency graph determines phase ordering:

```
Phase 1: Maze format conversion + validation
  └── Level parsing (text → Level), is_well_formatted(), WandB logging skeleton
  └── No dependencies on other new components

Phase 2: Buffer statistics extraction
  └── Depends on: Phase 1 (Level format confirmed)
  └── Read sampler dict from running training → format as ReferenceMaze[]

Phase 3: LLM generation + prompt building
  └── Depends on: Phase 2 (buffer stats available for prompts)
  └── MazeGenerator already exists; wire buffer_stats → prompt_builder

Phase 4: Decision gate integration
  └── Depends on: Phase 3 (candidates exist to gate)
  └── AgentEvaluator already exists; wire candidate_level → gate → accept/reject

Phase 5: Training loop hook + injection
  └── Depends on: Phases 1-4 (all pipeline components ready)
  └── LLMInjector class + maze_plr.py modification

Phase 6: Experiments and analysis
  └── Depends on: Phase 5 (injection working)
  └── Launch scripts, comparison scripts, ablations
```

**Critical path:** Level conversion correctness (Phase 1) blocks everything. Test it in isolation before wiring into the training loop.

## Scaling Considerations

| Concern | Current Scale (research) | If Scaling Up |
|---------|-------------------------|---------------|
| LLM API cost | ~$0.01-0.05 per injection batch (3-5 mazes) | Cache accepted mazes across runs; use cheaper model for initial filter |
| Buffer contamination | 4000-level buffer; LLM contributes ~10-50 mazes per run | Track source tag per level in `levels_extra` for attribution |
| Injection frequency | Every 500 steps = ~30 injections per 50k-step run | Adaptive: increase frequency when buffer diversity drops |
| JAX recompile on injection | insert() may trigger recompile if sampler["size"] changes traced shapes | Pre-fill buffer to capacity before injection; `donate_argnums` if needed |

## Sources

- Direct codebase inspection: `examples/maze_plr.py` (training loop, train_step, on_new/replay/mutate_levels)
- Direct codebase inspection: `src/jaxued/level_sampler.py` (LevelSampler — full file)
- Direct codebase inspection: `llm/maze_generator.py` (MazeGenerator, GenerationResult, GenerationConfig)
- Direct codebase inspection: `llm/decision_gate.py` (DecisionGate, DiversityThresholds, GateResult)
- Direct codebase inspection: `llm/agent_evaluator.py` (AgentEvaluator, rollout, trajectory format)
- Direct codebase inspection: `llm/prompt_builder.py` (ReferenceMaze, MetricEntry, build_generation_prompt)
- Direct codebase inspection: `.planning/codebase/ARCHITECTURE.md` (existing system layers)
- Direct codebase inspection: `.planning/codebase/STRUCTURE.md` (directory layout)
- Direct codebase inspection: `.planning/PROJECT.md` (project requirements and constraints)

---
*Architecture research for: LLM maze injection into JAXUED training pipeline*
*Researched: 2026-03-23*
