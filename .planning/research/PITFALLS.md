# Pitfalls Research

**Domain:** LLM-augmented Unsupervised Environment Design (UED) — adding LLM-based maze injection to a JAX RL training pipeline
**Researched:** 2026-03-23
**Confidence:** HIGH — grounded in direct codebase inspection of `llm/maze_generator.py`, `llm/decision_gate.py`, `src/jaxued/level_sampler.py`, `src/jaxued/environments/maze/level.py`, and `examples/maze_plr.py`

---

## Critical Pitfalls

### Pitfall 1: JAX JIT Boundary Violation — Calling Python-side LLM Inside a Traced Function

**What goes wrong:**
The training loop runs inside `jax.lax.scan` / `jax.lax.switch`, which traces Python code to XLA. If LLM generation is called anywhere within the scan body, JAX will try to trace the Python I/O (HTTP requests, subprocess calls) and either error out with "side-effect in traced function" or silently produce incorrect XLA that ignores the call entirely.

**Why it happens:**
The natural integration point looks like `train_step` — you want injection to happen every N steps. Developers insert an `if step % N == 0: inject()` inside `train_step`, not realizing the entire function body is traced. Python `if` on a traced value is silently evaluated at trace time (always False at step 0), so injection never runs in practice.

**How to avoid:**
LLM injection must happen *outside* the `jax.lax.scan` loop, in the Python orchestration layer. The correct pattern is:

```python
# CORRECT: injection in Python loop, not inside JAX scan
for outer_step in range(num_outer_steps):
    if outer_step % llm_interval == 0:
        new_levels = llm_generator.generate_batch(...)
        train_state = inject_levels_into_buffer(train_state, new_levels)
    # run inner JAX scan for N steps
    train_state, metrics = jax.lax.scan(train_step, (rng, train_state), ...)
```

The buffer injection itself (`level_sampler.insert`) is a JAX operation on pytrees — that is fine. Only the LLM API call must stay in Python.

**Warning signs:**
- Injection counter never increases in WandB logs even though `--llm_inject_interval` is set
- Training speed is identical with and without LLM enabled
- No API calls visible in network logs during training runs

**Phase to address:** Phase 1 (Integration scaffolding) — establish the Python/JAX boundary explicitly before writing any injection code.

---

### Pitfall 2: Score Initialization Trap — LLM Levels Inserted with Wrong Score Get Evicted Immediately

**What goes wrong:**
`LevelSampler._insert_new` only inserts a level if `score > scores[idx]` (lowest-weight slot score). At buffer capacity, the lowest-weight slot has a meaningful score. If you insert LLM levels with `score=0.0` or `score=-inf`, they either fail the insertion condition or get evicted on the very next insertion. You'll see 0% retention of LLM-generated levels in the buffer.

**Why it happens:**
The buffer `insert` API requires you to supply a score alongside the level. It is tempting to insert with a neutral score and let the training loop update it. But if the buffer is full (it reaches `capacity=4000` quickly), any neutral score is below the median score of existing levels and gets evicted before a single replay episode happens.

**How to avoid:**
Insert LLM levels with a score derived from the decision gate's regret or learnability metric computed *before* insertion. The gate already computes `regret_info.regret` or `learnability_info.learnability` as part of `evaluate_candidate`. Use that value as the insertion score:

```python
gate_result = evaluate_candidate(traj, refs, labels, thresholds)
if gate_result.accepted:
    score = gate_result.regret_info.regret  # or learnability
    sampler = level_sampler.insert(sampler, level, score)
```

This ensures the LLM level competes fairly for buffer slots. Do NOT use a hardcoded high score to force retention — that corrupts priority-based replay.

**Warning signs:**
- `llm_injected_count` WandB metric increments but buffer size stays flat
- `llm_retained_rate` metric shows ~0%
- Buffer composition plots show no LLM levels after the first few injection events

**Phase to address:** Phase 1 (Integration scaffolding) — buffer insertion must be designed correctly from the start.

---

### Pitfall 3: Maze Format Conversion — `is_well_formatted()` Passes but Training Crashes

**What goes wrong:**
`Level.is_well_formatted()` checks: binary wall_map, distinct agent/goal positions, valid agent direction, neither position on a wall, both within bounds. This is necessary but not sufficient for training. Three additional failure modes are not caught:

1. **Wall perimeter not solid**: The JAX maze environment assumes the border cells are walls. A maze where the LLM forgot to put walls on row 0, row 12, col 0, or col 12 lets the agent walk off the grid, producing OOB index errors inside `env.step` or NaN rewards.
2. **Maze is valid but trivially solvable in 1-2 steps**: Goal placed adjacent to agent. The level passes all checks but contributes zero gradient signal. At high injection rates this dilutes the curriculum.
3. **dtype mismatch between Level arrays**: `Level.from_str` stores `wall_map` as `jnp.bool_`, positions as `jnp.uint32`, `agent_dir` as `jnp.uint8`. If any code path reconstructs a Level with Python ints, `jax.lax.switch` will re-JIT the train branch for every novel dtype combination, causing O(N) compilation on injection events.

**Why it happens:**
`Level.from_str` is the only path from LLM text to Level objects. It uses `assert` statements that become `AssertionError` on malformed input. But once past parsing, no structural/gameplay checks are done. The `is_well_formatted()` method is defined on the `Level` struct but is not called automatically anywhere in the training loop.

**How to avoid:**
Add a validation wrapper between LLM output and buffer insertion:

```python
def validate_llm_level(level: Level) -> tuple[bool, str]:
    """Extended validation beyond is_well_formatted()."""
    if not level.is_well_formatted():
        return False, "is_well_formatted() failed"
    # Check border walls
    if not (level.wall_map[0, :].all() and level.wall_map[-1, :].all()
            and level.wall_map[:, 0].all() and level.wall_map[:, -1].all()):
        return False, "border walls missing"
    # Check minimum path length (> 5 steps)
    if not _bfs_path_length(level) > 5:
        return False, "maze trivially solvable"
    # Check dtype consistency
    assert level.wall_map.dtype == jnp.bool_, f"wall_map dtype: {level.wall_map.dtype}"
    assert level.agent_pos.dtype == jnp.uint32
    return True, ""
```

The generator already runs BFS solvability (`_bfs_solvable`) — extend it to also check the border and path length.

**Warning signs:**
- `JAXTracerError` or shape errors during the first training step after an injection event
- Unexpectedly fast episode completion (reward spikes to 1.0 on injected levels)
- Increasing XLA compilation time after injection events (dtype re-trace)

**Phase to address:** Phase 1 (Format conversion and validation) — implement `validate_llm_level()` as a hard gate before the decision gate.

---

### Pitfall 4: Buffer Contamination from Decision Gate Trajectory Staleness

**What goes wrong:**
The decision gate evaluates candidate levels by running the *current* agent on them via `AgentEvaluator`. The agent evolves throughout training. If a batch of LLM levels is generated, evaluated via the gate at step 5000, and then buffered — but injection is delayed or the levels are cached — by the time those levels are replayed at step 15000, the agent is substantially different. The gate's diversity/difficulty judgment is now stale. Levels that were genuinely novel at step 5000 may be trivially easy at step 15000 (the buffer will over-replay them until their PLR score decays naturally, which takes ~hundreds of episodes).

**Why it happens:**
The gate is designed as a standalone evaluator (friend's codebase). It takes an agent checkpoint path, loads the agent once at construction, and reuses it. If `AgentEvaluator` is constructed once at training start and shared across all injection events, it evaluates every LLM maze against the policy from step 0 — even at step 40000. The diversity metrics (td_error_emd, CENIE) will be computed against a policy that no longer represents current training.

**How to avoid:**
Refresh `AgentEvaluator` at each injection event, loading the most recent checkpoint. Alternatively, use the live `train_state` directly (pass the JAX params to a lightweight rollout function instead of loading from disk). The checkpoint-per-injection approach has ~1-2s overhead per injection event which is acceptable given injection happens every few thousand steps.

```python
# Refresh evaluator at each injection point
evaluator = AgentEvaluator(
    checkpoint_dir=config["checkpoint_dir"],
    checkpoint_step=-1,  # always load latest
)
```

**Warning signs:**
- Gate acceptance rate is high at step 1000 but collapses to near-zero by step 30000 (old agent gave generous gate scores)
- Or the opposite: gate acceptance rate stays high throughout because the stale agent finds everything "novel"
- PLR scores on LLM-injected levels decay rapidly after injection (agent has already surpassed them)

**Phase to address:** Phase 2 (Decision gate integration) — document the staleness assumption explicitly and design refresh schedule.

---

### Pitfall 5: Training Instability from OOD Injection Rate Mismatch

**What goes wrong:**
LLM-generated mazes are structurally different from ACCEL-mutated mazes (ACCEL flips 1-3 walls on existing buffer levels — it explores local structure). LLM levels can introduce global structural novelty: different room patterns, different maze topologies. If injection rate is too high relative to the agent's learning rate, the policy is presented with too many OOD levels simultaneously, gradient updates become incoherent, and solve rate on all levels collapses. This can look exactly like a hyperparameter issue.

**Why it happens:**
The 3-way training cycle (CMA-ES gen → replay → ACCEL mutate) is tuned for gradual curriculum progression. The `replay_prob=0.95` and `minimum_fill_ratio=1.0` parameters mean the agent replays ~95% of the time once the buffer is full. Injecting a large LLM batch (e.g., 32 levels) at once inserts many novel levels, changes the buffer composition significantly, and spikes the difficulty.

**How to avoid:**
- Limit batch size: inject 4-8 LLM levels per event rather than 32. The ACCEL popsize is 32, but LLM levels are structurally distinct and need gradual introduction.
- Use a warmup period: don't inject LLM levels until training step N_warmup (e.g., 10k steps), by which point the agent has a baseline policy.
- Track `mean_solve_rate` before and after injection. If it drops more than 0.1 in the first 500 steps post-injection, that is a signal the injection rate is too high.

**Warning signs:**
- Solve rate collapses after the first injection event and does not recover
- Loss curves show sudden spikes in policy entropy after injection
- Mean regret of buffer levels jumps discontinuously at injection timesteps

**Phase to address:** Phase 2 (Injection frequency tuning) — injection interval and batch size must be first-class hyperparameters, not hardcoded.

---

### Pitfall 6: Reproducibility Broken by Stochastic LLM Outputs

**What goes wrong:**
LLM outputs are non-deterministic (temperature > 0). Running the same experiment twice with the same JAX seed will diverge because LLM injection produces different levels at different timesteps, changing the buffer composition and therefore the entire training trajectory. This makes experiment comparison unreliable — you cannot tell if differences between runs are due to LLM quality or LLM stochasticity.

**Why it happens:**
JAX experiments use `jax.random.PRNGKey(seed)` for full reproducibility. But the LLM call uses `requests.post` to a cloud API — no seed mechanism exists. Even `temperature=0` (greedy decoding) is not guaranteed to be deterministic across API versions, server restarts, or model updates.

**How to avoid:**
Two complementary strategies:
1. **Cache generated levels per experiment run**: Serialize each accepted LLM Level to disk at injection time (store as `wall_map.npy` + metadata JSON). On re-run, load from cache instead of calling the API. This makes re-runs perfectly reproducible and also saves API cost.
2. **Log level hashes to WandB**: Compute a hash of each injected level's `wall_map` and log it. This makes it possible to detect reproducibility failures in post-hoc analysis.

```python
import hashlib
def level_hash(level: Level) -> str:
    return hashlib.sha256(np.array(level.wall_map).tobytes()).hexdigest()[:12]
```

**Warning signs:**
- Two runs with `--seed 0` diverge more than expected (check WandB seed control)
- Post-hoc analysis of injected levels shows different levels in "identical" runs
- Statistical variance across seeds is higher than baseline (ACCEL-only) experiments

**Phase to address:** Phase 3 (Experiment infrastructure) — implement level caching before running comparison experiments.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcode `llm_inject_interval=5000` | No tuning required | Different injection frequencies behave very differently; no ablation possible | Never — make it a CLI arg from day one |
| Insert LLM levels with fixed score=0.5 | Simple | Doesn't reflect actual difficulty; breaks PLR invariants | Never — use gate's regret/learnability score |
| Reuse AgentEvaluator across all injection events | Saves ~2s per injection | Gate evaluates with stale policy; distorts curriculum | Only for debugging, never for reported experiments |
| Skip solvability re-check after Level format conversion | Saves time | LLM mazes that pass ASCII parse but are unsolvable corrupt training | Never — BFS check costs microseconds |
| Use `assert` in level parsing (current behavior) | Simple | Asserts are stripped in `-O` mode; silent failures in production runs | Fix: raise `ValueError` explicitly |
| Cache LLM levels in RAM only (Python list) | Simple | Re-runs call API again; no reproducibility | Never for reported results — always serialize to disk |

---

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Ollama/LLM API | Timeout of 0 means no timeout — training hangs indefinitely if API is down | Set `timeout=60` and implement fallback (skip injection for this interval, log warning) |
| Ollama/LLM API | Not handling HTTP 429 (rate limit) by sleeping — retries hammer the API | Implement exponential backoff: `time.sleep(2 ** attempt)` |
| Ollama/LLM API | Checking `result.success` but not checking `result.level is not None` | Both checks are required — `success=True` with `level=None` can occur if parse succeeded but Level construction failed |
| AgentEvaluator | `load_agent` uses `sys.path` manipulation (lines 18-26 of `agent_evaluator.py`) — fails if script is not run from project root | Always run with `PYTHONPATH=/path/to/project` or install as editable package |
| LevelSampler | `insert_batch` uses `jax.lax.scan` internally — passing Python-side Level objects (not batched JAX arrays) causes retracing | Use `Level.stack([...])` to convert list of Levels to batched pytree before calling `insert_batch` |
| WandB | Logging LLM metadata (level strings, rejection reasons) as WandB `log` calls inside the injection loop bloats the run timeline and slows API calls | Use `wandb.log` once per injection event with a summary dict, not per-level |

---

## Performance Traps

Patterns that work at small scale but fail as training progresses.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Rebuilding `AgentEvaluator` (reload checkpoint + re-JIT) for every generated level in a batch | Injection of 8 levels takes 8x checkpoint load time + 8x JIT compilation | Build evaluator once per batch, call `evaluate_levels([...])` for the whole batch in one shot | At batch_size >= 4, single-level evaluation is noticeably slower than batch |
| Calling `_build_rollout_fn` with a different `num_levels` each time | XLA recompiles the rollout function for every new batch size | Fix `batch_size` for all evaluator calls (e.g., always evaluate 8 levels, padding with dummy levels if fewer available) | Every injection event triggers recompilation (~5-10s) |
| Extracting buffer statistics by transferring the entire buffer from GPU to CPU | `np.asarray(sampler["levels"])` on a 4000-level buffer copies 4000 × 13 × 13 booleans + metadata every injection event | Extract only what is needed (e.g., scores, timestamps, a random sample of K high-scoring levels) | Buffer size > 1000 levels — transfer latency becomes noticeable relative to LLM call time |
| Running `jax.lax.scan` for LLM injection evaluation inside the training loop body | JAX tries to trace the Python HTTP call | Keep all LLM code outside `jax.lax.scan` in the Python orchestration layer | Every single use — this is a correctness trap, not just performance |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **LLM integration "working":** Verify LLM injection actually runs during training by checking WandB `llm_injected_count` metric increments at the correct timesteps, not just that the code path exists
- [ ] **Decision gate "filtering":** Verify that the acceptance rate is < 100% — a gate that always accepts means thresholds are set too low or the evaluator is returning wrong metrics
- [ ] **Buffer "receiving LLM levels":** Verify LLM levels are not immediately evicted — check `llm_retained_rate` = (levels still in buffer after 1000 steps post-injection) / (levels injected). Should be > 50%.
- [ ] **Experiments "reproducible":** Verify two runs with the same seed produce identical solve rate curves up to the first injection event (JAX determinism check), then diverge as expected (LLM stochasticity)
- [ ] **Maze format conversion "complete":** Verify that `Level.from_str(level.to_str()) == level` round-trips correctly for all LLM-generated levels — any information loss (e.g., multi-goal handling) is caught here
- [ ] **Comparison experiment "apples-to-apples":** Verify all conditions (ACCEL-only, ACCEL+LLM) use identical seeds, buffer sizes, and evaluation sets — the existing eval prefabs in `level.py` must be identical across conditions

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| JAX JIT boundary violation (LLM inside scan) | LOW | Move injection call to outer Python loop; no rewrite required |
| Buffer eviction (LLM levels never retained) | LOW | Rerun with corrected score initialization; old results are invalid but easy to regenerate |
| Maze format crash in training | MEDIUM | Add `validate_llm_level()` wrapper; purge and re-generate any cached malformed levels |
| Stale AgentEvaluator producing wrong gate scores | MEDIUM | Re-run from most recent checkpoint with evaluator refresh enabled; compare new gate scores to logged ones |
| Training instability from OOD injection | MEDIUM | Reduce batch size to 4, increase warmup to 15k steps, re-run; may need 1-2 days compute |
| Irreproducible experiments (no level caching) | HIGH | Must re-run all comparison experiments with caching enabled; prior results cannot be compared rigorously |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| JAX JIT boundary violation | Phase 1: Integration scaffolding | `jax.make_jaxpr(train_step)` does not raise side-effect warnings; injection counter increments in WandB |
| Buffer score initialization trap | Phase 1: Buffer insertion design | `llm_retained_rate` > 50% in first 5k steps post-injection |
| Maze format validation gaps | Phase 1: Format conversion | All border cells are walls; BFS path length > 5; dtype checks pass |
| Decision gate trajectory staleness | Phase 2: Gate integration | Log gate evaluation timestep vs. current training timestep delta; delta should be < 1 injection interval |
| OOD injection instability | Phase 2: Injection frequency tuning | Solve rate does not drop > 0.1 within 500 steps of injection; ablate `--llm_batch_size` in [4, 8, 16] |
| Reproducibility from stochastic LLM | Phase 3: Experiment infrastructure | Level hashes logged to WandB; re-run divergence only after first injection event |
| AgentEvaluator sys.path fragility | Phase 1: Integration scaffolding | CI/test that runs `from llm.agent_evaluator import AgentEvaluator` from a non-root directory passes |

---

## Sources

- Direct code inspection: `llm/maze_generator.py` — `_parse_level`, `_bfs_solvable`, `generate_with_feedback`, `generate_batch`
- Direct code inspection: `llm/decision_gate.py` — `evaluate_candidate`, `DiversityThresholds`, `GateResult`
- Direct code inspection: `src/jaxued/level_sampler.py` — `_insert_new`, `insert_batch`, `level_weights`, score/eviction logic
- Direct code inspection: `src/jaxued/environments/maze/level.py` — `Level.from_str`, `is_well_formatted`, dtype contracts
- Direct code inspection: `examples/maze_plr.py` — `train_step`, `jax.lax.switch` branch logic, buffer initialization
- Direct code inspection: `llm/agent_evaluator.py` — `AgentEvaluator`, sys.path manipulation, rollout JIT rebuild
- Codebase concerns audit: `.planning/codebase/CONCERNS.md` — existing tech debt, known bugs, fragile areas
- Project context: `.planning/PROJECT.md` — integration requirements, constraints, out-of-scope items

---
*Pitfalls research for: LLM maze injection into JAXUED training pipeline*
*Researched: 2026-03-23*
