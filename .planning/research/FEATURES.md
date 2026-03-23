# Feature Research

**Domain:** LLM-augmented UED curriculum learning — periodic maze injection into ACCEL/PLR training
**Researched:** 2026-03-23
**Confidence:** HIGH (analysis based on existing codebase + PROJECT.md; no external lookup needed for integration features)

---

## Context

This research covers features for a **subsequent milestone**: integrating the existing standalone `llm/`
module (generator, decision gate, agent evaluator) into the live JAX training loop in `examples/maze_plr.py`.
The LLM code is complete and tested. The training loop is mature. The work is a bridge between them.

Features are evaluated from two perspectives:
1. **Does the integration work at all?** (functional correctness)
2. **Does the thesis story hold?** (scientific validity + explainability)

---

## Feature Landscape

### Table Stakes (Integration Must Have These)

Features that make the integration functionally correct. Missing any of these means the experiment
either does not run, produces invalid data, or cannot be compared against baselines.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Periodic injection hook in training loop | Without this, LLM never fires during training | MEDIUM | Outer Python loop at `eval_step` level (every `eval_freq` updates). Not inside JIT — avoids JAX boundary issues. Pattern already exists for `buffer_dump_interval`. |
| Configurable injection frequency (`--llm_inject_interval N`) | Allows ablation on injection rate; different schedules need comparison | LOW | Integer arg, default off (0 = disabled). Check `updates_so_far % llm_inject_interval == 0`. |
| Buffer statistics extraction for LLM prompt context | LLM needs to know what is currently in the buffer to generate diverse mazes | MEDIUM | Extract top-K mazes by regret from `sampler["levels"]` + `sampler["scores"]`. Convert Level → ASCII grid via existing `vae_level_utils` + `tokens_to_level`. Runs outside JIT. |
| Maze format conversion: ASCII text → Level object | LLM output is ASCII text; buffer expects `Level(wall_map, goal_pos, agent_pos, agent_dir)` | LOW | `llm/maze_generator.py` already does this via `_parse_level()` — returns a `Level` object on success. Wire it to `insert_batch()`. |
| Solvability + format validation before buffer insertion | Invalid mazes in the buffer corrupt training | LOW | Already in `MazeGenerator` via `_validate_format()` + `_bfs_solvable()`. Just confirm the `Level` passes `is_well_formatted()` before inserting. |
| Decision gate integration (difficulty + diversity filter) | Without filtering, injected mazes may be trivial or redundant, weakening thesis claim | HIGH | Requires running `AgentEvaluator` on each candidate against reference trajectories. The gate (`decision_gate.evaluate_candidate`) already exists — integration means calling it with the current agent checkpoint mid-training. |
| Agent checkpoint access mid-training | Gate requires evaluating the live agent (not a stored checkpoint) | MEDIUM | `AgentEvaluator` loads from file. During training, need to either dump a temp checkpoint or pass `train_state.params` directly. Temp checkpoint dump at injection time is simplest. |
| Buffer insert for accepted mazes | Accepted LLM mazes must enter the buffer to affect training | LOW | Use existing `level_sampler.insert_batch()`. Requires converting `Level` to batched pytree. Score can be initialized with buffer mean score or max score (configurable). |
| WandB logging of injection events | Without this, no evidence that injection actually happened; thesis needs observability | LOW | Log: `llm/injection_step`, `llm/candidates_generated`, `llm/acceptance_rate`, `llm/accepted_count`, `llm/diversity_score_mean`, `llm/regret_score_mean`. |
| Seed-controlled reproducibility | Experiments require reproducibility; LLM is stochastic but seed control on reference selection and gate thresholds matters | LOW | JAX PRNG for reference sampling; LLM temp is fixed in config (0.8); log config to WandB at run start. |
| Comparison experiment launch scripts | Thesis requires ACCEL-only vs ACCEL+LLM injection comparison | LOW | 2-3 new shell scripts (conditions × seeds). Pattern already established in `examples/launch_*.sh`. |

### Differentiators (Thesis-Competitive Advantage)

Features that go beyond making the integration work, and directly support the thesis claim that LLM
injection improves generalization beyond what automated search alone achieves.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Metric-informed prompt construction (regret + entropy + path overlay) | LLM generates structurally novel mazes precisely because it understands what patterns the agent already handles well | MEDIUM | Already implemented in `prompt_builder.py`. The differentiator is wiring the live buffer's agent trajectories into the prompt during training. Requires running `AgentEvaluator.evaluate_levels()` on reference mazes before each injection call. |
| TD-error EMD diversity gate (not just structural diversity) | Ensures injected mazes produce different *learning signals*, not just visually different wall layouts | HIGH | Already implemented in `decision_gate.py` with `td_error_emd`. The differentiator vs naive injection is proving the gate adds value — requires ablation (gate-on vs gate-off condition). |
| Diversity feedback loop (LLM retries on gate failure) | LLM receives specific metric feedback explaining why its candidate was rejected, and generates a better one | MEDIUM | Already in `MazeGenerator` via `build_diversity_feedback_prompt()` + `prior_messages`. The integration differentiator is verifying this actually improves acceptance rates during training. |
| Buffer statistics extraction: top-regret + structural diversity sampling | Reference maze selection strategy directly determines what the LLM learns from — top-regret surfaces the hardest mazes | MEDIUM | `strategy: top_regret` already in config. The differentiator is using *live* buffer scores (not a static dump) at each injection point. Need `strategy: diverse` variant for ablation. |
| ACCEL mutation of LLM-injected mazes | LLM provides seed diversity; ACCEL proliferates interesting variants — multiplicative effect | LOW | Already works: any level in the buffer can become a mutation source (Branch 2 in training loop). No code change needed. The differentiator is demonstrating in results that LLM-seeded mazes get mutated more than random mazes. |
| Injection timing analysis (early vs late training) | Injecting at different training phases (exploration vs exploitation) may have different effects | LOW | Implemented via `--llm_inject_start_step` parameter. Enables thesis claim about when LLM diversity is most valuable. |
| CENIE novelty gating (buffer-history-aware novelty) | Gate rejects mazes the agent has already experienced in hidden-state space, not just structurally similar ones | HIGH | Already in `decision_gate.py` with `cenie` option. Requires fitting `CENIEModel` on LSTM hidden states from buffer. Most sophisticated gate — good for thesis differentiation but adds latency. |

### Anti-Features (Deliberately NOT Build)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Async/concurrent LLM generation during JAX training | Sounds efficient — LLM runs in background while JAX trains | JAX compilation + Python GIL make true async complex; LLM calls (2-10s) are fast relative to injection interval (every 500+ steps = minutes); debugging async failures mid-experiment is costly | Synchronous blocking call at injection point. JAX training pauses for the LLM call. Acceptable because injection is infrequent. |
| Per-step LLM generation | More injections = more diversity | API cost O(steps); blocks JAX training every step; LLM cannot generate fast enough to keep up (step = milliseconds, LLM = seconds) | Periodic batch injection every N eval steps. Inject 4-8 mazes at once. |
| LLM-based mutation of existing mazes | Seems like a natural extension of LLM seeding | Mutation is already handled cheaply and efficiently by ACCEL (wall-flip). LLM mutation at $0.01-0.10/call multiplied by 32 environments × thousands of mutations = prohibitive cost | ACCEL handles mutation. LLM handles only seed-level generation. This is already the design — just enforce it. |
| Fine-tuning the LLM on accepted mazes | Personalize the LLM to the specific domain | Out of scope for thesis; adds a training regime for the LLM itself; Claude API does not expose fine-tuning | Use prompt engineering (reference mazes + metrics) to guide generation. Prompt quality is the variable to optimize. |
| Multi-provider A/B testing (Claude vs GPT vs Llama) | Compare LLM providers as an experiment dimension | Adds N×M experimental conditions; provider comparison is not the thesis question; thesis question is LLM-injection vs no-injection | Fix provider (claude-code / Sonnet) per PROJECT.md. Provider is not a variable. |
| Structural maze similarity gate (pixel diff / wall overlap) | Simple, fast pre-filter | Structural similarity is a weak proxy for behavioral similarity — two structurally different mazes can produce the same agent trajectory; the TD-error EMD gate already does behavioral gating | Use only behavioral gates (td_error_emd or CENIE). Structural checks only for hard validity (min_walls, min_path_distance). |
| Real-time buffer visualization during injection | Nice to see the buffer change live | Visualization is already done post-hoc in WandB; real-time adds latency to injection path and is not needed for research conclusions | Log injection events to WandB. Visualize buffer state at eval checkpoints (already implemented). |
| LLM-generated reward shaping or environment parameters | LLM influences more than level generation | Changes the RL problem, not the curriculum. Outside UED framework. Thesis is about curriculum quality, not reward design. | Keep LLM scope strictly to maze layout generation. Reward function is fixed (sparse goal-reach). |

---

## Feature Dependencies

```
[Periodic injection hook]
    └──requires──> [Configurable injection frequency]
    └──requires──> [Buffer statistics extraction]
                       └──requires──> [Agent checkpoint access mid-training]
    └──requires──> [Maze format conversion]
    └──requires──> [Buffer insert for accepted mazes]

[Decision gate integration]
    └──requires──> [Agent checkpoint access mid-training]
    └──requires──> [Buffer statistics extraction]  (for reference trajectories)
    └──enhances──> [Periodic injection hook]  (filters what gets inserted)

[Metric-informed prompt construction]
    └──requires──> [Buffer statistics extraction]
    └──requires──> [Agent checkpoint access mid-training]
    └──enhances──> [Periodic injection hook]  (improves LLM output quality)

[Diversity feedback loop]
    └──requires──> [Decision gate integration]
    └──enhances──> [Metric-informed prompt construction]

[WandB logging of injection events]
    └──requires──> [Periodic injection hook]
    └──enhances──> [Comparison experiment scripts]  (metrics to compare)

[ACCEL mutation of LLM-injected mazes]
    └──requires──> [Buffer insert for accepted mazes]
    (free — no code change; emerges from buffer mechanics)

[CENIE novelty gating]
    └──requires──> [Agent checkpoint access mid-training]
    └──requires──> [Decision gate integration]
    └──conflicts──> [TD-error EMD gate]  (use one or the other per run, not both)

[Injection timing analysis]
    └──requires──> [Periodic injection hook]
    └──requires──> [Configurable injection frequency]
```

### Dependency Notes

- **Agent checkpoint access mid-training requires careful implementation:** `AgentEvaluator` currently loads from file. During training, the simplest approach is a lightweight temp checkpoint dump (using existing Orbax path) at injection time, or passing `train_state` params directly to a rollout function without file I/O. The file-based approach reuses existing code at cost of disk I/O; direct-params approach avoids disk but requires refactoring `AgentEvaluator.__init__`. Recommend: file-based for v1 (simpler), direct params for v1.x (performance).

- **Buffer statistics extraction requires leaving JAX device:** The level buffer lives on JAX device (GPU/TPU). Extracting reference mazes means `jax.device_get()` on `sampler["levels"]` + `sampler["scores"]`. This is safe from Python at eval-step boundary (not inside `lax.scan`). Cost is a D2H transfer of ~4KB per maze × 6 references = ~24KB, negligible.

- **TD-error EMD conflicts with CENIE as gate metrics:** Both gate on behavioral diversity but via different mechanisms. TD-error EMD is pairwise (candidate vs each reference); CENIE is buffer-wide (candidate vs all past experiences via GMM). They are mutually exclusive in `DiversityThresholds.diversity_metric`. Choose per experiment condition, not both simultaneously.

- **Diversity feedback loop enhances prompt construction:** The feedback path in `build_diversity_feedback_prompt()` injects metric definitions and similarity analysis. This only has value if metric-informed prompts are already active. If running without metric injection, diversity feedback reduces to generic "make it different" prompts.

---

## MVP Definition

### Launch With (v1) — Thesis Experiment Baseline

Minimum needed to run the comparison experiment: ACCEL-only vs ACCEL+LLM injection.

- [ ] Periodic injection hook at `eval_step` boundary in `maze_plr.py` — core integration
- [ ] Configurable `--llm_inject_interval N` and `--llm_batch_size M` args — experiment control
- [ ] Buffer statistics extraction: top-K mazes by regret → `ReferenceMaze` objects — LLM context
- [ ] Agent checkpoint dump + `AgentEvaluator` call at injection time — gate evaluation
- [ ] Decision gate (td_error_emd, difficulty=regret) applied to each candidate — behavioral filtering
- [ ] `level_sampler.insert_batch()` call for accepted mazes — buffer population
- [ ] WandB logging: `llm/acceptance_rate`, `llm/injected_count`, `llm/diversity_score_mean` — observability
- [ ] Two launch scripts: `launch_llm_injection_baseline.sh`, `launch_accel_only_control.sh` — experiment execution

### Add After Validation (v1.x) — Strengthen Thesis Claims

Features to add once v1 shows measurable effect (or diagnose why effect is absent).

- [ ] Metric-informed prompts (per-step entropy, regret, path overlay) — trigger: v1 acceptance rate is too low, suggesting LLM needs more context to generate useful mazes
- [ ] Diversity feedback loop (LLM retries on gate failure) — trigger: high rejection rate in WandB logs; adds ~1-3 extra LLM calls per injection event
- [ ] Direct `train_state.params` passing to evaluator (no file I/O) — trigger: injection latency exceeds 30s, causing noticeable training pauses
- [ ] Injection timing analysis (`--llm_inject_start_step`) — trigger: v1 shows effect in late training but not early; worth ablating
- [ ] Reference maze selection strategy ablation (`diverse` vs `top_regret`) — trigger: v1 results are mixed, need to diagnose whether reference selection is a bottleneck

### Future Consideration (v2+) — Extended Thesis

Features requiring significant additional work, deferred until core results are in.

- [ ] CENIE novelty gate replacing TD-error EMD — requires GMM fitting on LSTM hidden states; adds ~5-10s per injection; justified only if TD-error EMD gate misses obvious behavioral redundancy
- [ ] Structural diversity of injection batch (generate N candidates, keep top-K by mutual diversity) — reduces per-injection redundancy; adds N×gate_cost; justified if accepted mazes cluster structurally
- [ ] Online prompt optimization (track which prompt strategies led to accepted+high-scoring mazes, weight accordingly) — essentially meta-learning for prompts; out of scope for thesis but interesting follow-up

---

## Feature Prioritization Matrix

| Feature | Research Value | Implementation Cost | Priority |
|---------|---------------|---------------------|----------|
| Periodic injection hook | HIGH — prerequisite for everything | MEDIUM | P1 |
| Configurable injection frequency | HIGH — ablation axis | LOW | P1 |
| Buffer statistics extraction | HIGH — LLM quality depends on context | MEDIUM | P1 |
| Agent checkpoint access mid-training | HIGH — gate requires live agent | MEDIUM | P1 |
| Decision gate (td_error_emd) | HIGH — core thesis claim (behavioral diversity) | HIGH | P1 |
| Buffer insert for accepted mazes | HIGH — prerequisite for any effect | LOW | P1 |
| WandB injection logging | HIGH — necessary for scientific reporting | LOW | P1 |
| Comparison launch scripts | HIGH — thesis requires controlled comparison | LOW | P1 |
| Metric-informed prompts (live) | MEDIUM — improves LLM quality but adds latency | MEDIUM | P2 |
| Diversity feedback loop | MEDIUM — improves acceptance rate | MEDIUM | P2 |
| Injection timing analysis | MEDIUM — additional ablation dimension | LOW | P2 |
| Reference selection strategy ablation | MEDIUM — diagnostic for mixed results | LOW | P2 |
| CENIE novelty gate | LOW — sophisticated but high latency | HIGH | P3 |
| Direct params evaluator (no file I/O) | LOW — optimization, not correctness | MEDIUM | P3 |

**Priority key:**
- P1: Must have for launch — v1 milestone
- P2: Should have — add after v1 validates concept
- P3: Nice to have — v2+ or future paper

---

## Comparable System Analysis

There are no direct open-source competitors implementing LLM injection into PLR/ACCEL specifically.
The closest comparable systems and their approaches:

| Feature | PAIRED/ACCEL (this codebase) | Voyager (LLM + RL for Minecraft) | PCG-LLM (generative curriculum papers) | Our Approach |
|---------|------------------------------|----------------------------------|----------------------------------------|--------------|
| Level generation method | Random mutation / CMA-ES | LLM proposes tasks via code | LLM proposes levels via text | LLM seed generation, ACCEL mutation |
| Curriculum signal | Regret / MaxMC | Success rate | Difficulty estimate | Regret + SFL learnability |
| Diversity enforcement | None explicit | None explicit | Structural diff | TD-error EMD (behavioral) |
| LLM feedback loop | N/A | Agent execution + skill library | None | Diversity gate feedback prompts |
| Integration depth | N/A | Full — LLM is primary generator | Offline pre-generation | Periodic injection, ACCEL proliferates |

The behavioral diversity gate (TD-error EMD, mode transition divergence) is the clearest differentiator
from naive LLM curriculum papers that use only structural novelty metrics.

---

## Sources

- Codebase: `llm/maze_generator.py`, `llm/decision_gate.py`, `llm/agent_evaluator.py`, `llm/prompt_builder.py`, `llm/config.yaml`
- Codebase: `examples/maze_plr.py` (training loop, injection hook points at lines 1050-1063)
- Codebase: `.planning/codebase/ARCHITECTURE.md` (training cycle, data flow, component boundaries)
- Codebase: `.planning/PROJECT.md` (project requirements, constraints, out-of-scope items)
- Training data reference (LOW confidence, used only for comparable system analysis): Voyager (Wang et al. 2023), general PCG-LLM literature

---

*Feature research for: LLM-augmented UED maze injection — ACCEL/PLR training pipeline*
*Researched: 2026-03-23*
