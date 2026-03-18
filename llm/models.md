# LLM Maze Generator — Model Evaluation Notes

Findings from testing various LLMs on 13x13 maze generation for the ACCEL RL pipeline.
All tests use the same prompt structure: 3 reference mazes from the replay buffer, metric injection, and solvability validation.

## Summary

### Recommended Models

| Model            | Provider     | Success | Time/Maze | Regret  | Solve Rate | $/Maze | Cost (in/out) |
|------------------|-------------|---------|-----------|---------|------------|--------|---------------|
| claude-sonnet-4  | OpenRouter  | 100%    | 6-16s     | 1.0-1.1 | 5-100%     | ~$0.02 | $3/$15 per M  |
| claude-opus-4    | OpenRouter  | 100%    | 22-163s   | 1.0-1.2 | 11-92%     | ~$0.23 | $15/$75 per M |
| gpt-5.4          | OpenRouter  | 67%     | 11s       | 1.0-1.1 | 97-100%    | ~$0.01 | $2.50/$10 per M |
| kimi-k2.5        | OpenRouter  | 67%     | 150-560s  | 1.0-1.04| 100%       | ~$0.02 | $0.60/$2 per M |
| deepseek-v3.2-speciale | OpenRouter | 33%  | 172s    | 1.05    | 100%       | ~$0.04 | $2.19/$8.76 per M |
| deepseek-v3.2    | OpenRouter  | 33%     | 5-23s     | 1.0     | 100%       | <$0.01 | $0.27/$1.10 per M |
| kimi-k2.5:cloud  | Ollama      | ~80%    | 300-400s  | 0.2-1.1 | 19-100%    | Free   | Free          |

### Not Recommended

| Model                  | Provider     | Success | Time/Maze | Failure Mode                       |
|------------------------|-------------|---------|-----------|-------------------------------------|
| deepseek-v3.2:cloud    | Ollama      | ~30%    | 8-12s     | Wrong row widths; copies references |
| qwen3.5:cloud          | Ollama      | ~80%    | 260-390s  | Slow; inconsistent difficulty       |
| qwen3:235b-a22b-cloud  | Ollama      | 0%      | 26s       | Can't produce valid grids           |
| claude-opus-4-6        | OpenRouter  | 100%    | 4s        | Low solve rates (0-20%)             |
| gpt-5.4-mini           | OpenRouter  | 33%     | 7s        | Format errors; low solve rate       |
| gpt-5.4-nano           | OpenRouter  | 0%      | 11s       | Can't produce valid 13x13 grids     |
| nemotron-3-super-120b  | OpenRouter  | 33%     | ~250s     | Mostly unparseable reasoning; 1 success after bug fix |
| minimax-m2.7           | OpenRouter  | 0%      | ~60s/call | Thinks hard (~13K chars), still can't count to 13 |
| glm-5-turbo            | OpenRouter  | 33%     | ~60s/call | Slow; mixed format errors; 1 success after bug fix  |

**Cost** = input/output per million tokens. Regret = MaxMC scalar regret from diversity gate.
Thinking traces available for: claude-opus-4 (`--thinking`), kimi-k2.5, qwen3.5, deepseek-v3.2 (via fallback).

## Detailed Notes

### kimi-k2.5:cloud (Ollama)
- **Thinking model** that puts output in `thinking` field (content is empty)
- Consistently produces valid 13x13 grids after the thinking→content fallback
- Slow (~300-400s per maze) because it's a reasoning model
- Good diversity when used with feedback loop
- Best model on the Ollama Cloud platform
- 100% solve rate on many runs; occasionally generates unsolvable mazes

### deepseek-v3.2:cloud (Ollama)
- Also a thinking model (output in `thinking` field)
- **Core problem**: consistently produces 12-char rows instead of 13
- When it does produce valid grids, they are near-copies of references (just swaps agent/goal)
- Very fast (~8s/maze) but the speed is wasted on bad output
- Not recommended for maze generation

### qwen3.5:cloud (Ollama)
- Thinking model with captured reasoning traces
- Decent quality when it produces valid output
- Slow (~260-390s per maze)
- Variable solve rates (0%-100%); inconsistent difficulty calibration

### claude-sonnet-4 (OpenRouter)
- **Best overall value** — 100% success rate, fast (~6s without feedback, ~16s with)
- Not a thinking model; no reasoning traces available
- Tends toward walled-border labyrinth style (different from buffer's open-border style)
- Common first-attempt error: uses `S` instead of `>v<^` for agent start (self-corrects on retry)
- Within-batch diversity is moderate — similar recursive subdivision patterns
- Feedback loop works well: regret improves from 0.2-0.7 to >1.0 after feedback
- Oscillation problem on hard mazes: swings between "too easy" and "too hard"

### claude-opus-4 (OpenRouter)
- **Highest quality** mazes; best reasoning about RL agent behavior
- With `--thinking`: captures detailed reasoning traces (5-13K chars)
  - Analyzes reference maze patterns (start positions, action dominance, entropy)
  - Manually traces paths through proposed mazes to verify solvability
  - Iterates on designs when first attempt is unsolvable
- First-attempt regret often >1.0 (passes diversity gate immediately)
- 92% solve rate on best maze; produces challenging but solvable designs
- Expensive (~$15/M input, ~$75/M output) and slower with thinking (~163s)
- Without thinking: still good quality, ~22s per maze

### claude-opus-4-6 (OpenRouter)
- Very fast (~4s/maze) but low solve rates (0-20%) in early tests
- May have been tested with different config (earlier in development)
- Needs re-evaluation with current prompt/config

### gpt-5.4 (OpenRouter)
- Good quality when it produces valid output (67% success)
- High regret (1.0-1.1) and high solve rates (97-100%)
- Cheaper than Claude Opus; good alternative to Sonnet
- Occasional format errors (wrong row count/width)

### gpt-5.4-mini / nano (OpenRouter)
- Mini: marginal (33% success), struggles with format
- Nano: complete failure (0% success) — too small for structured grid output
- Not recommended

### nvidia/nemotron-3-super-120b (OpenRouter, free)
- Thinking model with always-on reasoning; `content: None` bug caused 0% in initial runs
- After fix: 1/3 success — regret=1.052, **100% solve rate**, 29-step episodes
- Reasoning text mostly doesn't contain parseable grids (0 valid rows in most attempts)
- Very slow (~250s/attempt) and free tier may be rate-limited
- When it works, quality is good — but too unreliable for production use

### kimi-k2.5 (OpenRouter — moonshotai/kimi-k2.5)
- **Thinking model** — returns reasoning in `reasoning` field, content often `None`
- Required `content: None` fix (returns null instead of empty string; reasoning fallback needed)
- When reasoning fallback works, produces grids embedded in reasoning text
- 2/3 success: regret=1.013 and 1.039, both **100% solve rate**
- Maze 1 had regret=0.994 (0.006 under threshold!) — then reasoning fallback couldn't parse retries
- Slow (~150-560s/maze) due to thinking; cheaper than Ollama Cloud equivalent
- Good quality when it works but inconsistent grid extraction from reasoning text

### deepseek/deepseek-v3.2-speciale (OpenRouter)
- High-compute reasoning variant of DeepSeek V3.2
- Same `content: None` bug as Kimi — reasoning fallback needed
- 1/3 success: regret=1.049, **100% solve rate**, passed gate first check
- Failed mazes had 0-2 valid rows in reasoning text (model reasons about mazes but doesn't output clean grids)
- Very slow (~172s for the success, ~100s per failed attempt)
- ~$2.19/$8.76 per M tokens — expensive for 33% success rate

### minimax/minimax-m2.7 (OpenRouter)
- **Thinking model** with uncapped reasoning (~9K-18K chars per call, ~60s each)
- First valid maze had regret=0.750, second had 0.953 — improving but never hit 1.0 threshold
- Persistent 14-char row width problem (even after explicit error feedback and deep reasoning)
- Total wall clock for 1 maze attempt: **400s** (6.5 minutes), produced 0 successful mazes
- Despite more reasoning tokens than Opus (~51K total vs ~9K), worse outcomes
- Not viable: too slow, too expensive on reasoning tokens, unreliable grid formatting

## Key Findings

1. **Grid format precision is the #1 bottleneck** — most models fail because they can't consistently produce exactly 13 chars x 13 rows. This is a character-counting task that trips up smaller and thinking-heavy models.

2. **Thinking models are a double-edged sword** — Opus uses thinking to verify solvability (good), but Nemotron/DeepSeek waste tokens on reasoning and still produce broken grids.

3. **The feedback loop works** — when a model can produce valid grids, metric feedback consistently improves regret from 0.2-0.7 to >1.0. Path overlays and per-step regret curves give the LLM actionable information.

4. **Cost vs quality tradeoff**:
   - **Budget**: Sonnet ($3/$15) — 100% success, fast, good enough diversity
   - **Premium**: Opus ($15/$75) with `--thinking` — best quality, reasoning traces, but 10x cost
   - **Free**: kimi-k2.5 on Ollama — works but very slow (~400s/maze)

5. **Negative regret edge case** — when a maze is too hard for the agent (0% solve rate), max_return=0 but V(s_t)>0, producing negative regret. The `min_regret: 1` threshold filters these out.

6. **`content: None` bug (fixed)** — OpenRouter thinking models (Kimi K2.5, DeepSeek Speciale, GLM-5, Nemotron) return `content: null` with reasoning in `reasoning`/`reasoning_details` fields. Initial runs showed 0% success due to silent `NoneType` crashes. Fix: `message.get("content") or ""` + reasoning→content fallback (same as Ollama thinking path). Reruns after fix showed Kimi K2.5 at 67% and DeepSeek Speciale at 33%.
