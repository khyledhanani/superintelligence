# Trajectory Diversity Metrics

Metrics that measure behavioral diversity from the agent's perspective across maze levels. Used in the ACCEL replay buffer for level scoring and in the LLM diversity injection pipeline for generating novel mazes.

All metrics truncate trajectories at the first `done=True` flag to compare only the active episode.

## Metric Inventory

### Standalone (per-maze)

| Metric | Module | Input | Output | Measures | Status |
|--------|--------|-------|--------|----------|--------|
| Scalar Regret | `standalone/regret.py` | values, rewards, dones | `float` | Overall maze difficulty (MaxMC) | **Active gate** (difficulty_metric: regret) |
| SFL Learnability | `standalone/learnability.py` | solve_rate (multi-rollout) | `float` | Learning frontier (p×(1-p)) | Gate option (difficulty_metric: sfl) |
| CENIE Novelty | `standalone/cenie.py` | LSTM hstates, actions, dones | `float` NLL | Buffer-wide novelty (GMM density) | Gate option (diversity_metric: cenie) |
| Per-Step Regret | `standalone/per_step_regret.py` | values, rewards, dones | `(T,)` curve | Where difficulty spikes | Active (prompt) |
| Value Error Profile | `standalone/value_error.py` | values, rewards, dones | `(T,)` signed | Where & how the agent is wrong | Active (prompt, default: off) |
| Per-Step Entropy | `standalone/per_step_entropy.py` | entropy, dones | `(T,)` curve | Policy uncertainty at each step | Active (prompt) |
| Action Sequence | `standalone/per_step_action.py` | actions, dones | `(T,)` discrete | Behavioral fingerprint | Active (prompt) |

### Pairwise (maze-vs-maze)

| Metric | Module | Input | Output | Measures | Status |
|--------|--------|-------|--------|----------|--------|
| Position DTW | `pairwise/pos_dtw.py` | positions, dones | `float` distance | Spatial path similarity | Active (prompt), **not recommended for gating** |
| Regret DTW | `pairwise/regret_dtw.py` | values, rewards, dones | `float` distance | Difficulty profile similarity | Available, not in prompt |
| Action Distance | `pairwise/action_dtw_binary.py` | actions, dones | `float` distance | Behavioral divergence | Available, not in prompt |
| Mode Transition Divergence | `pairwise/mode_transition.py` | values, rewards, dones, entropy | `float` KL | Experiential diversity | New (default: off) |
| TD Error Distribution EMD | `pairwise/td_error_distribution.py` | values, rewards, dones | `float` EMD | Learning signal diversity | **Active gate** (diversity_metric: td_error_emd) |

## Detailed Descriptions

### Scalar Regret (MaxMC)
`mean_t[max_return - V(s_t)]` — the ACCEL scoring function itself.

- **Use:** Difficulty gate (default). Rejects trivially easy (regret < threshold) and unsolvable (negative regret) mazes.
- **Limitation:** Says nothing about diversity. Two mazes with regret=1.0 can be experientially identical. Only filters too-easy levels, not too-hard.
- **Config:** `gate.difficulty_threshold: 0.6`, `gate.difficulty_metric: regret`

### SFL Learnability
`p × (1-p)` where p = agent's solve rate across multiple rollouts.

- **Use:** Alternative difficulty gate. Automatically filters both too-easy (p≈1) and too-hard (p≈0) levels. Maximum at p=0.5 (learnability=0.25).
- **Intuition:** The variance of a Bernoulli(p). Greedily maximizing it is equivalent to maximizing expected improvement (Rutherford et al., 2024).
- **Limitation:** Requires multi-rollout evaluation (solve_rate). Says nothing about diversity.
- **Config:** `gate.difficulty_threshold: 0.1`, `gate.difficulty_metric: sfl`

### Per-Step Regret
`max_return - V(s_t)` as a time series — unsigned.

- **Use:** LLM prompt context. Shows where difficulty spikes (dead ends, chokepoints).
- **Limitation:** Unsigned — can't distinguish overconfidence (trap) from underconfidence (hidden shortcut). Depends on `max_return` estimate which is rollout-dependent.
- **Config:** `prompt_metrics.per_step_regret: true`

### Value Error Profile (NEW)
`V(s_t) - G_t` signed — positive = overconfident, negative = underconfident.

- **Use:** More informative replacement for per-step regret. Uses actual returns G_t instead of estimated max_return.
- **Key insight:** Sign reveals the *nature* of the error. Overconfident steps are where the agent walks into traps; underconfident steps are where it fails to recognize good positions.
- **Summary stats:** `mean_error`, `overconfident_frac`, `max_overconfident`, `max_underconfident`
- **Limitation:** G_t from a single rollout has high variance with stochastic policies.
- **Config:** `prompt_metrics.value_error: false` (not yet enabled by default)

### Per-Step Entropy
`H(π(·|s_t))` at each timestep — policy uncertainty.

- **Use:** LLM prompt context. Identifies decision points and confusing junctions.
- **Limitation:** Very sparse in practice (mean ~0.02 across our buffer). A well-trained agent is confident almost everywhere.
- **Config:** `prompt_metrics.per_step_entropy: true`

### Action Sequence
Discrete actions per step, with summary stats (dominant action, unique count).

- **Use:** Behavioral fingerprint for the LLM. Two mazes producing identical action sequences are functionally identical.
- **Limitation:** Our agent's narrow action repertoire (60% action 2 = down) limits discriminative power.
- **Config:** `prompt_metrics.action_sequence: true`

### Position DTW
DTW on start-relative (x,y) position traces — translation invariant.

- **Use:** Currently shown as pairwise metric in prompt.
- **Critical finding:** r=0.113 correlation with mode transition divergence. **Spatial diversity ≠ experiential diversity.** Different paths can produce identical learning signals.
- **Reports 0.00 in the diversity gate** in practice — not functioning as a useful gating signal.
- **Config:** `pairwise_metrics.position_dtw: true`, `gate.min_pos_dtw: null` (disabled)

### Regret DTW
DTW on per-step regret curves.

- **Use:** Available for analysis. High correlation with mode divergence (r=0.890) suggests it captures similar signal.
- **Limitation:** Same DTW alignment problems as position DTW. Redundant with mode transition divergence.

### Action Distance (Binary DTW)
DTW with binary mismatch cost on action sequences.

- **Use:** Available for analysis. Moderate correlation with mode divergence (r=0.614).
- **Limitation:** Binary cost throws away which actions differ. Narrow action repertoire limits range.

### Mode Transition Divergence (NEW)
Classifies each step into 5 experience modes, computes symmetric KL divergence between transition matrices.

**Modes:**
- `confident_correct` — low |error|, low entropy (agent knows what it's doing)
- `confident_wrong` — high |error|, low entropy (confidently mistaken — traps)
- `uncertain` — high entropy (decision point)
- `recovering` — |error| decreasing (figuring it out)
- `degrading` — |error| increasing (getting more confused)

- **Use:** Captures multi-step experience patterns (loops, traps, recovery) without temporal alignment. Human-interpretable (useful for LLM feedback prompts).
- **Thresholds are adaptive:** Computed from reference trajectories as mean + 1 std of |error| and entropy. No hardcoded magic numbers — scales automatically with agent capability and training stage.
- **Limitation:** The 5-mode taxonomy embeds assumptions about what "confident" and "recovering" mean. May not generalize to all RL domains. KL divergence is unbounded, making threshold setting non-intuitive. Not fully task-agnostic.
- **Config:** `pairwise_metrics.mode_transition: false` (not yet enabled by default)

### TD Error Distribution EMD
Earth Mover's Distance (Wasserstein-1) between the distributions of TD errors `δ_t = r_t + γV(s_{t+1}) - V(s_t)`.

- **Use:** Default pairwise diversity gate. Uses only values, rewards, and dones — any actor-critic has these. TD error is the raw learning signal; different distributions mean different gradient directions (in aggregate).
- **Key property:** No thresholds, no mode taxonomy, no architecture knowledge. Purely distributional comparison.
- **EMD interpretation:** The "cost" to reshape one histogram into the other. Bounded, interpretable, well-defined even for very different episode lengths.
- **Gate behavior:** Uses min-distance to closest reference (not mean). A candidate must be dissimilar from *every* reference to pass.
- **Key correlations:** r=0.301 with position DTW, r=0.687 with action DTW, r=0.614 with mode divergence, r=0.659 with CENIE novelty.
- **Limitation:** Collapses temporal structure entirely — a maze where the agent is confused early then recovers vs one that starts confident then gets lost would score identically if the overall δ distributions match. Typical EMD values are small (mean=0.023) so threshold sensitivity needs care.
- **Config:** `gate.diversity_metric: td_error_emd`, `gate.min_diversity: 0.02`

### CENIE Novelty
Negative mean log-likelihood of LSTM hidden state + action pairs under a GMM fitted on the training buffer.

Based on "Improving Environment Novelty Quantification for Effective Unsupervised Environment Design" (Teoh, Li, Varakantham, NeurIPS 2024).

- **Use:** Alternative diversity gate. Scores novelty against the *entire buffer* (via density model), not just selected references. Catches redundancy that pairwise metrics miss.
- **Key property:** Not pairwise — compares candidate against a GMM fitted on 50 buffer level trajectories. No blind spots from reference subsampling.
- **Features:** LSTM hidden state (256D, the agent's belief representation) + action (1D) = 257D per timestep. Per the paper, MiniGrid uses LSTM hidden states because of partial observability.
- **GMM fitting:** Diagonal covariance, silhouette-based K selection (2-10 components), fitted on concatenated state-action pairs from buffer trajectories.
- **Correlation with TD Error EMD:** Pearson r=0.659 (moderate), Spearman r=0.353 (weak rank). They share signal but disagree on edge cases — CENIE captures coverage gaps that pairwise EMD can't see.
- **Limitation:** Threshold is in NLL scale (typical range: -250 to -110 for buffer levels), not intuitive. Requires LSTM hidden states from agent rollouts (architecture-coupled). Not suitable for pairwise embedding — use TD Error EMD for visualization.
- **Config:** `gate.diversity_metric: cenie`, `gate.min_diversity: -200`

## Correlation Structure (16 buffer levels, 120 pairs, adaptive mode thresholds)

```
              Pos DTW   Action DTW   Regret DTW   Mode Div   Value Err   TD EMD
Pos DTW        1.000      0.333        0.163       0.113       —         0.301
Action DTW     0.333      1.000        0.699       0.614       —         0.687
Regret DTW     0.163      0.699        1.000       0.890       —         0.290
Mode Div       0.113      0.614        0.890       1.000      -0.078     0.614
Value Error      —          —            —        -0.078      1.000      0.310
TD EMD         0.301      0.687        0.290       0.614      0.310      1.000
```

**Key takeaways:**
- **Position DTW is weakly correlated with all experiential metrics** (r=0.11-0.30). Spatial path diversity ≠ learning diversity.
- **TD error EMD is partially independent from everything.** Its strongest correlation is with action DTW (r=0.687) — different actions tend to produce different TD errors. But it's only r=0.614 with mode divergence, meaning they capture distinct aspects of experience.
- **Mode divergence and value error are now nearly uncorrelated** (r=-0.078) with adaptive thresholds. The old r=0.885 was an artifact of fixed thresholds that artificially aligned the metrics.
- **No single metric captures everything.** The ideal gate would combine scalar regret (difficulty floor) with TD error EMD (learning signal diversity) — two orthogonal, task-agnostic signals.

## Recommendations

| Purpose | Metric | Why |
|---------|--------|-----|
| Gate: reject easy mazes | Scalar regret (difficulty_metric: regret) | Direct, proven |
| Gate: reject easy AND hard mazes | SFL learnability (difficulty_metric: sfl) | Filters both extremes, targets learning frontier |
| Gate: reject redundant mazes (vs refs) | TD error EMD (min-distance) | Task-agnostic, no thresholds/modes, captures learning signal diversity |
| Gate: reject redundant mazes (vs buffer) | CENIE novelty | Scores against full buffer via GMM density, no reference subsampling blind spots |
| LLM prompt: agent behavior | Path overlay + entropy + action sequence | Rich context for maze design |
| LLM prompt: difficulty | Value error profile or per-step regret | Shows where agent struggles |
| LLM feedback: diversity | Mode fractions + transition matrix | Human-interpretable "your maze creates the same confusion pattern as Maze A" |
| Analysis/visualization | All metrics | Different metrics reveal different aspects; correlation structure is informative |
| Drop from gating | Position DTW | r=0.11-0.30 with experiential metrics — wrong signal |

## Source

- Standalone metrics: `metrics/standalone/`
- Pairwise metrics: `metrics/pairwise/`
- Shared DTW algorithm: `metrics/dtw.py`
- Shared utilities: `metrics/utils.py`
- Configuration: `llm/config.yaml` (`prompt_metrics`, `pairwise_metrics`, `gate` sections)
- Visualization: `metrics/scripts/plot_metrics_demo.py` → `metrics/plots/`
