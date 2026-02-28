# Agent Verification Report

**Phase:** 01-Foundation
**Date:** 2026-02-28
**Compared against:** DCD repo (github.com/facebookresearch/dcd) arguments.py defaults

---

## PPO Hyperparameter Comparison

| Parameter | DCD default | Our default | Match? | Classification |
|-----------|-------------|-------------|--------|----------------|
| lr | 1e-4 | 1e-4 | YES | matches |
| gamma | 0.995 | 0.995 | YES | matches |
| gae_lambda | 0.95 | 0.98 | NO | potential-bug — higher lambda = lower bias/higher variance; 0.98 may reduce training stability slightly, but is a common value; warrants watching |
| clip_eps | 0.2 | 0.2 | YES | matches |
| entropy_coeff | 0.0 | 1e-3 | NO | intentional — small entropy bonus encourages exploration; common practice in maze navigation; DCD uses 0.0 which may be overly conservative |
| critic_coeff | 0.5 | 0.5 | YES | matches |
| max_grad_norm | 0.5 | 0.5 | YES | matches |
| epoch_ppo | 5 | 5 | YES | matches |
| num_minibatches | 1 | 1 | YES | matches |
| num_steps | 256 | 256 | YES | matches |
| num_train_envs | 32 | 32 | YES | matches |
| score_function | value_l1 (generic DCD) / MaxMC (DCD ACCEL config) | MaxMC | YES (vs ACCEL) | matches-accel-config — differs from DCD generic default but matches DCD ACCEL-specific config |

**Notes on parameter source locations:**
- All parameters above are defined in `examples/maze_plr.py` via `argparse` defaults (lines 1139–1206 in the codebase).
- `score_function` default is `"MaxMC"` (line 1154), confirmed by `parser.add_argument("--score_function", type=str, default="MaxMC", choices=["MaxMC", "pvl"])`.
- `gae_lambda` default: `parser.add_argument("--gae_lambda", type=float, default=0.98)` (line 1150).
- `entropy_coeff` default: `parser.add_argument("--entropy_coeff", type=float, default=1e-3)` (line 1151).

---

## PPO Algorithm Comparison

### GAE (compute_gae) — lines 69–104

**Our implementation:**
```python
def compute_gae_at_timestep(carry, x):
    gae, next_value = carry
    value, reward, done = x
    delta = reward + gamma * next_value * (1 - done) - value
    gae = delta + gamma * lambd * (1 - done) * gae
    return (gae, value), gae
```
Scan is run **in reverse** over the trajectory (`reverse=True`) with `unroll=16`.

**DCD formula (from dcd/utils/ppo.py):**
```
delta = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
gae_t = delta_t + gamma * lambda * (1 - done_t) * gae_{t+1}
```

**Assessment:** Formulas match exactly. The standard TD-lambda advantage formula is correctly implemented. The `(1 - done)` masking at episode boundaries is correct. The scan with `reverse=True` correctly computes advantages backward from terminal state.

**Differences:** None in algorithm; `gae_lambda` value differs (0.98 vs 0.95 DCD default — documented above).

---

### Update Epoch (update_actor_critic_rnn) — lines 235–317

**Our clipped surrogate objective:**
```python
ratio = jnp.exp(log_probs_pred - log_probs)
A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean()
```

**Our value loss (with clipping):**
```python
values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()
```

**Combined loss:**
```python
loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy
```

**Assessment vs DCD:**
- Clipped surrogate: matches DCD standard PPO-clip formula.
- Advantage normalization: `(A - mean) / (std + 1e-5)` — matches common PPO practice (DCD does the same).
- Value loss clipping: present — we clip value predictions within `clip_eps` of the old values, then take `max(unclipped_loss, clipped_loss)`. DCD uses the same value clipping strategy.
- Entropy bonus: `entropy_coeff=1e-3` (we) vs `0.0` (DCD default). Our implementation adds exploration incentive; DCD default does not.
- Loss formula structure: matches DCD (`l_clip + coeff*l_vf - coeff*entropy`).

**Linear learning rate schedule:** Our implementation uses a linear decay schedule:
```python
frac = 1.0 - (count // (num_minibatches * epoch_ppo)) / num_updates
lr_effective = lr * frac
```
DCD ACCEL also uses linear LR decay. Matches.

---

### Regret Computation

**Our path in `examples/maze_plr.py`:**
1. `max_returns = compute_max_returns(dones, rewards)` — computes the best return achieved across all episodes in the rollout (episodic max).
2. `scores = compute_score(config, dones, values, max_returns, advantages)` — when `score_function="MaxMC"`, calls `max_mc(dones, values, max_returns)` from `jaxued.utils`.

**jaxued `max_mc` function (jaxued.utils):**
- `max_mc` is the MaxMC (Maximum Monte Carlo) regret estimator from jaxued, which is the JAX reimplementation of DCD's ACCEL regret proxy.
- Formula: `regret = max_return - V(s_0)` where `V(s_0)` is the value estimate at the start of the episode.
- This is the standard ACCEL regret proxy, not the full PVL (Positive Value Loss) metric.

**In `es/regret_fitness.py`:**
```python
max_returns = compute_max_returns(dones, rewards)
regret = max_mc(dones, values, max_returns, incomplete_value=0.0)
```
Same jaxued utilities used consistently in both the training loop and the ES regret fitness evaluator.

**Assessment:** Regret computation matches DCD ACCEL's MaxMC approach. jaxued is a JAX reimplementation of DCD, so `max_mc` is the correct equivalent.

---

## Structural Differences

### Level Sampling / Mutation Strategy

**DCD ACCEL:**
- PLR buffer (prioritized level replay) with `replay_prob=0.8`, `staleness_coeff=0.3`
- When ACCEL branch triggered: calls `make_level_mutator_minimax(100)` — random minimax mutation (randomly flip tiles, keep walls if they increase regret)
- Mutation is entirely within original tile-space

**Our implementation:**
- PLR buffer with same hyperparameters (replay_prob=0.8, staleness_coeff=0.3)
- Three mutation modes available (selected via CLI flags):
  1. **Minimax mutation** (`default, no flag`): `make_level_mutator_minimax(100)` — same as DCD ACCEL
  2. **MAP-Elites + VAE latent mutation** (`--use_map_elites_mutation`): MAP-Elites archive in VAE latent space; parent selected from archive by fitness; child = decode(latent + Gaussian noise); archive indexed by behavior descriptors (BFS path length × obstacle count)
  3. **PLR-Weighted Latent Mutation** (`--use_plwm_mutation`): encode PLR replay parents to VAE/MazeAE latents; perturb; decode back to Level; insert into PLR buffer
- Classification: **INTENTIONAL** — MAP-Elites/PLWM mutation is the thesis contribution (ES-generated curriculum replacing minimax random mutation)

**Note:** When running with `--use_accel` alone (no `--use_map_elites_mutation` or `--use_plwm_mutation`), the code falls back to `make_level_mutator_minimax(100)`, which is identical to DCD ACCEL. The smoke test uses this baseline path.

### AutoReplayWrapper

- Both DCD and our implementation use `AutoReplayWrapper` — the agent replays the same level when an episode ends within a rollout, ensuring multiple episodes per rollout step.
- Source: `from jaxued.wrappers import AutoReplayWrapper`
- Classification: **matches** — both use jaxued wrapper.

### Network Architecture (ActorCritic)

Our ActorCritic uses:
- CNN layer: `Conv(16, kernel_size=(3,3), strides=(1,1), padding="VALID")` over image observation
- Direction embedding: `Dense(5)` applied to one-hot agent direction
- LSTM: `OptimizedLSTMCell(features=256)` wrapped in `ResetRNN` (resets hidden state on episode done)
- Actor head: `Dense(32) -> relu -> Dense(action_dim)` with orthogonal init
- Critic head: `Dense(32) -> relu -> Dense(1)` with orthogonal init

DCD ACCEL uses a similar recurrent actor-critic. The exact architecture matches the standard jaxued maze PPO configuration. Classification: **matches**.

### eval_freq and Training Loop

- Our default `eval_freq=250` (evaluates every 250 updates)
- Training loop: `for eval_step in range(num_updates // eval_freq): runner_state, metrics = train_and_eval_step(runner_state, None)`
- Inner scan: `jax.lax.scan(train_step, runner_state, None, eval_freq)` — JIT-compiled over `eval_freq` steps then logs.
- DCD structure is similar. Classification: **matches**.

---

## Summary of Differences

Flat list of all differences (one per line):

1. **gae_lambda = 0.98 (ours) vs 0.95 (DCD default)** — POTENTIAL-BUG: higher lambda reduces bias but increases variance; may slow convergence. Not a correctness error but warrants monitoring. Common alternative value used in practice.

2. **entropy_coeff = 1e-3 (ours) vs 0.0 (DCD default)** — INTENTIONAL: small entropy bonus to encourage exploration in maze navigation. Standard practice for sparse-reward environments. DCD uses 0.0 which is more conservative.

3. **score_function = MaxMC (ours) vs value_l1 / MaxMC (DCD)** — MATCHES-ACCEL-CONFIG: DCD's generic default is `value_l1`; DCD's ACCEL-specific config uses MaxMC. Our default is MaxMC, consistent with the ACCEL configuration we are building on.

4. **Level mutation: MAP-Elites + VAE latent space (ours) vs minimax random mutation (DCD ACCEL default)** — INTENTIONAL: this is the thesis contribution. Our ES/MAP-Elites curriculum replaces DCD's minimax mutation. Minimax fallback is still available (and used in smoke test baseline).

5. **PLWM (PLR-Weighted Latent Mutation) option** — INTENTIONAL: additional mutation strategy not in DCD. Requires `--use_plwm_mutation` flag.

---

## Smoke Test Results

### Command Run

```bash
cd /cs/student/project_msc/2025/csml/gmaralla/superintelligence
WANDB_MODE=offline python examples/maze_plr.py \
  --use_accel \
  --score_function MaxMC \
  --num_updates 5 \
  --num_train_envs 32 \
  --num_steps 256 \
  --seed 42 \
  --project FOUNDATION_SMOKE_TEST \
  --run_name smoke_test_accel_maxmc \
  --checkpoint_save_interval 999 \
  2>&1 | tee /tmp/smoke_test_output.txt
```

### Results

[To be filled in by Task 2 execution]
