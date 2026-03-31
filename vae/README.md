# VAE (Variational Autoencoder) for Maze Levels

A VAE that encodes 13x13 maze levels into a 64-dimensional latent space. Used for:

1. **Mutation strategies** -- Generate level variants by adding noise or interpolating in latent space (used in the LLM injection pipeline)
2. **Analysis & visualization** -- Compare LLM-injected vs organic ACCEL level distributions across training

## Architecture

**CluttrVAE** (defined in `vae_model.py`):
- **Encoder**: Embedding (300d) -> Highway stages (x2) -> Bidirectional LSTM -> mean/logvar projection -> **64d latent**
- **Decoder**: Bidirectional LSTM (2 stacks) -> Dense output -> 52 tokens
- **Vocab size**: 170, **Dropout**: 0.1

### Token format (52 tokens per level)

| Tokens | Content |
|--------|---------|
| 0-49 | Wall cell indices (1-based, sorted, 0-padded) |
| 50 | Goal position (1-based) |
| 51 | Agent position (1-based) |

### Key functions (`vae_level_utils.py`)

```python
from vae_level_utils import level_to_tokens, tokens_to_level, decode_latent_to_levels

level_to_tokens(level)                        # Level -> (52,) int32
tokens_to_level(tokens)                       # (52,) int32 -> Level
decode_latent_to_levels(z, vae, params)       # (N, 64) -> List[Level]
repair_tokens(tokens)                         # Fix invalid token sequences
```

---

## LLM injection plots -- complete reproduction guide

All LLM-related plots live under `vae/plots/`. Each subdirectory has its own README with additional details.

### Required data artifacts

| Artifact | Location |
|----------|----------|
| **Agent checkpoint (10k ACCEL+SFL baseline)** | `gcs_artifacts/agent/39` |
| **Original ACCEL buffer** | `gcs_artifacts/buffer/buffer_dump_final.npz` |
| **Merged buffers (pre-training)** | `/cs/student/project_msc/2025/csml/rhautier/injection_data/results/llm_inject_seed{0,1,2}/merged_buffer_{5,10,15,20,25}pct.npz` |
| **Training buffer dumps** | `/cs/student/project_msc/2025/csml/rhautier/injection_data/results/llm_inject_seed{0,1,2}/training_{5,10,15,20,25}pct/buffer_dumps/buffer_dump_{N}.npz` |
| **Eligible pools** | `/cs/student/project_msc/2025/csml/rhautier/injection_data/results/llm_inject_seed{0,1,2}/eligible_pool.npz` |
| **LLM seeds** | `/cs/student/project_msc/2025/csml/rhautier/injection_data/seeds/seeds.npz` |

### GCS bucket: `gs://ucl-ued-project-bucket`

| GCS path | What |
|----------|------|
| `llm-exp/checkpoints/accel_sfl_baseline_13x13/0` | ACCEL baseline checkpoint (seed 0) |
| `llm-exp/training/inject_llm_{5,10,15,20,25}pct_seed{0,1,2}/` | LLM injection training runs (checkpoints + buffer dumps) |
| `llm-exp/training/inject_wallflip_e5_20pct_seed{0,1}/` | Wall-flip baseline training runs |
| `llm-exp/injection/` | Mutation experiment outputs (merged buffers, eligible pools) |

---

### 1. `plots/embedding_evolution/` -- Initial buffer embeddings (unbalanced ancestors)

**Script**: `plot_embedding_evolution.py`

**What it shows**: PCA-2D and t-SNE-2D of the **pre-training merged buffers** (4000 levels each), showing where LLM-lineage levels sit relative to organic ACCEL levels.

**Methodology**:
1. For each (seed, pct) pair, load the merged buffer (`tokens` + `origins`)
2. Roll out the **initial 10k agent checkpoint** on every level via `sample_trajectories_rnn()`
3. Compute **257D embeddings** via `compute_insertion_embeddings()`:
   - **256D**: mean of LSTM hidden states across non-done timesteps
   - **1D**: mean action across non-done timesteps
4. **PCA**: fit globally across all 15 cells (3 seeds x 5 pcts) for consistent axes
5. **t-SNE**: fit independently per cell (perplexity=40, PCA-initialized)

**Colors**: light grey = organic ACCEL (origin=0), green = LLM mutations (origin=2), blue stars = LLM originals (origin=1)

**Layout**: 3 rows (seed 0/1/2) x 5 columns (5%/10%/15%/20%/25%)

**Note**: Uses **original unbalanced** merged buffers (mutations ranked purely by SFL, so a few LLM ancestors dominate).

**Reproduce**:
```bash
python vae/plot_embedding_evolution.py \
    --agent_checkpoint gcs_artifacts/agent/39 \
    --output_dir vae/plots/embedding_evolution \
    --cache_dir vae/plots/embedding_cache
```

---

### 2. `plots/embedding_evolution_balanced/` -- Initial buffer embeddings (balanced ancestors)

**Script**: `plot_embedding_evolution.py` (same script, different data)

**Identical methodology** but uses **balanced round-robin** merged buffers where each LLM seed ancestor contributes ~equal mutations (vs the unbalanced version where ancestor 6 dominates).

**Reproduce**:
```bash
python vae/plot_embedding_evolution.py \
    --agent_checkpoint gcs_artifacts/agent/39 \
    --output_dir vae/plots/embedding_evolution_balanced \
    --cache_dir vae/plots/embedding_cache_balanced
```

---

### 3. `plots/accel_vs_LLM_LSTM_embedding/` -- Buffer embeddings during training

**Script**: `plot_embedding_training.py`

**What it shows**: How LLM-lineage levels evolve within the PLR buffer during training. One grid per update step (250 through 10000).

**Methodology**:
1. Load buffer dumps from training runs (saved every 250 updates by `maze_plr.py --buffer_dump_interval 250`)
2. Each dump already contains **pre-computed 257D embeddings** -- computed by the **evolving agent** at that training step (NOT the initial checkpoint)
3. **Important**: The embedding space itself shifts over time as the agent learns
4. PCA fit **globally** across ALL update steps for consistent axes
5. t-SNE fit independently per cell

**Buffer dump `.npz` keys**: `tokens`, `scores`, `embeddings` (257D), `origins` (0/1/2), `origin_ids`, `ancestor_ids`, `size`, `update_num`

**Colors**: light blue = organic ACCEL (origin=0), green = LLM mutation (origin=2), red stars = LLM originals (origin=1)

**Output**: `grid_pca_u{N}.png`, `grid_tsne_u{N}.png` for each update step

**Reproduce**:
```bash
python vae/plot_embedding_training.py \
    --method both \
    --updates 250,500,750,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000 \
    --output_dir vae/plots/accel_vs_LLM_LSTM_embedding
```

---

### 4. `plots/accel_vs_LLM_env_embeddings/` -- Structural (non-behavioral) projections

**Script**: `plot_env_space.py`

**What it shows**: t-SNE-2D of buffer levels in **environment structure space** -- independent of agent behavior. Answers: "Are LLM-origin levels structurally different from ACCEL-organic levels (wall layout, positions)?"

**Methodology**:
1. Convert each level's 52-token encoding to a **173D structural feature vector**:
   - Tokens [0:50] wall indices -> **169D binary flat wall map** (13x13 grid)
   - Token [-2] goal position -> **2D normalized (x/12, y/12)**
   - Token [-1] agent position -> **2D normalized (x/12, y/12)**
2. **No agent rollout needed** -- purely structural
3. PCA fit globally; t-SNE fit independently per cell (perplexity=40)

**Output**: `grid_env_tsne_u{N}.png`, `grid_env_pca_u{N}.png`

**Reproduce**:
```bash
# Training snapshots
python vae/plot_env_space.py --source training --method tsne \
    --updates 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000 \
    --output_dir vae/plots/accel_vs_LLM_env_embeddings

# Initial buffers only
python vae/plot_env_space.py --source initial --method both \
    --output_dir vae/plots/accel_vs_LLM_env_embeddings
```

---

### 5. Standalone t-SNE / MDS injection visualization

**Script**: `tsne_buffer_embeddings.py`

**Output**: `vae/tsne_injection.png`, `vae/tsne_injection_mds.png`

**What it shows**: Buffer embeddings colored by SFL score (YlOrRd colormap), with reference mazes (blue circles, labeled R0-R4) and accepted LLM mazes (green stars). Dashed lines connect each accepted maze to its nearest reference in embedding space.

**Methodology**:
1. Load buffer with pre-computed `mean_embeddings` (257D)
2. If `--seeds`/`--eligible` provided: roll out agent to compute their embeddings
3. If no seeds provided: select k-medoids from buffer as references (PAM: greedy BUILD + SWAP)
4. Compute full pairwise L2 distance matrix
5. **t-SNE**: precomputed distance matrix, perplexity=30
6. **MDS**: precomputed dissimilarity matrix, normalized stress

**Reproduce**:
```bash
# Buffer-only
python vae/tsne_buffer_embeddings.py \
    --buffer /path/to/buffer_dump_emb_final.npz

# With seeds + eligible mazes
python vae/tsne_buffer_embeddings.py \
    --buffer /path/to/buffer_dump_emb_final.npz \
    --seeds /path/to/seeds.npz \
    --eligible /path/to/eligible_pool.npz \
    --agent_checkpoint gcs_artifacts/agent/39 \
    --n_refs 5 --n_buffer 500 --perplexity 30
```

---

## Embedding computation details

The 257D embedding is the core representation used across all LLM injection analysis. It captures **how the agent behaves** on a given maze:

```python
# From maze_plr.py:compute_insertion_embeddings()
def compute_insertion_embeddings(hstates, actions, dones):
    # hstates: (n_levels, max_steps, 256)  -- LSTM hidden states
    # actions: (n_levels, max_steps)
    # dones:   (n_levels, max_steps)
    
    # Mask out post-done timesteps
    mask = 1.0 - cumulative_done  # (n_levels, max_steps)
    
    # Mean LSTM hidden state (256D)
    mean_h = (hstates * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
    
    # Mean action (1D)
    mean_a = (actions * mask).sum(axis=1) / mask.sum(axis=1)
    
    return concat([mean_h, mean_a[:, None]], axis=-1)  # (n_levels, 257)
```

Two levels with similar embeddings means the agent exhibits similar behavior (hidden state trajectories + action distributions) on both.

---

## Usage in injection experiments

The VAE provides two mutation strategies in `experiments/mutation_strategies.py`:

| Strategy | How it works | GPU script |
|----------|-------------|------------|
| `vae_noise` | Encode seed -> add Gaussian noise in latent space -> decode | `experiments/gpu_scripts/run_vae_noise.sh` |
| `vae_interpolation` | Encode seed pairs -> interpolate latent vectors -> decode | `experiments/gpu_scripts/run_vae_interp.sh` |

Both require a trained VAE checkpoint (stored in `vae/runs/`).

## Dependencies

Beyond core jaxued:
```
scikit-learn    # PCA, t-SNE, MDS, pairwise_distances
matplotlib      # All plotting
pyyaml          # VAE config loading
```
