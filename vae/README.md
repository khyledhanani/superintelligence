# VAE (Variational Autoencoder) for Maze Levels

A VAE that encodes maze levels into a low-dimensional latent space. Supports both **13x13** (64D latent, vocab 170, seq_len 52) and **21x21** (96D latent, vocab 442, seq_len 152) grids. Used for:

1. **Mutation strategies** -- Generate level variants by adding noise or interpolating in latent space (used in the LLM injection pipeline)
2. **Analysis & visualization** -- Compare LLM-injected vs organic ACCEL level distributions across training

## Architecture

**CluttrVAE** (defined in `vae_model.py`):
- **Encoder**: Embedding -> Highway stages (x2) -> Bidirectional LSTM -> mean/logvar projection -> latent
- **Decoder**: Bidirectional LSTM (2 stacks) -> Dense output -> tokens
- **Dropout**: 0.1

| Parameter | 13x13 | 21x21 |
|-----------|-------|-------|
| `vocab_size` | 170 | 442 |
| `seq_len` | 52 | 152 |
| `latent_dim` | 64 | 96 |
| `embed_dim` | 300 | 300 |
| `enc_lstm_dim` | 300 | 400 |
| `dec_lstm_dim` | 400 | 512 |

VAE checkpoints: `/cs/student/project_msc/2025/csml/rhautier/vae_checkpoints/{13x13,21x21}/`

### Token format (seq_len tokens per level)

| Tokens | Content |
|--------|---------|
| 0 to max_walls-1 | Wall cell indices (1-based, sorted, 0-padded) |
| -2 | Goal position (1-based) |
| -1 | Agent position (1-based) |

For 13x13: max_walls=50, seq_len=52. For 21x21: max_walls=150, seq_len=152.

### Key functions (`vae_level_utils.py`)

All functions are parameterized by `grid_size`, `vocab_size`, `max_walls` with 13x13 defaults for backward compatibility:

```python
from vae_level_utils import level_to_tokens, tokens_to_level, decode_latent_to_levels

# 13x13 (defaults)
level_to_tokens(level)                                          # Level -> (52,) int32
tokens_to_level(tokens)                                         # (52,) int32 -> Level
decode_latent_to_levels(decode_fn, z_batch, rng)                # (N, 64) -> batched Level

# 21x21 (pass grid params explicitly)
level_to_tokens(level, grid_size=21, max_walls=150)             # Level -> (152,) int32
tokens_to_level(tokens, grid_size=21)                           # (152,) int32 -> Level
decode_latent_to_levels(decode_fn, z_batch, rng, grid_size=21, vocab_size=442)
```

---

## Behavioral embedding analysis during LLM injection training

The main analysis tracks how the PLR buffer's behavioral landscape evolves during training when LLM-generated levels are injected at various percentages (5%, 10%, 15%, 20%, 25%).

### What are the 257D behavioral embeddings?

Each level is characterized by **how the current agent behaves on it**, not by its structure. The embedding is computed by rolling out the agent on the level and summarizing its trajectory:

```python
# From maze_plr.py:compute_insertion_embeddings()
# 256D: mean LSTM hidden state across first-episode timesteps
# 1D:   mean action across first-episode timesteps
# Total: 257D per level
```

Two levels with similar embeddings means the agent exhibits similar internal representations and action patterns on both. This captures behavioral similarity that structural features (wall layout) cannot.

### The core question

When LLM-generated levels (and their wall-flip mutations) are injected into the ACCEL buffer, do they:
- Occupy a distinct behavioral region from organic ACCEL levels?
- Get absorbed into the existing distribution as the agent trains?
- Get evicted from the buffer by PLR's scoring mechanism?
- Shift the overall behavioral landscape?

### Data on GCS

All training data is stored on `gs://ucl-ued-project-bucket`:

| Data | GCS path |
|------|----------|
| Agent checkpoints (every 250 updates) | `llm-exp/training/inject_llm_{pct}pct_seed{s}/checkpoints/inject_llm_{pct}pct_seed{s}/{s}/models/{step}/` |
| Agent config | `llm-exp/training/inject_llm_{pct}pct_seed{s}/checkpoints/inject_llm_{pct}pct_seed{s}/{s}/config.json` |
| Buffer dumps (every 250 updates) | `llm-exp/training/inject_llm_{pct}pct_seed{s}/buffer_dumps/inject_llm_{pct}pct_seed{s}/{s}/buffer_dump_{N}.npz` |
| Warmstart agent (pre-injection) | `llm-exp/checkpoints/accel_sfl_baseline_13x13/0` |
| Mutation experiment outputs | `llm-exp/injection/` |
| Embedding cache (pre-computed) | `llm-exp/embedding_caches/tsne_training_cache/` |

**Available runs**: 3 seeds x 5 injection percentages = **15 training runs**
- `inject_llm_{5,10,15,20,25}pct_seed{0,1,2}`

Each run trains for 10k updates from the warmstart checkpoint, with buffer dumps and agent checkpoints saved every 250 updates.

**Checkpoint step mapping**: `step = (updates / 250) - 1` (e.g., 250 updates -> step 0, 1000 -> step 3, 10000 -> step 39).

### Buffer dump format

Each `buffer_dump_{N}.npz` contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `tokens` | (size, 52) | Token encoding of each level |
| `scores` | (size,) | PLR learnability scores |
| `origins` | (size,) | **0** = organic ACCEL, **1** = LLM original seed, **2** = LLM mutation descendant |
| `origin_ids` | (size,) | Deterministic hash per injected level |
| `ancestor_ids` | (size,) | Which LLM seed each descendant traces back to |
| `embeddings` | (size, 257) | Stale embeddings (computed at insertion time, not current agent) |
| `size` | scalar | Number of active levels in buffer |
| `update_num` | scalar | Training update count |

**Important**: The `embeddings` stored in buffer dumps are **stale** — they reflect the agent's behavior at the time each level was inserted, not the current agent. For accurate analysis, embeddings must be recomputed using the current agent checkpoint.

---

### t-SNE of buffer evolution during training (recommended analysis)

**Script**: `plot_tsne_training_evolution.py`

**What it produces**: A grid of t-SNE plots showing the buffer's behavioral landscape at each training timestep, with levels colored by provenance (organic vs LLM-injected).

**Methodology**:
1. For each (seed, timestep), download the **agent checkpoint** and **buffer dump** from GCS
2. Roll out the **current agent** (at that training step) on all ~4000 buffer levels
3. Average over **5 rollouts** per level for stable embeddings (action sampling is stochastic)
4. Compute fresh 257D behavioral embeddings
5. Fit **t-SNE independently per cell** — each cell has its own embedding distribution since the agent changes across timesteps
6. Plot colored by provenance

**Layout**: rows = seeds (0, 1, 2), columns = training timesteps

**Colors**: light grey (transparent) = organic ACCEL, green = LLM mutation descendants, blue stars = LLM original seeds

**Reproduce**:
```bash
cd /cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued

# Full run: all 5 pcts, 3 seeds, 13 timesteps (requires GPU for embedding computation)
python vae/plot_tsne_training_evolution.py \
    --cache_dir vae/plots/tsne_training_cache

# Single injection percentage
python vae/plot_tsne_training_evolution.py \
    --inject_pct 10pct \
    --cache_dir vae/plots/tsne_training_cache

# Plot from pre-computed embeddings only (CPU, no GCS needed)
python vae/plot_tsne_training_evolution.py \
    --cache_dir vae/plots/tsne_training_cache \
    --cache_only

# Subset of timesteps for faster iteration
python vae/plot_tsne_training_evolution.py \
    --cache_dir vae/plots/tsne_training_cache \
    --timesteps 250,1000,3000,5000,7000,10000

# Specific seeds only
python vae/plot_tsne_training_evolution.py \
    --seeds 0,1 \
    --cache_dir vae/plots/tsne_training_cache \
    --cache_only
```

**Output**: `vae/plots/tsne_training_evolution/tsne_evolution_{5,10,15,20,25}pct.png`

**Embedding cache**: Each (pct, seed, timestep) embedding is cached as `vae/plots/tsne_training_cache/emb_{pct}_s{seed}_t{timestep}.npz` after first computation. This avoids re-downloading checkpoints and re-rolling out levels on subsequent runs. The cache can also be pulled from GCS at `llm-exp/embedding_caches/tsne_training_cache/`.

---

### Other embedding analyses (reference)

These earlier analyses use either the warmstart agent or stale insertion-time embeddings. They are useful for understanding the initial buffer composition but do not reflect the evolving agent's perspective.

#### Pre-training buffer embeddings (`plot_embedding_evolution.py`)

Visualizes the merged buffer **before** training starts, using the warmstart agent. Shows where injected levels sit in the behavioral space relative to organic levels.

- `plots/embedding_evolution/` — unbalanced ancestor selection (few LLM ancestors dominate)
- `plots/embedding_evolution_balanced/` — balanced round-robin selection (equal contribution per ancestor)

#### Stale embedding training snapshots (`plot_embedding_training.py`)

Uses the pre-computed `embeddings` field from buffer dumps (computed at insertion time). Faster but less accurate — embeddings become stale as the agent evolves.

#### Structural projections (`plot_env_space.py`)

t-SNE of levels in structural feature space (169D wall map + positions). Agent-independent — purely based on maze layout.

#### Buffer embedding visualization (`tsne_buffer_embeddings.py`)

Standalone t-SNE/MDS of a single buffer snapshot with reference mazes and accepted LLM levels highlighted.

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
scikit-learn           # PCA, t-SNE, MDS, pairwise_distances
matplotlib             # All plotting
pyyaml                 # VAE config loading
google-cloud-storage   # GCS data fetching (for plot_tsne_training_evolution.py)
```
