# ES Environment Generation Pipeline

Evolves diverse CLUTTR maze environments in a 64-dim VAE latent space, scored by **MaxMC regret** from a frozen ACCEL agent.   search strategy is **MAP-Elites**, maintaining a grid of elite environments binned by behavior descriptors, guaranteeing structural diversity by construction.

##  Start

```bash
# Full MAP-Elites run (2000 iterations, ~1 min on GPU)
cd es/
python map_elites.py \
  --agent_checkpoint_dir agent_folder \
  --num_iterations 2000 \
  --batch_size 32 \
  --output_subdir _me_run1

#  test
python map_elites.py \
  --agent_checkpoint_dir agent_folder \
  --num_iterations 100 \
  --batch_size 16 \
  --init_pop 64 \
  --output_subdir _me_smoke
```

**Requirements**: Conda env with `jax`, `flax`, `jaxued`, `distrax`, `matplotlib`, `pyyaml`. Agent checkpoint at `agent_folder/agent_params.pkl`. VAE checkpoint at `../vae/model/checkpoint_420000.pkl`.

---

## Architecture

```
                    MAP-Elites Loop
                    ┌─────────────┐
                    │ Sample parent│
                    │ from archive │
                    └──────┬──────┘
                           │ z_parent
                           ▼
                    ┌─────────────┐
                    │   Mutate    │  z_child = z_parent + σ·N(0,I)
                    └──────┬──────┘
                           │ z_child (64-dim)
                           ▼
              ┌────────────────────────┐
              │  VAE Decoder           │  vae_decoder.py
              │  Gumbel-max sampling   │  temperature-controlled
              └────────────┬───────────┘
                           │ raw sequence (52)
                           ▼
              ┌────────────────────────┐
              │  Repair                │  vae_decoder.py
              │  Enforce CLUTTR rules  │  valid obstacle/agent/goal
              └────────────┬───────────┘
                           │ valid sequence (52)
                           ▼
              ┌────────────────────────┐
              │  Regret Fitness        │  regret_fitness.py
              │  CLUTTR → Maze Level   │  env_bridge.py
              │  Solvability check     │  flood_fill
              │  Complexity filter     │  min_obstacles, min_distance
              │  Agent rollout         │  agent_loader.py
              │  MaxMC regret          │  max_return - V(s_t)
              └────────────┬───────────┘
                           │ regret score + behavior descriptors
                           ▼
              ┌────────────────────────┐
              │  Archive Insert        │  map_elites.py
              │  If cell empty OR      │
              │  regret > existing     │
              │  → store in grid       │
              └────────────────────────┘
```

**Behavior descriptors** (2D grid, 8×6 = 48 cells):
- Axis 1: **num_obstacles** — bins: [5-10), [10-15), ..., [40-50]
- Axis 2: **manhattan_distance** (agent↔goal) — bins: [3-6), [6-9), ..., [18-24]

Each cell stores the highest-regret environment with those characteristics.

---

## MAP-Elites (Primary)

Based on Mouret & Clune (2015), "Illuminating search spaces by mapping elites."

### Algorithm

1. **Initialize**: Generate `init_pop` random latent vectors → decode → evaluate → insert valid ones into archive
2. **Loop** (each iteration):
   - Sample `batch_size` parents uniformly from occupied archive cells
   - Mutate: `z_child = z_parent + sigma * N(0, I)` in 64-dim latent space
   - Decode → repair → evaluate regret + compute behavior descriptors
   - Insert into archive if cell is empty or new regret > existing

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--agent_checkpoint_dir` | (required) | Path to agent checkpoint dir |
| `--num_iterations` | 2000 | Main loop iterations |
| `--batch_size` | 32 | Children per iteration |
| `--mutation_sigma` | 0.5 | Gaussian mutation std in latent space |
| `--decode_temperature` | 0.25 | Gumbel-max sampling temperature |
| `--rollout_steps` | 256 | Agent rollout length per environment |
| `--init_pop` | 256 | Random seeds for archive initialization |
| `--min_obstacles` | 5 | Complexity filter: min non-zero obstacles |
| `--min_distance` | 3 | Complexity filter: min agent-goal Manhattan dist |
| `--seed` | 42 | Random seed |
| `--log_freq` | 100 | Print progress every N iterations |
| `--output_subdir` | `_map_elites_run1` | Output directory name |

### Outputs

Each run writes to `es/<output_subdir>/`:

| File | Shape | Description |
|------|-------|-------------|
| `archive_envs.npy` | (N, 52) | CLUTTR sequences from filled cells |
| `archive_latents.npy` | (N, 64) | Latent vectors |
| `archive_fitness.npy` | (N,) | Regret values |
| `archive_descriptors.npy` | (N, 2) | [num_obstacles, manhattan_dist] per cell |
| `archive_heatmap.png` | — | 2D regret heatmap across behavior grid |
| `archive_gallery.png` | — | Top-12 environments visualized as grids |

### Results (2000 iterations, frozen ACCEL agent)

- **48/48 cells filled** (100% coverage)
- Regret range: 0.196 – 0.262 (mean 0.237)
- Obstacle counts: 5 – 48 across bins
- Manhattan distances: 4 – 19 across bins
- Visually diverse environments (sparse ↔ dense, short ↔ long paths)

---

## CMA-ES (Old... don't bother)

`evolve_envs.py` implements CMA-ES in the same latent space. Kept for reference **not recommended for diversity**: CMA-ES converges to a single max after ~200 generations (all environments become the same pattern).

```bash
python evolve_envs.py \
  --fitness_mode regret \
  --agent_checkpoint_dir agent_folder \
  --num_generations 200 \
  --pop_size 32 \
  --decode_temperature 0.25 \
  --output_subdir _cma_run
```

See `evolve_config.yml` for full configuration options.

---

## Shared Components

### regret_fitness.py
- `regret_fitness(rng, sequences, agent_params, network, env, env_params, ...)` → (fitness, info)
- `compute_complexity_mask(sequences, min_obstacles, min_distance)` → boolean mask
- MaxMC regret: time-averaged `max_return - V(s_t)` from agent rollouts
- Invalid environments (unsolvable or too simple) get adaptive penalty

### vae_decoder.py
- `decode_latent_to_env(decoder_params, z, rng_key, temperature)` → (batch, 52) sequences
- `repair_cluttr_sequence(seq)` → enforces CLUTTR structural constraints
- Temperature-controlled Gumbel-max sampling: `argmax(logits/T + gumbel_noise)`

### agent_loader.py
- `load_agent(checkpoint_dir, action_dim)` → (params, network)
- Pickle-first loading (portable), orbax fallback
- `verify_agent_contract(params, network)` — sanity check obs(5,5,3) → 7 actions

### env_bridge.py
- `cluttr_sequence_to_level(sequence, dir_key)` → jaxued Maze Level
- `flood_fill_solvable(wall_map, agent_pos, goal_pos)` → bool

### visualize_envs.py
- `sequence_to_grid(seq)` → 13×13 grid with walls/agent/goal
- `visualize_grid(ax, grid, title)` — matplotlib rendering
- `compute_stats(seq)` → {num_obstacles, manhattan_dist, valid}

### metrics.py
- `compute_latent_diversity(latents)` — mean pairwise L2
- `compute_sequence_diversity(sequences)` — mean pairwise Hamming

---



---

## Key Findings

1. **CMA-ES diversity collapse**: After ~200 generations, CMA-ES converges to one environment type. All environments share the same structure (agent top-left, goal middle-right, dist=18). Sigma drops, sequence diversity collapses.

2. **MAP-Elites solves this**: 100% archive coverage across all 48 behavior cells. Environments range from sparse (5 obstacles, short paths) to dense (48 obstacles, long paths).

3. **Regret ceiling with frozen agent**: Max regret plateaus at ~0.26. This is not a bug — regret measures how wrong the frozen agent's value predictions are. The agent can only be "so confused." For open-ended improvement, the agent must be in the training loop (co-evolution).

4. **Temperature matters**: T=0.25 produces the most structured environments. T=1.0 is too noisy, T=0.5 is a reasonable middle ground.

---

## Requirements

### Conda Environment

```
jax, jaxlib (CPU or CUDA)
flax, optax
jaxued
distrax
evosax (CMA-ES only)
matplotlib, pyyaml, numpy
```

### Agent Checkpoint

Place `agent_params.pkl` in `agent_folder/`. 

### VAE Checkpoint

Trained VAE at `../vae/model/checkpoint_420000.pkl` 

### GPU Notes

- **cream** (GLIBC 2.34, Quadro RTX 6000): works with JAX CUDA
- **blaze** (CentOS 7, GLIBC 2.17): cuDNN 9 incompatible — use CPU only or switch machine


