
## Files

| File | Purpose |
|------|---------|
| `vae_decoder.py` | Standalone VAE decoder extraction and repair utilities |
| `evolve_envs.py` | Main CMA-ES evolution script |
| `evolve_config.yml` | Evolution hyperparameters (pop_size, generations, fitness weights) |
| `visualize_envs.py` | Visualization tool for environments |
| `test_evolution.sh` | End-to-end test script |

## Quick Start

### 1. Install Dependencies

The code requires `evosax` (installed in the `jax_env` conda environment):

```bash
/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/pip install evosax
```

**Note**: evosax 0.2.0 will upgrade JAX to 0.6.2, which breaks CUDA. Pin JAX back:
```bash
/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/pip install jax==0.5.3 jaxlib==0.5.3
```

### 2. Run Evolution

Basic usage (200 generations, pop_size=32):
```bash
cd /cs/student/msc/csml/2025/gmaralla/superintelligence/vae
/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python evolve_envs.py
```

Custom settings:
```bash
python evolve_envs.py --num_generations 100 --pop_size 64 --seed 999
```

Outputs saved to `evolved/`:
- `evolved_envs.npy` — final population of environments (pop_size, 52)
- `best_env.npy` — best environment found (1, 52)
- `fitness_history.npy` — fitness over generations
- `best_latent.npy` — best latent vector (64,)

### 3. Visualize Results

Compare random vs evolved:
```bash
python visualize_envs.py --compare --num_envs 8 --output evolved/comparison.png
```

View only evolved:
```bash
python visualize_envs.py --evolved evolved/evolved_envs.npy --num_envs 8
```

View only random:
```bash
python visualize_envs.py --random --num_envs 8
```

### 4. Run Full Test Pipeline

```bash
bash test_evolution.sh
```

This runs evolution + visualization + statistics in one go.

## Technical Details

### VAE Decoder Extraction

The full `CluttrVAE` in `train_vae.py` has encoder+decoder in one module. We extract just the decoder:

**Full VAE param keys:**
- Encoder: `Embed_0`, `HighwayStage_0/1`, `LSTMCell_0/1` (300 hidden), `Dense_0` (600→128)
- Decoder: `LSTMCell_2/3` (BiLSTM 1, 400 hidden), `LSTMCell_4/5` (BiLSTM 2, 400 hidden), `Dense_1` (800→170)

**Standalone decoder remapping:**
```
LSTMCell_2 -> LSTMCell_0
LSTMCell_3 -> LSTMCell_1
LSTMCell_4 -> LSTMCell_2
LSTMCell_5 -> LSTMCell_3
Dense_1    -> Dense_0
```

### CMA-ES API (evosax v0.2.0)

```python
from evosax.algorithms import CMA_ES

es = CMA_ES(population_size=32, solution=jnp.zeros(64))
params = es.default_params
state = es.init(key, init_mean, params)

# Evolution loop
for gen in range(num_generations):
    population, state = es.ask(key_ask, state, params)
    fitness = evaluate(population)
    state, metrics = es.tell(key_tell, population, fitness, state, params)
```

**Note**: `tell()` requires a key as the first argument (unlike older evosax versions).

### Fitness Function (Placeholder)

Current fitness is a **structural complexity** heuristic:
- **Obstacle density**: (# obstacles) / 50
- **Manhattan distance**: agent-goal distance / (2 * grid_dim)
- **Validity**: goal/agent in [1,169] and distinct

Future work: replace with **RL-based evaluation** (agent failure rate, regret, MaxMC).

### Constraint Repair

After decoding (argmax over logits), we enforce CLUTTR constraints:
1. Clamp all values to [0, 169]
2. Clamp goal/agent to [1, 169]
3. If goal == agent, shift agent by +1 (wrapping)
4. Zero out obstacles colliding with goal/agent
5. Sort obstacles ascending

## Configuration

Edit `evolve_config.yml` to change defaults:

```yaml
es_strategy: "CMA_ES"
pop_size: 32
num_generations: 200
sigma_init: 1.0
seed: 42
warm_start: true  # encode random envs to seed initial CMA-ES mean

# Fitness weights
w_obstacles: 0.4
w_distance: 0.4
w_validity: 0.2

log_freq: 10
output_subdir: "evolved"
```
