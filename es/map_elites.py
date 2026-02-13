"""
MAP-Elites for diverse environment generation in VAE latent space.

Based on: Mouret & Clune (2015) "Illuminating search spaces by mapping elites"

Maintains a 2D archive of environments binned by behavior descriptors
(num_obstacles, manhattan_distance). Each cell stores the highest-regret
environment with those characteristics, guaranteeing diversity by construction.

Reuses the existing regret fitness pipeline — same agent, decoder, and
complexity filter as evolve_envs.py.

Usage:
    python es/map_elites.py --agent_checkpoint_dir agent_folder \
        --num_iterations 2000 --batch_size 32 --output_subdir _me_run1

    # Quick smoke test
    python es/map_elites.py --agent_checkpoint_dir agent_folder \
        --num_iterations 100 --batch_size 16 --init_pop 64 \
        --output_subdir _me_smoke
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import time
import argparse
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vae_decoder import (
    load_vae_params, extract_decoder_params,
    decode_latent_to_env, repair_cluttr_sequence,
)
from regret_fitness import regret_fitness
from agent_loader import load_agent, verify_agent_contract
from visualize_envs import sequence_to_grid, visualize_grid, compute_stats
from metrics import compute_latent_diversity, compute_sequence_diversity

from jaxued.environments.maze import Maze
from jaxued.wrappers import AutoReplayWrapper


# ---------------------------------------------------------------------------
# Behavior descriptor bins
# ---------------------------------------------------------------------------

OBS_BINS = np.array([5, 10, 15, 20, 25, 30, 35, 40, 50])   # 8 bins
DIST_BINS = np.array([3, 6, 9, 12, 15, 18, 24])             # 6 bins
N_OBS_BINS = len(OBS_BINS) - 1   # 8
N_DIST_BINS = len(DIST_BINS) - 1  # 6


def compute_behavior_descriptors(sequences):
    """Compute (num_obstacles, manhattan_dist) for a batch of sequences.

    Args:
        sequences: (batch, 52) integer arrays.

    Returns:
        obs_counts: (batch,) int array
        dists: (batch,) int array
    """
    seqs_np = np.asarray(sequences)
    obs_counts = (seqs_np[:, :50] > 0).sum(axis=1)
    goal_idx = seqs_np[:, 50]
    agent_idx = seqs_np[:, 51]
    goal_row = (goal_idx - 1) // 13
    goal_col = (goal_idx - 1) % 13
    agent_row = (agent_idx - 1) // 13
    agent_col = (agent_idx - 1) % 13
    dists = np.abs(goal_row - agent_row) + np.abs(goal_col - agent_col)
    return obs_counts, dists


def descriptor_to_cell(obs_count, dist):
    """Map a single (obs_count, dist) to archive grid indices.

    Returns (row, col) or None if out of bounds.
    """
    row = np.searchsorted(OBS_BINS, obs_count, side='right') - 1
    col = np.searchsorted(DIST_BINS, dist, side='right') - 1
    if 0 <= row < N_OBS_BINS and 0 <= col < N_DIST_BINS:
        return (row, col)
    return None


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

class Archive:
    """2D MAP-Elites archive storing elite environments per behavior cell."""

    def __init__(self, latent_dim=64):
        self.latent_dim = latent_dim
        self.grid = {}  # (row, col) -> {latent, sequence, regret}
        self.total_attempts = 0
        self.total_insertions = 0

    def try_insert(self, latent, sequence, regret, obs_count, dist):
        """Attempt to insert a solution. Returns True if inserted."""
        cell = descriptor_to_cell(obs_count, dist)
        if cell is None:
            return False

        self.total_attempts += 1
        existing = self.grid.get(cell)
        if existing is None or regret > existing['regret']:
            self.grid[cell] = {
                'latent': np.array(latent),
                'sequence': np.array(sequence),
                'regret': float(regret),
                'obs_count': int(obs_count),
                'dist': int(dist),
            }
            self.total_insertions += 1
            return True
        return False

    @property
    def num_filled(self):
        return len(self.grid)

    @property
    def total_cells(self):
        return N_OBS_BINS * N_DIST_BINS

    @property
    def coverage(self):
        return self.num_filled / self.total_cells

    def sample_parents(self, rng, n):
        """Sample n parent latent vectors from occupied cells (uniform)."""
        cells = list(self.grid.values())
        if not cells:
            return None
        indices = rng.integers(0, len(cells), size=n)
        return np.stack([cells[i]['latent'] for i in indices])

    def get_arrays(self):
        """Extract archive contents as numpy arrays."""
        if not self.grid:
            return None
        items = list(self.grid.values())
        return {
            'sequences': np.stack([it['sequence'] for it in items]),
            'latents': np.stack([it['latent'] for it in items]),
            'regrets': np.array([it['regret'] for it in items]),
            'descriptors': np.array([[it['obs_count'], it['dist']] for it in items]),
        }

    def regret_heatmap(self):
        """Build a 2D array of regret values for visualization."""
        hmap = np.full((N_OBS_BINS, N_DIST_BINS), np.nan)
        for (r, c), entry in self.grid.items():
            hmap[r, c] = entry['regret']
        return hmap


# ---------------------------------------------------------------------------
# Main MAP-Elites loop
# ---------------------------------------------------------------------------

def run_map_elites(config):
    key = jax.random.PRNGKey(config['seed'])
    rng_np = np.random.default_rng(config['seed'])

    # --- Setup (same as evolve_envs.py) ---
    print("Loading VAE checkpoint...")
    full_vae_params = load_vae_params(config['checkpoint_path'])
    decoder_params = extract_decoder_params(full_vae_params)

    print(f"Loading agent from {config['agent_checkpoint_dir']}...")
    agent_params, network = load_agent(config['agent_checkpoint_dir'], action_dim=7)
    verify_agent_contract(agent_params, network)

    maze_env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    wrapped_env = AutoReplayWrapper(maze_env)
    env_params = wrapped_env.default_params
    print("Agent and Maze environment initialized.")

    latent_dim = config['latent_dim']
    decode_temperature = config['decode_temperature']
    rollout_steps = config['rollout_steps']
    min_obstacles = config['min_obstacles']
    min_distance = config['min_distance']

    # --- JIT-compile evaluate function ---
    @jax.jit
    def evaluate_batch(eval_key, latents):
        decode_key, regret_key = jax.random.split(eval_key)
        sequences = decode_latent_to_env(
            decoder_params, latents, rng_key=decode_key, temperature=decode_temperature
        )
        sequences = jax.vmap(repair_cluttr_sequence)(sequences)
        fitness, info = regret_fitness(
            regret_key, sequences, agent_params, network, wrapped_env, env_params,
            num_steps=rollout_steps, deterministic=True,
            min_obstacles=min_obstacles, min_distance=min_distance,
        )
        return sequences, info['regret'], info['valid']

    # --- Initialize archive ---
    archive = Archive(latent_dim=latent_dim)
    init_pop = config['init_pop']
    batch_size = config['batch_size']

    print(f"\nInitializing archive with {init_pop} random latent vectors...")
    key, init_key = jax.random.split(key)
    z_init = jax.random.normal(init_key, (init_pop, latent_dim))

    # Evaluate in batches
    for start in range(0, init_pop, batch_size):
        end = min(start + batch_size, init_pop)
        key, eval_key = jax.random.split(key)
        z_batch = z_init[start:end]
        seqs, regrets, valid = evaluate_batch(eval_key, z_batch)

        seqs_np = np.asarray(seqs)
        regrets_np = np.asarray(regrets)
        valid_np = np.asarray(valid)
        z_np = np.asarray(z_batch)
        obs_counts, dists = compute_behavior_descriptors(seqs_np)

        for i in range(len(z_batch)):
            if valid_np[i]:
                archive.try_insert(z_np[i], seqs_np[i], regrets_np[i],
                                   obs_counts[i], dists[i])

    print(f"  Archive after init: {archive.num_filled}/{archive.total_cells} cells "
          f"({archive.coverage:.0%} coverage)")

    # --- Main MAP-Elites loop ---
    num_iterations = config['num_iterations']
    mutation_sigma = config['mutation_sigma']
    log_freq = config['log_freq']

    print(f"\nStarting MAP-Elites: {num_iterations} iterations, "
          f"batch_size={batch_size}, sigma={mutation_sigma}")
    print("-" * 80)

    t_total_start = time.time()

    for it in range(num_iterations):
        t_start = time.time()

        # 1. Sample parents from archive
        parents = archive.sample_parents(rng_np, batch_size)
        if parents is None:
            # Archive empty — generate random
            key, rand_key = jax.random.split(key)
            z_children = jax.random.normal(rand_key, (batch_size, latent_dim))
        else:
            # 2. Mutate: Gaussian perturbation in latent space
            noise = rng_np.standard_normal((batch_size, latent_dim)).astype(np.float32)
            z_children = jnp.array(parents + mutation_sigma * noise)

        # 3. Evaluate
        key, eval_key = jax.random.split(key)
        seqs, regrets, valid = evaluate_batch(eval_key, z_children)

        # 4. Try to insert into archive
        seqs_np = np.asarray(seqs)
        regrets_np = np.asarray(regrets)
        valid_np = np.asarray(valid)
        z_np = np.asarray(z_children)
        obs_counts, dists = compute_behavior_descriptors(seqs_np)

        insertions = 0
        for i in range(batch_size):
            if valid_np[i]:
                if archive.try_insert(z_np[i], seqs_np[i], regrets_np[i],
                                      obs_counts[i], dists[i]):
                    insertions += 1

        t_end = time.time()

        if it % log_freq == 0:
            data = archive.get_arrays()
            max_reg = data['regrets'].max() if data else 0
            mean_reg = data['regrets'].mean() if data else 0
            print(f"Iter {it:5d} | Cells: {archive.num_filled}/{archive.total_cells} "
                  f"({archive.coverage:.0%}) | Regret: max={max_reg:.4f} mean={mean_reg:.4f} | "
                  f"Insertions: {insertions}/{batch_size} | Time: {t_end-t_start:.2f}s")

    t_total = time.time() - t_total_start
    print("-" * 80)
    print(f"MAP-Elites complete. Total time: {t_total:.1f}s")

    # --- Save results ---
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    data = archive.get_arrays()
    if data is None:
        print("Archive is empty! No valid environments found.")
        return archive

    np.save(os.path.join(output_dir, 'archive_envs.npy'), data['sequences'])
    np.save(os.path.join(output_dir, 'archive_latents.npy'), data['latents'])
    np.save(os.path.join(output_dir, 'archive_fitness.npy'), data['regrets'])
    np.save(os.path.join(output_dir, 'archive_descriptors.npy'), data['descriptors'])

    print(f"\nArchive: {archive.num_filled}/{archive.total_cells} cells filled "
          f"({archive.coverage:.0%})")
    print(f"Regret: max={data['regrets'].max():.4f}, mean={data['regrets'].mean():.4f}, "
          f"min={data['regrets'].min():.4f}")
    print(f"Saved to {output_dir}/")

    # --- Visualize: heatmap ---
    hmap = archive.regret_heatmap()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(hmap, cmap='YlOrRd', aspect='auto', origin='lower',
                   interpolation='nearest')
    ax.set_xlabel('Manhattan Distance')
    ax.set_ylabel('Num Obstacles')
    ax.set_xticks(range(N_DIST_BINS))
    ax.set_xticklabels([f'{DIST_BINS[i]}-{DIST_BINS[i+1]}' for i in range(N_DIST_BINS)])
    ax.set_yticks(range(N_OBS_BINS))
    ax.set_yticklabels([f'{OBS_BINS[i]}-{OBS_BINS[i+1]}' for i in range(N_OBS_BINS)])
    plt.colorbar(im, label='Regret')

    # Annotate cells with regret values
    for (r, c), entry in archive.grid.items():
        ax.text(c, r, f'{entry["regret"]:.3f}', ha='center', va='center', fontsize=7,
                color='white' if entry['regret'] > hmap[~np.isnan(hmap)].mean() else 'black')

    ax.set_title(f'MAP-Elites Archive: {archive.num_filled}/{archive.total_cells} cells '
                 f'({archive.coverage:.0%} coverage)')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'archive_heatmap.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/archive_heatmap.png")
    plt.close()

    # --- Visualize: gallery of top environments ---
    sorted_items = sorted(archive.grid.values(), key=lambda x: x['regret'], reverse=True)
    n_show = min(12, len(sorted_items))
    ncols = min(4, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for i in range(n_show):
        it = sorted_items[i]
        grid = sequence_to_grid(it['sequence'])
        visualize_grid(axes[i], grid,
                       f"Regret={it['regret']:.4f}\n"
                       f"{it['obs_count']} obs, dist={it['dist']}")

    for j in range(n_show, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'MAP-Elites: Top {n_show} Environments (by regret)', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'archive_gallery.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/archive_gallery.png")
    plt.close()

    return archive


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MAP-Elites environment generation")
    parser.add_argument('--agent_checkpoint_dir', type=str, required=True)
    parser.add_argument('--num_iterations', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mutation_sigma', type=float, default=0.5)
    parser.add_argument('--decode_temperature', type=float, default=0.25)
    parser.add_argument('--rollout_steps', type=int, default=256)
    parser.add_argument('--init_pop', type=int, default=256)
    parser.add_argument('--min_obstacles', type=int, default=5)
    parser.add_argument('--min_distance', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--output_subdir', type=str, default='_map_elites_run1')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # VAE config
    vae_config_path = os.path.join(script_dir, '..', 'vae', 'vae_train_config.yml')
    with open(vae_config_path, 'r') as f:
        vae_config = yaml.safe_load(f)

    checkpoint_path = os.path.join(script_dir, '..', 'vae', 'model', 'checkpoint_420000.pkl')

    config = {
        **vars(args),
        'latent_dim': vae_config['latent_dim'],
        'checkpoint_path': checkpoint_path,
        'output_dir': os.path.join(script_dir, args.output_subdir),
    }

    run_map_elites(config)
