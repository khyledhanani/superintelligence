"""Pluggable mutation strategies for generating level variants from LLM seeds.

Each strategy takes a list of seed Level objects and produces a list of mutant
Level objects. All strategies share the same interface:

    mutants: List[Level] = strategy.mutate(seeds, n_per_seed, rng)

Strategies:
    - WallFlipMutator: random wall flips (ACCEL-style), BFS solvability filter
    - VAENoiseMutator: encode seed → add Gaussian noise in latent → decode
    - VAEInterpolationMutator: encode seeds → interpolate between pairs → decode
"""
import os
import sys
import pickle
from abc import ABC, abstractmethod
from typing import List, Optional
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [
    os.path.join(_project_root, "src"),
    os.path.join(_project_root, "vae"),
    _project_root,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from jaxued.environments.maze import Level
from jaxued.environments.maze.util import make_level_mutator_minimax
from vae_level_utils import (
    level_to_tokens, tokens_to_level, repair_tokens,
    decode_latent_to_levels, GRID_SIZE,
)


def _bfs_solvable(level) -> bool:
    """Check if agent can reach goal via BFS on the wall_map."""
    wm = np.asarray(level.wall_map)
    ax, ay = int(level.agent_pos[0]), int(level.agent_pos[1])
    gx, gy = int(level.goal_pos[0]), int(level.goal_pos[1])
    if wm[ay, ax] or wm[gy, gx]:
        return False
    h, w = wm.shape
    visited = set()
    queue = deque([(ax, ay)])
    visited.add((ax, ay))
    while queue:
        x, y = queue.popleft()
        if x == gx and y == gy:
            return True
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited and not wm[ny, nx]:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False


class MutationStrategy(ABC):
    """Base class for mutation strategies."""

    @abstractmethod
    def mutate(
        self,
        seeds: List,
        n_per_seed: int,
        rng: jax.Array,
    ) -> List:
        """Generate mutant levels from seed levels.

        Args:
            seeds: List of Level objects (LLM-generated seeds)
            n_per_seed: Number of mutants to produce per seed
            rng: JAX PRNGKey

        Returns:
            List of (mutant_level, seed_index) tuples
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class WallFlipMutator(MutationStrategy):
    """ACCEL-style random wall-flip mutations with BFS solvability filter.

    Uses make_level_mutator_minimax from the JAXUED codebase.
    Generates batches of mutations, filters for solvability, repeats until
    n_per_seed solvable mutants are found (with a safety cap).
    """

    def __init__(self, num_edits: int = 5, max_rounds: int = 10):
        self.num_edits = num_edits
        self.max_rounds = max_rounds
        self._mutate_fn = make_level_mutator_minimax(100)

    @property
    def name(self) -> str:
        return f"wall_flip_e{self.num_edits}"

    def mutate(self, seeds, n_per_seed, rng):
        results = []
        for seed_idx, seed in enumerate(seeds):
            found = []
            for _ in range(self.max_rounds):
                if len(found) >= n_per_seed:
                    break
                batch_size = min((n_per_seed - len(found)) * 3, 100)
                rng, mut_rng = jax.random.split(rng)
                mut_rngs = jax.random.split(mut_rng, batch_size)
                mutants = jax.vmap(
                    lambda r: self._mutate_fn(r, seed, self.num_edits)
                )(mut_rngs)

                for j in range(batch_size):
                    if len(found) >= n_per_seed:
                        break
                    m = jax.tree_util.tree_map(lambda x: x[j], mutants)
                    if _bfs_solvable(m):
                        found.append(m)

            results.extend((m, seed_idx) for m in found)
            print(f"  [WallFlip] Seed {seed_idx}: {len(found)}/{n_per_seed} solvable mutants")
        return results


class VAENoiseMutator(MutationStrategy):
    """Encode seed to VAE latent, add Gaussian noise, decode.

    Produces mutations that stay on the learned manifold of valid mazes.
    """

    def __init__(self, vae_checkpoint_path: str, vae_config_path: str,
                 sigma: float = 0.5):
        self.sigma = sigma
        self._load_vae(vae_checkpoint_path, vae_config_path)

    @property
    def name(self) -> str:
        return f"vae_noise_s{self.sigma}"

    def _load_vae(self, ckpt_path, cfg_path):
        import yaml
        with open(cfg_path) as f:
            vae_cfg = yaml.safe_load(f)

        from vae_model import CluttrVAE
        self.vae = CluttrVAE(
            vocab_size=vae_cfg["vocab_size"],
            embed_dim=vae_cfg["embed_dim"],
            latent_dim=vae_cfg["latent_dim"],
            seq_len=vae_cfg["seq_len"],
        )
        self.latent_dim = vae_cfg["latent_dim"]

        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        self.vae_params = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt

    def _encode(self, level):
        """Encode a single Level to latent mean vector."""
        tokens = level_to_tokens(level)
        tokens_batch = tokens[jnp.newaxis, :]  # (1, 52)
        mean, _ = self.vae.apply(
            {"params": self.vae_params}, tokens_batch,
            train=False, method=self.vae.encode,
        )
        return mean[0]  # (latent_dim,)

    def _decode_batch(self, z_batch, rng):
        """Decode batch of latent vectors to Levels."""
        def decode_fn(z):
            return self.vae.apply(
                {"params": self.vae_params}, z, method=self.vae.decode,
            )
        return decode_latent_to_levels(decode_fn, z_batch, rng)

    def mutate(self, seeds, n_per_seed, rng):
        results = []
        for seed_idx, seed in enumerate(seeds):
            z_seed = self._encode(seed)  # (latent_dim,)
            rng, noise_rng, decode_rng = jax.random.split(rng, 3)

            # Over-generate to filter for solvability
            n_candidates = n_per_seed * 3
            noise = jax.random.normal(noise_rng, (n_candidates, self.latent_dim)) * self.sigma
            z_mutants = z_seed[jnp.newaxis, :] + noise

            mutant_levels = self._decode_batch(z_mutants, decode_rng)

            found = []
            for j in range(n_candidates):
                if len(found) >= n_per_seed:
                    break
                m = jax.tree_util.tree_map(lambda x: x[j], mutant_levels)
                if _bfs_solvable(m):
                    found.append(m)

            results.extend((m, seed_idx) for m in found)
            print(f"  [VAENoise] Seed {seed_idx}: {len(found)}/{n_per_seed} solvable mutants "
                  f"(sigma={self.sigma})")
        return results


class VAEInterpolationMutator(MutationStrategy):
    """Interpolate between pairs of seeds in VAE latent space.

    For each pair of seeds (i, j), generates n interpolated levels along the
    line segment z_i → z_j. When there are fewer seed pairs than needed,
    also interpolates seed→seed+noise as fallback.
    """

    def __init__(self, vae_checkpoint_path: str, vae_config_path: str,
                 noise_sigma: float = 0.3):
        self.noise_sigma = noise_sigma
        self._load_vae(vae_checkpoint_path, vae_config_path)

    @property
    def name(self) -> str:
        return "vae_interpolation"

    def _load_vae(self, ckpt_path, cfg_path):
        import yaml
        with open(cfg_path) as f:
            vae_cfg = yaml.safe_load(f)

        from vae_model import CluttrVAE
        self.vae = CluttrVAE(
            vocab_size=vae_cfg["vocab_size"],
            embed_dim=vae_cfg["embed_dim"],
            latent_dim=vae_cfg["latent_dim"],
            seq_len=vae_cfg["seq_len"],
        )
        self.latent_dim = vae_cfg["latent_dim"]

        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        self.vae_params = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt

    def _encode(self, level):
        tokens = level_to_tokens(level)
        tokens_batch = tokens[jnp.newaxis, :]
        mean, _ = self.vae.apply(
            {"params": self.vae_params}, tokens_batch,
            train=False, method=self.vae.encode,
        )
        return mean[0]

    def _decode_batch(self, z_batch, rng):
        def decode_fn(z):
            return self.vae.apply(
                {"params": self.vae_params}, z, method=self.vae.decode,
            )
        return decode_latent_to_levels(decode_fn, z_batch, rng)

    def mutate(self, seeds, n_per_seed, rng):
        # Encode all seeds
        z_seeds = jnp.stack([self._encode(s) for s in seeds])  # (K, latent_dim)
        K = len(seeds)

        results = []
        total_per_seed = n_per_seed

        for seed_idx in range(K):
            z_i = z_seeds[seed_idx]
            z_targets = []

            # Interpolation targets: other seeds
            for j in range(K):
                if j != seed_idx:
                    z_targets.append(z_seeds[j])

            # If only 1 seed, use noisy versions as targets
            if not z_targets:
                rng, noise_rng = jax.random.split(rng)
                noise = jax.random.normal(noise_rng, (3, self.latent_dim)) * self.noise_sigma
                z_targets = [z_i + n for n in noise]

            # Generate interpolation points
            n_candidates = total_per_seed * 3  # over-generate for solvability
            rng, alpha_rng, decode_rng = jax.random.split(rng, 3)
            alphas = jax.random.uniform(alpha_rng, (n_candidates,), minval=0.1, maxval=0.9)

            # Cycle through targets
            z_batch = []
            for k in range(n_candidates):
                z_j = z_targets[k % len(z_targets)]
                alpha = float(alphas[k])
                z_interp = (1 - alpha) * z_i + alpha * z_j
                z_batch.append(z_interp)
            z_batch = jnp.stack(z_batch)

            mutant_levels = self._decode_batch(z_batch, decode_rng)

            found = []
            for j in range(n_candidates):
                if len(found) >= total_per_seed:
                    break
                m = jax.tree_util.tree_map(lambda x: x[j], mutant_levels)
                if _bfs_solvable(m):
                    found.append(m)

            results.extend((m, seed_idx) for m in found)
            print(f"  [VAEInterp] Seed {seed_idx}: {len(found)}/{total_per_seed} solvable mutants")

        return results


def get_strategy(
    name: str,
    vae_checkpoint_path: str = "",
    vae_config_path: str = "",
    **kwargs,
) -> MutationStrategy:
    """Factory function for mutation strategies.

    Args:
        name: One of "wall_flip", "vae_noise", "vae_interpolation"
        vae_checkpoint_path: Required for VAE strategies
        vae_config_path: Required for VAE strategies
        **kwargs: Strategy-specific params (sigma, num_edits, etc.)
    """
    if name == "wall_flip":
        return WallFlipMutator(
            num_edits=kwargs.get("num_edits", 3),
            max_rounds=kwargs.get("max_rounds", 10),
        )
    elif name == "vae_noise":
        return VAENoiseMutator(
            vae_checkpoint_path=vae_checkpoint_path,
            vae_config_path=vae_config_path,
            sigma=kwargs.get("sigma", 0.5),
        )
    elif name == "vae_interpolation":
        return VAEInterpolationMutator(
            vae_checkpoint_path=vae_checkpoint_path,
            vae_config_path=vae_config_path,
            noise_sigma=kwargs.get("noise_sigma", 0.3),
        )
    else:
        raise ValueError(
            f"Unknown mutation strategy: {name!r}. "
            "Expected 'wall_flip', 'vae_noise', or 'vae_interpolation'."
        )
