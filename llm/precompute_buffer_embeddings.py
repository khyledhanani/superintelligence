#!/usr/bin/env python3
"""Precompute TD errors for all buffer levels for use in diversity embedding plots.

Rolls out the agent on every buffer level (single rollout each) and saves the
TD error array per level. This avoids re-rolling out at plot time.

The saved data enables the diversity embedding plot to include buffer levels
as background context points alongside generated/rejected mazes.

Usage:
    PYTHONPATH=src:. python3 llm/precompute_buffer_embeddings.py

    # Custom paths
    PYTHONPATH=src:. python3 llm/precompute_buffer_embeddings.py \
        --buffer-path gcs_artifacts/buffer/buffer_dump_final.npz \
        --agent-dir gcs_artifacts/agent/cmaes_vae_beta2.0_seed0_198 \
        --output llm/buffer_td_errors.npz

    # Subsample for faster computation
    PYTHONPATH=src:. python3 llm/precompute_buffer_embeddings.py --max-levels 500
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from vae.vae_level_utils import tokens_to_level
from llm.agent_evaluator import AgentEvaluator
from metrics.pairwise.td_error_distribution import compute_td_errors

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Precompute buffer TD errors for embedding plots")
    parser.add_argument("--buffer-path", default="gcs_artifacts/buffer/buffer_dump_final.npz")
    parser.add_argument("--agent-dir", default="gcs_artifacts/agent/cmaes_vae_beta2.0_seed0_198")
    parser.add_argument("--output", default="gcs_artifacts/buffer/buffer_td_errors.npz")
    parser.add_argument("--max-levels", type=int, default=None,
                        help="Max levels to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for agent rollouts")
    args = parser.parse_args()

    # Load buffer
    logger.info(f"Loading buffer from {args.buffer_path}...")
    data = np.load(args.buffer_path)
    tokens = data["tokens"]
    scores = data.get("scores", np.zeros(len(tokens)))
    size = int(data.get("size", len(tokens)))
    logger.info(f"Buffer: {size} active levels")

    n_levels = min(size, args.max_levels) if args.max_levels else size
    if args.max_levels and args.max_levels < size:
        # Random subsample
        np.random.seed(42)
        indices = np.sort(np.random.choice(size, n_levels, replace=False))
        logger.info(f"Subsampled {n_levels} of {size} levels")
    else:
        indices = np.arange(n_levels)

    # Load agent
    print(f"Loading agent from {args.agent_dir}...", flush=True)
    evaluator = AgentEvaluator(args.agent_dir)
    print("Agent loaded.", flush=True)

    # Roll out in batches
    logger.info(f"Rolling out agent on {n_levels} levels (batch_size={args.batch_size})...")
    all_td_errors = []  # list of 1D arrays (variable length)
    all_ep_lens = []
    t_start = time.time()

    for batch_start in range(0, n_levels, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n_levels)
        batch_indices = indices[batch_start:batch_end]
        batch_size = len(batch_indices)

        # Convert tokens to Level objects
        levels = []
        for idx in batch_indices:
            tok = jnp.array(tokens[idx], dtype=jnp.int32)
            levels.append(tokens_to_level(tok))

        # Batch rollout
        trajs = evaluator.evaluate_levels(levels)

        for traj in trajs:
            td = compute_td_errors(traj["values"], traj["rewards"], traj["dones"], gamma=1.0)
            all_td_errors.append(td)
            all_ep_lens.append(len(td))

        elapsed = time.time() - t_start
        done = batch_end
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n_levels - done) / rate if rate > 0 else 0
        print(f"  {done}/{n_levels} levels ({rate:.0f} levels/s, ETA {eta:.0f}s)", flush=True)

    total_time = time.time() - t_start
    logger.info(f"Rollouts complete: {n_levels} levels in {total_time:.1f}s")

    # Pack TD errors into a padded array + lengths array
    max_len = max(all_ep_lens)
    td_padded = np.zeros((n_levels, max_len), dtype=np.float32)
    ep_lens = np.array(all_ep_lens, dtype=np.int32)
    for i, td in enumerate(all_td_errors):
        td_padded[i, :len(td)] = td

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        td_errors=td_padded,       # (N, max_ep_len) padded TD errors
        ep_lens=ep_lens,           # (N,) actual episode lengths
        indices=indices,           # (N,) buffer indices used
        scores=scores[indices],    # (N,) regret scores
        n_levels=n_levels,
    )
    file_size = os.path.getsize(args.output) / 1024 / 1024
    logger.info(f"Saved {args.output} ({file_size:.1f} MB)")
    logger.info(f"  {n_levels} levels, max_ep_len={max_len}, mean_ep_len={np.mean(ep_lens):.0f}")


if __name__ == "__main__":
    main()
