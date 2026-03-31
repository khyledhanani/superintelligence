"""Convert LLM-generated maze .txt files into seeds.npz for injection experiments.

Scans all maze_*.txt files in the generated_levels directory, parses them into
Level objects, converts to token representation, and saves as seeds.npz.

Usage:
    python experiments/prepare_llm_seeds.py \
        --input_dir llm/generated_levels \
        --output_dir /tmp/injection_data/seeds
"""
import argparse
import glob
import os
import sys

import jax.numpy as jnp
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [
    os.path.join(_project_root, "src"),
    os.path.join(_project_root, "examples"),
    os.path.join(_project_root, "vae"),
    _project_root,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from jaxued.environments.maze import Level
from vae_level_utils import level_to_tokens


def main():
    parser = argparse.ArgumentParser(description="Convert LLM maze .txt files to seeds.npz")
    parser.add_argument("--input_dir", type=str, default="llm/generated_levels",
                        help="Directory containing subdirectories with maze_*.txt files")
    parser.add_argument("--output_dir", type=str, default="/tmp/injection_data/seeds",
                        help="Output directory for seeds.npz")
    args = parser.parse_args()

    # Find all maze .txt files
    maze_files = sorted(glob.glob(os.path.join(args.input_dir, "**/maze_*.txt"), recursive=True))
    if not maze_files:
        print(f"No maze_*.txt files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(maze_files)} maze files")

    tokens_list = []
    for path in maze_files:
        with open(path) as f:
            grid = f.read().strip()

        try:
            level = Level.from_str(grid)
            tok = np.asarray(level_to_tokens(level))
            tokens_list.append(tok)
            print(f"  OK: {path}")
        except Exception as e:
            print(f"  SKIP: {path} ({e})")

    if not tokens_list:
        print("No valid mazes parsed!")
        sys.exit(1)

    tokens = np.stack(tokens_list)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "seeds.npz")
    np.savez(out_path, tokens=tokens)
    print(f"\nSaved {len(tokens_list)} seeds to {out_path}")


if __name__ == "__main__":
    main()
