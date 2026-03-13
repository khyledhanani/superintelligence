#!/usr/bin/env python3
"""Run TRACED and TRACED+PCA over multiple seeds.

Default behavior matches:
  python examples/maze_traced.py --seed <seed>
  python examples/maze_traced.py --traced_use_pca_mutation --seed <seed>

for seeds 0..2.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class RunSpec:
    label: str
    cmd: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start_seed", type=int, default=0, help="First seed (inclusive).")
    parser.add_argument("--num_seeds", type=int, default=3, help="Number of consecutive seeds to run.")
    parser.add_argument(
        "--shared_args",
        type=str,
        default="",
        help='Extra args appended to both commands. Example: --shared_args "--num_updates 10000"',
    )
    parser.add_argument(
        "--traced_args",
        type=str,
        default="",
        help="Extra args appended only to the baseline TRACED command.",
    )
    parser.add_argument(
        "--pca_args",
        type=str,
        default="",
        help="Extra args appended only to the TRACED+PCA command.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Keep running remaining jobs if one command fails.",
    )
    return parser.parse_args()


def build_seed_runs(
    seed: int,
    shared_extra: list[str],
    traced_extra: list[str],
    pca_extra: list[str],
) -> list[RunSpec]:
    return [
        RunSpec(
            label=f"seed={seed} traced",
            cmd=[
                sys.executable,
                "examples/maze_traced.py",
                "--seed",
                str(seed),
                *shared_extra,
                *traced_extra,
            ],
        ),
        RunSpec(
            label=f"seed={seed} traced_pca",
            cmd=[
                sys.executable,
                "examples/maze_traced.py",
                "--traced_use_pca_mutation",
                "--seed",
                str(seed),
                *shared_extra,
                *pca_extra,
            ],
        ),
    ]


def main() -> None:
    args = parse_args()
    if args.num_seeds < 1:
        raise ValueError("--num_seeds must be >= 1")

    shared_extra = shlex.split(args.shared_args)
    traced_extra = shlex.split(args.traced_args)
    pca_extra = shlex.split(args.pca_args)

    all_runs: list[RunSpec] = []
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        all_runs.extend(build_seed_runs(seed, shared_extra, traced_extra, pca_extra))

    print(f"Running {len(all_runs)} jobs ({args.num_seeds} seeds x 2 variants) sequentially.")
    failures: list[tuple[RunSpec, int]] = []

    for i, run in enumerate(all_runs, start=1):
        rendered = " ".join(shlex.quote(part) for part in run.cmd)
        print(f"\n[{i}/{len(all_runs)}] {run.label}")
        print(f"  {rendered}")

        if args.dry_run:
            continue

        result = subprocess.run(run.cmd, check=False)
        if result.returncode != 0:
            failures.append((run, result.returncode))
            if not args.continue_on_error:
                break

    if failures:
        print("\nFailures:")
        for run, code in failures:
            rendered = " ".join(shlex.quote(part) for part in run.cmd)
            print(f"  (exit {code}) {run.label}")
            print(f"    {rendered}")
        raise SystemExit(1)

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()
