#!/usr/bin/env python3
"""Run controlled ACCEL/PLWM ablations for 25k updates.

Runs:
1) base_accel
2) plwm_struct_k8
3) plwm_task_k8
4) plwm_task_k1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_runs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    ckpt = str(Path(args.plwm_mae_checkpoint).resolve())

    common = [
        "examples/maze_plr.py",
        "--mode", "train",
        "--project", args.project,
        "--seed", str(args.seed),
        "--num_updates", str(args.num_updates),
        "--use_accel",
    ]

    runs: list[tuple[str, list[str]]] = []

    # 1) Base ACCEL (no PLWM mutation)
    runs.append((
        "base_accel",
        common + [
            "--run_name", f"{args.run_prefix}_base_accel",
            "--no-use_plwm_mutation",
        ],
    ))

    # 2) PLWM + structural surrogate, k=8
    runs.append((
        "plwm_struct_k8",
        common + [
            "--run_name", f"{args.run_prefix}_plwm_struct_k8",
            "--use_plwm_mutation",
            "--plwm_use_maze_ae",
            "--plwm_surrogate_guided",
            "--no-plwm_task_aware_guided",
            "--plwm_surrogate_num_candidates", "8",
            "--plwm_mae_checkpoint", ckpt,
        ],
    ))

    # 3) PLWM + task-aware, k=8
    runs.append((
        "plwm_task_k8",
        common + [
            "--run_name", f"{args.run_prefix}_plwm_task_k8",
            "--use_plwm_mutation",
            "--plwm_use_maze_ae",
            "--plwm_surrogate_guided",
            "--plwm_task_aware_guided",
            "--plwm_surrogate_num_candidates", "8",
            "--plwm_mae_checkpoint", ckpt,
        ],
    ))

    # 4) PLWM + task-aware, k=1 (single candidate)
    runs.append((
        "plwm_task_k1",
        common + [
            "--run_name", f"{args.run_prefix}_plwm_task_k1",
            "--use_plwm_mutation",
            "--plwm_use_maze_ae",
            "--plwm_surrogate_guided",
            "--plwm_task_aware_guided",
            "--plwm_surrogate_num_candidates", "1",
            "--plwm_mae_checkpoint", ckpt,
        ],
    ))

    return runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=str, default="JAXUED_TASKAWARE_ABLATIONS_25K")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_updates", type=int, default=25000)
    p.add_argument("--run_prefix", type=str, default="ablate25k_seed0")
    p.add_argument(
        "--plwm_mae_checkpoint",
        type=str,
        default="vae/model_maze_ae_taskaware_stage1/checkpoint_final.pkl",
    )
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs = build_runs(args)

    print(f"Project: {args.project}")
    print(f"num_updates: {args.num_updates}, seed: {args.seed}")
    print(f"checkpoint: {Path(args.plwm_mae_checkpoint).resolve()}")
    print("-" * 80)

    for i, (name, cmd) in enumerate(runs, start=1):
        print(f"[{i}/{len(runs)}] {name}")
        print("  " + " ".join([sys.executable] + cmd))
        if args.dry_run:
            continue
        subprocess.run([sys.executable] + cmd, check=True)

    print("Done.")


if __name__ == "__main__":
    main()
