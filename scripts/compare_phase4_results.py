#!/usr/bin/env python3
"""
Phase 6: Compare PCA-space CMA-ES vs Phase 4 results.

Usage:
    python scripts/compare_phase4_results.py
    python scripts/compare_phase4_results.py --entity myusername
    python scripts/compare_phase4_results.py --entity myusername --project JAXUED_COMPARISON

Queries WandB JAXUED_COMPARISON project for three groups and prints solve_rate comparison.
"""
import argparse
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare Phase 6 PCA-CMA-ES vs Phase 4 baselines via WandB")
    parser.add_argument("--entity", type=str, default=None,
                        help="WandB entity (username/org). If omitted, uses logged-in user.")
    parser.add_argument("--project", type=str, default="JAXUED_COMPARISON",
                        help="WandB project name (default: JAXUED_COMPARISON)")
    parser.add_argument("--metric", type=str, default="solve_rate/mean",
                        help="Metric to compare (default: solve_rate/mean)")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. Run: pip install wandb", file=sys.stderr)
        sys.exit(1)

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project

    groups = {
        "PCA-CMA-ES + ACCEL (Phase 6)": "pca-cmaes-accel",
        "CMA-ES + CNN-VAE + ACCEL (Phase 4)": "cmaes-cnn-vae-accel",
        "ACCEL baseline (Phase 4)": "accel-baseline",
    }

    print(f"\n{'='*70}")
    print(f"Solve Rate Comparison: JAXUED_COMPARISON project")
    print(f"Metric: {args.metric} (final value per run)")
    print(f"{'='*70}\n")

    results = {}
    for label, group in groups.items():
        print(f"Querying group '{group}'...")
        try:
            runs = api.runs(project_path, filters={"group": group})
        except Exception as e:
            print(f"  ERROR querying {group}: {e}")
            results[label] = []
            continue

        final_vals = []
        for run in runs:
            try:
                hist = run.history(keys=[args.metric], x_axis="num_updates")
                if len(hist) > 0 and args.metric in hist.columns:
                    val = float(hist[args.metric].dropna().iloc[-1])
                    final_vals.append(val)
                    print(f"  {run.name}: final {args.metric} = {val:.4f}")
                else:
                    print(f"  {run.name}: no data for {args.metric}")
            except Exception as e:
                print(f"  {run.name}: ERROR - {e}")

        if not final_vals:
            print(f"  WARNING: group '{group}' returned 0 runs with data.")
        results[label] = final_vals

    print(f"\n{'='*70}")
    print(f"{'Condition':<42} {'Mean':>8} {'Std':>8} {'N':>4}  Seeds")
    print(f"{'-'*70}")
    for label, vals in results.items():
        if vals:
            seeds_str = ", ".join(f"{v:.3f}" for v in vals)
            print(f"{label:<42} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} {len(vals):>4}  [{seeds_str}]")
        else:
            print(f"{label:<42} {'NO DATA':>8} {'':>8} {0:>4}")
    print(f"{'='*70}")
    print("\nRun complete.")


if __name__ == "__main__":
    main()
