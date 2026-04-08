"""Plot eval solve rates from WandB runs with mean + min/max bands.

Pulls solve_rate/mean from finished WandB runs, groups by wandb_group,
and plots smoothed mean with min-max confidence bands.

Usage:
    python analysis/solve_rate_plotting/run_plot_solve_rates.py
    python analysis/solve_rate_plotting/run_plot_solve_rates.py --groups accel_baseline llm_injection_fresh
    python analysis/solve_rate_plotting/run_plot_solve_rates.py --window 15 --project JAXUED_LLM_INJECTION
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb


DEFAULT_COLORS = [
    'tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
]


def smooth(y, window=11):
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=str, default="JAXUED_LLM_INJECTION",
                        help="WandB project name")
    parser.add_argument("--groups", nargs="+", default=["accel_baseline", "llm_injection_fresh"],
                        help="WandB group names to plot")
    parser.add_argument("--metric", type=str, default="solve_rate/mean",
                        help="WandB metric key to plot")
    parser.add_argument("--window", type=int, default=11,
                        help="Smoothing window size")
    parser.add_argument("--eval_freq", type=int, default=250,
                        help="Eval frequency (to convert eval steps to training updates)")
    parser.add_argument("--output", type=str, default="analysis/solve_rate_plotting/solve_rate_comparison.png",
                        help="Output plot path")
    args = parser.parse_args()

    api = wandb.Api()

    # Collect finished runs per group
    data = {}
    for r in api.runs(args.project):
        if r.state == 'finished' and r.group in args.groups:
            hist = r.history(keys=[args.metric, '_step'], samples=500, pandas=False)
            steps = [h['_step'] for h in hist if args.metric in h and h[args.metric] is not None]
            vals = [h[args.metric] for h in hist if args.metric in h and h[args.metric] is not None]
            if steps:
                if r.group not in data:
                    data[r.group] = []
                order = np.argsort(steps)
                data[r.group].append((r.name, np.array(steps)[order], np.array(vals)[order]))
                print(f"  {r.name} | {r.group} | {len(steps)} pts")

    if not data:
        print("No data found. Check --project and --groups.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for gi, group in enumerate(args.groups):
        if group not in data:
            print(f"No finished runs for group '{group}', skipping.")
            continue

        color = DEFAULT_COLORS[gi % len(DEFAULT_COLORS)]
        runs = data[group]

        # Common step grid
        max_step = min(s.max() for _, s, _ in runs)
        min_step = max(s.min() for _, s, _ in runs)
        common_steps = np.arange(min_step, max_step + 1)

        # Interpolate all runs to common grid
        interp_vals = []
        for _, steps, vals in runs:
            interp_vals.append(np.interp(common_steps, steps, vals))
        interp_vals = np.array(interp_vals)

        # Smooth each run, then compute stats
        w = args.window
        half = w // 2
        smoothed_runs = np.array([smooth(v, window=w) for v in interp_vals])
        s_steps = common_steps[half:half + smoothed_runs.shape[1]]
        s_updates = s_steps * args.eval_freq

        mean_curve = smoothed_runs.mean(axis=0)
        min_curve = smoothed_runs.min(axis=0)
        max_curve = smoothed_runs.max(axis=0)

        ax.fill_between(s_updates, min_curve, max_curve, color=color, alpha=0.15)
        ax.plot(s_updates, mean_curve, color=color, linewidth=2.5,
                label=f'{group} ({len(runs)} seeds)')

    ax.set_xlabel('Training Updates', fontsize=12)
    ax.set_ylabel(args.metric, fontsize=12)
    ax.set_title(f'Eval {args.metric}: {" vs ".join(args.groups)}', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
