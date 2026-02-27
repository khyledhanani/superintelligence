"""
sweep_beta.py — Run 4 sequential VAE training runs with beta_max ∈ {0.5, 1.0, 1.5, 2.0}.

Usage:
    python sweep_beta.py
"""
import yaml
import subprocess
import sys
import copy

CONFIG_PATH = "vae_train_config.yml"
BETA_VALUES = [1.0, 1.5, 2.0]


def run_sweep():
    # Load the base config once
    with open(CONFIG_PATH, "r") as f:
        base_config = yaml.safe_load(f)

    base_tag = base_config.get("run_tag", "")

    for beta in BETA_VALUES:
        tag = f"{base_tag}_beta{beta:.1f}".strip("_")
        print(f"\n{'='*60}")
        print(f"  SWEEP: beta_max={beta}  run_tag={tag}")
        print(f"{'='*60}\n")

        # Write a modified config for this run
        cfg = copy.deepcopy(base_config)
        cfg["beta_max"] = beta
        cfg["run_tag"] = tag
        cfg["resume_run_id"] = None  # always a fresh run

        with open(CONFIG_PATH, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        # Launch train_vae.py as a subprocess so CONFIG is re-read fresh
        result = subprocess.run(
            [sys.executable, "train_vae.py"],
            cwd=sys.path[0] or ".",
        )

        if result.returncode != 0:
            print(f"[SWEEP] train_vae.py exited with code {result.returncode} "
                  f"for beta_max={beta}. Stopping sweep.")
            break

        print(f"\n[SWEEP] Finished beta_max={beta}\n")

    # Restore the original config so we don't leave it in a modified state
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(base_config, f, default_flow_style=False, sort_keys=False)
    print("[SWEEP] Restored original config. All done.")


if __name__ == "__main__":
    run_sweep()
