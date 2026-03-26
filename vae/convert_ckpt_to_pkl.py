"""Convert an Orbax agent checkpoint to a simple pickle file.

Run this on the SAME device type (GPU/TPU) where the checkpoint was saved.
Then load the .pkl file anywhere without topology issues.

Usage:
    python vae/convert_ckpt_to_pkl.py /tmp/warmstart_ckpt /tmp/agent_params.pkl
"""
import sys
import pickle
import numpy as np
import jax
import orbax.checkpoint as ocp


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <orbax_ckpt_dir> <output.pkl>")
        sys.exit(1)

    ckpt_dir = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading checkpoint from {ckpt_dir}...")
    manager = ocp.CheckpointManager(
        ckpt_dir,
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    step = manager.latest_step()
    print(f"Latest step: {step}")

    ckpt = manager.restore(step)
    print(f"Restored. Type: {type(ckpt)}")

    # Convert all arrays to numpy
    def to_numpy(x):
        if hasattr(x, 'numpy'):
            return x.numpy()
        if isinstance(x, jax.Array):
            return np.array(x)
        return x

    ckpt_np = jax.tree_util.tree_map(to_numpy, ckpt)

    # Extract params
    if isinstance(ckpt_np, dict) and "params" in ckpt_np:
        params = ckpt_np["params"]
    elif hasattr(ckpt_np, "params"):
        params = ckpt_np.params
    else:
        params = ckpt_np

    print(f"Saving params to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(params, f)

    # Verify
    with open(output_path, "rb") as f:
        loaded = pickle.load(f)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(loaded))
    print(f"Done. {n_params:,} parameters saved.")


if __name__ == "__main__":
    main()
