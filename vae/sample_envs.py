import jax
import jax.numpy as jnp
import numpy as np
import argparse
import os
import yaml

def generate_cluttr_batch_jax(key, batch_size, max_obs=50, inner_dim=13):
    total_cells = inner_dim ** 2  # 169

    # 1. Sample number of obstacles for each batch item
    key, subkey = jax.random.split(key)
    num_obs_batch = jax.random.randint(subkey, (batch_size,), 0, max_obs + 1)

    # 2. Sample unique indices (1-169)
    def sample_one(k):
        return jax.random.permutation(k, jnp.arange(1, total_cells + 1))

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, batch_size)
    shuffled_indices = jax.vmap(sample_one)(keys)

    # 3. Construct sequences using masking
    def construct_seq(indices, n_obs):
        agent_idx = indices[-1]
        goal_idx = indices[-2]
        obs_pool = indices[:max_obs]

        mask = jnp.arange(max_obs) < n_obs
        obs_masked = jnp.where(mask, obs_pool, 0)
        obs_sorted = jnp.sort(obs_masked)

        return jnp.concatenate([obs_sorted, jnp.array([goal_idx, agent_idx])])

    sequences = jax.vmap(construct_seq)(shuffled_indices, num_obs_batch)
    return key, sequences

def main():
    parser = argparse.ArgumentParser(description="Generate CLUTTR environment sequences using JAX.")
    parser.add_argument("--num_envs", type=int, default=20000, help="Number of environments to generate.")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for JAX generation (to manage VRAM).")
    parser.add_argument("--max_obs", type=int, default=50, help="Maximum number of obstacles.")
    parser.add_argument("--inner_dim", type=int, default=13, help="Dimension of the inner grid (e.g., 13 for a 15x15 total grid).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--filename", type=str, default="sample_envs.npy", help="Environments file.")

    args = parser.parse_args()
    
    # Open the config file
    with open('vae_train_config.yml', 'r') as f:
        CONFIG = yaml.safe_load(f)

    output_dir = os.path.join(CONFIG["working_path"],
                              CONFIG["vae_folder"])
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    key = jax.random.PRNGKey(args.seed)
    
    # Calculate how many full batches we need
    num_batches = args.num_envs // args.batch_size
    remainder = args.num_envs % args.batch_size
    
    all_sequences = []

    print(f"Generating {args.num_envs} environments...")

    for i in range(num_batches):
        key, batch = generate_cluttr_batch_jax(key, args.batch_size, args.max_obs, args.inner_dim)
        all_sequences.append(np.array(batch))
        if (i + 1) % 5 == 0:
            print(f"Generated {(i + 1) * args.batch_size} sequences...")

    if remainder > 0:
        key, batch = generate_cluttr_batch_jax(key, remainder, args.max_obs, args.inner_dim)
        all_sequences.append(np.array(batch))

    # Concatenate and save
    final_data = np.concatenate(all_sequences, axis=0)
    #save_path = os.path.join(output_dir, CONFIG["data_path"])
    save_path = os.path.join(output_dir, 'datasets', )
    np.save(save_path, final_data)

    print(f"\nSuccess! Saved dataset of shape {final_data.shape} to {save_path}")

if __name__ == "__main__":
    main()