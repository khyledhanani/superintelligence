#!/bin/bash
# TPU VM setup script: install deps and transfer code for CMA-ES training.
# Run this on the TPU VM after SSH'ing in.
set -e

echo "=== TPU Setup ==="
echo "Hostname: $(hostname)"
echo "Python: $(python3 --version)"

# Enable transparent hugepages (v5e/v6e performance)
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled" 2>/dev/null || true

# Install JAX for TPU (uses pre-installed libtpu from runtime image)
pip install --upgrade 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Verify TPU
python3 -c "
import jax
devices = jax.devices()
print(f'JAX backend: {jax.default_backend()}')
print(f'TPU devices: {len(devices)}')
for d in devices:
    print(f'  {d}')
"

# Install project dependencies
pip install flax distrax optax chex wandb

# Install jaxued from source (not on PyPI)
pip install git+https://github.com/DramaCow/jaxued.git

echo ""
echo "=== Setup complete ==="
echo "Now scp the project code and run training."
