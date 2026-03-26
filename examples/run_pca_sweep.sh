#!/bin/bash
set -e

for PCA_DIMS in 2 4 8 16; do
    echo "=========================================="
    echo "Starting run with pca_dims=${PCA_DIMS}"
    echo "=========================================="

    PYTHONUNBUFFERED=1 python3 examples/maze_plr.py \
        --project JAXUED_VAE_COMPARISON \
        --num_updates 10000 --eval_freq 250 \
        --use_cmaes --score_function sfl \
        --cmaes_sigma_min 0.1 \
        --cmaes_kl_threshold 0.01 \
        --cmaes_pca_dims ${PCA_DIMS} --cmaes_pca_fitness_aware \
        --cmaes_pca_buffer_only \
        --warmstart_checkpoint /tmp/warmstart/models \
        --warmstart_buffer /tmp/warmstart/buffer_dump_10k.npz \
        --warmstart_updates 10000 \
        --vae_checkpoint_path /tmp/vae_beta10/checkpoint_500000.pkl \
        --vae_config_path /tmp/vae_beta10/config.yaml \
        --gcs_bucket ucl-ued-project-bucket --gcs_prefix accel \
        --seed 1 --run_name "cmaes_warmstart_fitness_pca${PCA_DIMS}_kl"

    echo "Finished pca_dims=${PCA_DIMS}"
    echo ""
done

echo "All runs complete."
