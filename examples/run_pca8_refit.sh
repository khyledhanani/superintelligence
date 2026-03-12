#!/bin/bash
set -e

for SEED in 1 2 3; do
    echo "=========================================="
    echo "Starting: pca_dims=8, refit every 2000 updates, seed=${SEED}"
    echo "=========================================="

    PYTHONUNBUFFERED=1 python3 examples/maze_plr.py \
        --project JAXUED_VAE_COMPARISON \
        --num_updates 10000 --eval_freq 250 \
        --use_cmaes --score_function sfl \
        --cmaes_sigma_min 0.1 \
        --cmaes_kl_threshold 0.01 \
        --cmaes_pca_dims 8 --cmaes_pca_fitness_aware \
        --cmaes_pca_buffer_only \
        --cmaes_pca_refit_interval 8 \
        --warmstart_checkpoint /tmp/warmstart/models \
        --warmstart_buffer /tmp/warmstart/buffer_dump_10k.npz \
        --warmstart_updates 10000 \
        --vae_checkpoint_path /tmp/vae_beta10/checkpoint_500000.pkl \
        --vae_config_path /tmp/vae_beta10/config.yaml \
        --gcs_bucket ucl-ued-project-bucket --gcs_prefix accel \
        --seed ${SEED} --run_name "cmaes_warmstart_fitness_pca8_refit2k_kl_s${SEED}"

    echo "Finished seed=${SEED}"
    echo ""
done

echo "All seeds complete."
