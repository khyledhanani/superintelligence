#!/bin/bash
set -e

for SEED in 1 2 3; do
    echo "=========================================="
    echo "Starting: free CMA-ES -> PCA after 10k, pca_dims=20, seed=${SEED}"
    echo "=========================================="

    PYTHONUNBUFFERED=1 python3 examples/maze_plr.py \
        --project JAXUED_VAE_COMPARISON \
        --num_updates 30000 --eval_freq 250 \
        --use_cmaes --score_function sfl \
        --cmaes_sigma_min 0.1 \
        --cmaes_kl_threshold 0.01 \
        --cmaes_kl_data /tmp/train_1M_envs.npy \
        --cmaes_kl_samples 20000 \
        --cmaes_pca_dims 20 \
        --cmaes_pca_buffer_only \
        --cmaes_pca_refit_interval 40 \
        --cmaes_pca_start_after 10000 \
        --vae_checkpoint_path /tmp/vae_beta10/checkpoint_500000.pkl \
        --vae_config_path /tmp/vae_beta10/config.yaml \
        --gcs_bucket ucl-ued-project-bucket --gcs_prefix accel \
        --seed ${SEED} --run_name "cmaes_pca20_start10k_refit10k_kl_s${SEED}"

    echo "Finished seed=${SEED}"
    echo ""
done

echo "All seeds complete."
