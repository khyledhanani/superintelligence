#!/bin/bash
# Generate t-SNE embedding evolution plots for ACCEL vs CMA-ES runs.
# Produces both behavioral (257D agent) and structural (173D env) plots.
#
# Behavioral embeddings require GPU for first run. Structural embeddings
# are CPU-only (no agent rollout needed).
#
# Usage:
#   bash vae/run_cmaes_comparison.sh              # full (GPU + GCS)
#   bash vae/run_cmaes_comparison.sh --cache_only  # cached only (CPU)

cd /cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued

EXTRA_ARGS="$@"
CACHE_BEHAV=/cs/student/project_msc/2025/csml/rhautier/embedding_caches/cmaes_compare
CACHE_ENV=/cs/student/project_msc/2025/csml/rhautier/embedding_caches/cmaes_compare_env
PLOTS=vae/plots

# === Behavioral embeddings (257D, requires GPU) ===
echo "============================="
echo "  BEHAVIORAL EMBEDDINGS"
echo "============================="

echo "=== ACCEL ==="
python vae/plot_tsne_compare_runs.py \
    --config vae/compare_accel.yaml \
    --cache_dir $CACHE_BEHAV \
    --output $PLOTS/tsne_accel.png \
    $EXTRA_ARGS

echo ""
echo "=== CMA-ES + ACCEL mutations ==="
python vae/plot_tsne_compare_runs.py \
    --config vae/compare_cmaes_accel.yaml \
    --cache_dir $CACHE_BEHAV \
    --output $PLOTS/tsne_cmaes_accel.png \
    $EXTRA_ARGS

echo ""
echo "=== CMA-ES + Latent mutations ==="
python vae/plot_tsne_compare_runs.py \
    --config vae/compare_cmaes_latent.yaml \
    --cache_dir $CACHE_BEHAV \
    --output $PLOTS/tsne_cmaes_latent.png \
    $EXTRA_ARGS

# === Structural embeddings (173D, CPU only) ===
echo ""
echo "============================="
echo "  STRUCTURAL EMBEDDINGS"
echo "============================="

echo "=== ACCEL ==="
python vae/plot_tsne_compare_env.py \
    --config vae/compare_accel.yaml \
    --cache_dir $CACHE_ENV \
    --output $PLOTS/tsne_env_accel.png

echo ""
echo "=== CMA-ES + ACCEL mutations ==="
python vae/plot_tsne_compare_env.py \
    --config vae/compare_cmaes_accel.yaml \
    --cache_dir $CACHE_ENV \
    --output $PLOTS/tsne_env_cmaes_accel.png

echo ""
echo "=== CMA-ES + Latent mutations ==="
python vae/plot_tsne_compare_env.py \
    --config vae/compare_cmaes_latent.yaml \
    --cache_dir $CACHE_ENV \
    --output $PLOTS/tsne_env_cmaes_latent.png

# === Upload to GCS ===
echo ""
echo "=== Uploading to GCS ==="
python3 << 'PYEOF'
from google.cloud import storage
import os, glob

client = storage.Client(project="open-endedness-ued-project")
bucket = client.bucket("ucl-ued-project-bucket")
gcs_base = "llm-exp/embedding_caches/cmaes_compare"

# Upload behavioral cache
n = 0
for f in sorted(glob.glob("/cs/student/project_msc/2025/csml/rhautier/embedding_caches/cmaes_compare/*.npz")):
    bucket.blob(f"{gcs_base}/behavioral/{os.path.basename(f)}").upload_from_filename(f)
    n += 1
print(f"Uploaded {n} behavioral cache files")

# Upload structural cache
n = 0
for f in sorted(glob.glob("/cs/student/project_msc/2025/csml/rhautier/embedding_caches/cmaes_compare_env/*.npz")):
    bucket.blob(f"{gcs_base}/structural/{os.path.basename(f)}").upload_from_filename(f)
    n += 1
print(f"Uploaded {n} structural cache files")

# Upload plots
for name in ["tsne_accel.png", "tsne_cmaes_accel.png", "tsne_cmaes_latent.png",
             "tsne_env_accel.png", "tsne_env_cmaes_accel.png", "tsne_env_cmaes_latent.png"]:
    path = os.path.join("vae/plots", name)
    if os.path.exists(path):
        bucket.blob(f"{gcs_base}/plots/{name}").upload_from_filename(path)
        print(f"Uploaded {name}")
PYEOF

echo ""
echo "Done. 6 plots saved to $PLOTS/ and uploaded to GCS."
