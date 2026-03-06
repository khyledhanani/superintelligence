# Latent Analysis on vae_og Dataset

## Setup
- Dataset root: `/Users/khyledhanani/Documents/superintelligence/vae/datasets/vae_og`
- ES module root: `/Users/khyledhanani/Documents/superintelligence`
- Sampled 18000 examples (15000 train, 3000 val)
- CLUTTR checkpoint: `/Users/khyledhanani/Documents/superintelligence/vae/model/checkpoint_420000.pkl`
- MazeAE checkpoint: `/Users/khyledhanani/Documents/openend/superintelligence/vae/model_maze_ae/checkpoint_final.pkl`

## Cross-Model Geometry
- Pairwise distance Pearson: **0.116** (from 199988 sampled pairs)
- Pairwise distance Spearman: **0.106**
- kNN overlap@10: **0.024**
- Top-1 neighbor agreement: **0.018**

## PCA Variance
- CLUTTR: PC1=0.043, PC2=0.042
- MazeAE: PC1=0.020, PC2=0.020

## Strongest Structural Alignments
- CLUTTR (all-sample metrics):
  - `wall_count` best dim 24 corr=-0.697
  - `free_count` best dim 24 corr=0.697
  - `branch_points` best dim 24 corr=0.695
  - `dead_ends` best dim 24 corr=-0.545
  - `manhattan` best dim 29 corr=-0.071
- MazeAE (all-sample metrics):
  - `wall_count` best dim 11 corr=-0.231
  - `free_count` best dim 11 corr=0.231
  - `branch_points` best dim 11 corr=0.230
  - `dead_ends` best dim 55 corr=-0.192
  - `manhattan` best dim 32 corr=0.075
- CLUTTR (BFS subset metrics):
  - `path_slack_subset` best dim 24 corr=-0.174
  - `bfs_path_len_subset` best dim 27 corr=-0.088
- MazeAE (BFS subset metrics):
  - `bfs_path_len_subset` best dim 8 corr=0.090
  - `path_slack_subset` best dim 60 corr=-0.082

## Artifacts
- `summary.json`
- `dataset_metric_summary.csv`
- `latent_norm_summary.csv`
- `rank_shift_exemplars.csv`

Interpretation note: correlations identify dominant axes in the chosen sample, not causal factors.
