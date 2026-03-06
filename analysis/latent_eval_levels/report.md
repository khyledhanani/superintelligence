# Latent Analysis for New Eval Mazes

## Models
- CLUTTR VAE checkpoint: `/Users/khyledhanani/Documents/openend/superintelligence/vae/model/checkpoint_420000.pkl`
- Secondary model kind: `maze_ae`
- Secondary checkpoint: `/Users/khyledhanani/Documents/openend/superintelligence/vae/model_maze_ae/checkpoint_final.pkl`
- Note: Using MazeAE checkpoint.

## Cross-Model Geometry Agreement
- Spearman correlation of pairwise distances: **0.394**
- PCA variance (CLUTTR): PC1=0.562, PC2=0.164
- PCA variance (Secondary): PC1=0.243, PC2=0.171

## Strongest Structural Alignments
- CLUTTR best metric->latent alignments:
  - `wall_count` with latent dim 6: corr=-0.941
  - `free_count` with latent dim 6: corr=0.941
  - `dead_ends` with latent dim 37: corr=-0.806
  - `manhattan` with latent dim 23: corr=-0.739
  - `bfs_path_len` with latent dim 23: corr=-0.602
- Secondary best metric->latent alignments:
  - `branch_points` with latent dim 20: corr=-0.922
  - `wall_count` with latent dim 33: corr=0.826
  - `free_count` with latent dim 33: corr=-0.826
  - `manhattan` with latent dim 0: corr=0.767
  - `dead_ends` with latent dim 9: corr=0.653

## Levels with Biggest Isolation Rank Shift
- `PerimeterRun`: delta=-10.0, nn_cluttr=`ZigZagTunnel`, nn_second=`SpiralPocket`
- `ParallelCorridors`: delta=+9.0, nn_cluttr=`DeadendFan`, nn_second=`DeadendFan`
- `SpiralPocket`: delta=-8.0, nn_cluttr=`PerimeterRun`, nn_second=`PerimeterRun`
- `ZigZagTunnel`: delta=-7.0, nn_cluttr=`PerimeterRun`, nn_second=`PerimeterRun`
- `CentralChoke`: delta=+6.0, nn_cluttr=`NarrowBridge`, nn_second=`SpiralPocket`

## Artifacts
- `level_latent_summary.csv`
- `pairwise_dist_cluttr.csv`
- `pairwise_dist_second.csv`
- `summary.json`

Interpretation caution: only 15 levels were analyzed, so dimension-level correlations are directional, not definitive.
