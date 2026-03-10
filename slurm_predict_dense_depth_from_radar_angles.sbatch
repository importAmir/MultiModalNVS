#!/bin/bash
# Example Slurm script for running sharded dense-depth prediction on a cluster.
#
# This launches NUM_SHARDS tasks (processes). Each task computes a disjoint shard
# and writes an output file with suffix `.shardXXXofYYY.npz`.
#
# After the array/job completes, run the merge command once to combine shards.
#
# Adjust paths/partitions/modules to your cluster.

#SBATCH --job-name=radar_depth_sharded
#SBATCH --output=logs/%x_%j_%t.out
#SBATCH --error=logs/%x_%j_%t.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#
# If you want GPU acceleration, uncomment and set resources:
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu
#
# If you want job arrays for sharding:
#SBATCH --array=0-31

set -euo pipefail

NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD_RANK="${SLURM_ARRAY_TASK_ID:-0}"

# Optional: control BLAS threading to avoid oversubscription when you shard.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Activate your environment (edit):
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

POINTS_DIR="/path/to/radar/training/velodyne"
CALIB_DIR="/path/to/radar/training/calib"
INDEX="750"

OUT_DIR="/path/to/predictions"
mkdir -p "${OUT_DIR}" logs

# Target image resolution (edit to match your dataset camera images):
IMAGE_H="1248"
IMAGE_W="1920"

python3 predict_dense_depth.py \
  --points-bin "${POINTS_DIR}/$(printf '%05d' "${INDEX}").bin" \
  --calib-txt "${CALIB_DIR}/$(printf '%05d' "${INDEX}").txt" \
  --image-size "${IMAGE_H}" "${IMAGE_W}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-rank "${SHARD_RANK}" \
  --backend numpy \
  --num-workers 0 \
  --stride 1 \
  --output "${OUT_DIR}/depth_pred_${INDEX}.npz"

# For GPU backend (requires PyTorch + GPU nodes), switch to:
#   --backend torch --device cuda
#
# Then each task uses its local GPU via SLURM_LOCALID/LOCAL_RANK.

# Merge shards after the job finishes (run once, not in the array):
#
# python3 predict_dense_depth.py \
#   --merge-shards \
#   --shard-files "${OUT_DIR}/depth_pred_${INDEX}.npz".shard*of*.npz \
#   --merged-output "${OUT_DIR}/depth_pred_${INDEX}_merged.npz"

