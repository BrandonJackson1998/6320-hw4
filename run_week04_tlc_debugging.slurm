#!/bin/bash
#SBATCH --partition=utucset-gpu-grn
#SBATCH --qos=utucset-gpu-grn
#SBATCH --account=larsenc
#SBATCH --job-name=jackson-hw4-tlc-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-full-%j.out
#SBATCH --error=logs/slurm-full-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
cd "${SCRIPT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

module load uv
module load python/3.12.4
nvidia-smi -L || true

mkdir -p logs outputs outputs/plots

DATA_DIR="${DATA_DIR:-/scratch/general/vast/u0090307/ut-cs6320/datasets}"
FULL_DATA="${FULL_DATA:-${DATA_DIR}/week04_tlc_trip_duration_full.parquet}"

if [[ ! -r "${FULL_DATA}" ]]; then
  echo "${FULL_DATA} not found."
  exit 1
fi

uv run --project . python3 scripts/train_debugging_experiments.py \
  --data "${FULL_DATA}" \
  --output-dir outputs \
  --epochs 12 \
  --batch-size 512 \
  --seed 6320 \
  --device auto

uv run --project . python3 scripts/plot_learning_curves.py \
  --history-dir outputs \
  --output-dir outputs/plots
