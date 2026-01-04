#!/bin/bash
#SBATCH --job-name=CurrNN_Part2
#SBATCH --output=logs/chunk_train/unbalanced_part2_%j.out
#SBATCH --error=logs/chunk_train/unbalanced_part2_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu

set -eo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TMPDIR="${TMPDIR:-/tmp/${USER}_currnn_${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
mkdir -p logs/chunk_train

module load anaconda/anaconda-2023.09
cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn

set +u
conda activate myenv
set -u

# --- SETTINGS ---
PYTHON_SCRIPT="chunk_train2.py"
DATA_DIR="./data/star20_kh1_100_30_n100_80000_noise0"
MODEL_NAME="unbalanced" # Must match Part 1

EPOCHS_PER_K=70
K_START=19       # Second Half
K_END=29

CHUNK_FILES=16
NUM_WORKERS=8
SAVE_EVERY_EPOCHS=25

# --- STANDARD RUN BLOCK ---
# This will automatically find the checkpoint from the end of Part 1
CKPT_PATH="${DATA_DIR}/${MODEL_NAME}/checkpoints/ckpt_latest.pt"

if [[ -f "$CKPT_PATH" ]]; then
  echo "[INFO] Found checkpoint: $CKPT_PATH"
  echo "[INFO] Resuming training (Part 2)..."
  python "$PYTHON_SCRIPT" \
    --dirname "$DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --resume_ckpt "$CKPT_PATH" \
    --epochs "$EPOCHS_PER_K" \
    --k_start "$K_START" \
    --k_end "$K_END" \
    --chunk_files "$CHUNK_FILES" \
    --num_workers "$NUM_WORKERS" \
    --save_every_epochs "$SAVE_EVERY_EPOCHS" \
    --shuffle_files
else
  echo "[ERROR] No checkpoint found from Part 1!"
  exit 1
fi
