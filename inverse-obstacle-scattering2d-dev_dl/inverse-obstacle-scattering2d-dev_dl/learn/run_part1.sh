#!/bin/bash
#SBATCH --job-name=CurrNN_Part1
#SBATCH --output=logs/chunk_train/part1_%j.out
#SBATCH --error=logs/chunk_train/part1_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00  # Reduced to 5 days (safe for ~4.6 days of work)
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
# Ensure this points to the MULTIPROCESSING script
PYTHON_SCRIPT="chunk_train_new_loss.py" 
DATA_DIR="./data/star20_kh1_100_30_n100_80000_noise0"
MODEL_NAME="new_loss" # I changed this so you don't overwrite your previous test

EPOCHS_PER_K=70  # As requested
K_START=0
K_END=14         # First Half

CHUNK_FILES=16
NUM_WORKERS=8    # Critical for speed
SAVE_EVERY_EPOCHS=25

# --- STANDARD RUN BLOCK (Resumes if crashed, starts fresh otherwise) ---
CKPT_PATH="${DATA_DIR}/${MODEL_NAME}/checkpoints/ckpt_latest.pt"

if [[ -f "$CKPT_PATH" ]]; then
  echo "[INFO] Found checkpoint: $CKPT_PATH"
  echo "[INFO] Resuming training..."
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
  echo "[INFO] No checkpoint found. Starting fresh..."
  python "$PYTHON_SCRIPT" \
    --dirname "$DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --epochs "$EPOCHS_PER_K" \
    --k_start "$K_START" \
    --k_end "$K_END" \
    --chunk_files "$CHUNK_FILES" \
    --num_workers "$NUM_WORKERS" \
    --save_every_epochs "$SAVE_EVERY_EPOCHS" \
    --shuffle_files
fi
