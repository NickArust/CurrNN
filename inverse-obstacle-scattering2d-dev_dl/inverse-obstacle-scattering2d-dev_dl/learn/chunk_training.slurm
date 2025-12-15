#!/bin/bash
#SBATCH --job-name=CurrNN_train_chunked
#SBATCH --output=logs/chunk_train/train_%j.out
#SBATCH --error=logs/chunk_train/train_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=7-12:30:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu

set -eo pipefail

###############################################################################
# 0) basic env / threading sanity
###############################################################################
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export TMPDIR="${TMPDIR:-/tmp/${USER}_currnn_${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
mkdir -p logs

###############################################################################
# 1) Load modules / activate env
###############################################################################
module load anaconda/anaconda-2023.09

cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn

# conda's Qt activation scripts sometimes reference unset vars; avoid set -u crash
set +u
conda activate myenv
set -u

which python
python --version

###############################################################################
# 2) Run settings
###############################################################################
DATA_DIR="./data/star20_kh1_100_30_n100_80000_noise0"
MODEL_NAME="test"

EPOCHS_PER_K=10

K_START=0
K_END=29

CHUNK_FILES=2
NUM_WORKERS=0
SAVE_EVERY_EPOCHS=25

###############################################################################
# 3) Auto-resume from latest checkpoint if it exists
###############################################################################
CKPT_PATH="${DATA_DIR}/${MODEL_NAME}/checkpoints/ckpt_latest.pt"

if [[ -f "$CKPT_PATH" ]]; then
  echo "[INFO] Found checkpoint: $CKPT_PATH"
  echo "[INFO] Resuming training..."
  python chunk_train.py \
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
  python chunk_train.py \
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

###############################################################################
# 4) Clean up (optional)
###############################################################################
set +u
conda deactivate
set -u

