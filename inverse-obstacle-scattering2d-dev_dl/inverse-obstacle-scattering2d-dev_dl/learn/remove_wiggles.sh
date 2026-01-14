#!/bin/bash
#SBATCH --job-name=CurrNN_Relaxation
#SBATCH --output=logs/chunk_train/relax_%j.out
#SBATCH --error=logs/chunk_train/relax_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=23:59:00        # Reduced time (only 100 epochs needed)
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu

set -eo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TMPDIR="${TMPDIR:-/tmp/${USER}_currnn_relax_${SLURM_JOB_ID}}"
mkdir -p "$TMPDIR"
mkdir -p logs/chunk_train

module load anaconda/anaconda-2023.09
cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn

set +u
conda activate myenv
set -u

# --- SETTINGS ---
PYTHON_SCRIPT="chunk_train_new_loss.py"
# The folder where your "jittery" but successful run lives:
DATA_DIR="./data/star20_kh1_100_30_n100_80000_noise0"
OLD_MODEL_NAME="new_loss"
NEW_MODEL_NAME="relaxation_test"

# --- FREQUENCY SETTINGS ---
# Your logs show training finished at k=29. 
# We want to re-run this final step with NO penalty to fix the phase shift.
K_START=29
K_END=29  
EPOCHS_PER_K=100
CHUNK_FILES=16
NUM_WORKERS=8 
SAVE_EVERY_EPOCHS=10 

# --- RESUME FROM LATEST CHECKPOINT ---
# We load the FINAL weights from the previous run (the jittery ones)
RELATIVE_CKPT_PATH="${DATA_DIR}/${OLD_MODEL_NAME}/model.pt"
CKPT_PATH=$(readlink -f "$RELATIVE_CKPT_PATH")

if [[ -f "$CKPT_PATH" ]]; then
  echo "[INFO] Found jittery checkpoint: $CKPT_PATH"
  echo "[INFO] Starting Relaxation Phase at k=$K_START (Lambda=0.0)..."
  
  python "$PYTHON_SCRIPT" \
    --dirname "$DATA_DIR" \
    --model_name "$NEW_MODEL_NAME" \
    --retrain "$CKPT_PATH" \
    --train_cfg_path "./configs/train_nc20.json" \
    --epochs "$EPOCHS_PER_K" \
    --k_start "$K_START" \
    --k_end "$K_END" \
    --chunk_files "$CHUNK_FILES" \
    --num_workers "$NUM_WORKERS" \
    --save_every_epochs "$SAVE_EVERY_EPOCHS" \
    --shuffle_files

else
  echo "[ERROR] Could not find $CKPT_PATH"
  echo "Please check that your 'new_loss' run actually finished and saved checkpoints."
  exit 1
fi
