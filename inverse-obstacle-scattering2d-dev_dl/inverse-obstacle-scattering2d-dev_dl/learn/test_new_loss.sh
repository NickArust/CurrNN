#!/bin/bash
#SBATCH --job-name=CurrNN_Test_Wiggles
#SBATCH --output=logs/chunk_train/test_%j.out
#SBATCH --error=logs/chunk_train/test_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00 
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
PYTHON_SCRIPT="chunk_train_new_loss.py"
DATA_DIR="./data/star20_kh1_100_30_n100_80000_noise0"
MODEL_NAME="test_full_run" 

# --- FREQUENCY SETTINGS ---
# We want to TRAIN at k=25.
# So we load the weights from the end of k=24.
K_START=25  
K_END=25    # Just run this one frequency step for the test
EPOCHS_PER_K=50
CHUNK_FILES=16
NUM_WORKERS=8     
SAVE_EVERY_EPOCHS=10 

# --- RESUME FROM PREVIOUS K ---
# Instead of ckpt_latest, we target the specific file from the previous step.
# Make sure model_k24.pt exists in your folder!
PREV_K=$((K_START - 1))
RELATIVE_CKPT_PATH="${DATA_DIR}/${MODEL_NAME}/model_k${PREV_K}.pt"
CKPT_PATH=$(readlink -f "$RELATIVE_CKPT_PATH")
if [[ -f "$CKPT_PATH" ]]; then
  echo "[INFO] Found specific checkpoint: $CKPT_PATH"
  echo "[INFO] Rewinding to k=$PREV_K and starting training at k=$K_START with Spectral Loss..."
  
  python "$PYTHON_SCRIPT" \
    --dirname "$DATA_DIR" \
    --model_name "$MODEL_NAME" \
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
  echo "Please check your directory to see which 'model_k*.pt' files exist."
  echo "If you only have ckpt_latest.pt, change the script to use that, but 'model_k' is better."
  exit 1
fi
