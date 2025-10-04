#!/bin/bash

#SBATCH --job-name=inverse_wrapper
#SBATCH --output=logs/ps_inv/inv_%a.out
#SBATCH --error=logs/ps_inv/inv_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu
#SBATCH --gres=gpu:1
#SBATCH --array=0-760%10

module load anaconda/anaconda-2023.09 matlab
cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn
conda activate myenv

# build a sorted list of every model_* subdirectory
mapfile -t MODEL_DIRS < <(
  find data/ps -mindepth 2 -maxdepth 2 -type d -name 'model_*' | sort
)
OFFSET=951

# pick the one for this array task
TASK_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))
TARGET_DIR=${MODEL_DIRS[$TASK_ID]}

echo "[$(date)] Inverting in: $TARGET_DIR"


# pass it into MATLAB as a function argument
matlab -nodisplay -nosplash -nodesktop -r "\
    run('startup.m'); \
    inverse_wrapper('$TARGET_DIR','$SLURM_ARRAY_TASK_ID'); \
    exit;"

conda deactivate
