#!/bin/bash

#SBATCH --job-name=CurrNN_training
#SBATCH --output=logs/retrain_%j.out   # %j = job ID
#SBATCH --error=logs/retrain_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu

###############################################################################
# 1) Load any necessary modules
###############################################################################
module load anaconda/anaconda-2023.09

###############################################################################
# 2) Move into your working directory & activate your conda environment
###############################################################################
cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn

# Activate the environment we created. Make sure this environment is created 
# (Option A from above) so that it already exists.
conda activate myenv

# Optional: print Python version to confirm you're using the environment's Python
which python
python --version

###############################################################################
# 3) Train
###############################################################################
python train.py \
    --dirname ./data/star10_kh9_10_10_n48_2000_noise0\
    --model_name test

###############################################################################
# 4) Example: Retrain/fine-tune if needed
###############################################################################
: << 'EOF'
python train.py \
    --dirname ./data/star10_khr1000_khi10_n48_2000 \
    --model_name test2 \
    --retrain /home/karustamyan/JCP_Version/inverse-obstacle-scattering2d-dev_dl/learn/data/star10_kh10_n48_2000/test/model.pt
EOF

###############################################################################
# 5) Deactivate environment (optional)
###############################################################################
conda deactivate

