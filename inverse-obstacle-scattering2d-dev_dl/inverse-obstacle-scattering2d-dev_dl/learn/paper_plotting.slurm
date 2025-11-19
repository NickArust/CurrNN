#!/bin/bash

#SBATCH --job-name=inverse_wrapper
#SBATCH --output=logs/ps_inv/inv_%j.out
#SBATCH --error=logs/ps_inv/inv_%j.err
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu

module load anaconda/anaconda-2023.09 matlab
cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn
conda activate myenv

# build a sorted list of every model_* subdirectory

# pass it into MATLAB as a function argument
matlab -nodisplay -nosplash -nodesktop -r "\
    run('startup.m'); \
    S = load('src_info_33.mat'); \
    for_plotting('data/ps/star10_kh9_10_10_n48_1000_noise0/model_2_60/',0,0,S.src_info); \
    exit;"

conda deactivate
