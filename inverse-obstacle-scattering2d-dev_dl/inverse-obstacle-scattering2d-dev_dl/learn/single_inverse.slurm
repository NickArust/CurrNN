#!/bin/bash

#SBATCH --job-name=inverse_wrapper
#SBATCH --output=logs/single_inv/inv_%a.out
#SBATCH --error=logs/single_inv/inv_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu
#SBATCH --gres=gpu:1


module load anaconda/anaconda-2023.09 matlab
cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn
conda activate myenv

# build a sorted list of every model_* subdirectory


# pass it into MATLAB as a function argument
matlab -nodisplay -nosplash -nodesktop -r "\
    run('startup.m'); \
    inverse_wrapper2('star20_kh1_100_30_n100_80000_noise0'); \
    exit;"

conda deactivate
