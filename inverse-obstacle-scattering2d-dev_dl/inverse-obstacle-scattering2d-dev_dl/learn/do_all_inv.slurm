#!/bin/bash

#SBATCH --job-name=inverse_wrapper       # Job name
#SBATCH --output=logs/all10_inv__%j.out  # Output file. %j expands to job ID
#SBATCH --error=logs/all10_inv__%j.err   # Error file. %j expands to job ID
#SBATCH --time=05:00:00            # Time limit hrs:min:sec
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --mem=64G                  # Total memory limit
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu
#SBATCH --gres=gpu:1

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


module load matlab

# Ensure path is correctly set, compile C files, and run MATLAB scripts
matlab -nodisplay -nosplash -nodesktop -r "run('startup.m'); run('inverse_wrapper.m'); exit;"

conda deactivate
