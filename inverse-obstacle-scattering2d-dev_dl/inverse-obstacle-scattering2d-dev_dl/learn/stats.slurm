#!/bin/bash

#SBATCH --job-name=stats       # Job name
#SBATCH --output=logs/stats/only_deltak_%j.out  # Output file. %j expands to job ID
#SBATCH --error=logs/stats/only_deltak_%j.err   # Error file. %j expands to job ID
#SBATCH --time=00:30:00            # Time limit hrs:min:sec
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --mem=32G                  # Total memory limit
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


module load matlab

# Ensure path is correctly set, compile C files, and run MATLAB scripts
matlab -nodisplay -nosplash -nodesktop -r  "run('only_change_deltak.m'); exit;"

conda deactivate
