#!/bin/bash

#SBATCH --job-name=gen_data       # Job name
#SBATCH --output=logs/gen_data/gen_data_%j.out  # Output file. %j expands to job ID
#SBATCH --error=logs/gen_data/gen_data_%j.err   # Error file. %j expands to job ID
#SBATCH --time=12:30:00  # Time limit hrs:min:sec
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2          # Number of CPU cores per task
#SBATCH --mem=64G                  # Total memory limit
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

export MW_DISABLE_SERVICE_HOST=1

# Ensure path is correctly set, compile C files, and run MATLAB scripts
matlab -nodisplay -nosplash -nodesktop -r  "run('starn_forward_singlefreq.m'); exit;"

conda deactivate
