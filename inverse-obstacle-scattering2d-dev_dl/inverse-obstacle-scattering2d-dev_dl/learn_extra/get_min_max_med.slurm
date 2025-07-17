#!/bin/bash

#SBATCH --job-name=inverse_wrapper       # Job name
#SBATCH --output=logs/JCP_inv__%j.out  # Output file. %j expands to job ID
#SBATCH --error=logs/JCP_inv__%j.err   # Error file. %j expands to job ID
#SBATCH --time=00:05:00            # Time limit hrs:min:sec
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --mem=4G                  # Total memory limit
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu

cd /lustre/fs1/home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn


# Load MATLAB modules

module load anaconda/anaconda-2023.09

module load python/python-3.11.4-gcc-12.2.0

module load matlab

# Ensure path is correctly set, compile C files, and run MATLAB scripts
matlab -nodisplay -nosplash -nodesktop -r "run('startup.m'); run('step_stats.m'); exit;"

deactivate
