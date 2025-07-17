#!/bin/bash

#SBATCH --job-name=make_files      # Job name
#SBATCH --output=make_files__%j.out  # Output file. %j expands to job ID
#SBATCH --error=make_files_%j.err   # Error file. %j expands to job ID
#SBATCH --time=00:05:00            # Time limit hrs:min:sec
#SBATCH --cpus-per-task=2          # Number of CPU cores per task
#SBATCH --mem=4G                  # Total memory limit
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu

cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn

# Load MATLAB module
module load matlab

# Ensure path is correctly set and run MATLAB scripts
matlab -nodisplay -nosplash -nodesktop -r "compile_c_files;  exit;"

