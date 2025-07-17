#!/bin/bash

#SBATCH --job-name=gen_nc_10       # Job name
#SBATCH --output=logs/%x_%j.out    # Output file. %x expands to job name, %j to job ID
#SBATCH --error=logs/%x_%j.err     # Error file. %x expands to job name, %j to job ID
#SBATCH --time=24:30:00            # Time limit hrs:min:sec
#SBATCH --cpus-per-task=16         # Number of CPU cores per task
#SBATCH --mem=64G                  # Total memory limit
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu

cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn

# Load MATLAB module
module load matlab

# Ensure path is correctly set, compile C files, and run MATLAB scripts
matlab -nodisplay -nosplash -nodesktop -r " run('starn_forward_singlefreq'); exit;"

