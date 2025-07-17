#!/bin/bash

#SBATCH --job-name=matlab_test      # Job name
#SBATCH --output=matlab_test_%j.out # Output file. %j expands to job ID
#SBATCH --error=matlab_test_%j.err  # Error file. %j expands to job ID
#SBATCH --time=00:10:00             # Time limit hrs:min:sec
#SBATCH --cpus-per-task=1           # Number of CPU cores per task
#SBATCH --mem=2GB                   # Total memory limit

# Load MATLAB module
module load matlab

# Run the MATLAB script
matlab -nodisplay -nosplash -nodesktop -r "run('test_mat.m'); exit;"
