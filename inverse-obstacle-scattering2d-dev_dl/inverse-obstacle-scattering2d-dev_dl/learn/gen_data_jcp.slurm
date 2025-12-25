#!/bin/bash

#SBATCH --job-name=gen_nc_10       # Job name
#SBATCH --output=logs/testing/gen_nc_10_%j.out  # Output file. %j expands to job ID
#SBATCH --error=logs/testing/gen_nc_10_%j.err   # Error file. %j expands to job ID
#SBATCH --time=01:00:00            # Time limit hrs:min:sec
#SBATCH --cpus-per-task=16          # Number of CPU cores per task
#SBATCH --mem=64G                  # Total memory limit
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu
#SBATCH --array=0-4


cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn


module load anaconda/anaconda-2023.09
# Load MATLAB module
module load matlab

conda activate myenv

# Calculate noise level (0.0, 0.25, 0.5, 0.75, 1.0)
NOISE_VAL=$(python -c "print($SLURM_ARRAY_TASK_ID / 4)")

echo "Running Job Array Index: $SLURM_ARRAY_TASK_ID"
echo "Passing argument: $NOISE_VAL"

# RUN COMMAND CHANGED:
# We now call the function with the value inside parentheses
matlab -nodisplay -nosplash -nodesktop -r "run('startup.m'); noise_level_input = $NOISE_VAL; run('starn_forward_singlefreq.m'); exit;"
