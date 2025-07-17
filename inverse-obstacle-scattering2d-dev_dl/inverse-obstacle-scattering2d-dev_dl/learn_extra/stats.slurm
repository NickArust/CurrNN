#!/bin/bash

#SBATCH --job-name=stats_inv       # Job name
#SBATCH --output=logs/stats_%j.out  # Output file. %j expands to job ID
#SBATCH --error=logs/stats_inv_%j.err   # Error file. %j expands to job ID
#SBATCH --time=00:30:00            # Time limit hrs:min:sec
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=32G                  # Total memory limit
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu
#SBATCH --gres=gpu:1
cd /lustre/fs1/home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn/


# Load MATLAB module

module load anaconda/anaconda-2023.09

module load python/python-3.11.4-gcc-12.2.0

source ./tensorboardenv/bin/activate

module load matlab

# Ensure path is correctly set, compile C files, and run MATLAB scripts
matlab -nodisplay -nosplash -nodesktop -r "run('startup.m'); run('stats_script.m'); exit;"

deactivate
