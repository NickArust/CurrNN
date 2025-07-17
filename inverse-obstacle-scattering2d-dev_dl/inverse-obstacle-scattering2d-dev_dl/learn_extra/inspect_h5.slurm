#!/bin/bash
#
##SBATCH --job-name=Inspect_MAT_File
#SBATCH --output=inspect_mat_file_%j.out  # %j will be replaced by the job ID
#SBATCH --error=inspect_mat_file_%j.err
#SBATCH --cpus-per-task=1  # Reduced CPUs as file inspection is less intensive
#SBATCH --mem=4G  # Reduced memory requirement as the task is lighter
#SBATCH --time=01:00:00  # Shorter time, adjust based on expected runtime
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu

# Load Python environment
module load anaconda/anaconda3

# Activate the specific Python environment if needed
source /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn/tensorboardenv/bin/activate

# Navigate to the directory containing the inspect script
cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn

# Run the Python script to inspect the .mat file
python3 inspect_h5_files.py


# Deactivate the environment
deactivate

