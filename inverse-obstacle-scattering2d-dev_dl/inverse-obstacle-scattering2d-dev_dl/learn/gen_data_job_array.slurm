#!/bin/bash

#SBATCH --job-name=gen_data_array     # Job name
#SBATCH --array=1-4                   # CRITICAL: Create 4 jobs (IDs 1, 2, 3, 4)
#SBATCH --output=logs/gen_data/job_%A_%a.out  # %A=Job ID, %a=Array ID (e.g., job_123_1.out)
#SBATCH --error=logs/gen_data/job_%A_%a.err
#SBATCH --time=36:00:00               # Increased to 36 hours (Safety margin for ~28hr runtime)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17            # 16 Workers + 1 Controller
#SBATCH --mem=32G                     # 32GB RAM is plenty
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu

###############################################################################
# 1) Load Modules & Cleanup
###############################################################################
module load anaconda/anaconda-2023.09
module load matlab

# Clean up stale MATLAB locks (Critical for Lustre)
rm -rf ~/.MathWorks/ServiceHost
rm -rf ~/.matlab/local_cluster_jobs

# Explicit unset
unset MW_DISABLE_SERVICE_HOST

###############################################################################
# 2) Environment Setup
###############################################################################
cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn

# Activate environment
conda activate myenv

###############################################################################
# 3) Run MATLAB with Array ID
###############################################################################
# We pass ${SLURM_ARRAY_TASK_ID} into the function.
# This variable will be 1 for the first job, 2 for the second, etc.

echo "Starting Job Array Task ID: ${SLURM_ARRAY_TASK_ID}"

matlab -nodisplay -nosplash -nodesktop -r "startup; starn_forward_singlefreq_edited(${SLURM_ARRAY_TASK_ID}); exit;"

conda deactivate
