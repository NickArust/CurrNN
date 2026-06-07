#!/bin/bash

#SBATCH --job-name=n20k15     # Job name
#SBATCH --output=logs/gen_data/n20k15_%A.out  # %A=Job ID, %a=Array ID (e.g., job_123_1.out)
#SBATCH --error=logs/gen_data/n20k15_%A.err
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

cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn

# Activate environment
conda activate myenv



matlab -nodisplay -nosplash -nodesktop -r "startup; starn_forward_singlefreq; exit;"

conda deactivate
