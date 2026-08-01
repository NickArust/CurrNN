#!/bin/bash
#SBATCH --job-name=currnn_analysis
#SBATCH --output=logs/pipeline/analysis_%j.out
#SBATCH --error=logs/pipeline/analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

module load matlab
cd "$SLURM_SUBMIT_DIR"
matlab -nodisplay -nosplash -nodesktop -r "addpath('automation_pipeline/analysis'); run_analysis; exit;"
