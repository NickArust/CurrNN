#!/bin/bash
#SBATCH --job-name=aggregate_inverse
#SBATCH --output=logs/pipeline/aggregate_%j.out
#SBATCH --error=logs/pipeline/aggregate_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
module load matlab

PROJECT_DIR="$SLURM_SUBMIT_DIR"
DATASET_DIR="$PROJECT_DIR/data/star20_kh1_100_15_n100_60000_noise0__60k_20260726_180341"

cd "$PROJECT_DIR"

matlab -nodisplay -nosplash -nodesktop -r "\
addpath('automation_pipeline/analysis'); \
aggregate_inversion_results('$DATASET_DIR', { 'all-frequencies-mse','equal-mse', 'equal-new-loss', 'up-mse', 'up-new-loss','down-mse', 'down-new-loss'} );exit;"
