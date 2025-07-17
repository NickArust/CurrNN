#!/bin/bash

#SBATCH --output=my_training_job_%j.out  # %j will be replaced by the job ID
#SBATCH --error=my_training_job_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL

# Load the Anaconda module
module load anaconda/anaconda3

# Navigate to the project directory
cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn

# Activate the Python virtual environment
source ./tensorboardenv/bin/activate

# Print the current working directory
pwd

# Run the test script to check tensorboard import
python3 test_tensorboard.py

# Deactivate the virtual environment
deactivate

