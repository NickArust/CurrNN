#!/bin/bash

#SBATCH --job-name=15_trial_5_15_training
#SBATCH --output=logs/trial_5_15_nc15_my_training_job_%j.out  # %j will be replaced by the job ID
#SBATCH --error=logs/trial_5_15_nc15_my_training_job_%j.err
#SBATCH --cpus-per-task=64
#SBATCH	--nodes=1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=06:30:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ni406436@ucf.edu

module load anaconda/anaconda-2023.09


cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn
source ./tensorboardenv/bin/activate


# cd to the right directory
cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn

# Run the training script
python3 train.py --dirname './data/star15_kh25_n100_20000'

deactivate
