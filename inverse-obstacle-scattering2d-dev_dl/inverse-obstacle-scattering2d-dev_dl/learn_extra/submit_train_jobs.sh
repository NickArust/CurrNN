#!/bin/bash

# List of dirnames (you can provide this list or read it from a file)
dirnames=(
    "./data/star3_kh10_n48_50"
    "./data/star3_kh10_n48_60"
    "./data/star3_kh10_n48_70"
    "./data/star3_kh10_n48_80"
    "./data/star3_kh10_n48_90"
    "./data/star3_kh10_n48_100"
    )

# Define the prefix
prefix="trial_5_3"  # Change this value as needed

# Loop over each dirname
for dirname in "${dirnames[@]}"; do
    # Extract 'nc' value from the dirname (assumes 'star<number>_...')
    nc=$(basename "$dirname" | grep -oP 'star\K\d+')

    # Create a new SLURM script for this run
    slurm_script="train_nc${nc}_job.slurm"
    cp train_slurm.slurm "$slurm_script"

    # Modify the --dirname in the SLURM script
    sed -i "s|--dirname './data/[^']*'|--dirname '$dirname'|" "$slurm_script"

    # Modify the job name to {nc}_{prefix}_training
    sed -i "s|#SBATCH --job-name=.*|#SBATCH --job-name=${nc}_${prefix}_training|" "$slurm_script"

    # Modify the output and error log paths to include the nc value and prefix number
    sed -i "s|logs/70_nc15_my_training_job_%j.out|logs/${prefix}_nc${nc}_my_training_job_%j.out|" "$slurm_script"
    sed -i "s|logs/70_nc15_my_training_job_%j.err|logs/${prefix}_nc${nc}_my_training_job_%j.err|" "$slurm_script"

    # Submit the modified SLURM script
    sbatch "$slurm_script"

done

