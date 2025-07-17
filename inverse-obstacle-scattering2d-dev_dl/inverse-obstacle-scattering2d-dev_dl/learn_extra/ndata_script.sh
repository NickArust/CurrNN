#!/bin/bash

# Check if nc value is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <nc_value>"
    exit 1
fi

nc=$1
original_config="configs/nc${nc}-25.json"

# Percentages
declare -a percentages=(50 60 70 80 90 100)

# Get max ndata based on nc value
case $nc in
    3) max_ndata=100 ;;
    5) max_ndata=500 ;;
    10) max_ndata=2000 ;;
    15) max_ndata=20000 ;;
    *) echo "Invalid nc value"; exit 1 ;;
esac

# Iterate through each percentage
for pct in "${percentages[@]}"; do
    ndata=$(( max_ndata * pct / 100 ))
    temp_config="configs/temp_nc${nc}_pct${pct}.json"

    # Create a temporary config file
    cp "$original_config" "$temp_config"
    sed -i "s/\"ndata\": [0-9]*/\"ndata\": $ndata/" "$temp_config"

    # Submit the SLURM job, passing the temp config file and percentage as mat_id
    echo "Submitting SLURM job for ${pct}% with temp config: $temp_config"
    sbatch --export=CONFIG_FILE="$temp_config",MAT_ID=0 gen_data.slurm
done

