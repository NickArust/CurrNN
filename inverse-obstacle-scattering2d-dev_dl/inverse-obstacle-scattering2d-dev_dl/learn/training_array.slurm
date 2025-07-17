#!/bin/bash
#SBATCH --job-name=ml_train
#SBATCH --output=logs/ps_training/train_%a.out
#SBATCH --error=logs/ps_training/train_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-799%10    # 800 configs × 3 epoch‐multipliers = 2400

# 1) build list of your 800 datagen config files
mapfile -t CONFIGS < <(ls configs/data_gen_ps/nc*_ndata*_ks*_dk*_noise*.json | sort)

# 2) list your epoch‐multipliers
MUL=(1.0)

N=${#CONFIGS[@]}        # should be 800

# 3) decode which config + multiplier this task is
idx=$SLURM_ARRAY_TASK_ID
mult_idx=$(( idx / N ))
conf_idx=$(( idx % N ))

CONFIG_FILE=${CONFIGS[$conf_idx]}
MULT=${MUL[$mult_idx]}

# 4) parse fields out of the filename
base=$(basename "$CONFIG_FILE" .json)
# base="nc10_ndata1000_ks6_dk0.1_noise0.0"
IFS='_' read -r nc ndata ks dk noise <<< "$base"
nc=${nc#nc}         # "10"
ndata=${ndata#ndata}   # "1000"
ks=${ks#ks}         # "6"
dk=${dk#dk}         # "0.1"
noise=${noise#noise}   # "0.0"

# turn noise=0.1 → noise_pct=10, noise=0.0 → 0, etc.
noise_pct=$(python - <<EOF
print(int(round($noise * 100)))
EOF
)

dk_int=$(python - <<EOF
print(int(round($dk * 100)))
EOF
)

# 5) assemble the data folder name:
#    star{nc}_kh{ks}_{gridX}_{gridY}_n{nSensors}_{ndata}_noise{noise_pct}
#    (you can also pull grid & nSensors from the JSON, but if they're fixed:)
gridX=10
gridY=10
nSensors=48

DATA_FOLDER="data/ps/star${nc}_kh${ks}_${dk_int}_${gridY}_n${nSensors}_${ndata}_noise${noise_pct}"

# 6) compute n_k, base_epoch, epoch, save_every_nepoch exactly as before:
kh_start=$ks   # same thing
dk_val=$dk

n_k=$(python - <<EOF
import math
print(int(round((10 - $kh_start)/$dk_val)) + 1)
EOF
)

base_epoch=$(python - <<EOF
import math
n_k = $n_k
cands = [i for i in range(0, 2000) if i % n_k == 0]
print(min(cands, key=lambda x: abs(x - 1000)))
EOF
)

epoch=$(python - <<EOF
print(int(round($MULT * $base_epoch)))
EOF
)

save_every_nepoch=$(( epoch / n_k ))

echo "$DATA_FOLDER"

echo "$epoch"
echo "$save_every_nepoch"


###############################################################################
# 1) Load any necessary modules
###############################################################################
module load anaconda/anaconda-2023.09

###############################################################################
# 2) Move into your working directory & activate your conda environment
###############################################################################
cd /home/karustamyan/CurrNN_ISP/inverse-obstacle-scattering2d-dev_dl/inverse-obstacle-scattering2d-dev_dl/learn

# Activate the environment we created. Make sure this environment is created 
# (Option A from above) so that it already exists.
conda activate myenv

# Optional: print Python version to confirm you're using the environment's Python
which python
python --version
# 7) run your training
python train.py \
  --model_name=model_${MULT}_${SLURM_ARRAY_TASK_ID} \
  --dirname   "$DATA_FOLDER" \
  --epochs        $epoch \
  --save-every-nepoch $save_every_nepoch
  --kh-start $kh_start


conda deactivate

