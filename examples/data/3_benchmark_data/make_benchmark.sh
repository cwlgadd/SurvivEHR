#!/bin/bash -l

#SBATCH --account=gokhalkm-optimal
#SBATCH --qos=bbdefault
#SBATCH --time=10:0:0
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36

#SBATCH --output=out/NE_CVD_%A.out
#SBATCH --job-name=regional_run

# ---  ---
REPO_DIR="/rds/homes/g/gaddcz/Projects/SurvivEHR"

set -euo pipefail

echo "$SLURM_JOB_PARTITION"
nvidia-smi || echo "no nvidia-smi"

# --- Diagnostics (optional) ---
echo "HOST=$(hostname)"
echo "OS=$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release)"
echo "CPU=$(lscpu | sed -n 's/^Model name: *//p')"
echo "PARTITION=${SLURM_JOB_PARTITION:-}"
echo "CONSTRAINT=${SLURM_JOB_CONSTRAINT:-}"


# 
echo "Making cross-sectional dataset"

# Competing-Risk
# python make_xsectional_datasets.py  --experiment=mm_ne --seed=1
# python make_xsectional_datasets.py  --experiment=hypertension_ne --seed=1

# --- Run code inside the container with the venv ---
bash "$REPO_DIR/containers/run_in_container.sh" \
    examples/data/3_benchmark_data/make_xsectional_datasets.py \
  --experiment=cvd_ne \
  --seed=1