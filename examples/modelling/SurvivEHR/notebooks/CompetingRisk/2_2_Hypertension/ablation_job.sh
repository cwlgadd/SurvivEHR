#!/bin/bash -l
#SBATCH --account=gokhalkm-optimal

#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake

# SBATCH --qos=bbgpupriority3
# SBATCH --gres=gpu:a100:1

#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=regional_run
#SBATCH --output=out/job-%j.out

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

# --- Parse/validate args (same as your old script) ---
PRE_TRAINED_MODEL=${1:-}
SEED=${2:-}
SWEEP=${3:-}

if [[ -z "$PRE_TRAINED_MODEL" || -z "$SEED" || -z "$SWEEP" ]]; then
  echo "Usage: sbatch slurm/ablation_job.sbatch <pre-trained-model> <seed> <sweep>"
  exit 1
fi

echo "Fine-tuning model='$PRE_TRAINED_MODEL' for task=Hypertension SR, seed='$SEED', sweep='$SWEEP'"

# --- Run code inside the container with the venv ---
bash "$REPO_DIR/containers/run_in_container.sh" \
    examples/modelling/SurvivEHR/notebooks/CompetingRisk/2_2_Hypertension/ablation_run.py \
  --pre-trained-model "$PRE_TRAINED_MODEL" \
  --seed "$SEED" \
  --sweep "$SWEEP"
