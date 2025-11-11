#!/bin/bash -l
#SBATCH --account=gokhalkm-optimal

#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake

# SBATCH --qos=bbgpupriority3
# SBATCH --gres=gpu:a100:1

#SBATCH --time=05:00:00
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
TASK=${1:-}
PRE_TRAINED_MODEL=${2:-}
TRAIN_DATA=${3:-}
EVAL_DATA=${4:-}
SEED=${5:-}
SWEEP=${6:-}

if [[ -z "$TASK" || -z "$PRE_TRAINED_MODEL" || -z "$TRAIN_DATA" || -z "$EVAL_DATA" || -z "$SEED" ]]; then
  echo "Usage: sbatch slurm/regional_job.sbatch <task> <pre-trained-model> <training-region> <evaluation-region> <seed>"
  exit 1
fi

echo "Fine-tuning model='$PRE_TRAINED_MODEL' for task='$TASK', train='$TRAIN_DATA', eval='$EVAL_DATA', seed='$SEED', sweep='$SWEEP'"

# --- Run code inside the container with the venv ---
bash "$REPO_DIR/containers/run_in_container.sh" \
    examples/modelling/SurvivEHR/notebooks/CompetingRisk/3_Regional_analaysis/regional_run.py \
  --task "$TASK" \
  --pre-trained-model "$PRE_TRAINED_MODEL" \
  --train-set "$TRAIN_DATA" \
  --eval-set "$EVAL_DATA" \
  --seed "$SEED" \
  --sweep "$SWEEP"
