#!/bin/bash -l
#SBATCH --account=gokhalkm-optimal
#SBATCH --qos=bbdefault
#SBATCH --time=30:00:0
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --output=build_NE_pretrain_BEHRT_dataset_output.out

set -e   # Exit on first error

export SQLITE_TMPDIR=/rds/projects/g/gokhalkm-optimal/DataforCharles
export TMPDIR=/rds/projects/g/gokhalkm-optimal/DataforCharles
echo $SQLITE_TMPDIR
echo $TMPDIR

echo $TERM
echo $BB_CPU

echo "HOST=$(hostname)"
echo "OS=$(cat /etc/os-release | sed -n 's/^PRETTY_NAME=//p')"
echo "CPU=$(lscpu | sed -n 's/^Model name: *//p')"
echo "PARTITION=$SLURM_JOB_PARTITION"
echo "MODULEPATH=$MODULEPATH"
echo "LMOD_SYSTEM_NAME=$LMOD_SYSTEM_NAME"

# Activate the virtual environment
export VENV_PATH="/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/virtual_envs/SurvivEHR-3.10.4"
source ${VENV_PATH}/bin/activate


# 
echo "Build BEHRT pre-training dataset from the existing SurvivEHR dataset for North-East England"

# Execute your Python scripts
python build_NE_pretrain_BEHRT_dataset.py;