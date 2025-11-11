#!/bin/bash -l
#SBATCH --account=gokhalkm-optimal
#SBATCH --qos=bbdefault
#SBATCH --time=15:00:0
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --output=build_cvd_NE_BEHRT_dataset_output.out

set -e   # Exit on first error

# module purge; module load bluebear
# module load bear-apps/2022a/live 
# module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
# module load PyTorch-Lightning/2.1.0-foss-2022a-CUDA-11.7.0
# module load sklearn-pandas/2.2.0-foss-2022a
# module load Hydra/1.3.2-GCCcore-11.3.0
# module load polars/0.17.12-foss-2022a
# module load wandb/0.13.6-GCC-11.3.0
# module load Seaborn/0.12.1-foss-2022a
# module load umap-learn/0.5.3-foss-2022a
# # export VENV_PATH="/rds/homes/g/gaddcz/Projects/CPRD/virtual-env-${BB_CPU}"
# export VENV_PATH="/rds/homes/g/gaddcz/Projects/CPRD/virtual-envTorch2.0-icelake"
# echo $VENV_PATH
# # Activate the virtual environment
# source ${VENV_PATH}/bin/activate

echo "HOST=$(hostname)"
echo "OS=$(cat /etc/os-release | sed -n 's/^PRETTY_NAME=//p')"
echo "CPU=$(lscpu | sed -n 's/^Model name: *//p')"
echo "PARTITION=$SLURM_JOB_PARTITION"
echo "MODULEPATH=$MODULEPATH"
echo "LMOD_SYSTEM_NAME=$LMOD_SYSTEM_NAME"

# Activate the virtual environment
export VENV_PATH="/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/virtual_envs/SurvivEHR-3.10.4"
source ${VENV_PATH}/bin/activate

export SQLITE_TMPDIR=/rds/projects/g/gokhalkm-optimal/DataforCharles
export TMPDIR=/rds/projects/g/gokhalkm-optimal/DataforCharles
echo $SQLITE_TMPDIR
echo $TMPDIR

echo $TERM
echo $BB_CPU

# 
echo "Build BEHRT fine-tuning dataset from the existing SurvivEHR dataset"
# cd /rds/homes/g/gaddcz/Projects/CPRD/examples/data/4_BEHRT_data/Study1_T2D/CVD/

# Execute your Python scripts
python build_cvd_BEHRT_dataset.py;