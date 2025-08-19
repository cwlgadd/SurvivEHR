#!/bin/bash -l
# CPU setup
# SBATCH --account=gokhalkm-optimal
# SBATCH --qos=bbdefault
# SBATCH --time=15:00:0
# SBATCH --ntasks=1
# SBATCH --nodes=1
# SBATCH --cpus-per-task=48

# GPU setup
#SBATCH --account=gokhalkm-optimal
#SBATCH --qos=bbgpupriority3
#SBATCH --time=2:00:0
#SBATCH --gres gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12

#SBATCH --output=build_dataset_output2.out

set -e   # Exit on first error

module purge; module load bluebear
module load bear-apps/2022a/live 
module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
module load PyTorch-Lightning/2.1.0-foss-2022a-CUDA-11.7.0
module load sklearn-pandas/2.2.0-foss-2022a
module load Hydra/1.3.2-GCCcore-11.3.0
module load polars/0.17.12-foss-2022a
module load wandb/0.13.6-GCC-11.3.0
module load Seaborn/0.12.1-foss-2022a
module load umap-learn/0.5.3-foss-2022a

export SQLITE_TMPDIR=/rds/projects/g/gokhalkm-optimal/DataforCharles
export TMPDIR=/rds/projects/g/gokhalkm-optimal/DataforCharles
echo $SQLITE_TMPDIR
echo $TMPDIR

echo $TERM
echo $BB_CPU

# export VENV_PATH="/rds/homes/g/gaddcz/Projects/CPRD/virtual-env-${BB_CPU}"
export VENV_PATH="/rds/homes/g/gaddcz/Projects/CPRD/virtual-envTorch2.0-icelake"
echo $VENV_PATH

# # Check if virtual environment exists and stop submission if not
# if [[ ! -d ${VENV_PATH} ]]; then
#     exit 1
# fi

# Activate the virtual environment
source ${VENV_PATH}/bin/activate

# 
echo "Build dataset from CPRD.db database of DeXTER output"
cd /rds/homes/g/gaddcz/Projects/CPRD/examples/data/2_build_pre_training_dataset/

# Execute your Python scripts
python build_dataset.py;