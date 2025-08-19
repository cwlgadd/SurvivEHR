#!/bin/bash -l

#SBATCH --account=gokhalkm-optimal
#SBATCH --qos=bbdefault
#SBATCH --time=10:0:0

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36

#SBATCH --output=/rds/homes/g/gaddcz/Projects/CPRD/examples/data/0_benchmark_data/out_%A.out

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

echo $BB_CPU

export VENV_PATH="/rds/homes/g/gaddcz/Projects/CPRD/virtual-envTorch2.0-${BB_CPU}"
echo $VENV_PATH

# Activate the virtual environment
source ${VENV_PATH}/bin/activate

# 
echo "Making cross-sectional dataset"

# Competing-Risk
python make_xsectional_datasets.py  --experiment=mm --seed=5 --n_sample=20000