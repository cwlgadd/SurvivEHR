#!/bin/bash -l

#SBATCH --account=gokhalkm-optimal
#SBATCH --qos=bbgpupriority3
#SBATCH --gres gpu:a100:1
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=pretrain_London_run
#SBATCH --output=out/London-job-%j.out

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
echo "Training SurvivEHR Foundation Model on patients from London in England"
cd /rds/homes/g/gaddcz/Projects/CPRD/examples/modelling/SurvivEHR

# Execute your Python scripts

# Competing-Risk
python run_experiment.py  --config-name=config_CompetingRisk11M \
    experiment.project_name="SurvivEHR-London" \
    experiment.run_id="SurvivEHR-London-v1-v1" \
    experiment.train=false \
    experiment.test=true \
    data.path_to_ds="/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/ByRegion/PreTrain_London/" \
    optim.learning_rate=1e-4 \


    # 3e-4, 1.5e-4, 1e-4, 5e-5
    
# optim.scheduler_warmup=False \
# 
