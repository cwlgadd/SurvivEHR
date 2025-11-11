#!/bin/bash -l
#SBATCH --output=out/%x-%j.out

pre_trained_models=('from-scratch' 'SurvivEHR-cr-small-debug7_exp1000-v1-v4-v1') 
# 'SurvivEHR-NorthEast-v1-v1-v1-v1''
experiments=('CVD') 
# 'Hypertension' 'MM'
training_sets=('North East')
eval_set="auto"

# sweeps=('optim.accumulate_grad_batches=1', 'optim.accumulate_grad_batches=2', 'optim.accumulate_grad_batches=5', 'optim.accumulate_grad_batches=10') 

declare -a sweeps=(
    'data.subsample_training=null'
  # "experiment.project_name=SurvivEHR-regional-fine-tuning"
  # "optim.early_stop_patience=20"
  # 'data.repeating_events=True'
  # "optim.val_check_interval=0.05"
  # "optim.val_check_interval=0.125"
  # "optim.val_check_interval=0.25"
  # "optim.val_check_interval=0.5"
  # "optim.val_check_interval=1.0"
)

# Each pre-trained model we want to consider
for experiment in "${experiments[@]}"; do
    for training_set in "${training_sets[@]}"; do
        for pre_trained_model in "${pre_trained_models[@]}"; do
            for seed in 1 2 3 4 5; do
                for sweep in "${sweeps[@]}"; do
            
                    echo "exp=$experiment model=$pre_trained_model train=$training_set eval=$eval_set seed=$seed sweep $sweep"

                    sbatch ./regional_job.sh \
                        "$experiment" \
                        "$pre_trained_model" \
                        "$training_set" \
                        "$eval_set" \
                        "$seed" \
                        "$sweep"
                done
            done
        done
    done
done
