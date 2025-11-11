#!/bin/bash -l
#SBATCH --output=out/%x-%j.out

declare -a sweeps=(
    'data.subsample_training=2999'
    #'data.subsample_training=5296'
    #'data.subsample_training=9351'
    #'data.subsample_training=16509'
    #'data.subsample_training=29148'
    #'data.subsample_training=51461'
    #'data.subsample_training=90856'
    #'data.subsample_training=160407'
    #'data.subsample_training=283203'
    #'data.subsample_training=500000'
)


# Each pre-trained model we want to consider
for pre_trained_model in 'from-scratch' 'SurvivEHR-cr-small-debug7_exp1000-v1-v4-v1'; do
    for seed in 1 3 5; do  # 
        for sweep in "${sweeps[@]}"; do
    
            echo "model=$pre_trained_model seed=$seed sweep $sweep"

            sbatch ./ablation_job.sh \
                "$pre_trained_model" \
                "$seed" \
                "$sweep"
        done
    done
done
