import argparse
import os
import pathlib
import pickle
import torch
import wandb
from hydra import compose, initialize

from SurvivEHR.examples.modelling.SurvivEHR.run_experiment import run


def record_results(summary, task, pre_trained_model, train_set, eval_set, seed, recorded_metrics):
    
    num_to_record = len(recorded_metrics)
    
    # Add each metric
    metric_results = []
    for metric in recorded_metrics:
        metric_results.append(summary.get(metric))
        
    results = {
        "Task": [task for _ in range(num_to_record)],
        "Pre-trained": [pre_trained_model for _ in range(num_to_record)],
        "Training sub-population": [train_set for _ in range(num_to_record)],
        "Metric": recorded_metrics,
        "Seed": [seed for _ in range(num_to_record)],
        "Evaluation sub-population": [eval_set for _ in range(num_to_record)],
        "Metric value": metric_results
    }

    return results

def run_job(task, pre_trained_model, train_set, eval_set, seed, sweep, record_metrics=None):
    """
    Note: If viewing results via W&B interface, note that the results evaluated on the 
            out-of-distribution test set, in the second run() call, will log to the same 
            name as the in-distribution results.
    """

    if train_set == "North East":
        if eval_set == "auto":        
            eval_set = "London"
    elif train_set == "London":
        if eval_set == "auto":
            eval_set = "North East"
    else:
        raise ValueError(f"Unknown train_set '{train_set}'")

        
    if record_metrics is None:
        record_metrics = ["Test:OutcomePerformanceMetricsctd",
                          "Test:OutcomePerformanceMetricsibs",
                          "Test:OutcomePerformanceMetricsinbll",
                          "test_loss_desurv"
                         ]

    # Get sweep logging title
    sweep_arg, sweep_target = sweep.split("=")
    sweep_name = sweep_arg.split(".")[-1] + "=" + sweep_target                   

    # load the configuration file, override any settings 
    with initialize(version_base=None, config_path="../../../confs", job_name="testing_notebook"):
        cfg = compose(config_name="config_CompetingRisk11M", 
                      overrides=[# Experiment setup
                                 "experiment.project_name='SurvivEHR-regional-fine-tuning'",
                                 f"experiment.run_id='{pre_trained_model}'",
                                 f"experiment.fine_tune_id='RegionalTask{task}_tr{train_set}/ce9091d_refactored_{seed}_{sweep_name}'",
                                 f"experiment.seed={seed}",
                                 # Dataloader
                                 f"data.path_to_ds=/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/ByRegion/{task}_{train_set}/",
                                 "data.batch_size=512",
                                 "data.meta_information_path=/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/PreTrain/meta_information_QuantJenny.pickle",
                                 "data.min_workers=12",
                                 "data.global_diagnoses=True",
                                 "data.repeating_events=False",
                                 # Optimisation
                                 "optim.num_epochs=500",
                                 "optim.scheduler_warmup=False",
                                 "optim.scheduler=reduceonplateau",
                                 "optim.learning_rate=1e-4",
                                 "optim.val_check_interval=0.125",
                                 "optim.limit_val_batches=null",
                                 "optim.limit_test_batches=null",
                                 "optim.early_stop=True",
                                 "optim.early_stop_patience=5",
                                 "optim.accumulate_grad_batches=1",
                                 # Model
                                 "transformer.block_size=512",
                                 # Head
                                 "fine_tuning.head.learning_rate=5e-4",
                                 "fine_tuning.use_callbacks.hidden_embedding.num_batches=0",
                                 "fine_tuning.compression_layer=384",
                                 sweep,
                                ]
                     )

    
    if task == "CVD":
        cfg.experiment.type = "fine-tune-cr"
        cfg.fine_tuning.fine_tune_outcomes=[
            "IHDINCLUDINGMI_OPTIMALV2",
            "ISCHAEMICSTROKE_V2",
            "MINFARCTION",
            "STROKEUNSPECIFIED_V2",
            "STROKE_HAEMRGIC"
        ]
    elif task == "Hypertension":
        cfg.experiment.type = "fine-tune-sr"
        cfg.fine_tuning.fine_tune_outcomes=[
            "HYPERTENSION"
        ]
    elif task == "MM":
        cfg.experiment.type = "fine-tune-sr"
        cfg.fine_tuning.custom_outcome_method._target_="CPRD.examples.modelling.SurvivEHR.helpers.custom_mm_outcomes"
        cfg.data.subsample_training=20000
    else:
        raise NotImplementedError

    # Run train job 
    model, dm = run(cfg)
    # Record results
    if wandb.run:
        train_run_id = wandb.run.id
        summary = dict(wandb.run.summary)
        results_train = record_results(summary, task, pre_trained_model, train_set, train_set, seed, record_metrics)
    wandb.finish()

    # Save results
    save_path = "examples/modelling/SurvivEHR/notebooks/CompetingRisk/3_Regional_analaysis/"
    pathlib.Path(save_path + "results").mkdir(exist_ok=True)
    with open(save_path + f"results/{train_run_id}.pkl", "wb") as f:
        pickle.dump(results_train, f)
    
    if eval_set is None:
        return

    ###############
    # Update config to new evaluation dataset, tell config to not re-train, but to evaluate on the new dataset
    ###############
    cfg.experiment.train = False
    cfg.experiment.test = True
    cfg.data.path_to_ds = data_root + task + "_" + eval_set + "/"
    # Run eval job
    model, dm = run(cfg)
    # Record results
    if wandb.run:
        eval_run_id = wandb.run.id
        summary = dict(wandb.run.summary)
        results_eval = record_results(summary, task, pre_trained_model, train_set, eval_set, seed, record_metrics)
    wandb.finish()

    # Save all results
    with open(save_path + f"results/{eval_run_id}.pkl", "wb") as f:
        pickle.dump(results_eval, f)

    return


if __name__ == "__main__":

    os.environ["SLURM_NTASKS_PER_NODE"] = "28"

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["CVD", "Hypertension", "MM"])
    parser.add_argument("--pre-trained-model", dest="pre_trained_model", required=True)
    parser.add_argument("--train-set", dest="train_set", required=True, choices=["London", "North East"])
    parser.add_argument("--eval-set", dest="eval_set", required=True, choices=["London", "North East", "auto", None])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sweep", type=str, default="experiment.project_name='SurvivEHR-regional-fine-tuning'")
    args = parser.parse_args()

    torch.manual_seed(1337)
    torch.set_float32_matmul_precision('medium')

    run_job(args.task, args.pre_trained_model, args.train_set, args.eval_set, args.seed, args.sweep)
