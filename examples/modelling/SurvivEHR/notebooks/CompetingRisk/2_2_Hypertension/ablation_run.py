import argparse
import os
import pathlib
import pickle
import torch
import wandb
from hydra import compose, initialize

from SurvivEHR.examples.modelling.SurvivEHR.run_experiment import run


def record_results(summary, task, pre_trained_model, seed, recorded_metrics):
    
    num_to_record = len(recorded_metrics)
    
    # Add each metric
    metric_results = []
    for metric in recorded_metrics:
        metric_results.append(summary.get(metric))

    results = {
        "Task": [task for _ in range(num_to_record)],
        "Pre-trained": [pre_trained_model for _ in range(num_to_record)],
        "Metric": recorded_metrics,
        "Seed": [seed for _ in range(num_to_record)],
        "Metric value": metric_results
    }

    return results

def run_job(pre_trained_model, seed, sweep, record_metrics=None):
    """
    Note: If viewing results via W&B interface, note that the results evaluated on the 
            out-of-distribution test set, in the second run() call, will log to the same 
            name as the in-distribution results.
    """

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
                                 "experiment.type='fine-tune-sr'",
                                 "experiment.project_name='SurvivEHR-hypertension-fine-tuning'",
                                 f"experiment.run_id='{pre_trained_model}'",
                                 f"experiment.fine_tune_id='HypertensionAblation/{seed}_{sweep_name}'",
                                 f"experiment.seed={seed}",
                                 # Dataloader
                                 "data.path_to_ds=/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/FineTune_Hypertension/",
                                 "data.batch_size=128",
                                 "data.meta_information_path=/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/PreTrain/meta_information_QuantJenny.pickle",
                                 "data.min_workers=12",
                                 "data.global_diagnoses=True",
                                 "data.repeating_events=False",
                                 # Optimisation
                                 "optim.num_epochs=500",
                                 "optim.scheduler_warmup=False",
                                 "optim.scheduler=reduceonplateau",
                                 "optim.learning_rate=1e-3",
                                 "optim.val_check_interval=0.125",
                                 "optim.limit_val_batches=0.1",
                                 "optim.limit_test_batches=null",
                                 "optim.early_stop=True",
                                 "optim.early_stop_patience=10",
                                 "optim.accumulate_grad_batches=5",
                                 # Model
                                 "transformer.block_size=512",
                                 # Head
                                 "fine_tuning.head.learning_rate=1e-3",
                                 "fine_tuning.use_callbacks.hidden_embedding.num_batches=0",
                                 sweep,
                                ]
                     )

    cfg.fine_tuning.fine_tune_outcomes=[
        "HYPERTENSION"
    ]

    # Run train job 
    model, dm = run(cfg)
    # Record results
    if wandb.run:
        train_run_id = wandb.run.id
        summary = dict(wandb.run.summary)
        results_train = record_results(summary, "Hypertension-ablation", pre_trained_model, seed, record_metrics)
    wandb.finish()

    # Save results
    pathlib.Path("results").mkdir(exist_ok=True)
    with open(f"results/{train_run_id}.pkl", "wb") as f:
        pickle.dump(results_train, f)

    return


if __name__ == "__main__":

    os.environ["SLURM_NTASKS_PER_NODE"] = "28"

    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-trained-model", dest="pre_trained_model", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sweep", type=str, default="experiment.project_name='SurvivEHR-hypertension-fine-tuning'")
    args = parser.parse_args()

    torch.manual_seed(1337)
    torch.set_float32_matmul_precision('medium')

    run_job(args.pre_trained_model, args.seed, args.sweep)
