# SurvivEHR Pre-Training

This directory contains resources for **pre-training SurvivEHR**. 

---

## Overview

We pre-train the **SurvivEHR foundation model** using competing-risk survival objectives.  

SurvivEHR was trained using the BlueBEAR HPC cluster using **SLURM (`sbatch`) job submission**. Jobs were submitted an epoch at a time, whilst tapering learning rates. The jobs are designed for the BlueBEAR cluster with access to A100 GPUs.

- **Job system**: SLURM (`sbatch`)
- **Compute**: 2 × NVIDIA A100 GPUs, 12 CPUs
- **Runtime limit**: 48h
- **Software stack**: PyTorch 2.0.1, PyTorch Lightning 2.1.0, Polars, Hydra, wandb, Seaborn, UMAP, sklearn-pandas
- **Virtual environment**: `virtual-envTorch2.0-${BB_CPU}` (activated inside the job)

---

## Usage

Submit a training job with:

```bash
sbatch pre_train_job.sh
```

## Configuring Pre-Training Jobs

The behaviour of SurvivEHR pre-training runs is controlled by **Hydra configuration files**.  
Parameters can be **overridden directly in the `sbatch` job script** when calling `python run_experiment.py`.  

For example, in a low resource environment where we can trade-off performance for speed, we may want to try:

```bash
python run_experiment.py --config-name=config_CompetingRisk11M \
    experiment.run_id=SurvivEHR-low-memory \
    transformer.n_embd=192 \
    transformer.n_layer=6 \
	transformer.block_size=128 \
	data.batch_size=16 \
	data.repeating_events=False \
	optim.accumulate_grad_batches=4 \
	
```

| Section          | Parameter                                         | Description                                                  | Example Values                                |
| ---------------- | ------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| **transformer**  | `block_type`                                      | Transformer architecture variant                             | `Neo`, `Nano`                                 |
|                  | `block_size`                                      | Max context window length                                    | `256`, `512`                                  |
|                  | `n_layer`                                         | Number of transformer layers (depth)                         | `6`, `8`, `12`                                |
|                  | `n_head`                                          | Number of attention heads                                    | `6`, `8`, `12`                                |
|                  | `n_embd`                                          | Embedding dimension                                          | `192`, `384`, `768`                           |
|                  | `attention_type`                                  | Attention mechanism (Neo supports `"local"`)                 | `global`, `local`                             |
|                  | `dropout` / `attention_dropout` / `resid_dropout` | Dropout rates                                                | `0.0`, `0.1`                                  |
|                  | `use_fine_tune_adapter`                           | Freeze backbone & add adapters (fine-tuning only)            | `True`, `False`                               |
| **head**         | `SurvLayer`                                       | Survival head type                                           | `"cr"` (competing-risk), `"sr"` (single-risk - deprecated) |
|                  | `surv_weight`                                     | Weight for survival prediction loss                          | `1.0`                                         |
|                  | `value_weight`                                    | Weight for value prediction loss                             | `0.0`, `0.1`                                  |
| **data**         | `batch_size`                                      | Training batch size                                          | `32`, `64`                                    |
|                  | `min_workers`                                     | Number of dataloader workers                                 | `12`                                          |
|                  | `repeating_events`                                | Keep repeated measurements (`True`) or only latest (`False`) | `True`, `False`                               |
|                  | `global_diagnoses`                                | Append truncated diagnoses to context window                 | `True`, `False`                               |
|                  | `path_to_db`                                      | Path to FastEHR SQLite database                              | `/path/to/cprd.db`                            |
|                  | `path_to_ds`                                      | Path to FastEHR Polars dataset                               | `/path/to/dataset/`                           |
|                  | `meta_information_path`                           | Metadata produced by FastEHR(tokenisation, bounds, etc.)     | `/path/to/meta.pickle`                        |
| **optim**        | `num_epochs`                                      | Number of epochs                                             | `1`, `10`, `100`                              |
|                  | `learning_rate`                                   | Initial AdamW learning rate                                  | `3e-4`, `1e-3`                                |
|                  | `scheduler`                                       | Learning rate scheduler                                      | `decaycawarmrestarts`, `CosineAnnealingLR`    |
|                  | `scheduler_warmup`                                | Enable warmup                                                | `True`, `False`                               |
|                  | `scheduler_periods`                               | Period length for cyclic schedulers                          | `10000`                                       |
|                  | `learning_rate_decay`                             | Decay factor between restarts for relevant schedulers        | `0.8`                                         |
|                  | `val_check_interval`                              | Batches between validation checks                            | `2500`                                        |
|                  | `early_stop`                                      | Enable early stopping                                        | `True`, `False`                               |
|                  | `early_stop_patience`                             | Patience (validation intervals) before stopping              | `30`                                          |
|                  | `log_every_n_steps`                               | Number of steps between logging                              | `30`                                          |
|                  | `limit_val_batches`                               | Portion of validation set to use                             | `1.0`, `0.1`                                  |
|                  | `limit_test_batches`                              | Portion of testing set to use                                | `1.0`, `0.1`                                  |
|                  | `accumulate_grad_batches`                         | Gradient accumulation steps                                  | `1`, `2`, `4`                                 |
| **experiment**   | `type`                                            | Experiment type                                              | `pre-train`, `zero-shot`, `fine-tune`         |
|                  | `run_id`                                          | Run identifier (name of your pre-trained model)              | `SurvivEHR-CPRD-1.0`                          |
|                  | `seed`                                            | Random seed                                                  | `1337`                                        |
|                  | `log`                                             | Whether to log using W&B                                     | `True`, `False`                               |
|                  | `log_dir`                                         | Directory for logs                                           | `/path/to/logs/`                              |
|                  | `ckpt_dir`                                        | Directory for checkpoints                                    | `/path/to/checkpoints/`                       |


