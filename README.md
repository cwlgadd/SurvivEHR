[![medRxiv](https://img.shields.io/badge/medRxiv-10.1101%2F2025.08.04.25332916v1-blue)](https://www.medrxiv.org/content/10.1101/2025.08.04.25332916v1)

# SurvivEHR

SurvivEHR is a research project developing a foundation model for survival analysis on electronic health records (EHR). It introduces a decoder-only transformer trained on time-to-event prediction tasks, with a novel competing risks survival objective. This repository contains the code associated with the SurvivEHR model, as described in the medRxiv preprint "SurvivEHR: An EHR Foundation Model for Time-to-Event Prediction with Survival Objectives" (available on medRxiv: DOI: 10.1101/2025.08.04.25332916).

SurvivEHR is trained on a large corpus of EHR data -- over 7.6 billion coded events from 23 million patients in UK primary care. This large-scale pretraining enables the model to learn rich representations of patient histories which can be used in downstream fine-tuning tasks.

# Project Overview

SurvivEHR is a generative transformer-based foundation model trained on over 7.6 billion EHR events from 23 million patients in UK primary care. The model introduces a competing risks, time-to-event pretraining objective that enables forecasting of future clinical events (e.g. new diagnoses, lab investigations, medications) and patient mortality. By learning from longitudinal health records, SurvivEHR can capture complex patient trajectories across multiple long-term conditions. In our experiments, SurvivEHR demonstrated strong risk stratification performance and outperformed traditional survival models across multiple prediction tasks. It also showed effective transfer learning to specific prognostic tasks (especially in low-resource settings), highlighting its potential as a reusable EHR foundation model for clinical risk prediction

> Note: This repository provides the research code for training and evaluating SurvivEHR as described in the preprint. Due to data privacy and size constraints, pre-trained model weights are not included. Users can reproduce training or fine-tune SurvivEHR on their own data following the instructions below.

# Installation and Setup Instructions

To use this code, please ensure you have a suitable Python environment prepared (tested on Python 3.10.4). We recommend using a virtual environment, or the apptainer below. Follow these steps to install SurvivEHR and its dependencies:


## Clone repository

First, clone the SurvivEHR repository:

```bash
git clone https://github.com/cwlgadd/SurvivEHR.git
mv SurvivEHR CPRD 
cd CPRD
```

## Option 1: Create a virtual environment

Requirements for this project can be found in `requirements.txt`. 

Setting up the virtual environment can be done in many ways. For example, this can be done using ![Astral's UV packaging tools](https://docs.astral.sh/uv/):

```bash
VENV_DIR="/path/to/virtual_envs"
VENV_PATH="$VENV_DIR/SurvivEHR-3.10.4"              # SurvivEHR's virtual environment will be created here
mkdir -p "$VENV_DIR"

uv python install 3.10.4
uv venv --python 3.10.4 "$VENV_PATH"
source "$VENV_PATH/bin/activate"
```

Optionally, choose to

```bash
# Create a symlink to the external env to help editors that expect to find .venv
ln -sfn "$VENV_PATH" .venv      

# Tell UV to cache in another directory (for example if personal directory has limited space)
export UV_CACHE_DIR="/path/to/virtal_envs/scratch/directory/uv-cache"
mkdir -p "$UV_CACHE_DIR"
uv cache dir

# Give more time for UV to build from default
export UV_HTTP_TIMEOUT=180

```

You can then build the venv using the provided, generated, requirements.lock file

```bash
uv pip sync /path/to/project/root/requirements.lock
```

To activate the environment use

```bash
source ${VENV_PATH}/bin/activate
```

Note, this creates a new version of python which can lead to conflicts if

## Option 2: Create and run in an apptainer

Apptainer is used instead of Docker on high performance computing systems due to the administrative privileges that are required to run the latter.
Ensure you are not already running an apptainer (e.g. interactive nodes).

1) Build container image (saved to /rds, not repo)
```bash
bash containers/container_build.sh
```

3) Create/update venv on /rds and sync dependencies
```bash
bash containers/env_bootstrap.sh
```

5) Run commands in the container using the venv, e.g.
```bash
bash containers/run_in_container.sh python -V
```

## Install FastEHR (Data Pipeline)

Whilst custom data pipelines can be used, we recommend using FastEHR. SurvivEHR uses the FastEHR library for EHR data preprocessing (providing tools for data pre-processing, loading, dataset construction, etc.). FastEHR is a high-performance pipeline for transforming raw EHR events into ML-ready format. 

Clone and add to your python path

```bash
git clone https://github.com/cwlgadd/FastEHR.git
```

FastEHR shares the same dependencies as SurvivEHR.

# Examples Directory Overview

The repository includes an examples/ directory containing scripts and Jupyter notebooks demonstrating how to prepare data and run the SurvivEHR model on various tasks. Below is an overview of the contents and their usage:

- **Data Preparation Examples (examples/data/)**: This folder contains scripts and notebooks for building the model input datasets from raw EHR data (using FastEHR outputs).
    - **1_build_database**: How to build an SQLite database for your Electronic Health Records to enable fast data pipeline building.
    - **2_build_pre_training_dataset/**: How to construct the large-scale pre-training dataset.
        - Combining events from all patients into a sequential format
            - Splitting data into cross-validation splits (including creation or loading of splits used elsewhere in the pipeline to avoid practice leakage).
            - Pull and adapt meta information to be used elsewhere in the data pipeline (e.g. tokenisation, vocabulary truncation, outlier removal).
            - Additionally includes examples for experiments which stratify by region (e.g., separate cohorts for different geographic regions like North-East vs. London).
    - **3_benchmark_data/**: Convert supervised FastEHR datasets for cross-sectional benchmarks (e.g. DeepHit, DeSurv, Random Survival Forest)
    - **4_convert_BEHRT_data/**: Convert self-supervised and supervised FastEHR datasets for time series benchmarks using FastEHR's adapter framework (e.g. BEHRT)
- **SurvivEHR Examples (examples/modelling/)**: This folder contains scripts and notebooks for running SurvivEHR's experiments
    - **/**: The root directory contains wrapper scripts for the models given in `src/models/`.
        - `run_experiment.py` wraps all experiments and calls: `setup_causal_experiment.py` for pre-training; `set_fewshot_experiment.py` for supervised cases which do use the pre-training architecture, such as zero-shot experiments; and `setup_finetune_experiment.py` for supervised experiments which replace the pre-training head.
    - **notebooks**: Each folder in the notebooks contains the various experiments presented throughout the accompanying manuscript.
      


# Citation

If you use or reference SurvivEHR in your research or work, please cite the accompanying paper:

```Charles Gadd et al. (2025). SurvivEHR: a competing risks, time-to-event foundation model for multiple long-term conditions from primary care electronic health records. medRxiv. DOI: 10.1101/2025.08.04.25332916.```

# Copyright

© 2025 Charles Gadd, University of Oxford. All rights reserved where applicable.

Distributed under the GNU GPL v3.0 (or later). See LICENSE for details.
