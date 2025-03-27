import os
from pathlib import Path
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import wandb
from tqdm import tqdm
import pickle
from hydra import compose, initialize
from omegaconf import OmegaConf
from CPRD.examples.modelling.SurvivEHR.run_experiment import run
from FastEHR.dataloader.foundational_loader import FoundationalDataModule
import pickle 

from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from CPRD.src.modules.head_layers.survival.desurv import ODESurvSingle
from pycox.evaluation import EvalSurv

import time
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import os
import polars as pl
import pandas as pd

from contextlib import redirect_stdout
import argparse


def make_dataset(datamodule, target_tokens, vocab_size, split='train', n=None):

    # X = pd.DataFrame(columns=[f'static_{_idx}' for _idx in range(16)] + [f'{datamodule.train_set.tokenizer._itos[_idx]}' for _idx in range(2,vocab_size)])
    Y = []

    match split:
        case 'train':
            dataloader = datamodule.train_dataloader()
        case 'val':
            dataloader = datamodule.val_dataloader()
        case 'test':
            dataloader = datamodule.test_dataloader()
        case _:
            raise NotImplementedError

    dfs = []
    for b_idx, batch in tqdm(enumerate(dataloader), total=n, desc=f"Creating {split} cross-sectional dataset to be used for benchmarking."):

        # Input
        ########
        # Static variables are already processed into categories where required
        static = batch["static_covariates"].numpy()
    
        # Get a binary vector of vocab_size elements, which indicate if patient has any history of a condition (at any time, as long as it fits study criteria)
        # Note, 0 and 1 are PAD and UNK tokens which arent required
        input_tokens = batch["tokens"]
        token_binary = np.zeros((static.shape[0], vocab_size-2))
        for s_idx in range(static.shape[0]):
            for tkn_idx in range(2, vocab_size):
                if tkn_idx in input_tokens[s_idx, :]:
                    token_binary[s_idx, tkn_idx-2] = 1
    
        batch_input = np.hstack((static, token_binary))
        batch_df = pd.DataFrame(batch_input, columns=[f'static_{_idx}' for _idx in range(16)] + [f'{datamodule.train_set.tokenizer._itos[_idx]}' for _idx in range(2,vocab_size)])
        dfs.append(batch_df)
        
        # Target
        ########
        targets = batch["target_token"].numpy()
        for s_idx in range(static.shape[0]):
            # default to 0
            target = 0

            # replace with target if its in the outcome set
            for idx_outcome, outcome in enumerate(target_tokens):
                if targets[s_idx] == outcome:
                    target = idx_outcome + 1
            
            Y.append((target, batch["target_age_delta"][s_idx] ))
    
        # if n is not None and b_idx >= n:
        #     break
    
    # X = pd.concat([X, batch_df])
    X = pd.concat(dfs, ignore_index=True)
    y = np.array(Y, dtype=[('Status', 'int'), ('Survival_in_days', '<f8')])

    return X, y

def run(experiment, sample_size, seed):
    
    # load the configuration file, override any settings 
    with initialize(version_base=None, config_path="../../modelling/SurvivEHR/confs", job_name="testing_notebook"):
        cfg = compose(config_name="config_CompetingRisk37M") 
        cfg.transformer.block_size=1000000     # Ensure all records get included

    match experiment.lower():
        case "cvd":
            cfg.data.path_to_ds="/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/FineTune_CVD/"
        case "hypertension":
            cfg.data.path_to_ds="/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/FineTune_Hypertension/"
        case "mm":
            # cfg.data.path_to_ds="/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/FineTune_MultiMorbidity2/"
            cfg.data.path_to_ds="/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/FineTune_MultiMorbidity50+/"
            

    supervised = True 
    logging.info("="*100)
    logging.info(f"# Loading DataModule for dataset {cfg.data.path_to_ds}. This will be loaded in {'supervised' if supervised else 'causal'} form.")
    logging.info("="*100)
    dm = FoundationalDataModule(path_to_db=cfg.data.path_to_db,
                                path_to_ds=cfg.data.path_to_ds,
                                load=True,
                                tokenizer="tabular",
                                batch_size=cfg.data.batch_size,
                                max_seq_length=cfg.transformer.block_size,
                                global_diagnoses=cfg.data.global_diagnoses,
                                freq_threshold=cfg.data.unk_freq_threshold,
                                min_workers=cfg.data.min_workers,
                                overwrite_meta_information=cfg.data.meta_information_path,
                                supervised=supervised,
                                subsample_training=sample_size,
                                seed=seed,
                               )
    if sample_size is not None:
        print(dm.train_set.subsample_indicies)
    # Get required information from initialised dataloader
    # ... vocab size
    vocab_size = dm.train_set.tokenizer.vocab_size
    # ... Extract the measurements, using the fact that the diagnoses are all up upper case. This is needed for automatically setting the configuration below
    #     encode into the list of univariate measurements to model with Normal distribution
    # measurements_for_univariate_regression = [record for record in dm.tokenizer._event_counts["EVENT"] if record.upper() != record]
    # cfg.head.tokens_for_univariate_regression = dm.encode(measurements_for_univariate_regression) #
    measurements_for_univariate_regression = dm.train_set.meta_information["measurement_tables"][dm.train_set.meta_information["measurement_tables"]["count_obs"] > 0]["event"].to_list()
    cfg.head.tokens_for_univariate_regression = dm.encode(measurements_for_univariate_regression)
    logging.debug(OmegaConf.to_yaml(cfg))

    match experiment.lower():
        case "cvd":
            conditions = ["IHDINCLUDINGMI_OPTIMALV2", "ISCHAEMICSTROKE_V2", "MINFARCTION", "STROKEUNSPECIFIED_V2", "STROKE_HAEMRGIC"]
            cfg.experiment.fine_tune_outcomes=conditions
        case "hypertension":
            conditions = ["HYPERTENSION"]
            cfg.experiment.fine_tune_outcomes=conditions
        case "mm":
            conditions = (
                dm.tokenizer._event_counts.filter((pl.col("COUNT") > 0) &
                    (pl.col("EVENT").str.contains(r'^[A-Z0-9_]+$')))
                  .select("EVENT")
                  .to_series()
                  .to_list()
            )
            cfg.experiment.fine_tune_outcomes=conditions
    
    target_tokens = dm.encode(conditions)

    # print(OmegaConf.to_yaml(cfg))

    if sample_size is not None:
        save_path = cfg.data.path_to_ds + f"benchmark_data/N={sample_size}_seed{seed}.pickle" 
    else:
        save_path = cfg.data.path_to_ds + "benchmark_data/all.pickle"
    
    try:
        # Load the pickled file for testing
        print(f"Trying to load {save_path}")
        
        with open(save_path, "rb") as handle:
            data = pickle.load(handle)
    
    except:
        print(f"Loading failed, creating dataset")
        
        # Training set
        n_train =  len(dm.train_dataloader()) 
        X_train, y_train = make_dataset(dm, target_tokens, vocab_size, split='train', n=n_train)    

        data = {
            "X_train": X_train,
            "y_train": y_train,
        }

         # Test and validation sets - only for the full dataset version, as there is no point repeating this operation
        if sample_size is None:
            
            n_val = len(dm.val_dataloader())  
            X_val, y_val = make_dataset(dm, target_tokens, vocab_size, split='val', n=n_val)
            
            n_test = len(dm.test_dataloader())  
            X_test, y_test = make_dataset(dm, target_tokens, vocab_size, split='test', n=n_test)
            
            print(X_test)
            print(X_test.head())
            
            data = {**data,
                    "X_val": X_val,
                    "y_val": y_val,
                    "X_test": X_test,
                    "y_test": y_test
                    }
        
        with open(save_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saving to {save_path}")

def main():

    torch.manual_seed(1337)
    torch.set_float32_matmul_precision('medium')
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Run evaluation script")
    parser.add_argument("--experiment", type=str, required=True, help="Which experiment's data to convert (cvd, hypertension, mm)")
    parser.add_argument("--n_sample", type=int, default=None, help="Number of sub-samples (for ablation). Default is to convert all the data.",)
    parser.add_argument("--seed", type=int, default=32, help="Seed for splitting data.",)
    
    args = parser.parse_args()

    run(args.experiment, args.n_sample, args.seed)

if __name__ == "__main__":
    main()
    
