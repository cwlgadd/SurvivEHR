import os
from pathlib import Path
import sys
node_type = os.getenv('BB_CPU')
venv_dir = f'/rds/homes/g/gaddcz/Projects/CPRD/virtual-env-{node_type}'
venv_site_pkgs = Path(venv_dir) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
if venv_site_pkgs.exists():
    sys.path.insert(0, str(venv_site_pkgs))
    print(f"Added path '{venv_site_pkgs}' at start of search paths.")
else:
    print(f"Path '{venv_site_pkgs}' not found. Check that it exists and/or that it exists for node-type '{node_type}'.")


import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from CPRD.data.foundational_loader import FoundationalDataModule
import logging
import time


if __name__ == "__main__":

    torch.manual_seed(1337)
    logging.basicConfig(level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_threads = 5
    print(f"Using device: {device}.")
    print(f"Fitting dataset over {num_threads} threads")

    # load the configuration file, override any settings 
    with initialize(version_base=None, config_path="../modelling/SurvStreamGPT/confs", job_name="dataset_creation_notebook"):
        cfg = compose(config_name="config_CompetingRisk37M")
    print(OmegaConf.to_yaml(cfg))

    # Build 
    # overwrite_meta_information:
    #   There is no need to over-write this yet.
    #   In creating the dataset, we collect values which can be used by default, we can then change these, and pass them into it again to load the dataset.
    dm = FoundationalDataModule(path_to_db=cfg.data.path_to_db,
                                path_to_ds=cfg.data.path_to_ds,
                                load=False,
                                include_diagnoses=True,                            
                                include_measurements=True,
                                drop_missing_data=False,
                                drop_empty_dynamic=True,
                                tokenizer="tabular",
                                practice_inclusion_conditions=["COUNTRY = 'E'"],
                                overwrite_meta_information=None,         
                                num_threads=num_threads
                               )
    
    vocab_size = dm.train_set.tokenizer.vocab_size
    
    print(f"{len(dm.train_set)} training patients")
    print(f"{len(dm.val_set)} validation patients")
    print(f"{len(dm.test_set)} test patients")
    print(f"{vocab_size} vocab elements")
    
    