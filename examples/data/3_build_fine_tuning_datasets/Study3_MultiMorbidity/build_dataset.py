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
import logging
import time
from CPRD.examples.data.study_criteria import multimorbidity_inclusion_method
from FastEHR.dataloader.foundational_loader import FoundationalDataModule

if __name__ == "__main__":

    torch.manual_seed(1337)
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Building study dataset on {os.cpu_count()} CPUs and {torch.cuda.device_count()} GPUs")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_threads = 1
    print(f"Using device: {device}.")
    print(f"Fitting dataset over {num_threads} threads")

    # load the configuration file, override any settings 
    with initialize(version_base=None, config_path="../../../modelling/SurvivEHR/confs", job_name="dataset_creation_multimorbidity_job"):
        cfg = compose(config_name="config_CompetingRisk11M", overrides=[])
        
    # Create new dataset 
    cfg.data.path_to_ds = "/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/FineTune_MultiMorbidity/"
    print(OmegaConf.to_yaml(cfg))

    # Build 
    dm = FoundationalDataModule(path_to_db=cfg.data.path_to_db,
                                path_to_ds=cfg.data.path_to_ds,
                                load=False,
                                include_diagnoses=True,
                                include_measurements=True,
                                drop_missing_data=False,
                                drop_empty_dynamic=True,
                                tokenizer="tabular",
                                overwrite_practice_ids = "/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/PreTrain/practice_id_splits.pickle",
                                overwrite_meta_information=cfg.data.meta_information_path,
                                study_inclusion_method=multimorbidity_inclusion_method(),  # min_events=50
                                num_threads=num_threads
                               )
    
    vocab_size = dm.train_set.tokenizer.vocab_size
    
    print(f"{len(dm.train_set)} training patients")
    print(f"{len(dm.val_set)} validation patients")
    print(f"{len(dm.test_set)} test patients")
    print(f"{vocab_size} vocab elements")
    
    