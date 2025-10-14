import os
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
    num_threads = 5
    logging.info(f"Fitting dataset over {num_threads} threads")

    # load the configuration file, override any settings 
    with initialize(version_base=None, 
                    config_path="../../../../modelling/SurvivEHR/confs", 
                    job_name="dataset_creation_stratified_MM_job"):
        cfg = compose(config_name="config_CompetingRisk11M", overrides=[])

    # Build for each group 
    authority_groups = [["North East"], 
                        ["London"]]

    for auth_group in authority_groups:

        save_path = "/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/ByRegion/"
        path_to_ds = save_path + f"MM_{'_'.join(auth_group)}/"
        path_to_split = save_path + f'practice_id_splits_{"_".join(auth_group)}.pickle'

        # Create directory to store created dataset in
        os.makedirs(path_to_ds, exist_ok=True)
        for split_dir in ["train", "test", "val"]:
            os.makedirs(path_to_ds + f"split={split_dir}", exist_ok=True)
	
        # Create new dataset 
        cfg.data.path_to_ds = path_to_ds
        logging.info(OmegaConf.to_yaml(cfg))
    
        # Build 
        dm = FoundationalDataModule(
            path_to_db=cfg.data.path_to_db,
            path_to_ds=cfg.data.path_to_ds,
            load=False,
            include_diagnoses=True,
            include_measurements=True,
            drop_missing_data=False,
            drop_empty_dynamic=True,
            tokenizer="tabular",
            overwrite_practice_ids = path_to_split,
            overwrite_meta_information=cfg.data.meta_information_path,
            study_inclusion_method=multimorbidity_inclusion_method(),
            num_threads=num_threads,
        )
        
        vocab_size = dm.train_set.tokenizer.vocab_size
        
        logging.info(f"{len(dm.train_set)} training patients")
        logging.info(f"{len(dm.val_set)} validation patients")
        logging.info(f"{len(dm.test_set)} test patients")
        logging.info(f"{vocab_size} vocab elements")
