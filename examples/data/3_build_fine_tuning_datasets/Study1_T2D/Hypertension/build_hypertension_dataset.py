import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
import logging
import time

from CPRD.examples.data.study_criteria import t2d_inclusion_method
from FastEHR.dataloader.foundational_loader import FoundationalDataModule


if __name__ == "__main__":

    torch.manual_seed(1337)
    logging.basicConfig(level=logging.INFO)
    num_threads = 5
    logging.info(f"Fitting dataset over {num_threads} threads")

    # load the configuration file, override any settings 
    with initialize(version_base=None, 
                    config_path="../../../../modelling/SurvivEHR/confs", 
                    job_name="dataset_creation_multimorbidity_job"):
        cfg = compose(config_name="config_CompetingRisk11M", overrides=[])
        
    # Create new dataset 
    cfg.data.path_to_ds = "/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/FineTune_Hypertension_example/"
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
        overwrite_practice_ids = "/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/PreTrain/practice_id_splits.pickle",
        overwrite_meta_information=cfg.data.meta_information_path,
        study_inclusion_method=t2d_inclusion_method(
            outcomes=["HYPERTENSION"]
        ),
        num_threads=num_threads,
    )
    
    vocab_size = dm.train_set.tokenizer.vocab_size
    
    logging.info(f"{len(dm.train_set)} training patients")
    logging.info(f"{len(dm.val_set)} validation patients")
    logging.info(f"{len(dm.test_set)} test patients")
    logging.info(f"{vocab_size} vocab elements")
    
    