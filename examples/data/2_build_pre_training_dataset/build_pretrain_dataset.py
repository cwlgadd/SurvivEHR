import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
import logging
import time

from FastEHR.dataloader import FoundationalDataModule


if __name__ == "__main__":

    torch.manual_seed(1337)
    logging.basicConfig(level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_threads = 1
    logging.info(f"Using device: {device}.")
    logging.info(f"Fitting dataset over {num_threads} threads")

    # load the configuration file, override any settings 
    with initialize(version_base=None, 
                    config_path="../../modelling/SurvivEHR/confs", 
                    job_name="pretrain_dataset_creation_job"
                   ):
        cfg = compose(config_name="config_CompetingRisk11M")
    # Overwrite dataset name to example
    cfg.data.path_to_ds = "/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/PreTrain_example/"
    print(OmegaConf.to_yaml(cfg))

    # Build 
    # overwrite_meta_information:
    #   There is no need to over-write this yet.
    #   In creating the dataset, we collect values which can be used by default,
    #   we can then change these, and pass them into it again to load the dataset.
    dm = FoundationalDataModule(
        path_to_db=cfg.data.path_to_db,
        path_to_ds=cfg.data.path_to_ds,
        load=False,
        include_diagnoses=True,
        include_measurements=True,
        drop_missing_data=False,
        drop_empty_dynamic=True,
        tokenizer="tabular",
        practice_inclusion_conditions=["COUNTRY = 'E'"],
        num_threads=num_threads
    )
    
    vocab_size = dm.train_set.tokenizer.vocab_size
    
    logging.info(f"{len(dm.train_set)} training patients")
    logging.info(f"{len(dm.val_set)} validation patients")
    logging.info(f"{len(dm.test_set)} test patients")
    logging.info(f"{vocab_size} vocab elements")
