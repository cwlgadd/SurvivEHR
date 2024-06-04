from omegaconf import DictConfig, OmegaConf
import os
import hydra
import torch
import logging
from CPRD.data.foundational_loader import FoundationalDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wandb")

@hydra.main(version_base=None, config_path="confs", config_name="default")
def run(cfg : DictConfig):

    logging.info(f"Running dataloader test for {cfg.head.SurvLayer} experiment config, on {os.cpu_count()} CPUs and {torch.cuda.device_count()} GPUs")

    # Global settings
    torch.manual_seed(cfg.experiment.seed)
    torch.set_float32_matmul_precision('medium')
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # make dataloader
    dm = FoundationalDataModule(path_to_db="/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/",
                                load=True,
                                tokenizer="tabular",
                                batch_size=cfg.data.batch_size,
                                max_seq_length=cfg.transformer.block_size,
                                unk_freq_threshold=cfg.data.unk_freq_threshold,
                                min_workers=cfg.data.min_workers,
                                inclusion_conditions=["COUNTRY = 'E'"],
                               )
    # Get required information from initialised dataloader
    # ... vocab size
    vocab_size = dm.train_set.tokenizer.vocab_size
    logging.info(f"{vocab_size} vocab elements")
    # ... Extract the measurements, using the fact that the diagnoses are all up upper case. This is needed for automatically setting the configuration below
    #     encode into the list of univariate measurements to model with Normal distribution
    measurements_for_univariate_regression = [record for record in dm.tokenizer._event_counts["EVENT"] if record.upper() != record]
    cfg.head.tokens_for_univariate_regression = dm.encode(measurements_for_univariate_regression) 
    logging.debug(OmegaConf.to_yaml(cfg))
    
    # Run over batch to ensure loader is working
    for loader in [dm.train_dataloader, dm.val_dataloader, dm.test_dataloader]:
        for batch in loader():
            pass
        
    return

if __name__ == "__main__":
    run()
    