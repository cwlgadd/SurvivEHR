from omegaconf import DictConfig, OmegaConf
import os
import hydra
import torch
import logging
from CPRD.data.foundational_loader import FoundationalDataModule
from CPRD.examples.modelling.SurvStreamGPT.setup_experiment import setup_survival_experiment, SurvivalExperiment

@hydra.main(version_base=None, config_path="confs", config_name="default")
def run(cfg : DictConfig):

    logging.basicConfig(level=logging.DEBUG)
    logging.info(f"Running {cfg.head.SurvLayer} experiment on {os.cpu_count()} CPUs and {torch.cuda.device_count()} GPUs")

    # Global settings
    torch.manual_seed(cfg.experiment.seed)
    torch.set_float32_matmul_precision('medium')
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # make dataloader
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
    
    # Create experiment
    experiment, trainer = setup_survival_experiment(cfg=cfg, dm=dm, vocab_size=vocab_size)

    # Train/load
    ckpt_path = cfg.experiment.log_dir + f'checkpoints/{cfg.experiment.run_id}.ckpt'
    if cfg.experiment.train:
        logging.info(f"Training model. Checkpointing to {ckpt_path}")
        trainer.fit(experiment, datamodule=dm)
        
    # checkpoint = trainer.checkpoint_callback.dirpath + f"/{cfg.experiment.run_id}.ckpt"
    logging.info(f"Loading from cached checkpoint {ckpt_path}")
    experiment = SurvivalExperiment.load_from_checkpoint(ckpt_path)

    # Test model
    if cfg.experiment.test:
        trainer.test(experiment, dataloaders=dm.test_dataloader())

    return experiment.model, dm

if __name__ == "__main__":
    run()
    