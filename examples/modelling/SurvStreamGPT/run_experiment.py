from omegaconf import DictConfig, OmegaConf
import os
import hydra
import torch
import logging
from pathlib import Path
from CPRD.data.foundational_loader import FoundationalDataModule
from CPRD.examples.modelling.SurvStreamGPT.setup_causal_experiment import setup_causal_experiment, CausalExperiment
from CPRD.examples.modelling.SurvStreamGPT.setup_zeroshot_experiment import setup_zeroshot_experiment, ZeroShotExperiment

from CPRD.src.models.survival.task_heads.causal import SurvStreamGPTForCausalModelling


@hydra.main(version_base=None, config_path="confs", config_name="default")
def run(cfg : DictConfig):

    logging.basicConfig(level=logging.DEBUG)
    logging.info(f"Running {cfg.head.SurvLayer} on {os.cpu_count()} CPUs and {torch.cuda.device_count()} GPUs")

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
    # measurements_for_univariate_regression = [record for record in dm.tokenizer._event_counts["EVENT"] if record.upper() != record]
    # cfg.head.tokens_for_univariate_regression = dm.encode(measurements_for_univariate_regression) #
    measurements_for_univariate_regression = dm.train_set.meta_information["measurement_tables"][dm.train_set.meta_information["measurement_tables"]["count_obs"] > 0]["event"].to_list()
    cfg.head.tokens_for_univariate_regression = dm.encode(measurements_for_univariate_regression)
    logging.debug(OmegaConf.to_yaml(cfg))
    
    # Create experiment
    match cfg.experiment.type.replace('-', '').replace(' ', '').lower():
        case "pretrain":
            logging.info("Running Causal experiment. This will create a pre-trained Foundation Model")
            experiment_instance, Experiment, trainer = setup_causal_experiment(cfg=cfg, dm=dm, vocab_size=vocab_size)
        case "zeroshot":
            logging.info("Running Zero-Shot experiment. This will evaluate the pre-trained Foundation Model on a clinical prediction model task.")
            experiment_instance, Experiment, trainer = setup_zeroshot_experiment(cfg=cfg, dm=dm, vocab_size=vocab_size)
        case "finetune":
            raise NotImplementedError
        case _:
            raise NotImplementedError

    # Train/load
    ckpt_path = cfg.experiment.log_dir + f'checkpoints/{cfg.experiment.run_id}.ckpt'
    
    if cfg.experiment.train:
        logging.info(f"Training model. Checkpointing to {ckpt_path}")

        # If this file exists from a previous run, raise an exception. (TODO: LBYL)
        #   This is to avoid accidentally creating a new model with a -V1 etc suffix, then re-loading a previous run's best checkpoint later
        if Path(ckpt_path).is_file():
            raise FileExistsError(f"A pre-trained experiment with the checkpoint path {ckpt_path} already exists.")
            
        trainer.fit(experiment_instance, datamodule=dm)
        
    # checkpoint = trainer.checkpoint_callback.dirpath + f"/{cfg.experiment.run_id}.ckpt"
    logging.info(f"Loading from cached checkpoint {ckpt_path}")
    experiment_instance = Experiment.load_from_checkpoint(ckpt_path)

    # Test model
    if cfg.experiment.test:
        logging.info(f"Testing model.")
        trainer.test(experiment_instance, dataloaders=dm.test_dataloader())

    return experiment_instance.model, dm

if __name__ == "__main__":
    run()
    