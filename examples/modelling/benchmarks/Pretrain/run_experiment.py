from omegaconf import DictConfig, OmegaConf
import os
import hydra
import torch
import pytorch_lightning as pl
import logging
from pathlib import Path
from FastEHR.dataloader.foundational_loader import FoundationalDataModule
# Load the different experiment types
from CPRD.examples.modelling.benchmarks.Pretrain.setup_causal_t_mlp_experiment import CausalTMLPExperiment, setup_t_mlp_experiment
from CPRD.examples.modelling.benchmarks.Pretrain.setup_causal_mlp_experiment import CausalMLPExperiment, setup_mlp_experiment


@hydra.main(version_base=None, config_path="../../SurvivEHR/confs", config_name="default")
def run(cfg : DictConfig):
    logging.info(f"Running {cfg.head.SurvLayer} on {os.cpu_count()} CPUs and {torch.cuda.device_count()} GPUs")

    # Create logger
    log_id = cfg.experiment.run_id
    if cfg.experiment.fine_tune_id is not None:
        log_id += "_" + cfg.experiment.fine_tune_id
    logger = pl.loggers.WandbLogger(project=cfg.experiment.project_name, name=log_id, save_dir=cfg.experiment.log_dir, notes=cfg.experiment.notes, tags=cfg.experiment.tags)
    logging.basicConfig(level=logging.DEBUG)
    
    # Global settings
    torch.manual_seed(cfg.experiment.seed)
    torch.set_float32_matmul_precision('medium')
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # make dataloader
    supervised = True if (cfg.fine_tuning.fine_tune_outcomes is not None) or (cfg.fine_tuning.custom_outcome_method._target_ is not None) else False    
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
                                repeating_events=cfg.data.repeating_events,
                                freq_threshold=cfg.data.unk_freq_threshold,
                                min_workers=cfg.data.min_workers,
                                overwrite_meta_information=cfg.data.meta_information_path,
                                supervised=supervised,
                                subsample_training=cfg.data.subsample_training,
                                seed=cfg.experiment.seed,
                               )
    
    # Get required information from initialised dataloader
    # ... vocab size
    vocab_size = dm.train_set.tokenizer.vocab_size
    # ... Extract the measurements, using the fact that the diagnoses are all up upper case. This is needed for automatically setting the configuration below
    #     encode into the list of univariate measurements to model with Normal distribution
    # measurements_for_univariate_regression = [record for record in dm.tokenizer._event_counts["EVENT"] if record.upper() != record]
    # cfg.head.tokens_for_univariate_regression = dm.encode(measurements_for_univariate_regression) #
    measurements_for_univariate_regression = dm.train_set.meta_information["measurement_tables"][dm.train_set.meta_information["measurement_tables"]["count_obs"] > 0]["event"].to_list()
    cfg.head.tokens_for_univariate_regression = dm.encode(measurements_for_univariate_regression)
    logging.info(OmegaConf.to_yaml(cfg))
    
    # Experiment pre-trained model checkpoint path
    pre_trained_ckpt_path = cfg.experiment.ckpt_dir + cfg.experiment.run_id + ".ckpt"
    
    # Create experiment
    experiment_type = cfg.experiment.type.replace('-', '').replace(' ', '').lower()
    match experiment_type:
    # (TODO: LBYL)
        case "pretrain" | "causal" | "selfsupervised":
            
            # Training (or causal evaluation) a pre-trained model
            logging.info("="*100)
            logging.info(f"# Pre-training experiment")
            logging.info("="*100)
            
            if Path(pre_trained_ckpt_path).is_file():
                # Load existing experiment from checkpoint
                logging.info(f"Loading a pre-trained model with the checkpoint path {pre_trained_ckpt_path}.")

                # Catch cases where user loads a pre-trained model to pre-train it further
                #   (it will result in checkpointing to a new pre_train_ckpt-V2.ckpt file and then re-loading the original after training)
                if cfg.experiment.train:
                    logging.warning(f"Further training on a checkpoint {pre_trained_ckpt_path} which will create a new checkpoint. Ensure evaluation is not on the original checkpoint.")
                    
                load_from_checkpoint = pre_trained_ckpt_path
                new_checkpoint = None  # Not implemented a versioning control on iterative checkpointing
                
            else:
                # Create new experiment
                logging.info(f"Creating new pre-trained model at the path {pre_trained_ckpt_path}.")
                
                assert dm.is_supervised == False, f"If you are training a new pre-trained model, the data module must not be supervised. Got {dm.is_supervised}."
                assert cfg.experiment.train is True, f"If you are not training a new pre-trained model, please load a valid checkpoint. {pre_trained_ckpt_path} is not valid."
                
                logging.info(f"# This will create / evaluate a pre-trained Foundation Model on a causal (next-event prediction) modelling task.")
                load_from_checkpoint = None
                new_checkpoint = pre_trained_ckpt_path
                
            if cfg.static:
                experiment_instance, Experiment, trainer = setup_mlp_experiment(cfg=cfg,
                                                                                dm=dm, 
                                                                                vocab_size=vocab_size,
                                                                                checkpoint=load_from_checkpoint,
                                                                                logger=logger,
                                                                                )
            elif cfg.static == False:
                experiment_instance, Experiment, trainer = setup_t_mlp_experiment(cfg=cfg,
                                                                                  dm=dm, 
                                                                                  vocab_size=vocab_size,
                                                                                  checkpoint=load_from_checkpoint,
                                                                                  logger=logger,
                                                                                  )
            else:
                logging.warning("To use this benchmarking script you must add ``static`` to config with '+static=<bool>'")
                raise NotImplementedError

        case "zeroshot":
            raise NotImplementedError

        case "fewshot":
            raise NotImplementedError

        case "finetune" | "finetunecr" | "finetunesr" :
            raise NotImplementedError
            
        case _:
            raise NotImplementedError
    
    if cfg.experiment.train:
        logging.info(f"Training model.")
        trainer.fit(experiment_instance, datamodule=dm)

        # Ensure we evaluate on the best/latest version of the model - particularly if we just trained then load the new best checkpoint
        logging.info(f"Re-loading from best cached checkpoint {new_checkpoint}")
        experiment_instance = Experiment.load_from_checkpoint(new_checkpoint)

    # Test model
    if cfg.experiment.test:
        logging.info(f"Testing model.")
        trainer.test(experiment_instance, dataloaders=dm.test_dataloader())

    return experiment_instance, dm

if __name__ == "__main__":
    run()