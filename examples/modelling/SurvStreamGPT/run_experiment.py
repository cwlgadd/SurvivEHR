from omegaconf import DictConfig, OmegaConf
import os
import hydra
import torch
import logging
from pathlib import Path
from CPRD.data.foundational_loader import FoundationalDataModule
from CPRD.examples.modelling.SurvStreamGPT.setup_causal_experiment import setup_causal_experiment, CausalExperiment
# from CPRD.examples.modelling.SurvStreamGPT.setup_zeroshot_experiment import setup_zeroshot_experiment, ZeroShotExperiment
from CPRD.examples.modelling.SurvStreamGPT.setup_supervised_experiment import setup_supervised_experiment, SupervisedExperiment

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
    supervised = True if cfg.experiment.fine_tune_outcomes is not None else False
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
                                supervised=supervised
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
    logging.debug(OmegaConf.to_yaml(cfg))

    # Experiment pre-trained model checkpoint path
    pre_trained_ckpt_path = cfg.experiment.ckpt_dir + cfg.experiment.run_id + ".ckpt"
    
    # Create experiment
    match cfg.experiment.type.replace('-', '').replace(' ', '').lower():
    # (TODO: LBYL)
        case "pretrain" | "causal" | "selfsupervised":
            if Path(pre_trained_ckpt_path).is_file():
                # Load existing experiment from checkpoint
                
                if cfg.experiment.train:
                    # Catch cases where user loads a pre-trained model and tries to pre-train it further, as this edge case is not supported 
                    #   (it will result in checkpointing to a new pre_trian_ckpt-V2.ckpt file and then re-loading the original after training)
                    raise FileExistsError(f"A pre-trained causal experiment with the checkpoint path {pre_trained_ckpt_path} already exists. Further training on a checkpoint is not yet supported.")
                    
                logging.info("="*100)
                logging.info(f"# A pre-trained model with the checkpoint path {pre_trained_ckpt_path} already exists, loading.")
                logging.info("="*100)
                load_from_checkpoint = pre_trained_ckpt_path
                
            else:
                logging.info(f"A pre-trained model cannot be found at checkpoint path {pre_trained_ckpt_path}.")
                # Create new experiment
                assert dm.is_supervised == False, f"If you are training a new pre-trained model, the data module must not be supervised. Got {dm.is_supervised}."
                assert cfg.experiment.train is True, f"If you are not training a new pre-trained model, please load a valid checkpoint. {pre_trained_ckpt_path} is not valid."
                
                logging.info("="*100)
                logging.info(f"# Creating a new pre-trained/causal experiment.")
                logging.info(f"# This will create / evaluate a pre-trained Foundation Model on a causal (next-event prediction) modelling task.")
                logging.info(f"# This can be found at {pre_trained_ckpt_path}")
                logging.info("="*100)
                load_from_checkpoint = None
                
            experiment_instance, Experiment, trainer = setup_causal_experiment(cfg=cfg, dm=dm, vocab_size=vocab_size, checkpoint=load_from_checkpoint)
            new_checkpoint = pre_trained_ckpt_path
            
        case "finetune" | "supervised" | "clinicalpredictionmodel" | "cpm":

            # Ensure the pre-trained model exists
            if not Path(pre_trained_ckpt_path).is_file():
                raise FileExistsError(f"The pre-trained model with the checkpoint path {pre_trained_ckpt_path} does not exist.")

            fine_tune_run_id =  cfg.experiment.run_id + f"_{cfg.data.path_to_ds.split('/')[-2]}"   # run id + dataset folder name (i.e. CR_11M_FineTune_CVD)
            fine_tune_ckpt_path = cfg.experiment.ckpt_dir + fine_tune_run_id + ".ckpt"
            if Path(fine_tune_ckpt_path).is_file():
                # Load existing fine-tuned experiment from checkpoint

                if cfg.experiment.train:
                    # Catch cases where user loads a fine-tuned model and tries to fine-tune it further, as this edge case is not supported 
                    #   (it will result in checkpointing to a new fine_tune_ckpt-V2.ckpt file and then re-loading the original after training)
                    raise FileExistsError(f"A fine-tuned supervised experiment with the checkpoint path {fine_tune_ckpt_path} already exists. Further training on a checkpoint is not yet supported.")

                load_from_checkpoint = fine_tune_ckpt_path
                
            else:
                # Create new fine-tuning experiment
                logging.info("="*100)
                logging.info(f"# Creating a new supervised experiment from the existing pre-trained model")
                logging.info(f"# This pre-trained causal experiment can be found at {pre_trained_ckpt_path}.")
                if cfg.experiment.train:
                    logging.info(f"# This will fine-tune the pre-trained Foundation Model on a supervised (clinical prediction model) modelling task.")
                logging.info(f"# This can be found at {fine_tune_ckpt_path}")
                logging.info("="*100)
                load_from_checkpoint = pre_trained_ckpt_path
                
            experiment_instance, Experiment, trainer = setup_supervised_experiment(cfg=cfg, dm=dm, checkpoint=load_from_checkpoint)
            new_checkpoint = fine_tune_ckpt_path
            
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

    return experiment_instance.model, dm

if __name__ == "__main__":
    run()
    