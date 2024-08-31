from omegaconf import DictConfig, OmegaConf
import os
import hydra
import torch
import logging
from pathlib import Path
from CPRD.data.foundational_loader import FoundationalDataModule
from CPRD.examples.modelling.SurvStreamGPT.setup_causal_experiment import setup_causal_experiment, CausalExperiment
from CPRD.examples.modelling.SurvStreamGPT.setup_zeroshot_experiment import setup_zeroshot_experiment, ZeroShotExperiment
from CPRD.examples.modelling.SurvStreamGPT.setup_finetune_experiment import setup_finetune_experiment, FineTuneExperiment

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
    # ... Extract the measurements, using the fact that the diagnoses are all up upper case. This is needed for automatically setting the configuration below
    #     encode into the list of univariate measurements to model with Normal distribution
    # measurements_for_univariate_regression = [record for record in dm.tokenizer._event_counts["EVENT"] if record.upper() != record]
    # cfg.head.tokens_for_univariate_regression = dm.encode(measurements_for_univariate_regression) #
    measurements_for_univariate_regression = dm.train_set.meta_information["measurement_tables"][dm.train_set.meta_information["measurement_tables"]["count_obs"] > 0]["event"].to_list()
    cfg.head.tokens_for_univariate_regression = dm.encode(measurements_for_univariate_regression)
    logging.debug(OmegaConf.to_yaml(cfg))

    # Experiment pre-trained model checkpoint path
    pre_trained_ckpt_path = cfg.experiment.ckpt_dir + cfg.experiment.run_id + ".ckpt"
    checkpoint = pre_trained_ckpt_path
    
    # Create experiment
    match cfg.experiment.type.replace('-', '').replace(' ', '').lower():
    # (TODO: LBYL)
    #   Assertions are to avoid accidentally creating a new model with a -V1 etc suffix, then re-loading a previous run's best checkpoint later
        case "pretrain":
            logging.info("\nCreating Causal experiment. This will create / evaluate a pre-trained Foundation Model on a causal (next-event prediction) modelling task")
            logging.info(f"This model can be found at checkpoint {pre_trained_ckpt_path}\n")
            experiment_instance, Experiment, trainer = setup_causal_experiment(cfg=cfg, dm=dm, vocab_size=vocab_size)
            
            # Ensure no existing model
            if cfg.experiment.train:
                if Path(pre_trained_ckpt_path).is_file():
                    raise FileExistsError(f"A pre-trained experiment with the checkpoint path {pre_trained_ckpt_path} already exists.")
            else:
                assert Path(pre_trained_ckpt_path).is_file(), f"If we are not training a new pre-trained model there must already be a valid checkpoint {pre_trained_ckpt_path} to load."
                
        case "zeroshot":
            
            logging.info("\nCreating Zero-Shot experiment. This will evaluate the pre-trained Foundation Model on a clinical prediction model task.")
            experiment_instance, Experiment, trainer = setup_zeroshot_experiment(pre_trained_ckpt_path, cfg=cfg, dm=dm)
            # 
            assert cfg.experiment.train == False, f"The zero-shot experiment is only for evaluation, not training. Ensure config.experiment.train==False, not {cfg.experiment.train}."
            assert Path(pre_trained_ckpt_path).is_file(), f"To evaluate a pre-trained model, please ensure there is a valid pre-trained checkpoint {pre_trained_ckpt_path} to load."


        case "finetune":
            logging.info("\nCreating Fine-Tuning experiment. This will create / evaluate a fine-tuned model on a pre-existing pre-trained Foundation Model on a clinical prediction model task.")
            fine_tune_checkpoint_name = cfg.experiment.run_id + f"_{cfg.data.path_to_ds.split('/')[-2]}"    # run id + dataset folder name (i.e. CR_11M_FineTune_CVD)
            logging.info(f"The fine-tuned model can be found at {fine_tune_checkpoint_name}.ckpt\n")
            
            experiment_instance, Experiment, trainer = setup_finetune_experiment(pre_trained_ckpt_path, cfg=cfg, dm=dm, checkpoint_finetune=fine_tune_checkpoint_name)
            
            # Ensure there is an existing pre-trained model to fine-tune
            if cfg.experiment.train:
                if not Path(pre_trained_ckpt_path).is_file():
                    raise FileExistsError(f"The pre-trained model with the checkpoint path {pre_trained_ckpt_path} does not exist.")
                if Path(fine_tune_checkpoint_name).is_file():
                    raise FileExistsError(f"A fine-tuned model with the checkpoint path {fine_tune_checkpoint_name} already exists.")
            else:
                assert Path(fine_tune_checkpoint_name).is_file(), f"If we are not training a new fine-tuned model there must already be a valid checkpoint {fine_tune_checkpoint_name} to load."

            # If we are fine-tuning the model, then after any training we will load in the fine_tune_checkpoint_path checkpoint
            checkpoint = fine_tune_checkpoint_name
            
        case _:
            raise NotImplementedError
    
    if cfg.experiment.train:
        logging.info(f"Training model.")
        trainer.fit(experiment_instance, datamodule=dm)

        # Ensure we evaluate on the best/latest version of the model - particularly if we just trained then load the new best checkpoint
        logging.info(f"Re-loading from best cached checkpoint {checkpoint}")
        experiment_instance = Experiment.load_from_checkpoint(checkpoint)

    # Test model
    if cfg.experiment.test:
        logging.info(f"Testing model.")
        trainer.test(experiment_instance, dataloaders=dm.test_dataloader())

    return experiment_instance.model, dm

if __name__ == "__main__":
    run()
    