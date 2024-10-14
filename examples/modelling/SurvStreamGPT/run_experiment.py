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
    logging.info(OmegaConf.to_yaml(cfg))

    # Experiment pre-trained model checkpoint path
    pre_trained_ckpt_path = cfg.experiment.ckpt_dir + cfg.experiment.run_id + ".ckpt"
    
    # Create experiment
    experiment_type = cfg.experiment.type.replace('-', '').replace(' ', '').lower()
    match experiment_type:
    # (TODO: LBYL)
        case "pretrain" | "causal" | "selfsupervised":
            
            if Path(pre_trained_ckpt_path).is_file():
                # Load existing experiment from checkpoint
                logging.info("="*100)
                logging.info(f"# Loading a pre-trained model with the checkpoint path {pre_trained_ckpt_path}. Evaluating causal performance...")
                logging.info("="*100)
                
                if cfg.experiment.train:
                    # Catch cases where user loads a pre-trained model and tries to pre-train it further, as this edge case is not supported 
                    #   (it will result in checkpointing to a new pre_trian_ckpt-V2.ckpt file and then re-loading the original after training)
                    raise FileExistsError(f"Further training on a checkpoint is not supported.")
                    
                load_from_checkpoint = pre_trained_ckpt_path
                
            else:
                # Create new experiment
                logging.info("="*100)
                logging.info(f"Creating new pre-trained model at the path {pre_trained_ckpt_path}.")
                logging.info("="*100)
                
                assert dm.is_supervised == False, f"If you are training a new pre-trained model, the data module must not be supervised. Got {dm.is_supervised}."
                assert cfg.experiment.train is True, f"If you are not training a new pre-trained model, please load a valid checkpoint. {pre_trained_ckpt_path} is not valid."
                
                logging.info(f"# This will create / evaluate a pre-trained Foundation Model on a causal (next-event prediction) modelling task.")
                load_from_checkpoint = None
                
            experiment_instance, Experiment, trainer = setup_causal_experiment(cfg=cfg, dm=dm, vocab_size=vocab_size, checkpoint=load_from_checkpoint)
            new_checkpoint = pre_trained_ckpt_path

            
        case "zeroshot":
            # Evaluate an existing pre-trained experiment
            logging.info("="*100)
            logging.info(f"# Loading a pre-trained model with the checkpoint path {pre_trained_ckpt_path}. Evaluating supervised performance")
            logging.info("="*100)
            
            # Ensure the pre-trained model exists
            if not Path(pre_trained_ckpt_path).is_file():
                raise FileExistsError(f"The pre-trained model with the checkpoint path {pre_trained_ckpt_path} does not exist.")
            
            # Load existing pre-trained checkpoint
            assert cfg.experiment.train is False, f"The zero-shot experiment evaluates a pre-trained causal model on a supervised task without additional training. Ensure training is set to False"

            load_from_checkpoint = pre_trained_ckpt_path
                
            experiment_instance, Experiment, trainer = setup_supervised_experiment(cfg=cfg, dm=dm, checkpoint=load_from_checkpoint)
            new_checkpoint = pre_trained_ckpt_path

        case "finetune" | "fewshot":
            # Create/evaluate a fine-tuned model

            # Get ckpt path for each experiment type
            supervised_run_id =  cfg.experiment.run_id + "_" + cfg.experiment.fine_tune_id   # run id + dataset folder name (i.e. CR_11M_FineTune_CVD)
            supervised_ckpt_path = cfg.experiment.ckpt_dir + supervised_run_id + ".ckpt"
            
            logging.info("="*100)
            logging.info(f"# Fine-tuning experiment, at checkpoint {supervised_ckpt_path}")
            logging.info("="*100)
            
            # Ensure the pre-trained model exists
            if not Path(pre_trained_ckpt_path).is_file():
                raise FileExistsError(f"The pre-trained model with the checkpoint path {pre_trained_ckpt_path} does not exist.")
            
            if Path(supervised_ckpt_path).is_file():
                # Load existing fine-tuned experiment from checkpoint
                logging.info(f"Loading a fine-tuned model with the checkpoint path {supervised_ckpt_path}. Evaluating supervised performance")

                # Catch cases where user loads a fine-tuned model and tries to fine-tune it further, as this edge case is not supported 
                #   (it will result in checkpointing to a new fine_tune_ckpt-V2.ckpt file and then re-loading the original after training)
                assert cfg.experiment.train is False, f"Further training on a checkpoint is not supported."

                load_from_checkpoint = supervised_ckpt_path
                
            else:
                # Create new fine-tuning experiment
                logging.info(f"Creating new fine-tuned model at the path {supervised_ckpt_path}.")
                logging.info(f"This is trained from a checkpointed pre-trained causal experiment, which can be found at {pre_trained_ckpt_path}.")
                
                assert cfg.experiment.train is True, f"If you are not training a new fine-tuned model, please load a valid checkpoint. {pre_trained_ckpt_path} is not valid."
                
                load_from_checkpoint = pre_trained_ckpt_path
                
            experiment_instance, Experiment, trainer = setup_supervised_experiment(cfg=cfg, dm=dm, checkpoint=load_from_checkpoint)
            new_checkpoint = supervised_ckpt_path

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
    