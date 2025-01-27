import pytorch_lightning as pl
import torch
import logging
import numpy as np
from CPRD.src.modules.head_layers.survival.desurv import ODESurvMultiple, ODESurvSingle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from CPRD.src.models.survival.custom_callbacks.single_risk_clinical_prediction_model_eval import PerformanceMetrics

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class DeSurvExperiment(pl.LightningModule):

    def __init__(self,
                 outcome_tokens,
                 num_covariates,
                 vocab_size=265,
                 risk_model='single risk',         # 'single-risk', 'competing-risk', or (TODO) False (for case of predicting value but not survival risk)
                ):
        
        super().__init__()
        self.save_hyperparameters()
        self.outcome_tokens = outcome_tokens
        self.vocab_size = vocab_size

        # Create new survival head
        match risk_model.replace('-', '').replace(' ', '').lower():
            # Removing padding token from vocab size as this is not considered an event in either case
            case "singlerisk" | "sr":
                # Combine each of the given outcomes into a single event, e.g. all events may constitute some form of cardiovascular disease and treat it as a single risk
                # self.reduce_to_outcomes = lambda x: sum([torch.where(x==i, 1, 0) for i in outcome_tokens])
                # self.reduced_outcome_tokens = [1]
                raise NotImplementedError
            case "competingrisk" | "cr":
                self.sr_ode = ODESurvMultiple(cov_dim=num_covariates+vocab_size-2,
                                              hidden_dim=32,
                                              num_risks=len(outcome_tokens)+1, 
                                              device="cuda") 
                self.reduce_to_outcomes = lambda x: sum([torch.where(x==i, idx+1, 0) for idx, i in enumerate(outcome_tokens)])
                self.reduced_outcome_tokens = [idx + 1 for idx, i in enumerate(outcome_tokens)]
            case _:
                raise ValueError(f"Survival head must be either 'single-risk' or 'competing-risk'")

        self._time_scale = 365*5
    

    def make_batch_cross_sectional(self, batch):
        
        # inputs
        covariates = batch["static_covariates"].to(self.device)
        tokens = batch['tokens'].to(self.device)                           # torch.Size([bsz, seq_len])       
        ages = batch['ages'].to(self.device)                               # torch.Size([bsz, seq_len])
        values = batch['values'].to(self.device)                           # torch.Size([bsz, seq_len])
        attention_mask = batch['attention_mask'].to(self.device)           # torch.Size([bsz, seq_len])
        
        # targets
        target_token = batch['target_token'].reshape((-1,1)).to(self.device)               # torch.Size([bsz, 1])
        target_token = self.reduce_to_outcomes(target_token)                               # torch.Size([bsz, 1])
        target_age_delta = batch['target_age_delta'].reshape((-1)).to(self.device)       # torch.Size([bsz, 1]),
        target_value = batch['target_value'].reshape((-1,1)).to(self.device)               # torch.Size([bsz, 1])

        bsz = covariates.shape[0]
        
        # Get a binary vector of vocab_size elements, which indicate if patient has any history of a condition (at any time, as long as it fits study criteria)
        # Note, 0 and 1 are PAD and UNK tokens which arent required
        token_binary = torch.zeros((bsz, self.vocab_size-2), device=self.device)
        for sample_idx in range(bsz):
            for tkn_idx in range(2, self.vocab_size):
                if tkn_idx in tokens[sample_idx, :]:
                    token_binary[sample_idx, tkn_idx-2] = 1
    
        batch_input = torch.hstack((covariates, token_binary))
        
        # Target
        ########
        # targets = batch["target_token"].numpy()
        batch_targets = torch.zeros((bsz,1), device=self.device)
        for sample_idx in range(bsz):
            batch_targets[sample_idx] = 1 if target_token[sample_idx] in self.outcome_tokens else 0

        return batch_input.to(self.device), target_age_delta.to(self.device), batch_targets.to(self.device)
        
    def forward(self, batch, is_generation=False, return_loss=True, return_generation=False):
        # Because of how DeSurv is coded we have the loss returned in the forward, so we have some redundancy

        X, t, k = self.make_batch_cross_sectional(batch)
        t /= self._time_scale
        bsz = X.shape[0]

        losses_desurv = [self.sr_ode.loss(X, t, k) / bsz]

        if return_loss:
            loss = torch.sum(torch.stack(losses_desurv))                                  # losses are returned as a list, as the Single-Risk head is many DeSurv models in parallel, combine

            if torch.isnan(loss):
                logging.warning(f"Invalid loss {loss}: with inputs {x}, target times {t} and target events {k}")
                raise NotImplementedError
        else:
            loss = None
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch)   
        self.log(f"train_loss", loss, prog_bar=False, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)   
        self.log(f"val_loss", loss, prog_bar=False, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self(batch)   
        self.log(f"test_loss", loss, prog_bar=False, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):

        parameters = self.parameters()
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)

        logging.info("Using ReduceLROnPlateau scheduler")
        scheduler = ReduceLROnPlateau(optimizer, factor=0.9, min_lr=1e-4, patience=20)
        freq = 50

        lr_scheduler_config = {
            "frequency": freq,                                                          # How many epochs/steps should pass between calls to `scheduler.step()`
            "scheduler": scheduler,                                                     # The scheduler instance
            "interval": "step",                                                         # The unit of the scheduler's step size
            "monitor": "val_loss",                                                      # Metric to monitor for scheduler, if needed
            "strict": False,                                                            # Enforce that "val_loss" is available when the scheduler is updated
            "name": 'Scheduler',                                                        # For `LearningRateMonitor`, specify a custom logged name
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }

def setup_desurv_experiment(cfg, dm, risk_model="cr"):

    assert dm.is_supervised, "Datamodule for must be supervised for `setup_finetune_experiment` ."
    assert cfg.experiment.fine_tune_outcomes is not None, "Must provide outcome list for `setup_finetune_experiment`."

    # Which tokens we want to predict as outcomes. 
    #    In the fine-tuning setting these are used to construct a new head which can be fine-tuned.
    #    TODO: a new clinical prediction model callback then needs to be made (or existing one editted) for this new case
    outcome_tokens =  dm.encode(cfg.experiment.fine_tune_outcomes)
    outcome_dict = {_key: _value for _key, _value in zip(cfg.experiment.fine_tune_outcomes, outcome_tokens)}
    logging.info(f"Creating {risk_model} DeSurv experiment with outcomes {outcome_dict}")

    desurv_experiment = DeSurvExperiment( outcome_tokens=outcome_tokens, num_covariates=16, vocab_size=dm.train_set.tokenizer.vocab_size, risk_model=risk_model)
    logging.debug(desurv_experiment)

    # Initialize wandb logger
    if cfg.experiment.log == True:
        logger = pl.loggers.WandbLogger(project=cfg.experiment.project_name,
                                        name=cfg.experiment.run_id, 
                                        save_dir=cfg.experiment.log_dir
                                        )
    else:
        logger = None

    # Make all callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.experiment.ckpt_dir,
        filename=cfg.experiment.run_id, 
        verbose=cfg.experiment.verbose,
        monitor="val_loss",
    )
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    callbacks = [checkpoint_callback,
                 lr_monitor,
                 ]

    # ... custom callbacks
    val_batch = next(iter(dm.val_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))

    if cfg.optim.early_stop:
        logging.debug("Creating early stopping callback")
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss", mode="min",
            min_delta=0,
            patience=cfg.optim.early_stop_patience,
            verbose=cfg.experiment.verbose,
        )
        callbacks.append(early_stop_callback)

    # Add callbacks which apply to outcome prediction tasks
    # metric_callback = PerformanceMetrics(outcome_tokens=finetune_experiment.reduced_outcome_tokens,
    #                                      log_ctd=True,
    #                                      log_ibs=True,
    #                                      log_inbll=True,
    #                                      plot_outcome_curves=True,
    #                                      reduce_to_outcomes = finetune_experiment.reduce_to_outcomes,
    #                                      )
    # callbacks.append(metric_callback)

    _trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.optim.num_epochs,
        log_every_n_steps=cfg.optim.log_every_n_steps,
        val_check_interval=cfg.optim.val_check_interval,
        limit_val_batches=cfg.optim.limit_val_batches,
        limit_test_batches=cfg.optim.limit_test_batches,
        # devices=torch.cuda.device_count(),
        # gradient_clip_val=0.5
    )

    return desurv_experiment, DeSurvExperiment, _trainer