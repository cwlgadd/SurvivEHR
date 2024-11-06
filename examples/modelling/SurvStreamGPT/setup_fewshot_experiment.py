import pytorch_lightning as pl
import torch
import logging
from CPRD.src.models.survival.task_heads.causal import SurvStreamGPTForCausalModelling
# from CPRD.data.foundational_loader import convert_batch_to_none_causal
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR, LambdaLR, SequentialLR, ChainedScheduler
from CPRD.src.models.base_callback import Embedding
# from CPRD.src.models.survival.custom_callbacks.single_risk_clinical_prediction_model_eval import PerformanceMetrics
from CPRD.src.models.survival.custom_callbacks.clinical_prediction_model import PerformanceMetrics

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class FewShotExperiment(pl.LightningModule):

    def __init__(self,
                 cfg,
                 vocab_size,
                 freeze_body=False
                ):
        
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = SurvStreamGPTForCausalModelling(cfg, vocab_size)

        self.freeze_body = freeze_body
        if self.freeze_body:
            logging.info(f"Fixing Transformer parameters and training only head.")
        self.parameters_head = list(self.model.surv_layer.parameters())
        if self.model.value_weight > 0:
            self.parameters_head += list(self.model.value_layer.parameters())
        self.parameters_head = set(self.parameters_head)
        self.parameters_body = (param for param in self.model.parameters() if param not in self.parameters_head)
        
    def forward(self, batch, return_loss=True, return_generation=False):

        # inputs
        covariates = batch["static_covariates"].to(self.device)
        tokens = batch['tokens'].to(self.device)                           # torch.Size([bsz, seq_len])       
        ages = batch['ages'].to(self.device)                               # torch.Size([bsz, seq_len])
        values = batch['values'].to(self.device)                           # torch.Size([bsz, seq_len])
        attention_mask = batch['attention_mask'].to(self.device)           # torch.Size([bsz, seq_len])
        
        # targets
        target_token = batch['target_token'].reshape((-1,1)).to(self.device)               # torch.Size([bsz, 1])
        target_age_delta = batch['target_age_delta'].reshape((-1,1)).to(self.device)       # torch.Size([bsz, 1]),
        target_value = batch['target_value'].reshape((-1,1)).to(self.device)               # torch.Size([bsz, 1])
        bsz, seq_len = tokens.shape

        # Get hidden representation along sequence:                         torch.Size([bsz, seq_len, hid_dim])
        hidden_states =  self.model.transformer(tokens=tokens,
                                                ages=ages,
                                                values=values,
                                                covariates=covariates,
                                                attention_mask=attention_mask
                                                )

        # We can take the last hidden state as all padded states share the same values as the last observed sequence element's hidden state
        in_hidden_state = hidden_states[:,[-1],:self.model.n_embd - self.model.n_embd_private]       # torch.Size([bsz, 1, n_embd])

        # survival time to event head (survival curve until next token)
        surv_dict, losses_desurv = self.model.surv_layer.predict(in_hidden_state,
                                                                 target_tokens=target_token,
                                                                 target_ages=target_age_delta, 
                                                                 attention_mask=None,
                                                                 is_generation=True,
                                                                 return_loss=return_loss,
                                                                 return_cdf=return_generation,
                                                                 )
        
        if return_loss:
            loss = torch.sum(torch.stack(losses_desurv))         # losses are returned as a list, as the Single-Risk head is many DeSurv models in parallel, combine

            if torch.isnan(loss):
                logging.warning(f"Invalid loss {loss}: with target tokens {target_tokens}, target values {target_values} and target ages {target_ages}")
                logging.warning(f"from hidden state {torch.sum(in_hidden_state, axis=2)/128}")
                raise NotImplementedError
        else:
            loss = None
            
        return {"surv": surv_dict}, {"loss": loss}, in_hidden_state[:, [-1], :]
            
        # # regression head (values of next token if applicable)
        # values_dist, loss_values = self.model.value_layer.predict(in_hidden_state[:,:, self.model.n_embd_private:],
        #                                                           target_tokens=target_tokens,
        #                                                           target_values=target_values,
        #                                                           attention_mask=target_attention_mask,
        #                                                           is_generation=is_generation,
        #                                                           return_loss=return_loss,
        #                                                           return_value_dist=return_generation,
        #                                                           )

        # if return_loss:
        #      # losses are returned as a list, as the Single-Risk head is many DeSurv models in parallel, combine
        #     loss_desurv = torch.sum(torch.stack(losses_desurv))
        #     # and take weighted sum of heads
        #     loss = (self.model.surv_weight * loss_desurv)  + (self.model.value_weight * loss_values)

        #     if torch.isnan(loss):
        #         logging.warning(f"Invalid loss {loss}: ({loss_desurv} and {loss_values}), " + \
        #                         f"with target tokens {target_tokens}, target values {target_values} and target ages {target_ages}")
        #         raise NotImplementedError
        # else:
        #     loss_desurv = None
        #     loss = None
            
        # outputs = {"surv": surv_dict,
        #            "values_dist": values_dist}
        # losses = {"loss": loss,
        #           "loss_desurv": loss_desurv,                  
        #          }
        # if self.model.value_weight > 0:
        #     losses = {**losses,
        #               "loss_values": loss_values}
        
        # return outputs, losses, in_hidden_state

    def training_step(self, batch, batch_idx):
        _, loss_dict, _ = self(batch)   
        for _key in loss_dict.keys():
            self.log(f"train_" + _key, loss_dict[_key], prog_bar=False, logger=True, sync_dist=True)
        return loss_dict['loss'] 

    def validation_step(self, batch, batch_idx):
        _, loss_dict, _ = self(batch)   
        for _key in loss_dict.keys():
            self.log(f"val_" + _key, loss_dict[_key], prog_bar=False, logger=True, sync_dist=True)
        return loss_dict['loss'] 

    def test_step(self, batch, batch_idx):
        _, loss_dict, _ = self(batch)   
        for _key in loss_dict.keys():
            self.log(f"test_" + _key, loss_dict[_key], prog_bar=False, logger=True, sync_dist=True)
        return loss_dict['loss'] 

    def configure_optimizers(self):

        parameters = self.parameters_head if self.freeze_body else self.parameters()
        optimizer = torch.optim.AdamW(parameters, lr=self.cfg.optim.learning_rate)

        freq = 1
        match self.cfg.optim.scheduler.lower():
            case 'cawarmrestarts':
                logging.info("Using CosineAnnealingWarmRestarts scheduler")
                scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                        T_0=int(self.cfg.optim.scheduler_periods),
                                                        T_mult=2,
                                                        eta_min=self.cfg.optim.learning_rate / 5)
            case 'cosineannealinglr':
                logging.info("Using CosineAnnealingLR scheduler")
                scheduler = CosineAnnealingLR(optimizer,
                                              T_max=self.cfg.optim.lr_cosine_decay_period / self.cfg.data.batch_size,
                                              eta_min=self.cfg.optim.learning_rate / 5)
            case 'reduceonplateau':
                logging.info("Using ReduceLROnPlateau scheduler")
                scheduler = ReduceLROnPlateau(optimizer, factor=0.9, min_lr=self.cfg.optim.learning_rate/10, patience=20)
                freq = self.cfg.optim.val_check_interval
            case _:
                raise NotImplementedError

        if self.cfg.optim.scheduler_warmup:
            logging.info(f"Using warm-up in scheduler for {self.cfg.optim.scheduler_periods} steps")
            # Create scheduler with linear warmup followed by Cosine Annealing with warm restarts.
            warmup = int(self.cfg.optim.scheduler_periods)
            lambda1 = lambda step: float(step) / warmup if step < warmup else 1
            scheduler_warm = LambdaLR(optimizer, lr_lambda=lambda1)
            scheduler = SequentialLR(optimizer, schedulers=[scheduler_warm, scheduler], milestones=[warmup])     
        else:
            logging.info("Not using warm-up in scheduler")
            
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

def setup_fewshot_experiment(cfg, dm, vocab_size, checkpoint=None):

    assert dm.is_supervised, "Datamodule for must be supervised for `setup_fewshot_experiment`."
    assert cfg.experiment.fine_tune_outcomes is not None, "Must provide outcome list for `setup_fewshot_experiment`."

    # Which tokens we want to predict as outcomes. 
    #    In the few-shot/zero-shot setting these are used in the clinical prediction model callback to collapse the `vocab_size` outcomes into outcomes and none-outcome
    #    but the model still predicts all `vocab_size` outcomes.
    outcome_tokens =  dm.encode(cfg.experiment.fine_tune_outcomes)
    logging.info(f"Fine tune outcomes encode into tokens {outcome_tokens}")

    if checkpoint is not None:
        fewshot_experiment = FewShotExperiment.load_from_checkpoint(checkpoint, cfg=cfg)
    else:
        fewshot_experiment = FewShotExperiment(cfg=cfg, vocab_size=vocab_size, freeze_body=False)
    logging.debug(fewshot_experiment)

    # Initialize wandb logger
    if cfg.experiment.log == True:
        logger = pl.loggers.WandbLogger(project=cfg.experiment.project_name,
                                        name=cfg.experiment.run_id + "_" + cfg.experiment.fine_tune_id, 
                                        save_dir=cfg.experiment.log_dir
                                        )
    else:
        logger = None

    # Make all callbacks
    ####################
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.experiment.ckpt_dir,
        filename=cfg.experiment.run_id + "_" + cfg.experiment.fine_tune_id, 
        verbose=cfg.experiment.verbose,
        monitor="val_loss",
    )
    # LR monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    # Add callbacks which apply to outcome prediction tasks
    metric_callback = PerformanceMetrics(outcome_tokens=outcome_tokens, log_ctd=True, log_ibs=True, log_inbll=True)
    # 
    callbacks = [checkpoint_callback, lr_monitor, metric_callback]

    # Optional callbacks
    if not fewshot_experiment.freeze_body:
        # Hidden state embedding
        logging.debug("Creating hidden state embedding callback - as we are fine-tuning the entire model the latent embeddings will be changed")
        # We do not want to plot every single validation/test sample, so pass in a batch of each
        embedding_callback = Embedding(val_batch=next(iter(dm.val_dataloader())),
                                       test_batch=next(iter(dm.test_dataloader()))
                                      )
        callbacks.append(embedding_callback)

    if cfg.optim.early_stop:
        # Early stopping callback
        logging.debug("Creating early stopping callback")
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss", mode="min",
            min_delta=0,
            patience=cfg.optim.early_stop_patience,
            verbose=cfg.experiment.verbose,
        )
        callbacks.append(early_stop_callback)

    _trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.optim.num_epochs,
        log_every_n_steps=cfg.optim.log_every_n_steps,
        val_check_interval=cfg.optim.val_check_interval,
        limit_val_batches=cfg.optim.limit_val_batches,
        limit_test_batches=cfg.optim.limit_test_batches,
    )

    return fewshot_experiment, FewShotExperiment, _trainer