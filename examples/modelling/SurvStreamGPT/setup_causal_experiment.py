import pytorch_lightning as pl
import torch
import logging
from CPRD.src.models.survival.task_heads.causal import SurvStreamGPTForCausalModelling
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR, LambdaLR, SequentialLR, ConstantLR, ChainedScheduler, ExponentialLR
from CPRD.src.models.base_callback import Embedding
import math
import numpy as np

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class CausalExperiment(pl.LightningModule):

    def __init__(self,
                 cfg,
                 vocab_size,
                ):
        
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = SurvStreamGPTForCausalModelling(cfg, vocab_size)

    def forward(self, batch, is_generation=False, return_loss=True, return_generation=False):
        # Because of how DeSurv is coded we have the loss returned in the forward, so we have some redundancy

        tokens = batch['tokens'].to(self.device)
        ages = batch['ages'].to(self.device)
        values = batch['values'].to(self.device)
        covariates = batch["static_covariates"].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)   

        return self.model(tokens,
                          ages,
                          values,
                          covariates,
                          attention_mask,
                          is_generation=is_generation,
                          return_loss=return_loss,
                          return_generation=return_generation
                          )

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.optim.learning_rate)

        schedulers = []
        milestones = []

        # Warm-up phase
        warmup_period = 0
        if self.cfg.optim.scheduler_warmup:
            logging.info(f"Using warm-up in scheduler for {self.cfg.optim.scheduler_periods} steps")
            
            # Create scheduler with linear warmup followed by Cosine Annealing with warm restarts.
            warmup_period = int(self.cfg.optim.scheduler_periods)
            lambda1 = lambda step: float(step) / warmup_period if step < warmup_period else 1
            scheduler_warm = LambdaLR(optimizer, lr_lambda=lambda1)
            
            schedulers.append(scheduler_warm)
            milestones.append(warmup_period)
        else:
            logging.info(f"Not using warm-up in scheduler")

        # Annealing phase (to avoid local optima)
        freq = 1
        match self.cfg.optim.scheduler.lower():
            case 'decaycawarmrestarts':
                logging.info(f"Using Decayed Cosine Annealing with Warm Restarts in scheduler")

                a = self.cfg.optim.scheduler_periods      # period of first restart
                r = 2.0
                scheduler = CosineAnnealingWarmRestartsDecay(optimizer, 
                                                        T_0=int(a),
                                                        T_mult=int(r),
                                                        eta_min=self.cfg.optim.learning_rate / 5,
                                                        decay=self.cfg.optim.learning_rate_decay)

                # If we want to add another phase after this, calculate how long this phase should be to not end half way through a restart
                # Forms a geometric series, calculate length based on how many restarts we want (hard coded for now - it probably makes v. little difference to our application)
                num_restarts = 5
                anneal_period = (a * (1- r**(num_restarts+1) )) / (1-r)
                
                schedulers.append(scheduler)
                
            case 'cawarmrestarts':
                logging.info(f"Using Cosine Annealing with Warm Restarts in scheduler")
                
                a = self.cfg.optim.scheduler_periods      # period of first restart
                r = 2.0
                scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                        T_0=int(a),
                                                        T_mult=int(r),
                                                        eta_min=self.cfg.optim.learning_rate / 5)

                # If we want to add another phase after this, calculate how long this phase should be to not end half way through a restart
                # Forms a geometric series, calculate length based on how many restarts we want (hard coded for now - it probably makes v. little difference to our application)
                num_restarts = 2
                anneal_period = (a * (1- r**(num_restarts+1) )) / (1-r)
                
                schedulers.append(scheduler)
                
            case 'cosineannealinglr':
                logging.info(f"Using Cosine Annealing in scheduler")
                
                period = self.cfg.optim.scheduler_periods
                scheduler = CosineAnnealingLR(optimizer,
                                              T_max=int(period),
                                              eta_min=self.cfg.optim.learning_rate / 5)

                schedulers.append(scheduler)
                
            case _:
                pass


        # Fine-tuning phase
        # TODO: this is more important if not using a decay
        fine_tuning_scheduler = False
        if fine_tuning_scheduler:
            logging.info("Followed by a fine-tuning scheduler for the remaining steps")
            
            # make scheduler
        
            # scheduler = ReduceLROnPlateau(optimizer)
            # scheduler = ConstantLR(optimizer, factor=1.0)
            # scheduler = CosineAnnealingLR(optimizer,
            #                               T_max=int(remaining_batches),
            #                               eta_min=self.cfg.optim.learning_rate / 2.5)
        
            schedulers.append(scheduler)
            milestones.append(anneal_period)
        

        # Combine
        scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

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

    def on_train_epoch_end(self):
        current_batchch = self.trainer.current_epoch

def setup_causal_experiment(cfg, dm, vocab_size, checkpoint=None):

    if checkpoint is None:
        causal_experiment = CausalExperiment(cfg=cfg, vocab_size=vocab_size)
    else:
        causal_experiment = CausalExperiment.load_from_checkpoint(checkpoint,
                                                                  cfg=cfg, 
                                                                  )
    logging.debug(causal_experiment)

    # Initialize wandb logger
    if cfg.experiment.log == True:
        logger = pl.loggers.WandbLogger(project=cfg.experiment.project_name,
                                        name=cfg.experiment.run_id,
                                        job_type='train',
                                        save_dir=cfg.experiment.log_dir
                                        )
    else:
        logger = None

    # Make all callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.experiment.ckpt_dir,
        filename=cfg.experiment.run_id,
        verbose=cfg.experiment.verbose,
        monitor="val_loss_desurv",
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    callbacks = [checkpoint_callback,
                 lr_monitor,
                 ]

    # ... custom callbacks
    val_batch = next(iter(dm.val_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))

    # Hidden state embedding
    logging.info("Creating hidden state embedding callback")
    embedding_callback = Embedding(val_batch=val_batch,
                                   test_batch=test_batch
                                  )
    # callbacks.append(embedding_callback)

    # ... optional callbacks
    if cfg.optim.early_stop:
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss_desurv", mode="min",
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
        # devices=torch.cuda.device_count(),
        # gradient_clip_val=0.5
    )

    return causal_experiment, CausalExperiment, _trainer


class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    def __init__(self, 
                 optimizer,
                 T_0, 
                 T_mult=1,
                 eta_min=0, 
                 last_epoch=-1, 
                 verbose=False, 
                 decay=1):
        
        super().__init__(optimizer,
                         T_0, 
                         T_mult=T_mult,
                         eta_min=eta_min, 
                         last_epoch=last_epoch, 
                         verbose=verbose)
        
        self.decay = decay
        self.initial_lrs = self.base_lrs
        self._eta_min = eta_min
        
    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            else:
                n = 0

            new_base_lrs = [np.maximum(self._eta_min, initial_lrs * (self.decay**n)) for initial_lrs in self.initial_lrs]
            
            self.base_lrs = new_base_lrs # [initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs]

        super().step(epoch)