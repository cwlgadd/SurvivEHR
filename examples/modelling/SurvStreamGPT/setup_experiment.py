import pytorch_lightning as pl
import torch
import logging
from CPRD.src.models.survival.task_heads.causal import SurvStreamGPTForCausalModelling
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR, LambdaLR, SequentialLR, ChainedScheduler
from CPRD.src.models.base_callback import Embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wandb")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = "cpu"    # if more informative debugging statements are needed

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class SurvivalExperiment(pl.LightningModule):

    def __init__(self,
                 cfg,
                 vocab_size
                ):
        
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        self.model = SurvStreamGPTForCausalModelling(cfg, vocab_size)

    def forward(self, batch, is_generation=False):
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
                          is_generation=is_generation
                          )

    def training_step(self, batch, batch_idx):
        _, loss_dict, _ = self(batch)        
        for _key in loss_dict.keys():
            self.log(f"train_" + _key, loss_dict[_key], prog_bar=False, logger=True)
        return loss_dict['loss'] 

    def validation_step(self, batch, batch_idx):
        _, loss_dict, _ = self(batch)        
        for _key in loss_dict.keys():
            self.log(f"val_" + _key, loss_dict[_key], prog_bar=False, logger=True)
        return loss_dict['loss'] 

    def test_step(self, batch, batch_idx):
        _, loss_dict, _ = self(batch)        
        for _key in loss_dict.keys():
            self.log(f"test_" + _key, loss_dict[_key], prog_bar=False, logger=True)
        return loss_dict['loss'] 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.optim.learning_rate)
        # Create scheduler with linear warmup followed by Cosine Annealing with warm restarts.
        warmup = int(320000 / self.cfg.data.batch_size)
        lambda1 = lambda step: float(step) / warmup if step < warmup else 1
        scheduler1 = LambdaLR(optimizer, lr_lambda=lambda1)
        scheduler2 = CosineAnnealingWarmRestarts(optimizer, 
                                                 T_0=warmup,
                                                 T_mult=2,
                                                 eta_min=self.cfg.optim.learning_rate / 5)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup])      
        lr_scheduler_config = {
            # ReduceLROnPlateau
            # "scheduler": ReduceLROnPlateau(optimizer),         
            # "interval": "step",                                                         # The unit of the scheduler's step size
            # "frequency": 2 * self.cfg.optim.val_check_interval,                         # How many epochs/steps should pass between calls to `scheduler.step()`
            # "monitor": "val_loss",                                                      # Metric to monitor for scheduler
            # "strict": False,                                                            # Enforce that "val_loss" is available when the scheduler is updated
            "scheduler": scheduler,                                                     # The scheduler instance
            "interval": "step",                                                         # The unit of the scheduler's step size
            "frequency": 1,                                                             # How many epochs/steps should pass between calls to `scheduler.step()`
            "name": 'Scheduler',                                                        # For `LearningRateMonitor`, specify a custom logged name
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }

def setup_survival_experiment(cfg, dm, vocab_size):

    _model = SurvivalExperiment(cfg=cfg, vocab_size=vocab_size)
    logging.debug(_model)

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
        monitor="val_loss",
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    callbacks = [checkpoint_callback,
                 lr_monitor,
                 ]

    # ... custom callbacks
    val_batch = next(iter(dm.val_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))

    # Hidden state embedding
    embedding_callback = Embedding(val_batch=val_batch,
                                   test_batch=test_batch
                                  )
    callbacks.append(embedding_callback)

    # ... optional callbacks
    if cfg.optim.early_stop:
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
        devices=torch.cuda.device_count(),
        # gradient_clip_val=0.5
    )

    return _model, _trainer