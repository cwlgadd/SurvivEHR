import pytorch_lightning as pl
import torch
import logging
from CPRD.src.models.survival.task_heads.causal import SurvStreamGPTForCausalModelling

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

    def forward(self, batch):
        # Because of how DeSurv is coded we have the loss returned in the forward, so we have some redundancy

        tokens = batch['tokens'].to(self.device)
        ages = batch['ages'].to(self.device)
        values = batch['values'].to(self.device)
        covariates = batch["static_covariates"].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)   

        (surv, values_dist), (losses_desurv, loss_values), loss = self.model(tokens,
                                                                             ages,
                                                                             values,
                                                                             covariates,
                                                                             attention_mask,
                                                                             is_generation=False
                                                                            )

        losses = {"loss": loss,
                  "losses_desurv": losses_desurv,
                  "loss_values": loss_values
                 }

        return (surv, values_dist), losses

    def training_step(self, batch, batch_idx):
        _, loss_dict = self(batch)        
        self.log(f"train_loss", loss_dict['loss'], prog_bar=True, logger=True)
        self.log(f"train_loss_desurv", loss_dict['losses_desurv'], prog_bar=True, logger=True)
        self.log(f"train_loss_values", loss_dict['loss_values'], prog_bar=True, logger=True)
        return loss_dict['loss'] 

    def validation_step(self, batch, batch_idx):
        _, loss_dict = self(batch)        
        self.log(f"val_loss", loss_dict['loss'], prog_bar=True, logger=True)
        self.log(f"val_loss_desurv", loss_dict['losses_desurv'], prog_bar=True, logger=True)
        self.log(f"val_loss_values", loss_dict['loss_values'], prog_bar=True, logger=True)
        return loss_dict['loss'] 

    def test_step(self, batch, batch_idx):
        _, loss_dict = self(batch)        
        self.log(f"test_loss", loss_dict['loss'], prog_bar=True, logger=True)
        self.log(f"test_loss_desurv", loss_dict['losses_desurv'], prog_bar=True, logger=True)
        self.log(f"test_loss_values", loss_dict['loss_values'], prog_bar=True, logger=True)
        return loss_dict['loss'] 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.optim.learning_rate)
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),         # The scheduler instance
            "interval": "step",                                                         # The unit of the scheduler's step size
            "frequency": 2 * self.cfg.optim.val_check_interval,                         # How many epochs/steps should pass between calls to `scheduler.step()`
            "monitor": "val_loss",                                                      # Metric to monitor for scheduler
            "strict": False,                                                            # Enforce that "val_loss" is available when the scheduler is updated
            "name": 'ReduceLROnPlateau',                                                # For `LearningRateMonitor`, specify a custom logged name
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }

def setup_survival_experiment(cfg, vocab_size):

    # Get validation and test hook batch
    # val_data = next(iter(data_module.val_dataloader()))
    # test_data = next(iter(data_module.test_dataloader()))

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

    # optional callbacks
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
    )

    return _model, _trainer