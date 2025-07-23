import logging
import math
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR, LambdaLR, SequentialLR, ConstantLR, ChainedScheduler, ExponentialLR
from CPRD.src.models.survival.task_heads.causal import SurvStreamGPTForCausalModelling
from CPRD.examples.modelling.SurvivEHR.helpers import is_interactive

import importlib
import functools
from itertools import islice
from CPRD.src.models.base_callback import Embedding
from CPRD.src.models.survival.custom_callbacks.causal_eval import PerformanceMetrics


class CausalExperiment(pl.LightningModule):
    r"""
    PyTorch Lightning module for causal survival modeling using SurvStreamGPT.

    This experiment class wraps the SurvStreamGPTForCausalModelling model and integrates
    training, validation, and test steps with support for various learning rate schedulers,
    including warm-up and cosine annealing with or without decay.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Configuration object containing optimizer, scheduler, and experiment parameters.
    vocab_size : int
        The vocabulary size used by the token embedding layer in the model.
    concurrent_strategy : str, optional
        Strategy for handling concurrent events (e.g., "add_noise"), by default "add_noise".

    Attributes
    ----------
    cfg : omegaconf.DictConfig
        The configuration object.
    model : SurvStreamGPTForCausalModelling
        The wrapped survival model.

    Methods
    -------
    forward(batch, is_generation=False, return_loss=True, return_generation=False)
        Forward pass of the model.
    training_step(batch, batch_idx)
        Executes a training step and logs loss metrics.
    validation_step(batch, batch_idx)
        Executes a validation step and logs loss metrics.
    test_step(batch, batch_idx)
        Executes a test step and logs loss metrics.
    configure_optimizers()
        Configures the optimizer and learning rate scheduler(s).
    """

    def __init__(self,
                 cfg,
                 vocab_size,
                 concurrent_strategy="add_noise"
                ):
        
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = SurvStreamGPTForCausalModelling(cfg, vocab_size,
                                                     concurrent_strategy=concurrent_strategy)

    def forward(self, batch, is_generation=False, return_loss=True, return_generation=False):
        # Because of how DeSurv is coded we have the loss returned in the forward, so we have some redundancy

        tokens = batch['tokens'].to(self.device)
        ages = batch['ages'].to(self.device)
        values = batch['values'].to(self.device)
        covariates = batch["static_covariates"].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device) if is_generation is False else None  # None for hidden callback

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
                
            case 'cosineannealinglr':
                logging.info(f"Using Cosine Annealing in scheduler")
                
                period = self.cfg.optim.scheduler_periods * 10
                scheduler = CosineAnnealingLR(optimizer,
                                              T_max=int(period),
                                              eta_min=self.cfg.optim.learning_rate / 5)

            case _:
                raise NotImplementedError

        schedulers.append(scheduler)
        
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


def setup_causal_experiment(cfg, dm, vocab_size, checkpoint=None, logger=None):
    """
    Set up the causal experiment module, trainer, and callbacks for training or evaluation.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra configuration object with optimizer, trainer, and logging parameters.
    dm : LightningDataModule
        A PyTorch Lightning data module that provides train/val/test dataloaders. For formatting see https://github.com/cwlgadd/FastEHR
    vocab_size : int
        Size of the input vocabulary used in the model.
    checkpoint : str or None, optional
        Path to a checkpoint file to resume from. If None, initializes a new model.
    logger : pl.loggers.Logger or None, optional
        Logger instance (e.g., WandB or TensorBoard). If None or logging is disabled, no logger is used.

    Returns
    -------
    causal_experiment : CausalExperiment
        The initialized CausalExperiment module.
    CausalExperiment : type
        The class reference for CausalExperiment.
    _trainer : pl.Trainer
        The configured PyTorch Lightning trainer with callbacks.
    """

    #########################################################
    # Load existing pre-trained model,                      #
    #     overriding config where necessary                 #
    #########################################################
    if checkpoint is None:
        causal_experiment = CausalExperiment(cfg=cfg, vocab_size=vocab_size)
    else:
        causal_experiment = CausalExperiment.load_from_checkpoint(checkpoint,
                                                                  cfg=cfg, 
                                                                  )
    # if torch.cuda.is_available():
    #     causal_experiment = torch.compile(causal_experiment)
    
    logging.debug(causal_experiment)

    ####################
    # Use given logger #
    ####################
    logger = logger if cfg.experiment.log == True else None

    #############################
    # Make experiment callbacks #
    #############################
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

    # Early stopping
    if cfg.optim.early_stop:
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss_desurv", mode="min",
            min_delta=0,
            patience=cfg.optim.early_stop_patience,
            verbose=cfg.experiment.verbose,
        )
        callbacks.append(early_stop_callback)
    else:
        logging.warning(f"Early stopping is not being used: {cfg.optim.early_stop}")
            
    ########################
    # Validation callbacks #
    ########################
    
    # Get method for patient stratification to be used in some of the callbacks 
    if cfg.fine_tuning.custom_stratification_method._target_ is not None:
        
        # Extract the list of targets
        custom_strat_targets = OmegaConf.to_container(cfg.fine_tuning.custom_stratification_method._target_, resolve=True)

        # For each target, create a partially initialised stratification strategy and label for the plot.
        custom_stratification_methods, stratification_method_labels = [], []
        for custom_strat_target in custom_strat_targets:
            # partially initialise stratification method
            module_name, function_name = custom_strat_target.rsplit(".", 1)
            stratification_method = getattr(importlib.import_module(module_name), function_name)
            # Add plotting strategy
            custom_stratification_methods.append(functools.partial(stratification_method, dm=dm))
            stratification_method_labels.append(function_name.replace("_", " "))
    else:
        custom_stratification_methods = [None]
        stratification_method_labels = [" "]

    # Hidden state embedding
    ########################
    config_embed = cfg.fine_tuning.use_callbacks.hidden_embedding
    if config_embed.num_batches > 0:
        # Collect data to be used for hidden embedding
        num_batches = int(config_embed.num_batches)
        val_batches = list(islice(iter(dm.val_dataloader()), num_batches))
        test_batches = list(islice(iter(dm.test_dataloader()), num_batches))

        # Create each callback, built on each different stratification strategy
        for custom_stratification_method, stratification_method_label in zip(custom_stratification_methods, stratification_method_labels):
            embedding_callback = Embedding(val_batch                    = val_batches,
                                           test_batch                   = test_batches,
                                           custom_stratification_method = custom_stratification_method,
                                           stratification_title         = stratification_method_label,
                                           mask_static                  = config_embed.mask_static,
                                           mask_value                   = config_embed.mask_value,
                                          )
            callbacks.append(embedding_callback)
            logging.info(f"Created hidden state embedding callback {stratification_method_label}")
        
    # Performance metric
    ########################
    if cfg.fine_tuning.use_callbacks.performance_metrics:
        # Add callbacks which apply to outcome prediction tasks- should already be sorted, but sort again 
        #   NOTE: by default the tokenizer already orders them by frequency, and so prevelance_based_risk_score
        #         will just be an ordered list
        event_counts = dm.tokenizer._event_counts.sort("FREQUENCY", descending=False)
        prevelance_based_risk_score = []
        for row in event_counts.rows(named=True):
            next_most_prevalent_k = dm.encode([row["EVENT"]])[0]
            prevelance_based_risk_score.append(next_most_prevalent_k)
        metric_callback = PerformanceMetrics(prevelance_based_risk_score, log_concordance=True)
        callbacks.append(metric_callback)

    ######################
    # Set up the Trainer #
    ######################
    logging.info(f"Interactive job = {is_interactive()}")
    _trainer = pl.Trainer(
        logger=logger,
        # precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed",
        strategy="auto" if is_interactive() else "ddp",
        callbacks=callbacks,
        max_epochs=cfg.optim.num_epochs,
        log_every_n_steps=cfg.optim.log_every_n_steps,
        val_check_interval=cfg.optim.val_check_interval,
        limit_val_batches=cfg.optim.limit_val_batches,
        limit_test_batches=cfg.optim.limit_test_batches,
        # accumulate_grad_batches=cfg.optim.accumulate_grad_batches,
        # gradient_clip_val=1.0
    )

    return causal_experiment, CausalExperiment, _trainer


class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    """
    Modified CosineAnnealingWarmRestarts scheduler with multiplicative decay of base learning rate at each restart.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    T_0 : int
        Number of iterations for the first restart.
    T_mult : int, optional
        Multiplicative factor of T_0 after each restart, by default 1.
    eta_min : float, optional
        Minimum learning rate value, by default 0.
    last_epoch : int, optional
        The index of last epoch, by default -1.
    verbose : bool, optional
        If True, prints learning rate updates, by default False.
    decay : float, optional
        Multiplicative factor by which base_lrs are decayed at each restart, by default 1 (no decay).

    Notes
    -----
    - At the end of each restart cycle, the base learning rates are multiplied by `decay`.
    - This allows cosine annealing with a decaying amplitude over time.
    """
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
