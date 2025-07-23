import pytorch_lightning as pl
import torch
import logging
from CPRD.src.models.survival.task_heads.causal import SurvStreamGPTForCausalModelling
# from CPRD.data.foundational_loader import convert_batch_to_none_causal
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR, LambdaLR, SequentialLR, ChainedScheduler

import importlib
import functools
from itertools import islice
from CPRD.src.models.base_callback import Embedding
from CPRD.src.models.survival.custom_callbacks.clinical_prediction_model import PerformanceMetrics

class FewShotExperiment(pl.LightningModule):

    def __init__(self,
                 cfg,
                 vocab_size,
                 use_adapter=False    # If True, we freeze the body and use an `Adapter` module to allow parameter efficient fine-tuning. If False, we train all parameters
                ):
        
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.use_adapter = use_adapter

        # Pre-trained Transformer
        adapter_dim = cfg.transformer.adapter_dim if use_adapter is True else False
        self.model = SurvStreamGPTForCausalModelling(cfg, vocab_size, use_adapter=adapter_dim)
                    
        # Freeze the pre-trained model body and only fine-tune the new head
        if self.use_adapter:
            logging.info(f"Fixing Transformer parameters and fine-tuning using an Adapter mechanism.")
            for name, param in self.model.named_parameters():
                if ("adapter" in name.lower() 
                    or "ln_" in name.lower()
                    or "layernorm" in name.lower()                     
                    or "surv_layer" in name.lower() 
                    or "value_layer" in name.lower()
                   ):
                    # If parameter is in adapter or a layer_norm then fine-tune it
                    param.requires_grad = True
                else:
                    # If the parameter is in the body, fix it.
                    param.requires_grad = False
        else:
            logging.info(f"Training all Transformer parameters")
            for name, param in self.model.named_parameters():
                param.requires_grad = True
                    
    def _reinit_weights(self):
        # Else, if it is in the (pre-train architecure) head, re-initialise and re-train. 
        #    This is important in the Few-Shot setting to not get stuck in local optima.
        #    As we transition from the task of predicting the `next` event risk, to the 
        #    task of predicting risk within X years, the time-scales vary drastically.
        logging.info(f"Re-initialising value/survival head parameters.")
        # reset_module_parameters(self.model.surv_layer)
        # reset_module_parameters(self.model.value_layer)
        for name, param in self.model.named_parameters():
            if "surv_layer" in name or "value_layer" in name:
                if param.dim() > 1:              # weight parameters 
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.zeros_(param)  # Initialize biases or other 1D parameters
            else:
                # Transformer params
                pass

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
                logging.warning(f"Invalid loss {loss}: with target tokens {target_token}, target values {target_value} and target ages {target_age_delta}")
                logging.warning(f"DeSurv losses {losses_desurv}")
                logging.warning(f"from hidden state {torch.sum(in_hidden_state, axis=2)/128}")
                raise NotImplementedError
        else:
            loss = None
            
        return {"surv": surv_dict}, {"loss": loss}, in_hidden_state[:, [-1], :]
            
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

        # parameters = self.parameters_head if self.freeze_body else self.parameters()
        # optimizer = torch.optim.AdamW(parameters, lr=self.cfg.optim.learning_rate)

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.optim.learning_rate)

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

def setup_fewshot_experiment(cfg, dm, vocab_size, checkpoint=None, logger=None, **kwargs):

    assert dm.is_supervised, "Datamodule for must be supervised for `setup_fewshot_experiment`."
    
    #########################################################
    # Get outcomes of interest                              #
    #########################################################
    # Which tokens we want to predict as outcomes. 
    #    In the fine-tuning setting these are used to construct a new head which can be fine-tuned.
    #    TODO: a new clinical prediction model callback then needs to be made (or existing one editted) for this new case
    if cfg.fine_tuning.fine_tune_outcomes is not None:
        # Use outcomes provided directly
        outcomes = cfg.fine_tuning.fine_tune_outcomes
        logging.info("Setting outcome list based on cfg.fine_tuning.fine_tune_outcomes")
    elif cfg.fine_tuning.custom_outcome_method._target_ is not None:
        # Use outcomes decided by some custom criteria
        module_name, function_name = cfg.fine_tuning.custom_outcome_method._target_.rsplit(".", 1)
        outcome_method = getattr(importlib.import_module(module_name), function_name)
        outcomes = outcome_method(dm)
        logging.info("Setting outcome list based on cfg.fine_tuning.custom_outcome_method._target_")
    else:
        # As this is a supervised CPM experiment, we require there to be some outcomes.
        raise NotImplementedError
    outcome_tokens = dm.encode(outcomes)
    outcome_dict = {_key: _value for _key, _value in zip(outcomes, outcome_tokens)}
    logging.info(f"Running few-shot experiment with outcomes {outcome_dict}")

    #########################################################
    # Load pre-trained model,                               #
    #     overriding config where necessary                 #
    #########################################################
    if checkpoint is not None:
        logging.info("Loading from checkpoint")
        use_adapter = cfg.transformer.use_fine_tune_adapter
        fewshot_experiment = FewShotExperiment.load_from_checkpoint(checkpoint, cfg=cfg, use_adapter=use_adapter, strict=not use_adapter)
        if cfg.experiment.train:
            fewshot_experiment._reinit_weights()                        # Re-initialise the survival head weights
    else:
        logging.info("Creating new experiment")
        fewshot_experiment = FewShotExperiment(cfg=cfg, vocab_size=vocab_size, use_adapter=False)
    logging.debug(fewshot_experiment)

    ####################
    # Use given logger #
    ####################
    logger = logger if cfg.experiment.log == True else None

    #############################
    # Make experiment callbacks #
    #############################
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.experiment.ckpt_dir,
        filename=cfg.experiment.run_id + "_" + cfg.experiment.fine_tune_id, 
        verbose=cfg.experiment.verbose,
        monitor="val_loss",
    )
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    callbacks = [checkpoint_callback,
                 lr_monitor,
                 ]

    # Early stopping
    if cfg.optim.early_stop:
        logging.debug("Creating early stopping callback")
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_loss", mode="min",
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
        logging.info(f"custom_stratification_method._target_ {cfg.fine_tuning.custom_stratification_method._target_}")
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

    
    # CPM erformance metrics
    ########################
    if cfg.fine_tuning.use_callbacks.performance_metrics:
        # Create a hash map which maps the tokens of interset to their corresponding desurv output index
        # For few-shot this is simply converting token to the index (PAD token takes value zero so we shift)
        outcome_token_to_desurv_output_index = {token: token - 1 for token in outcome_tokens}        
        metric_callback = PerformanceMetrics(outcome_token_to_desurv_output_index=outcome_token_to_desurv_output_index,
                                             log_individual=True if cfg.head.SurvLayer.lower() == "sr" else False,
                                             log_combined=True, # True if cfg.head.SurvLayer.lower() == "cr" else False,
                                             log_ctd=True, 
                                             log_ibs=True,
                                             log_inbll=True)
        callbacks.append(metric_callback)
  
    
    ######################
    # Set up the Trainer #
    ######################
    _trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.optim.num_epochs,
        log_every_n_steps=cfg.optim.log_every_n_steps,
        val_check_interval=cfg.optim.val_check_interval,
        limit_val_batches=cfg.optim.limit_val_batches,
        limit_test_batches=cfg.optim.limit_test_batches,
        accumulate_grad_batches=cfg.optim.accumulate_grad_batches,
        # gradient_clip_val=0.5
    )

    return fewshot_experiment, FewShotExperiment, _trainer