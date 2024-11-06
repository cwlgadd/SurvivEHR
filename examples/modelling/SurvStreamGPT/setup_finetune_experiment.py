import pytorch_lightning as pl
import torch
import logging
from CPRD.src.models.survival.task_heads.causal import SurvStreamGPTForCausalModelling
from CPRD.src.modules.head_layers.survival.competing_risk import ODESurvCompetingRiskLayer
from CPRD.src.modules.head_layers.survival.single_risk import ODESurvSingleRiskLayer
from CPRD.examples.modelling.SurvStreamGPT.setup_causal_experiment import CausalExperiment
# from CPRD.data.foundational_loader import convert_batch_to_none_causal
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR, LambdaLR, SequentialLR, ChainedScheduler
from CPRD.src.models.base_callback import Embedding
from CPRD.src.models.survival.custom_callbacks.clinical_prediction_model import PerformanceMetrics
from CPRD.examples.modelling.SurvStreamGPT.setup_causal_experiment import CausalExperiment

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class FineTuneExperiment(pl.LightningModule):

    def __init__(self,
                 cfg,
                 outcome_tokens,
                 risk_model,         # 'single-risk', 'competing-risk', or (TODO) False (for case of predicting value but not survival risk)
                 vocab_size=265,
                 freeze_body=False
                ):
        
        super().__init__()
        self.save_hyperparameters()        
        self.cfg = cfg
        self.model = SurvStreamGPTForCausalModelling(cfg, vocab_size)

        # Create new survival head
        match risk_model.replace('-', '').replace(' ', '').lower():

            case "singlerisk" | "sr":
                # Combine each of the given outcomes into a single event, and treat it as a single risk
                # e.g. This could be a single event, or all events that constitute some form of umbrella, e.g. cardiovascular disease
                self.surv_layer = ODESurvSingleRiskLayer(self.model.n_embd - self.model.n_embd_private, [32,32], num_risks=1, device="cuda")
                
                # Create a method which reduces from the causal k={1,2,3,4,5,...} form to the competing risk form k={\null, 1,2,3,..., K}
                self.reduce_to_outcomes = lambda x: sum([torch.where(x==i, 1, 0) for i in outcome_tokens])
                
                # and then record what these K outcomes are
                self.reduced_outcome_tokens = [1]
                
            case "competingrisk" | "cr":
                # Treat each risk as a competing risk
                self.surv_layer = ODESurvCompetingRiskLayer(self.model.n_embd - self.model.n_embd_private, [32,32], num_risks=len(outcome_tokens) + 1, device="cuda")
                
                # Create a method which reduces from the causal k={1,2,3,4,5,...} form to the competing risk form k={\null, 1,2,3,..., K}
                self.reduce_to_outcomes = lambda x: sum([torch.where(x==i, idx+1, 0) for idx, i in enumerate(outcome_tokens)])
                
                # and then record what these K outcomes are, e.g. = [1,2,3,4,K=5]
                self.reduced_outcome_tokens = [idx + 1 for idx, i in enumerate(outcome_tokens)]
                
            case _:
                raise ValueError(f"Survival head must be either 'single-risk' or 'competing-risk'")

        # Freeze the pre-trained model body and only fine-tune the new head
        self.freeze_body = freeze_body
        if self.freeze_body:
            self.parameters_head = list(self.surv_layer.parameters())
            self.parameters_head = set(self.parameters_head)
            for param in self.model.parameters():
                if param not in self.parameters_head:
                    param.requires_grad = False
        logging.info(f"Trainable parameters: {'DeSurv head' if self.freeze_body else 'all'} parameters")

        self.dropout = torch.nn.Dropout(p=0.0)

    def forward(self, batch, is_generation=False, return_loss=True, return_generation=False):
        # Because of how DeSurv is coded we have the loss returned in the forward, so we have some redundancy

        # inputs
        covariates = batch["static_covariates"].to(self.device)
        tokens = batch['tokens'].to(self.device)                           # torch.Size([bsz, seq_len])       
        ages = batch['ages'].to(self.device)                               # torch.Size([bsz, seq_len])
        values = batch['values'].to(self.device)                           # torch.Size([bsz, seq_len])
        attention_mask = batch['attention_mask'].to(self.device)           # torch.Size([bsz, seq_len])
        
        # targets
        target_token = batch['target_token'].reshape((-1,1)).to(self.device)               # torch.Size([bsz, 1])
        target_token = self.reduce_to_outcomes(target_token)                               # torch.Size([bsz, 1])
        target_age_delta = batch['target_age_delta'].reshape((-1,1)).to(self.device)       # torch.Size([bsz, 1]),
        target_value = batch['target_value'].reshape((-1,1)).to(self.device)               # torch.Size([bsz, 1])
        bsz, seq_len = tokens.shape
        
        # torch.Size([bsz, seq_len, hid_dim])
        hidden_states =  self.model.transformer(tokens=tokens,
                                                ages=ages,
                                                values=values,
                                                covariates=covariates,
                                                attention_mask=attention_mask
                                                )

        hidden_states = self.dropout(hidden_states)

        # Convert attention mask to a mask which we can use to predict only the final transition
        #   this mask is 1 if last observation, 0 otherwise. This ensures we can only push the last observation through.
        #   this is required because of padding leaving variable sequence lengths
        _att_mask_tmp =  torch.hstack((attention_mask, torch.zeros((bsz,1), device=attention_mask.device)))
        gen_mask = attention_mask - _att_mask_tmp[:,1:]            # torch.Size([bsz, seq_len])

        
        # Get the hidden states of the last input temporal event
        in_hidden_state = torch.zeros((bsz, hidden_states.shape[-1]), device=self.device)
        for idx in range(bsz):
            assert sum(gen_mask[idx, :]) == 1
            in_hidden_state[idx, :] = hidden_states[idx, gen_mask[idx, :]==1, :]

        # The hidden states, made of the last hidden state of input sequence, and a padded zero
        # Note, we add the hidden state again as the padding target, as in generation this will be what is forwarded
        in_hidden_state = torch.stack((in_hidden_state, in_hidden_state), axis=1)         # bsz, seq_len=2, embd_dim

        
        # The target states, made of a padded zero, and the target states
        target_tokens = torch.hstack((torch.zeros((bsz,1), device=self.device), target_token))              # bsz, seq_len=2

        # The target ages, made of a padded zero, and the target ages
        target_ages = torch.hstack((torch.zeros((bsz,1), device=self.device), target_age_delta))            # bsz, seq_len=2

        # The target ages, made of a padded zero, and the target ages
        target_values = torch.hstack((torch.zeros((bsz,1), device=self.device), target_value))

        # Attention matrix. As we have reduced to only the transition of last seen input to sequence target, nothing is masked
        target_attention_mask = torch.ones_like(target_tokens, device=self.device) == 1

        # survival time to event head (survival curve until next token)
        surv_dict, losses_desurv = self.surv_layer.predict(in_hidden_state[:,:,:self.model.n_embd - self.model.n_embd_private],
                                                           target_tokens=target_tokens,
                                                           target_ages=target_ages, 
                                                           attention_mask=target_attention_mask,
                                                           is_generation=is_generation,
                                                           return_loss=return_loss,
                                                           return_cdf=return_generation,
                                                           )

        if return_loss:
            loss = torch.sum(torch.stack(losses_desurv))                                  # losses are returned as a list, as the Single-Risk head is many DeSurv models in parallel, combine

            if torch.isnan(loss):
                logging.warning(f"Invalid loss {loss}: with target tokens {target_tokens}, target values {target_values} and target ages {target_ages}")
                logging.warning(f"from hidden state {torch.sum(in_hidden_state, axis=2)/128}")
                raise NotImplementedError
        else:
            loss = None
            
        return {"surv": surv_dict}, {"loss": loss}, in_hidden_state

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

def setup_finetune_experiment(cfg, dm, mode, risk_model, checkpoint=None):

    assert dm.is_supervised, "Datamodule for must be supervised for `setup_finetune_experiment` ."
    assert cfg.experiment.fine_tune_outcomes is not None, "Must provide outcome list for `setup_finetune_experiment`."

    # Which tokens we want to predict as outcomes. 
    #    In the fine-tuning setting these are used to construct a new head which can be fine-tuned.
    #    TODO: a new clinical prediction model callback then needs to be made (or existing one editted) for this new case
    outcome_tokens =  dm.encode(cfg.experiment.fine_tune_outcomes)
    outcome_dict = {_key: _value for _key, _value in zip(cfg.experiment.fine_tune_outcomes, outcome_tokens)}
    logging.info(f"Running {risk_model} fine-tuning experiment with outcomes {outcome_dict}")
    
    # Load pre-trained model, overriding config if necessary
    match mode:
        case "load_from_finetune":
            assert checkpoint is not None
            logging.info(f"Loading fine-tuned checkpoint from {checkpoint}")
            finetune_experiment = FineTuneExperiment.load_from_checkpoint(checkpoint, cfg=cfg, outcome_tokens=outcome_tokens, risk_model=risk_model)
        case "load_from_pretrain":
            assert checkpoint is not None
            logging.info(f"Loading pre-trained checkpoint from {checkpoint}")
            pretrained_experiment = CausalExperiment.load_from_checkpoint(checkpoint, cfg=cfg)
            finetune_experiment = FineTuneExperiment(cfg, outcome_tokens, risk_model=risk_model) 
            finetune_experiment.model = pretrained_experiment.model
        case "no_load":
            logging.info(f"Fine-tuning from scratch")
            finetune_experiment = FineTuneExperiment(cfg, outcome_tokens, risk_model=risk_model) 
        case _:
            raise NotImplementedError
            
    logging.debug(finetune_experiment)

    # Initialize wandb logger
    if cfg.experiment.log == True:
        logger = pl.loggers.WandbLogger(project=cfg.experiment.project_name,
                                        name=cfg.experiment.run_id + "_" + cfg.experiment.fine_tune_id, 
                                        save_dir=cfg.experiment.log_dir
                                        )
    else:
        logger = None

    # Make all callbacks
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

    # ... custom callbacks
    val_batch = next(iter(dm.val_dataloader()))
    test_batch = next(iter(dm.test_dataloader()))
    
    # Hidden state embedding
    if not finetune_experiment.freeze_body:
        logging.debug("Creating hidden state embedding callback: We are fine-tuning the entire model and so latent embeddings will be changed")
        embedding_callback = Embedding(val_batch=val_batch,
                                       test_batch=test_batch
                                      )
        callbacks.append(embedding_callback)

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
    metric_callback = PerformanceMetrics(outcome_tokens=finetune_experiment.reduced_outcome_tokens,
                                         log_ctd=True,
                                         log_ibs=True,
                                         log_inbll=True,
                                         reduce_to_outcomes = finetune_experiment.reduce_to_outcomes,
                                         )
    callbacks.append(metric_callback)

    _trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.optim.num_epochs,
        log_every_n_steps=cfg.optim.log_every_n_steps,
        val_check_interval=cfg.optim.val_check_interval,
        limit_val_batches=cfg.optim.limit_val_batches,
        limit_test_batches=cfg.optim.limit_test_batches,
    )

    return finetune_experiment, FineTuneExperiment, _trainer