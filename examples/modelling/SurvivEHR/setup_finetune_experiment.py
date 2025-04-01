import logging
from hydra.utils import instantiate
import functools
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR, LambdaLR, SequentialLR, ChainedScheduler
from CPRD.src.models.survival.task_heads.causal import SurvStreamGPTForCausalModelling
from CPRD.src.modules.head_layers.survival.competing_risk import ODESurvCompetingRiskLayer
from CPRD.src.modules.head_layers.survival.single_risk import ODESurvSingleRiskLayer
from CPRD.src.modules.head_layers.value_layers import GaussianRegressionLayer

from CPRD.src.models.base_callback import Embedding
from CPRD.src.models.survival.custom_callbacks.clinical_prediction_model import PerformanceMetrics
from CPRD.src.models.survival.custom_callbacks.mm_clinical_prediction_model import RestrictedMeanSurvivalTime
import importlib


class FineTuneExperiment(pl.LightningModule):

    def __init__(self,
                 cfg,
                 outcome_tokens,
                 risk_model,         # 'single-risk', 'competing-risk', or (TODO) False (for case of predicting value but not survival risk)
                 vocab_size=265,
                 use_adapter=False    # If True, we freeze the body and use an `Adapter` module to allow parameter efficient fine-tuning. If False, we train all parameters
                ):
        
        super().__init__()
        self.save_hyperparameters()        
        self.cfg = cfg
        self.use_adapter = use_adapter
        self.reduced_hidden_dim = 384
        
        # Load the pre-trained Transformer
        ###################################
        adapter_dim = cfg.transformer.adapter_dim if use_adapter is True else False
        self.model = SurvStreamGPTForCausalModelling(cfg, vocab_size, use_adapter=adapter_dim)

        # Approaches for freezing the pre-trained model body whilst fine-tuning the new head
        if self.use_adapter is True:
            logging.info(f"Fixing Transformer parameters and fine-tuning using an Adapter mechanism.")
            for name, param in self.model.named_parameters():
                if "adapter" in name.lower() or "ln_" in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        elif self.use_adapter == "fix":
            logging.info(f"Fixing Transformer parameters.")
            for name, param in self.model.named_parameters():
                if ("ln_" in name.lower() or "layernorm" in name.lower()):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                
        else:
            logging.info(f"Training all Transformer parameters")
            for name, param in self.model.named_parameters():
                param.requires_grad = True

        # Drop down network to compress GPT hidden dimensions before new heads
        self.reduce_hidden = torch.nn.Sequential(
            torch.nn.Linear(self.model.n_embd, self.reduced_hidden_dim),
            torch.nn.ReLU()
        )

        # Create new heads
        ##################
        total_weight = cfg.fine_tuning.head.surv_weight + cfg.fine_tuning.head.value_weight
        self.surv_weight = cfg.fine_tuning.head.surv_weight / total_weight
        self.value_weight = cfg.fine_tuning.head.value_weight / total_weight
                    
        # Create a new survival head
        hidden_dimensions = self.reduced_hidden_dim #self.model.n_embd - self.model.n_embd_private
        match risk_model.replace('-', '').replace(' ', '').lower():
            case "singlerisk" | "sr":
                # Combine each of the given outcomes into a single event, and treat it as a single risk
                #    e.g. This could be a single event, or all events that constitute some form of umbrella, e.g. cardiovascular disease
                self.surv_layer = ODESurvSingleRiskLayer(outcome_tokens, hidden_dimensions, [32, 32], device="cuda")
                # Create a method which reduces batch["tokens"] from the causal k={1,2,3,4,5,...\vocab_size} form to the single risk 
                #    form k={\null, 1} that surv_layer is expecting
                self.reduce_to_outcomes = lambda target_token: target_token
                
            case "competingrisk" | "cr":
                # Treat each risk as a competing risk
                self.surv_layer = ODESurvCompetingRiskLayer(hidden_dimensions, [32, 32], num_risks=len(outcome_tokens), device="cuda")
                # Create a method which reduces batch["tokens"] from the causal k={1,2,3,4,5,...\vocab_size} form to the competing risk 
                #    form k={1,2,3,..., K} that surv_layer is expecting
                self.reduce_to_outcomes = lambda target_token: sum([torch.where(target_token==i, idx+1, 0) for idx, i in enumerate(outcome_tokens)])

            case _:
                raise ValueError(f"Survival head must be either 'single-risk' or 'competing-risk'")

        # Create a new value head
        if self.value_weight > 0:
            self.value_layer = GaussianRegressionLayer(hidden_dimensions,
                                                       measurement_tokens=outcome_tokens,
                                                       base_hidden_dim=32,
                                                       )
            logging.debug(f"Created value layer:\n{self.value_layer}")
        else:
            logging.debug(f"Did not create value layer as weighting set to zero")

        

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

        # Reduce hidden dimension
        in_hidden_state = self.reduce_hidden(in_hidden_state)
        
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
        surv_dict, losses_desurv = self.surv_layer.predict(in_hidden_state,
                                                           target_tokens=self.reduce_to_outcomes(target_tokens),
                                                           target_ages=target_ages,
                                                           attention_mask=target_attention_mask,
                                                           is_generation=is_generation,
                                                           return_loss=return_loss,
                                                           return_cdf=return_generation,
                                                           )

        # regression head (values of next token if applicable)
        if self.value_weight > 0:
            values_dist, loss_values = self.value_layer.predict(in_hidden_states,
                                                                target_tokens=target_tokens,
                                                                target_values=target_values,
                                                                attention_mask=target_attention_mask,
                                                                is_generation=is_generation,
                                                                return_loss=return_loss,
                                                                return_value_dist=return_generation,
                                                                )
        else:
            values_dist = None
            loss_values = 0 

        if return_loss:
            loss_desurv = torch.sum(torch.stack(losses_desurv))                                  # losses are returned as a list, as the Single-Risk head is many DeSurv models in parallel, combine
            loss = (self.surv_weight * loss_desurv) + (self.value_weight * loss_values)          # Weight the loss
        else:
            loss_desurv = None
            loss = None

        outputs = {"surv": surv_dict,
                   "values_dist": values_dist
                  }
        losses = {"loss": loss,
                  "loss_desurv": loss_desurv,
                  "loss_values": loss_values
                 }
        
        return outputs, losses, in_hidden_state

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

        # Split parameters by those in new fine-tuning head and those in backbone
        head_params = list(self.surv_layer.parameters())
        if self.value_weight > 0:
            head_params += list(self.value_layer.parameters())
        body_params = [p for n, p in self.model.named_parameters() if p.requires_grad and not any(h is p for h in head_params)]
        
        match self.use_adapter:
            case "fix":
                optimizer = torch.optim.AdamW(head_params, lr=self.cfg.optim.learning_rate)
                logging.info(f"Fixing backbone, and using head learning rate {self.cfg.optim.learning_rate}")
                
            case True:
                body_lr = self.cfg.optim.learning_rate # * 0.1
                optimizer = torch.optim.AdamW([
                    {'params': body_params, 'lr': body_lr}, 
                    {'params': head_params, 'lr': self.cfg.optim.learning_rate}
                    ])
                logging.info(f"Using adpater on backbone with learning rate {body_lr}, and using head learning rate {self.cfg.optim.learning_rate}")
                
            case _:
                body_lr = self.cfg.optim.learning_rate # * 0.01
                optimizer = torch.optim.AdamW([
                    {'params': body_params, 'lr': body_lr}, 
                    {'params': head_params, 'lr': self.cfg.optim.learning_rate}
                    ])
                logging.info(f"Full training backbone with learning rate {body_lr}, and using head learning rate {self.cfg.optim.learning_rate}")
        
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
                raise ValueError(f"Invalid scheduler {self.cfg.optim.scheduler}.")

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
            "interval": "epoch",         #step                                                # The unit of the scheduler's step size
            "monitor": "val_loss",                                                      # Metric to monitor for scheduler, if needed
            "strict": True,              #False                                              # Enforce that "val_loss" is available when the scheduler is updated
            "name": 'Scheduler',                                                        # For `LearningRateMonitor`, specify a custom logged name
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }

def setup_finetune_experiment(cfg, dm, mode, risk_model, checkpoint=None, logger=None, **kwargs):

    assert dm.is_supervised, "Datamodule for must be supervised for `setup_finetune_experiment` ."

    # Which tokens we want to predict as outcomes. 
    #    In the fine-tuning setting these are used to construct a new head which can be fine-tuned.
    #    TODO: a new clinical prediction model callback then needs to be made (or existing one editted) for this new case
    if cfg.fine_tuning.fine_tune_outcomes is not None:
        # Use outcomes provided directly
        outcomes = cfg.experiment.fine_tune_outcomes
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
    logging.info(f"Running {risk_model} fine-tuning experiment with outcomes {outcome_dict}")

    #########################################################
    # Load pre-trained model,                               #
    #     overriding config where necessary                 #
    #########################################################
    match mode:
        case "load_from_finetune":
            assert checkpoint is not None
            logging.info(f"Loading fine-tuned checkpoint from {checkpoint}")
            finetune_experiment = FineTuneExperiment.load_from_checkpoint(checkpoint, cfg=cfg, outcome_tokens=outcome_tokens, risk_model=risk_model)
        case "load_from_pretrain":
            assert checkpoint is not None
            logging.info(f"Loading pre-trained model from checkpoint from {checkpoint}.")
            finetune_experiment = FineTuneExperiment.load_from_checkpoint(checkpoint, cfg=cfg, outcome_tokens=outcome_tokens, risk_model=risk_model,
                                                                          use_adapter=cfg.transformer.use_fine_tune_adapter, strict=False)
        case "no_load":
            assert cfg.transformer.use_fine_tune_adapter is False, "If fine-tuning from scratch do not freeze any Transformer parameters through the adapter module."
            logging.info(f"Fine-tuning from scratch")
            finetune_experiment = FineTuneExperiment(cfg, outcome_tokens, risk_model=risk_model, use_adapter=False, ) 
        case _:
            raise NotImplementedError

    # if torch.cuda.is_available():
    #     finetune_experiment = torch.compile(finetune_experiment)
            
    logging.debug(finetune_experiment)

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
        module_name, function_name = cfg.fine_tuning.custom_stratification_method._target_.rsplit(".", 1)
        stratification_method = getattr(importlib.import_module(module_name), function_name)
        custom_stratification_method = functools.partial(stratification_method, **{"tokens": outcome_tokens})
    else:
        custom_stratification_method = None
        
    # Hidden state embedding
    if cfg.fine_tuning.use_callbacks.hidden_embedding:
        logging.debug("Creating hidden state embedding callback")
        embedding_callback = Embedding(val_batch=next(iter(dm.val_dataloader())),
                                       test_batch=next(iter(dm.test_dataloader()))
                                      )
        callbacks.append(embedding_callback)

    # Add callbacks which apply to outcome prediction tasks
    if cfg.fine_tuning.use_callbacks.performance_metrics:
        # Create a hash map which maps the tokens of interset to their corresponding desurv output index
        #    For fine-tuning, where the token is condensed into a subset, this is a map from this new token value
        #    and the corresponding DeSurv output index
        # TODO:
        #    SingleRisk and Competing Risk models have a slightly different structure now, and so they are treated 
        #    a bit differently here also. TODO: update CompetingRisk to follow the same structure of taking in the
        #    target tokens
        if risk_model == "single-risk":
            outcome_token_to_desurv_output_index = {token: 0 for token_idx, token in enumerate(outcome_tokens)}
        if risk_model == "competing-risk":
            outcome_token_to_desurv_output_index = {token: token_idx for token_idx, token in enumerate(outcome_tokens)}
        # Construct callback
        metric_callback = PerformanceMetrics(outcome_token_to_desurv_output_index=outcome_token_to_desurv_output_index,
                                             log_combined=True,
                                             log_individual=False if risk_model == "single-risk" else False,
                                             log_ctd=True, 
                                             log_ibs=True,
                                             log_inbll=True)
        callbacks.append(metric_callback)


    if cfg.fine_tuning.use_callbacks.rmst:
        metric_callback = RestrictedMeanSurvivalTime(outcome_token_to_desurv_output_index=outcome_token_to_desurv_output_index,
                                                     log_combined=True,
                                                     log_individual=False,
                                                     custom_stratification_method=custom_stratification_method
                                                    )
        callbacks.append(metric_callback)

    ######################
    # Set up the Trainer #
    ######################
    _trainer = pl.Trainer(
        logger=logger,
        # precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed",
        callbacks=callbacks,
        min_epochs=30,
        max_epochs=cfg.optim.num_epochs,
        log_every_n_steps=cfg.optim.log_every_n_steps,
        val_check_interval=cfg.optim.val_check_interval,
        limit_val_batches=cfg.optim.limit_val_batches,
        limit_test_batches=cfg.optim.limit_test_batches,
        accumulate_grad_batches=cfg.optim.accumulate_grad_batches,
        # gradient_clip_val=1.0,
    )

    return finetune_experiment, FineTuneExperiment, _trainer