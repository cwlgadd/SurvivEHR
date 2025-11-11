import logging
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR, LambdaLR, SequentialLR, ChainedScheduler
import importlib
import functools
import math
from itertools import islice

from SurvivEHR.src.models.base_callback import Embedding
from SurvivEHR.src.models.survival.custom_callbacks.clinical_prediction_model import PerformanceMetrics
from SurvivEHR.src.models.survival.custom_callbacks.mm_clinical_prediction_model import RestrictedMeanSurvivalTime
from SurvivEHR.src.models.survival.task_heads.causal import SurvStreamGPTForCausalModelling
from SurvivEHR.src.modules.head_layers.survival.competing_risk import ODESurvCompetingRiskLayer
from SurvivEHR.src.modules.head_layers.survival.single_risk import ODESurvSingleRiskLayer
from SurvivEHR.src.modules.head_layers.value_layers import GaussianRegressionLayer
from SurvivEHR.examples.modelling.SurvivEHR.optimizers_fine_tuning import ConfigureFTOptimizers


class FineTuneExperiment(pl.LightningModule):

    def __init__(self,
                 cfg,
                 outcome_tokens,
                 risk_model,         # 'single-risk', 'competing-risk', or (TODO) False (for case of predicting value but not survival risk)
                 vocab_size=265,
                ):
        
        super().__init__()
        self.save_hyperparameters()        
        self.cfg = cfg

        # self.PEFT = 
        # self.probe_epochs = int(getattr(self.cfg.fine_tuning.backbone, "linear_probe_epochs", 0))
        # self.unfreeze_top_k = getattr(self.cfg.fine_tuning.backbone, "unfreeze_top_k", 1e6)
        
        ###################################
        # Load the pre-trained Transformer
        #  and remove previous causal heads
        ###################################
        adapter_dim = getattr(cfg.fine_tuning.PEFT, "adapter_dim", 8) if cfg.fine_tuning.PEFT.method == "adapter" else False
        self.model = SurvStreamGPTForCausalModelling(cfg, vocab_size, use_adapter=adapter_dim)
        self.model.surv_layer, self.model.value_layer = None, None

        # Initial block-wise freezing
        # if (self.PEFT.method == "fix") or (self.probe_epochs > 0) or (self.unfreeze_top_k == "gradual"):
        #     top_k = 0
        # elif isinstance(self.unfreeze_top_k, int):
        #     top_k = self.unfreeze_top_k
        # else:
        #     top_k = None
        # self.constrain_backbone(top_k=top_k)

        # Drop down network to compress GPT hidden dimensions before new heads
        self.use_compression = getattr(self.cfg.fine_tuning, "compression_layer", False)
        if self.use_compression:
            reduce_hidden_dim = self.use_compression if isinstance(self.use_compression, int) else 384
            self.reduce_hidden = torch.nn.Sequential(
                torch.nn.Linear(self.model.n_embd, reduce_hidden_dim),
                torch.nn.ReLU()
            )
            # Using Layer-Norm speeds up training, but will require new default configuration
            # self.reduce_hidden = torch.nn.Sequential(
            #     torch.nn.LayerNorm(self.model.n_embd),
            #     torch.nn.Linear(self.model.n_embd, reduce_hidden_dim),
            #     torch.nn.GELU(),
            #     torch.nn.Dropout(p=0.1),
            # )
            hidden_dimensions = reduce_hidden_dim
        else:
            hidden_dimensions = self.model.n_embd

        ##################
        # Create new heads
        ##################
        total_weight = cfg.fine_tuning.head.surv_weight + cfg.fine_tuning.head.value_weight
        self.surv_weight = cfg.fine_tuning.head.surv_weight / total_weight
        self.value_weight = cfg.fine_tuning.head.value_weight / total_weight
        assert self.surv_weight > 0 or self.value_weight > 0
                    
        # Create a new survival head
        if self.surv_weight > 0:
            desurv_device = "cuda" if torch.cuda.is_available() else "cpu"
            match risk_model.replace('-', '').replace(' ', '').lower():
                case "singlerisk" | "sr":
                    # Combine each of the given outcomes into a single event, and treat it as a single risk
                    #    e.g. This could be a single event, or all events that constitute some form of umbrella, e.g. cardiovascular disease
                    self.surv_layer = ODESurvSingleRiskLayer(outcome_tokens, hidden_dimensions, [32, 32], device=desurv_device)
                    # Create a method which reduces batch["tokens"] from the causal k={1,2,3,4,5,...\vocab_size} form to the single risk 
                    #    form k={\null, 1} that surv_layer is expecting
                    self.reduce_to_outcomes = lambda target_token: target_token
                    
                case "competingrisk" | "cr":
                    # Treat each risk as a competing risk
                    self.surv_layer = ODESurvCompetingRiskLayer(hidden_dimensions, [32, 32], num_risks=len(outcome_tokens), device=desurv_device)
                    # Create a method which reduces batch["tokens"] from the causal k={1,2,3,4,5,...\vocab_size} form to the competing risk 
                    #    form k={1,2,3,..., K} that surv_layer is expecting
                    self.reduce_to_outcomes = lambda target_token: sum([torch.where(target_token==i, idx+1, 0) for idx, i in enumerate(outcome_tokens)])
    
                case _:
                    raise ValueError(f"Survival head must be either 'single-risk' or 'competing-risk'")
        else:
            logging.debug(f"Did not create survival layer as weighting set to zero")

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

        # Replace the gen_mask loop with a vectorised gather?
        # lengths = attention_mask.sum(dim=1) - 1                                 # (bsz,)
        # idx = lengths.clamp(min=0).view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
        # in_hidden_state = hidden_states.gather(1, idx).squeeze(1)               # (bsz, hid)

        # Reduce hidden dimension
        if getattr(self, "reduce_hidden", False):
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
        if self.surv_weight > 0:
            surv_dict, losses_desurv = self.surv_layer.predict(in_hidden_state,
                                                               target_tokens=self.reduce_to_outcomes(target_tokens),
                                                               target_ages=target_ages,
                                                               attention_mask=target_attention_mask,
                                                               is_generation=is_generation,
                                                               return_loss=return_loss,
                                                               return_cdf=return_generation,
                                                               )
        else:
            surv_dict = None
            losses_desurv = [torch.zeros(1)]

        # regression head (values of next token if applicable)
        if self.value_weight > 0:
            values_dist, loss_values = self.value_layer.predict(in_hidden_state,
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

        ft_opt = ConfigureFTOptimizers(self)
        return ft_opt.get_optim_cfg
   
        
    # def constrain_backbone(
    #     self, 
    #     *, 
    #     top_k: int | None = None, 
    #     constrain_embeddings: bool = False,
    # ) -> None:
    #     """
    #     Constrain the Transformer backbone.

    #     TODO: move this into SurvStreamGPTForCausalModelling class
    
    #     Args:
    #         top_k:                How many top_k blocks are trainable; if None, unfreeze all
    #     """

    #     if top_k == None:
    #         return
            
    #     blocks = self.model.transformer.blocks
    #     top_k = top_k if top_k is not None else len(blocks)
    #     freeze_until = max(0, len(blocks) - int(top_k))

    #     # Start by freezing all parameters
    #     for p in self.model.transformer.parameters():
    #         p.requires_grad = False
        
    #     # Keep only the last top_k blocks trainable; keep earlier frozen
    #     for i, blk in enumerate(blocks):
    #         req = (i >= freeze_until)
    #         for n, p in blk.named_parameters():
    #             if self.PEFT.method in ["adapter"]:
    #                 # For adapters where we train a subset of old or new parameters
    #                 # Do gradual unfreezing of adaption parameters
    #                 p.requires_grad = req if "adapter" in n.lower() else False
                    
    #             elif self.PEFT.method in ["fix"]:
    #                 # If we fix backbone never unfreeze, regardless of top_k settings
    #                 p.requires_grad = False
                    
    #             else:
    #                 # Do gradual unfreezing of `all` parameters
    #                 p.requires_grad = req

    #     if constrain_embeddings:
    #         # Never unfreeze embeddings
    #         pass
    #     else:
    #         # Only unfreeze initial embedding layers if
    #         #    - we also unfreeze all blocks under full tuning
    #         #    - we have not specified to fix the entire backbone
    #         if freeze_until == 0 and self.PEFT.method not in ["fix"]:
    #             subs = ("wte", "wpe")
    #             for n, p in self.model.transformer.named_parameters():
    #                 if any(s in n for s in subs):
    #                     p.requires_grad = True

    #     # Always unfreeze, regardless of setup
    #     subs = ("layernorm", "ln_")
    #     for n, p in self.model.transformer.named_parameters():
    #         if any(s in n for s in subs):
    #             p.requires_grad = True

    #     print(f"Unfroze top-{len(blocks) - freeze_until} backbone layers.")

    #     # DEBUG: Report remaining unfrozen params
    #     # for n, p in self.model.transformer.named_parameters():
    #     #     if p.requires_grad == False:
    #     #         print(f"Frozen {n}")
    #     #     else:
    #     #         print(f"Trainable {n}")
    
    # def on_train_epoch_start(self):
    #     # If we have specified a scheme in which trainable parameters can change over the training cycle

    #     if self.unfreeze_top_k == 'gradual':
    #         # Gradual layer-wise unfreezing, with or without probe
    #         top_k = self.current_epoch - self.probe_epochs + 1
    #         if top_k >= 0:
    #             self.constrain_backbone(top_k=top_k)
    #             logging.info(f"Gradual layerwise unfreezing: Unfreezing top {top_k} layers")
            
    #     elif self.probe_epochs > 0 and self.current_epoch == self.probe_epochs:
    #         # No Gradual + linear probe, 
    #         self.constrain_backbone(top_k=self.unfreeze_top_k)
    #         logging.info(f"Linear probe ending: Unfreezing top {self.unfreeze_top_k} layers")
            
    #     else:
    #         # No gradual, no linear probe
    #         pass
            
def setup_finetune_experiment(cfg, dm, mode, risk_model, checkpoint=None, logger=None, **kwargs):
    """
    Set up the fine-tuning experiment module, trainer, and callbacks for training or evaluation.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra configuration object with optimizer, trainer, and logging parameters.
    dm : LightningDataModule
        A PyTorch Lightning data module that provides train/val/test dataloaders. For formatting see https://github.com/cwlgadd/FastEHR
    risk_model : 
        Whether to use single-risk ("single-risk") or competing risk ("competing-risk") survival head for fine-tuning.
    mode:
        Whether we want to train a new fine-tune model from scratch ("no_load"), from a pre-trained causal model ("load_from_pretrain"), 
        or load an already fine-tuned model ("load_from_finetune").
    checkpoint : str or None, optional
        Path to a checkpoint file to resume from. If None, initializes a new model.
    logger : pl.loggers.Logger or None, optional
        Logger instance (e.g., WandB or TensorBoard). If None or logging is disabled, no logger is used.

    Returns
    -------
    finetune_experiment : FineTuneExperiment
        The initialized CausalExperiment module.
    FineTuneExperiment : type
        The class reference for FineTuneExperiment.
    _trainer : pl.Trainer
        The configured PyTorch Lightning trainer with callbacks.
    """

    assert dm.is_supervised, "Datamodule for must be supervised for `setup_finetune_experiment` ."

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
            finetune_experiment = FineTuneExperiment.load_from_checkpoint(checkpoint, cfg=cfg, outcome_tokens=outcome_tokens, risk_model=risk_model, strict=False)
        case "no_load":
            assert cfg.fine_tuning.PEFT.method is None, "If fine-tuning from scratch do not use any PEFT such as the adapter module."
            logging.info(f"Fine-tuning from scratch")
            finetune_experiment = FineTuneExperiment(cfg, outcome_tokens, risk_model=risk_model) 
        case _:
            raise NotImplementedError

    # if torch.cuda.is_available():
    #     finetune_experiment = torch.compile(finetune_experiment)
            
    # logging.info(finetune_experiment)
    print(finetune_experiment)

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

    # Restricted Mean Survival Time callbacks
    ########################
    if cfg.fine_tuning.use_callbacks.rmst:
        # Create each callback, built on each different stratification strategy
        for custom_stratification_method, stratification_method_label in zip(custom_stratification_methods, stratification_method_labels):
            metric_callback = RestrictedMeanSurvivalTime(outcome_token_to_desurv_output_index=outcome_token_to_desurv_output_index,
                                                         log_combined=True,
                                                         log_individual=False,
                                                         custom_stratification_method=custom_stratification_method
                                                        )
            callbacks.append(metric_callback)
        
    ######################
    # Set up the Trainer #
    ######################
    if  torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"
    else:
        precision = 32
        
    _trainer = pl.Trainer(
        logger=logger,
        # precision=precision,
        callbacks=callbacks,
        max_epochs=cfg.optim.num_epochs,
        log_every_n_steps=cfg.optim.log_every_n_steps,
        val_check_interval=cfg.optim.val_check_interval,
        limit_val_batches=cfg.optim.limit_val_batches,
        limit_test_batches=cfg.optim.limit_test_batches,
        accumulate_grad_batches=cfg.optim.accumulate_grad_batches,
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm="norm",
    )

    return finetune_experiment, FineTuneExperiment, _trainer