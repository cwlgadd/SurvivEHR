import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import logging
from CPRD.examples.modelling.SurvivEHR.helpers import is_interactive
from CPRD.src.models.survival.custom_callbacks.causal_eval import PerformanceMetrics

class CausalMLPExperiment(pl.LightningModule):
    def __init__(
        self,
        cfg,
        vocab_size: int,
        static_dim: int = 17,
        hidden_dims: list[int] = [256, 128],
        lr: float = 1e-3,
    ):
        """
        Args:
            vocab_size:    size of token vocabulary V.
            hidden_dims:   list of hidden layer widths for the MLP.
            lr:            learning rate.
        """
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.vocab_size = vocab_size

        # MLP: input dim = n_static + vocab_size
        dims = [vocab_size + static_dim] + hidden_dims + [vocab_size]
        layers = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch, is_generation=False, return_loss=True, return_generation=False):
        r"""
        ARGS:
            tokens              (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Tokens for categorical elements of sequence modeling. Indices are selected in `[0, ..., config.vocab_size]`, including the padding index
                which defaults to 0 in the accompanying data module. These are not ignored (masked) by default and you should also 
                pass the `attention_mask`. With the attention mask the loss is only computed for labels in `[0, ..., config.vocab_size]`
                
            ages                (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Positions for each categorical element of the sequence.
                
            values              (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Possible values which match each token. For example, for a token of a measurement name this will include the measurement value. 
                When no value corresponds this will be None.

        KWARGS:
            attention_mask:     (`torch.Tensor` of shape `torch.Size([batch_size, sequence_length])`):
                The padding attention mask
                
            is_generation:
                Whether GPT model is in generation or training mode

            return_cdf:
                Whether (when is_generation=False) to also return the survival predicted CDF

        """

        tokens = batch['tokens'].to(self.device)
        static_covariates = batch["static_covariates"].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device) if is_generation is False else None  # None for hidden callback
        
        B, L = tokens.shape
        
        if not is_generation:
                
            assert tokens is not None
            assert attention_mask is not None

            # Get the competing risk event types. A list of len vocab_size-1 where each element of the list is an event
            #       The 1st element of list corresponds to 2nd vocab element (vocab index == 0 is the PAD token which is excluded)
            #       k \in {0,1} with 1 if the seq target is the same as the single risk ode's index (position in list), and 0
            #       otherwise
            k = tokens[:, 1:]                                                                           # torch.Size([bsz, seq_len - 1])
            
            # We are considering the delta of time, but each element in the seq_len just has the time of event. 
            # This means the output mask requires both the time at the event, and the time of the next event to be available.
            tte_obs_mask = attention_mask[:, :-1] & attention_mask[:, 1:]                               # shape: torch.Size([bsz, seq_len - 1])
            tte_obs_mask = tte_obs_mask.reshape(-1)                                                     # torch.Size([bsz * (seq_len-1)])

            # This part add the causal aspect (normally done inside Transformer)
            tokens_in = tokens[:,:-1].repeat_interleave(tokens.shape[-1] - 1, dim=0)                    # torch.Size([bsz * seq_len-1, seq_len-1])
            tokens_in = tokens_in.reshape((tokens.shape[-1] - 1, tokens.shape[-1] - 1, -1))             # torch.Size([bsz, seq_len-1, seq_len-1])
            tokens_in = torch.tril(tokens_in)                                                           # torch.Size([bsz, seq_len-1, seq_len-1])
            tokens_in = tokens_in.reshape((-1, tokens.shape[-1] - 1))                                   # torch.Size([bsz * seq_len-1, seq_len-1])
            assert tokens_in.shape[0] == B * (L-1)
            assert tokens_in.shape[1] == L-1

            inputs_obs = []
            for b in range(tokens_in.shape[0]):
                patient_idx = b // (tokens.shape[-1] - 1)
                
                tokens_sample = torch.unique(tokens_in[b, :])
                tokens_sample_obs = torch.bincount(tokens_sample, minlength=self.vocab_size + 2)[1:]    # bin counts of tokens {1,2,...} and remove count of PAD
                inputs_sample_obs = torch.concat([static_covariates[patient_idx, :], tokens_sample_obs])                 # torch.Size([cov_dim + vocab_size])
                inputs_obs.append(inputs_sample_obs)
            inputs_obs = torch.stack(inputs_obs)                                                        # torch.Size([bsz * seq_len-1, cov_dim + vocab_size])

            # and apply the observation mask
            k = k.flatten()[tte_obs_mask == 1]                                                          # torch.Size([K=bsz * (seq_len-1) - MASK])
            inputs_obs = inputs_obs[tte_obs_mask == 1]
            assert ((k > 0) & (k <= self.vocab_size - 1)).all(), f"tokens {torch.unique(k)}, vocab_size {self.vocab_size}"

            print(inputs_obs.shape)
            print(self.vocab_size)

            logits = self.mlp(inputs_obs)                                                               #  torch.Size([K, vocab_size])        
                                                                                                        
        else:
            raise NotImplementedError

        output_dict ={"k": [k],
                      "logits": [logits[:, i] for i in range(logits.shape[1])]                                # [torch.Size([K,]) for _ in range(vocab_size)]      
                     }

        if return_loss:
            loss = F.cross_entropy(logits, k)
        else:
            loss = None

        outputs = {"clf": output_dict}
        losses = {"loss": loss}
        
        return outputs, losses, (logits, k)
        
            # x, K = [], []
            # risk_preds = []
            # for b in range(B):
            #     batch_tokens = tokens[b, :]
            #     batch_attention_mask = attention_mask[b, :]
            #     obs_batch_tokens = batch_tokens[batch_attention_mask==1]
                
            #     for l in range(obs_batch_tokens.shape[0]-1):
            #         x_counts = torch.bincount(obs_batch_tokens[:l+1], minlength=self.vocab_size+1)
            #         x_bin = (x_counts > 0).long()
            #         x_stack = torch.cat([covariates[b,:], x_bin], dim=-1)  # (1, D_static+V)
            #         x.append(x_stack)
            #         K.append(batch_tokens[l+1])
    
            # x = torch.stack(x, dim=0)
            # K = torch.stack(K, dim=0)
    
            # logits = self.mlp(x))  # (B, V)
    
            # risk_preds = [logits[:, vocab_index].unsqueeze(1).tile(1000).cpu().detach().numpy() for vocab_index in range(logits.shape[-1])] 
            # return_dict = {"surv_CDF": risk_preds, "k": [K]}
            
            # return {"surv": return_dict}, logits, None
            


    def training_step(self, batch, batch_idx):
        _, loss_dict, _ = self(batch)   
        for _key in loss_dict.keys():
            self.log(f"train_" + _key, loss_dict[_key], prog_bar=False, logger=True, sync_dist=True)
        return loss_dict['loss']
        
    def validation_step(self, batch, batch_idx):
        _, loss_dict, (logits, k) = self(batch)   
        for _key in loss_dict.keys():
            self.log(f"val_" + _key, loss_dict[_key], prog_bar=False, logger=True, sync_dist=True)

        # Record accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == k).float().mean()
        self.log('val_acc', acc, prog_bar=True)
            
        return loss_dict['loss']
    

    def test_step(self, batch, batch_idx):
        _, loss_dict, (logits, k) = self(batch)   
        for _key in loss_dict.keys():
            self.log(f"test_" + _key, loss_dict[_key], prog_bar=False, logger=True, sync_dist=True)
        
        # Record accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == k).float().mean()
        self.log('test_acc', acc, prog_bar=True)
        
        return loss_dict['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.optim.learning_rate)
        return {
            "optimizer": optimizer,
        }


def setup_mlp_experiment(cfg, dm, vocab_size, checkpoint=None, logger=None):

    #########################################################
    # Load existing pre-trained model,                      #
    #     overriding config where necessary                 #
    #########################################################
    if checkpoint is None:
        next_event_experiment = CausalMLPExperiment(cfg=cfg,
                                                    vocab_size=vocab_size,
                                                    )
    else:
        next_event_experiment = CausalMLPExperiment.load_from_checkpoint(checkpoint,
                                                                         cfg=cfg)
    # if torch.cuda.is_available():
    #     next_event_experiment = torch.compile(next_event_experiment)
    
    logging.debug(next_event_experiment)

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

    return next_event_experiment, CausalMLPExperiment, _trainer