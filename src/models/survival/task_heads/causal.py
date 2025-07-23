import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from CPRD.src.models.TTE.base import TTETransformer
from CPRD.src.modules.head_layers.survival.competing_risk import ODESurvCompetingRiskLayer
from CPRD.src.modules.head_layers.survival.single_risk import ODESurvSingleRiskLayer
from CPRD.src.modules.head_layers.value_layers import GaussianRegressionLayer

from typing import Optional
import logging

class SurvStreamGPTForCausalModelling(nn.Module):
    r"""    
    """
    
    def __init__(self, 
                 cfg,
                 vocab_size,
                 use_adapter=False,
                 concurrent_strategy=None,
                ):
        super().__init__()
        
        total_weight = cfg.head.surv_weight + cfg.head.value_weight
        self.surv_weight = cfg.head.surv_weight / total_weight
        self.value_weight = cfg.head.value_weight / total_weight
        self.block_size = cfg.transformer.block_size
        
        self.n_embd = cfg.transformer.n_embd                                                      # Total number of embedded dimensions after MHA concatenation
        self.n_embd_per_head = cfg.transformer.n_embd // cfg.transformer.n_head                   # How many of these dimensions belong to each head
        self.n_embd_private = cfg.transformer.private_heads * self.n_embd_per_head                # and how many of these dimensions are private
        
        self.transformer = TTETransformer(cfg, vocab_size, use_adapter=use_adapter)

        match cfg.head.SurvLayer.lower():
            # Removing padding token from vocab size as this is not considered an event in either case.
            case "single-risk" | "sr":
                raise NotImplementedError    #  this has been replaced with a SingleRisk layer, which must now be wrapped with a new class for the causal case
                # self.surv_layer = ODESurvSingleRiskLayer(self.n_embd - self.n_embd_private, [], num_risks=vocab_size - 1, device="cuda")
            case "competing-risk" | "cr":

                self.surv_layer = ODESurvCompetingRiskLayer(self.n_embd - self.n_embd_private, 
                                                            hidden_dim=32,
                                                            num_risks=vocab_size-1,
                                                            concurrent_strategy=concurrent_strategy,
                                                            device='cuda' if torch.cuda.is_available() else 'cpu')
            case _:
                raise ValueError(f"Survival head must be either 'single-risk' or 'competing-risk'")


        # Regression layers, create a separate regression layer for each measurement
        #   In the case we want to include private_heads, then 
        self.value_layer = GaussianRegressionLayer(self.n_embd - self.n_embd_private,
                                                   measurement_tokens=cfg.head.tokens_for_univariate_regression,
                                                   base_hidden_dim=32,  # None
                                                   )

        # apply special scaled init to the residual projections, per GPT-2
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def forward(self, 
                tokens:                 torch.tensor,
                ages:                   torch.tensor,
                values:                 torch.tensor,
                covariates:             torch.tensor,
                attention_mask:         torch.tensor,
                is_generation:          bool = False,
                return_generation:      bool = False,
                return_loss:            bool = True,
                ):
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


        Note 1:
          Typically we have no way of computing the losses for the final token element of the sequence as we have no subsequent target.
          Therefore, we would remove the final sequence element's hidden state (as this has no target). This shift is done inside 
          each of the called modules where we predict the final seq_len - 1 elements from the fist seq_len - 1 hidden states. In 
          this case, the first element is not included as a target in the loss. 
          e.g. see https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L981

          This is true even for the survival head, as even though we could censor the target token, we do not have a time delta.

        """
        
        hidden_states = self.transformer(tokens=tokens, 
                                         ages=ages, 
                                         values=values,
                                         covariates=covariates,
                                         attention_mask=attention_mask)  # shape: (bsz, seq_len, n_embd)

        # survival time to event head (survival curve until next token)
        surv_dict, losses_desurv = self.surv_layer.predict(hidden_states[:,:,:self.n_embd - self.n_embd_private],
                                                           target_tokens=tokens,
                                                           target_ages=ages, 
                                                           attention_mask=attention_mask,
                                                           is_generation=is_generation,
                                                           return_loss=return_loss,
                                                           return_cdf=return_generation,
                                                          )
            
        # regression head (values of next token if applicable)
        values_dist, loss_values = self.value_layer.predict(hidden_states[:,:, self.n_embd_private:],
                                                            target_tokens=tokens,
                                                            target_values=values,
                                                            attention_mask=attention_mask,
                                                            is_generation=is_generation,
                                                            return_loss=return_loss,
                                                            return_value_dist=return_generation,
                                                            )

        if return_loss:
            loss_desurv = torch.sum(torch.stack(losses_desurv))                                  # losses are returned as a list, as the Single-Risk head is many DeSurv models in parallel, combine
            loss = (self.surv_weight * loss_desurv) + (self.value_weight * loss_values)          # Weight the loss
        else:
            loss_desurv = None
            loss = None

        outputs = {"surv": surv_dict,
                   "values_dist": values_dist}
        losses = {"loss": loss,
                  "loss_desurv": loss_desurv,
                  "loss_values": loss_values
                 }
        
        return outputs, losses, hidden_states
    
    def generate(self, 
                 tokens: torch.tensor,
                 ages: torch.tensor,
                 values: torch.tensor,
                 static_covariates: torch.tensor,
                 attention_mask: torch.tensor,
                 max_new_tokens: int = 50,
                 exceed_block_size: bool = False,
                 **kwargs
                 ):
        """ Generate future samples for the single-risk."""

        device = tokens.device
        batch_size, context_length = tokens.shape
        B = self.block_size
        
        # If we don't want to exceed block size then there is no point generating more new tokens than block_size
        # num_to_fill_lowest_record = self.block_size - torch.sum()
        max_new_tokens = max_new_tokens if exceed_block_size else np.min((self.block_size, max_new_tokens))
        
        for _ in range(max_new_tokens):
                    
            # for each example, grab the last B attended positions
            # build empty (pad-filled) windows:
            windowed_tokens = torch.full((batch_size, B), 0,    device=device, dtype=tokens.dtype)
            windowed_ages   = torch.full((batch_size, B), 0,    device=device, dtype=ages.dtype)
            windowed_values = torch.full((batch_size, B), torch.nan,    device=device, dtype=values.dtype)
            windowed_attention_mask = torch.full((batch_size, B), 0,    device=device, dtype=values.dtype)
            for i in range(batch_size):
                att_pos = attention_mask[i].nonzero(as_tuple=True)[0]   # all indices where mask==1
                if att_pos.numel() == 0:
                    continue
                sel = att_pos[-B:]                                      # at most B positions
                L   = sel.size(0)
                # copy into the *leftmost* L slots of our window
                windowed_tokens[i, :L] = tokens[i, sel]
                windowed_ages  [i, :L] = ages  [i, sel]
                windowed_values[i, :L] = values[i, sel]
                windowed_attention_mask[i, :L] = 1

            # Get the generation predictions
            outputs, _, hidden_states = self(tokens=windowed_tokens, 
                                             ages=windowed_ages,
                                             values=windowed_values, 
                                             covariates=static_covariates,
                                             attention_mask=windowed_attention_mask,
                                             is_generation=True,
                                             return_generation=True,
                                             return_loss=False,
                                             )
            pred_surv = outputs["surv"]["surv_CDF"]
            pred_values = outputs["values_dist"]

            # sample next event tokens and age-deltas
            next_tokens, next_delta_ages =  self.surv_layer.sample_surv(pred_surv)
            # build a tensor of next_values
            next_vals = []
            for sample_idx, tok in enumerate(next_tokens):
                if tok in self.value_layer.measurement_tokens:
                    dist = pred_values[self.value_layer.token_key(tok)]
                    next_vals.append(dist.sample()[sample_idx].item())
                else:
                    next_vals.append(torch.nan)
            next_values = torch.tensor(next_vals, device=device, dtype=values.dtype)
            
            # compute current lengths
            lengths = attention_mask.sum(dim=1).long()              # (bsz,)
            
            # Extend with space for new value
            tokens = torch.cat(
                [tokens, torch.full((batch_size, 1), 0, device=device, dtype=tokens.dtype)], dim=1
            )
            ages = torch.cat(
                [ages, torch.full((batch_size, 1), 0, device=device, dtype=ages.dtype)], dim=1
            )
            values = torch.cat(
                [values, torch.full((batch_size, 1), torch.nan, device=device, dtype=values.dtype)], dim=1
            )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros((batch_size, 1), device=device, dtype=attention_mask.dtype)],
                    dim=1
                )
                
            # Add new generated event
            for idx in range(batch_size):
                tokens[idx, lengths[idx]] = next_tokens[idx]
                ages[idx, lengths[idx]] = ages[idx, lengths[idx]-1] + next_delta_ages[idx]
                values[idx, lengths[idx]] = next_values[idx]
                attention_mask[idx, lengths[idx]] = 1

        if not exceed_block_size:
            tokens = tokens[:, :self.block_size]
            ages = ages[:, :self.block_size]
            values = values[:, :self.block_size]
            attention_mask[:, :self.block_size]

        return tokens, ages, values, attention_mask
