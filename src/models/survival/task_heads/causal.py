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
                 vocab_size):
        super().__init__()
        
        total_weight = cfg.head.surv_weight + cfg.head.value_weight
        self.surv_weight = cfg.head.surv_weight / total_weight
        self.value_weight = cfg.head.value_weight / total_weight
        self.block_size = cfg.transformer.block_size
        
        self.n_embd = cfg.transformer.n_embd                                                      # Total number of embedded dimensions after MHA concatenation
        self.n_embd_per_head = cfg.transformer.n_embd // cfg.transformer.n_head                   # How many of these dimensions belong to each head
        self.n_embd_private = cfg.transformer.private_heads * self.n_embd_per_head                # and how many of these dimensions are private
        
        self.transformer = TTETransformer(cfg, vocab_size)

        match cfg.head.SurvLayer.lower():
            # Removing padding token from vocab size as this is not considered an event in either case
            case "single-risk" | "sr":
                self.surv_layer = ODESurvSingleRiskLayer(self.n_embd - self.n_embd_private, [], num_risks=vocab_size - 1, device="cuda")
            case "competing-risk" | "cr":
                self.surv_layer = ODESurvCompetingRiskLayer(self.n_embd - self.n_embd_private, [], num_risks=vocab_size - 1, device="cuda")
            case _:
                raise ValueError(f"Survival head must be either 'single-risk' or 'competing-risk'")


        # Regression layers, create a separate regression layer for each measurement
        #   In the case we want to include private_heads, then 
        self.value_layer = GaussianRegressionLayer(self.n_embd - self.n_embd_private,
                                                   measurement_tokens=cfg.head.tokens_for_univariate_regression
                                                   )

        # apply special scaled init to the residual projections, per GPT-2
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def forward(self, 
                tokens:                 torch.tensor,
                ages:                   torch.tensor,
                values:                 torch.tensor,
                covariates:             Optional[torch.tensor] = None,
                attention_mask:         Optional[torch.tensor] = None,
                is_generation:          bool = True,
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
                 covariates: Optional[torch.tensor] = None,
                 eos_token: Optional[int] = None,               # add DEATH to determine EOS later?
                 max_new_tokens: int = 50,
                 ):
        """ Generate future samples for the single-risk
        
        # TODO: havent tested for batched generation
        """
        
        for _ in range(max_new_tokens):
            # crop tokens to the last block_size tokens 
            if tokens.shape[1] > self.block_size:
                logging.debug(r"Context window is greater than block size." + \
                              " This is not compatible with the `sparse` setting of `FoundationalDataset()` which enforces earlier diagnoses are prepended to batch windows.")
            tokens_window = tokens[:, -self.block_size:]
            ages_window = ages[:, -self.block_size:] 
            values_window = values[:, -self.block_size:] 

            # get the predictions
            outputs, _, _ = self(tokens=tokens_window, 
                                 ages=ages_window,
                                 values=values_window, 
                                 covariates=covariates,
                                 is_generation=True,
                                 return_generation=True,
                                 return_loss=False,
                                )

            # sample survival 
            surv = outputs["surv"]["surv_CDF"]
            token_next, delta_age =  self.surv_layer.sample_surv(surv)
            ages_next = ages[:, [-1]] + delta_age
            
            # values
            values_next = []
            for i in range(token_next.shape[0]):
                if token_next[i, 0].item() in self.value_layer.measurement_tokens:
                    values_next.append(outputs["values_dist"][self.value_layer.token_key(token_next[i, 0])].sample()[0])
                else:
                    values_next.append(torch.tensor([torch.nan], device=tokens.device))

            # print(values_next)
            values_next = torch.stack(values_next)    # (B, 1)
            # print(values_next.shape)
            
            # append generated samples to the running sequence
            tokens = torch.cat((tokens, token_next), dim=1) # (B, T+1)
            ages = torch.cat((ages, ages_next), dim=1) 
            values = torch.cat((values, values_next), dim=1) 

            # TODO: add death token as EOS token
            if token_next == eos_token:
                break

            if ages_next > 120*365:
                logging.warning("Breaking generation due to implausible age")
                break
            
        return tokens, ages, values
