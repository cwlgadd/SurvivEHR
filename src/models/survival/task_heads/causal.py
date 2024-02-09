import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from CPRD.src.models.TTE.base import TTETransformer
from CPRD.src.modules.head_layers.surv_layers import ODESurvSingleLayer
from CPRD.src.modules.head_layers.value_layers import GaussianRegressionLayer

from typing import Optional
import logging

class SurvStreamGPTForCausalModelling(nn.Module):
    r"""    
    """
    
    def __init__(self, 
                 config,
                 vocab_size):
        super().__init__()
        self.config = config
        self.surv_weight = 1/2
        self.value_weight = 1/2
        
        self.transformer = TTETransformer(config, vocab_size)

        match config.SurvLayer.lower():
            case "single-risk" | "sr":
                self.surv_layer = ODESurvSingleLayer(config.n_embd, [], num_events=vocab_size, device="cuda")
            case "competing-risk" | "cr":
                raise NotImplementedError
            case _:
                raise ValueError(f"Survival head must be either 'single-risk' or 'competing-risk'")


        # Regression layers, create a separate regression layer for each measurement
        self.value_layer = GaussianRegressionLayer(config.n_embd,
                                                   measurement_tokens=config.tokens_for_univariate_regression
                                                   )

        # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def forward(self, 
                tokens: torch.tensor,
                ages: torch.tensor,
                values: torch.tensor,
                attention_mask: Optional[torch.tensor] = None,
                is_generation: bool = False
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
                                         attention_mask=attention_mask)  # shape: (bsz, seq_len, n_embd)

        # survival time to event head (survival curve until next token)
        surv, losses_desurv = self.surv_layer.predict(hidden_states,
                                                      target_tokens=tokens,
                                                      target_ages=ages, 
                                                      attention_mask=attention_mask,
                                                      is_generation=is_generation)
            
        # regression head (values of next token if applicable)
        values_dist, loss_values = self.value_layer.predict(hidden_states,
                                                            target_tokens=tokens,
                                                            target_values=values,
                                                            attention_mask=attention_mask,
                                                            is_generation=is_generation,
                                                            )

        if not is_generation:
            loss = (self.surv_weight * torch.sum(losses_desurv)) + (self.value_weight * loss_values)
        else:
            loss = None

        return (surv, values_dist), (losses_desurv, loss_values), loss
    
    def generate(self, 
                 tokens: torch.tensor,
                 ages: torch.tensor,
                 values: torch.tensor,
                 # eos_token: Optional[int] = None,               # add this later
                 max_new_tokens: int = 50):
        """ Generate future samples for the single-risk
        
        # TODO: havent tested for batched generation
        """
        
        for _ in range(max_new_tokens):
            # crop tokens to the last block_size tokens
            tokens_window = tokens[:, -self.config.block_size:]
            ages_window = ages[:, -self.config.block_size:] 
            values_window = values[:, -self.config.block_size:] 

            # get the predictions
            (surv, value_dists), _, _ = self(tokens=tokens_window, 
                                             ages=ages_window,
                                             values=values_window, 
                                             is_generation=True)

            # sample survival 
            token_next, delta_age =  self.surv_layer.generate_sample(surv)
            ages_next = ages[:, [-1]] + delta_age
            
            # values
            values_next = []
            for i in range(token_next.shape[0]):
                if token_next[i, 0].item() in self.value_layer.measurement_tokens:
                    values_next.append(value_dists[self.value_layer.token_key(token_next[i, 0])].sample()[0])
                else:
                    values_next.append(torch.tensor([torch.nan], device=tokens.device))

            # print(values_next)
            values_next = torch.stack(values_next)    # (B, 1)
            # print(values_next.shape)
            
            # append generated samples to the running sequence
            tokens = torch.cat((tokens, token_next), dim=1) # (B, T+1)
            ages = torch.cat((ages, ages_next), dim=1) 
            values = torch.cat((values, values_next), dim=1) 

            # print(f"tokens {tokens}")
            # print(f"ages {ages}")
            # print(f"values {values}")

            # if token_next == eos_token:
            #     raise NotImplementedError
            #     break

            if ages_next > 120*365:
                logging.warning("TODO: add death token")
                break
            
        return tokens, ages, values
