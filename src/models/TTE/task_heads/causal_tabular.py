import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from CPRD.src.models.TTE.base import TTETransformer
from CPRD.src.models.TTE.tte_layers import GeometricTTELayer, ExponentialTTELayer
from CPRD.src.models.TTE.value_layers import GaussianRegressionLayer

from typing import Optional
import logging

class TTETransformerForCausalTimeSeriesModelling(nn.Module):
    r"""    
    """
    
    def __init__(self, 
                 config,
                 vocab_size):
        super().__init__()
        self.config = config
        self.token_weight = 1/3
        self.age_weight = 1/3
        self.value_weight = 1/3
        
        self.transformer = TTETransformer(config, vocab_size)

        # lm head with weight tying on embedding and softmax layer. 
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight   # See https://paperswithcode.com/method/weight-tying
        # self.apply(self.transformer._init_weights)          # initialise all the weights, before special layers

        # time-to-event layer
        match config.TTELayer.lower():
            case "geometric":
                self.tte_layer = GeometricTTELayer(config.n_embd)
            case "exponential":
                self.tte_layer = ExponentialTTELayer(config.n_embd)
            case _:
                raise ValueError(f"TTELayer must be either Geometric or Exponential")

        # Regression layers, create a separate regression layer for each measurement
        self.value_layer = GaussianRegressionLayer(config.n_embd,
                                                   measurement_tokens=config.tokens_for_univariate_regression
                                                   )

        # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    
    def _predict_token(self, 
                       hidden_states: torch.tensor,
                       target_tokens: Optional[torch.tensor] = None,
                       attention_mask: Optional[torch.tensor] = None,
                       is_generation: bool = False
                       ):
        r"""
        
        """
        
        if not is_generation:
            assert target_tokens is not None
            
            # if we are not generating samples, we calculate the loss
            logits = self.lm_head(hidden_states)                                 # shape: (bsz, seq_len, n_embd)
            logits = logits[:, :-1, :].contiguous()                              # shape: (bsz, seq_len - 1, n_embd)
            targets = torch.where(attention_mask[:, 1:] == 1, target_tokens[:, 1:], -100) if attention_mask is not None else target_tokens[:, 1:]

            # if we are given some desired targets also calculate the loss
            B, T, C = logits.shape
            logits = logits.contiguous().view(B*T, C)
            targets = targets.contiguous().view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=-100)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(hidden_states[:, [-1], :])    # note: using list [-1] to preserve the seq_len dim
            loss = None

        return logits, loss

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
            ages

            values

        KWARGS:
            attention_mask:

            is_generation:

        Note:
        We have no way of computing the losses for the final token element of the sequence as we have no subsequent target.
          Therefore, we must remove the final sequence element's hidden state (as this has no target). This shift is done inside 
          each of the called modules where we predict the final seq_len - 1 elements from the fist seq_len - 1 hidden states. In 
          this case, the first element is not included as a target in the loss. 
          e.g. see https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L981
        """
        
        hidden_states = self.transformer(tokens=tokens, 
                                         ages=ages, 
                                         values=values,
                                         attention_mask=attention_mask)  # shape: (bsz, seq_len, n_embd)

        # classifier head (next token)
        logits, loss_clf = self._predict_token(hidden_states, 
                                               target_tokens=tokens,
                                               attention_mask=attention_mask,
                                               is_generation=is_generation)

        # time to event head (time until next token)
        tte_dist, loss_tte = self.tte_layer.predict(hidden_states,
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
            loss = (self.token_weight * loss_clf) + (self.age_weight * loss_tte) + (self.value_weight * loss_values)
        else:
            loss = None

        return (logits, tte_dist, values_dist), (loss_clf, loss_tte, loss_values), loss
    
    def generate(self, 
                 tokens: torch.tensor,
                 ages: torch.tensor,
                 values: torch.tensor,
                 # eos_token: Optional[int] = None,               # add this later
                 max_new_tokens: int = 50):
        """ Generate future samples.
        
        # TODO: havent tested for batched generation
        """
        
        for _ in range(max_new_tokens):
            # crop tokens to the last block_size tokens
            tokens_window = tokens[:, -self.config.block_size:]
            ages_window = ages[:, -self.config.block_size:] 
            values_window = values[:, -self.config.block_size:] 

            # get the predictions
            (logits, delta_age, value_dists), _, _ = self(tokens=tokens_window, 
                                                          ages=ages_window, 
                                                          values=values_window, 
                                                          is_generation=True)
            
            # apply softmax to get probabilities,   (bsz, C)
            probs = F.softmax(logits.squeeze(1), dim=-1)                               
            token_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # print(token_next.shape)
            
            # sample from the distributions
            # age
            ages_next = ages[:, [-1]] + delta_age     # (B, 1)
            # print(f"ages next {ages_next}")
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
            
        return tokens, ages, values
