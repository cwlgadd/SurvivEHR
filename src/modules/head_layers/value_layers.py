# This module implements the TTE and regression generative emission layers used in the model.
# adapted from https://github.com/mmcdermott/EventStreamGPT/blob/main/EventStream/transformer/generative_layers.py
import torch
from torch import nn
from typing import Optional
import logging
import numpy as np
# from pytorch_lognormal_mixture import LogNormalMixtureDistribution


class GaussianRegressionLayer(torch.nn.Module):
    """A probabilistic regression layer that outputs a normal distribution for univariate measurement and test values.

    This module is used to predict value of a test or measurement event in the CausalTimeSeriesModelling set of heads. The input tensor is
    projected to get the implied normal distribution.
    
    Args:
        in_dim: The dimensionality of the input.
    """

    def __init__(self,
                 in_dim: int,
                 measurement_tokens: Optional[list[int]] = None,
                 base_hidden_dim: Optional[int] = None
                ):
        super().__init__()
        
        self.token_key = lambda token: f"Token {token.item() if isinstance(token, torch.Tensor) else token}"
        self.measurement_tokens = measurement_tokens

        # Optional shared base layers for each of the values predicted. 
        # This all results in a FC head. Structured like this as we want a dictionary module for easier code readability
        self.base_regression_layer = None
        if base_hidden_dim is not None:
            self.base_regression_layer = nn.Sequential(
                        nn.Linear(in_dim, base_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(base_hidden_dim, base_hidden_dim),
                        nn.ReLU()
            )
        
        # Create a separate network for each separate univariate Gaussian measurement that will be predicted
        self.regression_layers = torch.nn.ModuleDict({})
        if measurement_tokens is None:
            logging.warning("GaussianRegressionLayer has been initialised, but no tokens with values to be predicted. Check this is intended behaviour")
        else:
                
            for token in measurement_tokens:
                if self.token_key(token) in self.regression_layers:
                    raise ValueError(f"{self.token_key(token)} duplicated in configuration")

                if base_hidden_dim is None:
                    self.regression_layers[self.token_key(token)] = torch.nn.Linear(in_dim, 2)
                else:
                    self.regression_layers[self.token_key(token)] = nn.Sequential(
                        self.base_regression_layer,
                        torch.nn.Linear(base_hidden_dim, 2)
                    )
        
        logging.debug(f"Value base regression layer {self.base_regression_layer} " + \
                      f"regression layers {self.regression_layers}")

    def predict(self,
                hidden_states: torch.tensor,                    # shape: torch.Size([bsz, seq_len, n_embd])
                target_tokens: Optional[torch.tensor] = None,
                target_values: Optional[torch.tensor] = None, 
                attention_mask: Optional[torch.tensor] = None,
                is_generation: bool = False,                         # Whether we forward every step (True) of seq_len, or just the final step (False)
                return_value_dist: bool = False,
                return_loss: bool = True,
                ):
        r"""
    
        TODO: merge val_dists so only valid ones are returned
        
        TODO: At the moment we predict every possible measure at every point during training and generation
                In reality we know what the target token was and so during training only the relevant hidden
                states need to be forwarded.

        Note:
                In the generation case, we do not know yet what logit will be sampled as this sampling is
                performed after this. Consequently we return all forwarded regression layers for every hidden
                state. This may also be useful for analysis. and so we only need to forward the relevant ones 
                regression layers at those hidden states.
        """
        
        if not is_generation:
            

            assert target_tokens is not None
            assert target_values is not None
            assert attention_mask is not None
            
            # initialise loss
            loss = 0
            for token in self.measurement_tokens:

                # create empty value dist - not all of these will be filled (such as when the target is a diagnosis)
                value_dist = torch.distributions.normal.Normal(loc=torch.zeros_like(target_tokens[:, 1:]), 
                                                               scale=torch.ones_like(target_tokens[:, 1:]))  

                # Mask based on whether this token belongs to this layer head 
                token_mask = torch.where(target_tokens[:, 1:] == token, 1, 0)                
                # And add in value mask for missing (or removed in the case of outliers) values
                value_mask = torch.where(target_values[:, 1:].isnan(), 0, 1)
                # Add in attention mask (this is redundant but here for code clarity)                
                atn_mask = attention_mask[:, 1:] if attention_mask is not None else torch.ones_like(target_tokens[:, 1:])
                # combine
                mask = token_mask & value_mask & atn_mask
                
                # Pass the first N-1 hidden states through the token specific regression layer. 
                # We do not need the last hidden state as there is no target
                # TODO: We pass everything, even if it is later masked - this can be significantly optimised but kept like this for readability.
                # gives: Normal(mean: torch.Size([bsz, seq_len-1]), std: torch.Size([bsz, seq_len-1])) object
                token_value_dist = self(hidden_states[:, :-1, :], token_key=self.token_key(token))
                
                # update value_dist with token's entries
                value_dist.loc = torch.where(mask == 1, token_value_dist.loc, value_dist.loc)
                value_dist.scale = torch.where(mask == 1, token_value_dist.scale, value_dist.scale)
                
                # set target values that were masked or do not belong to current looped token to zero. 
                # They are masked in the loss, this just lets us pass the entire tensor through
                token_values = torch.where(mask == 1, target_values[:, 1:], 0) 

                # Calculate loss, including on masked values which were set to zero just to avoid errors
                log_prob = value_dist.log_prob(token_values)                 # shape: torch.Size([bsz, seq_len - 1])               

                # Mask and sum across sequence (so log likelihood factorises as a product along the sequence)
                #  As we do not filter to ensure that sequences have at least one token entry, we also add a small positive constant to 
                #  the denominator to avoid division by zero for sequences containing none of looped token.
                #  in those cases the numerator is also zero due to all entries being masked and so the ll is also zero
                token_ll_per_patient = (log_prob * token_mask.float()).sum(-1) / (token_mask.float().sum(-1) + 1e-5)  # shape: torch.Size([bsz])
                # print(token_ll_per_patient.shape)
                
                # average across batch
                loss += -token_ll_per_patient.mean() 
                
            # loss /= len(self.measurement_tokens)

        else:  

            if return_loss:

                assert target_tokens is not None
                assert target_values is not None
                assert attention_mask is not None
                
                # Forward the last (non-padded?) state. This will be used for fine-tuning a clinical prediction model, 
                # but another use case for is_generation = True is that we are simply generating future trajectories. 
                # In this case we want to just forward the last hidden state, irrespective of any potential padding
                raise NotImplementedError

            else:
                loss = None

            if return_value_dist:
                value_dist = {}
                for token in self.measurement_tokens:
                    # Pass every hidden state through the token specific regression layer
                    # inference-time mini-optimization: only forward the head on the very last position
                    token_value_dist = self(hidden_states[:, [-1], :],                 #    note: using list [-1] to preserve the seq_len dim
                                            token_key=self.token_key(token))           # Normal(mean: torch.Size([bsz, 1]), std: torch.Size([bsz, 1]))
                    
                    # Mask based on given attention mask and token mask (1=not masked and has valid token)
                    # token_mask = torch.where(tokens == token, torch.ones_like(tokens[:, [-1], :]), torch.zeros_like(tokens[:, [-1], :]))                
                    # and update value_dist with token's entries
                    # loc = torch.where(token_mask == 1, token_value_dist.loc, loc)
                    # scale = torch.where(token_mask == 1, token_value_dist.scale, scale)
                    # value_dist = torch.distributions.normal.Normal(loc=loc, scale=scale) 
                    
                    value_dist[self.token_key(token)] = token_value_dist
            else:
                value_dist = None
                
            

        return value_dist, loss

    
    def forward(self, 
                hidden_states: torch.Tensor,
                token_key: str) -> torch.distributions.exponential.Exponential:
        """Forward pass.

        Args:
            hidden_states: The input tensor of shape (batch_size, sequence_length, in_dim)

        Returns:
            The `torch.distributions.normal.Normal` distribution with parameters `self.proj(hidden_states)`,
            which will have output shape `(batch_size, sequence_length, 1)`.
        """
        # The projection has shape (batch_size, sequence_length, 1). We want to squeeze that last dimension.
        Z = self.regression_layers[token_key](hidden_states)
        Z_mean = Z[..., 0::2].squeeze(dim=-1)
        # torch.nn.functional.elu has idxmage (-1, 1), but we need our std parameter to be > 0. So we need to
        # add 1 to the output here. To ensure validity given numerical imprecision, we also add a buffer given
        # by the smallest possible positive value permissible given the type of `T`.
        Z_std = torch.nn.functional.elu(Z[..., 1::2]) + 1 + torch.finfo(hidden_states.dtype).tiny
        Z_std = Z_std.squeeze(dim=-1)
        
        return torch.distributions.normal.Normal(loc=Z_mean, scale=Z_std)        
