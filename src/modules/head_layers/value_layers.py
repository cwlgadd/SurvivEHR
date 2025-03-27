# This module implements the TTE and regression generative emission layers used in the model.
# adapted from https://github.com/mmcdermott/EventStreamGPT/blob/main/EventStream/transformer/generative_layers.py
import torch
from torch import nn
from typing import Optional
import logging
import numpy as np
# from pytorch_lognormal_mixture import LogNormalMixtureDistribution


class GaussianRegressionLayer(torch.nn.Module):
    """
    A probabilistic regression layer that models univariate measurements with a Normal distribution.
    
    This layer can generate mean and standard deviation parameters for each measurement token.
    Optionally, it uses a shared MLP (`base_regression_layer`) before per-measurement heads.

    Parameters
    ----------
    in_dim : int
        Dimensionality of the input embeddings.
    measurement_tokens : list of int, optional
        A list of measurement tokens for which separate Gaussian heads are created. If None or empty,
        no measurement heads are instantiated.
    base_hidden_dim : int, optional
        If provided, creates a shared MLP before each per-measurement output layer. If not provided,
        parameters for the Normal distribution are regressed directly from the input embeddings.

    Attributes
    ----------
    base_regression_layer : torch.nn.Sequential or None
        A shared MLP applied before each measurement-specific head. None if no shared MLP is used.
    regression_layers : torch.nn.ModuleDict
        A dictionary of measurement-token-specific regression heads. Each head outputs parameters
        for a Normal distribution (mean and standard deviation).

    Notes
    -----
    - Each measurement token has its own head that returns a Normal(loc, scale).
    - If `measurement_tokens` is not provided or empty, the layer logs a warning but will still
      instantiate successfully (no measurement heads).
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
        
    def __str__(self):
        s = "Gaussian Regression layer, with token keys:"
        for key, item in self.regression_layers.items():
            s += f"\n\t{key}"
        return s
    
    def predict(self,
                hidden_states: torch.tensor,                    # shape: torch.Size([bsz, seq_len, n_embd])
                target_tokens: Optional[torch.tensor] = None,
                target_values: Optional[torch.tensor] = None, 
                attention_mask: Optional[torch.tensor] = None,
                is_generation: bool = False,                         # Whether we forward every step (True) of seq_len, or just the final step (False)
                return_value_dist: bool = False,
                return_loss: bool = True,
                ):
        """
        Predict measurement values (mean, std) for each token and optionally compute log-likelihood loss.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape: (batch_size, seq_len, in_dim).
            The input embeddings or hidden states from which to predict measurement distributions.
        target_tokens : torch.Tensor, optional
            The integer tokens at each sequence position, shape: (batch_size, seq_len).
        target_values : torch.Tensor, optional
            Continuous target values for each token, shape: (batch_size, seq_len).
            Missing or invalid values should be NaN to ensure they're masked out of the loss.
        attention_mask : torch.Tensor, optional
            Binary mask for valid positions: 1 means valid, 0 means masked, shape: (batch_size, seq_len).
        is_generation : bool
            - If False, we process all positions except the final one for training.
            - If True, we might forward only the last state or handle generation-specific logic.
        return_value_dist : bool
            If True, returns the predicted Normal distributions. If False, returns None for distributions.
        return_loss : bool
            If True, computes and returns the negative log-likelihood loss. Otherwise, returns None for loss.

        Returns
        -------
        value_dist : dict of {str : torch.distributions.Normal} or None
            If `return_value_dist=True` and `is_generation=False`, a dict mapping token_key to Normal distributions
            (each has shape `[batch_size, seq_len-1]` in training, or `[batch_size, 1]` in generation). 
            If `is_generation=True`, may only contain the final position distribution.
            Returns None if `return_value_dist=False`.
        loss : torch.Tensor or None
            The average negative log-likelihood loss over the batch, or None if `return_loss=False`.

        Notes
        -----
        - When `is_generation=True` and `return_loss=True`, this method currently raises
          `NotImplementedError`. Implementation depends on your generation-time loss usage.
        - This method predicts distributions for all measurement tokens at every time step;
          you may further optimize by only forwarding the relevant tokens.

        Raises
        ------
        NotImplementedError
            If `is_generation=True` and `return_loss=True`, as that path is not implemented.            

        TODO
        -----
        - merge val_dists so only valid ones are returned
        - At the moment we predict every possible measure at every point during training and generation
          In reality we know what the target token was and so during training only the relevant hidden
          states need to be forwarded.
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
                
                # if mask.sum().item() < 1:
                #     logging.info("Ran value layer predict with no observed values")
                
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
                
                # average/sum across batch
                # loss += -token_ll_per_patient.sum() 
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
        """
        Forward pass for a single token's regression head, returning a Normal distribution.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape: (batch_size, sequence_length, in_dim).
            The embedding or hidden representation from which we predict measurement parameters.
        token_key : str
            A key (e.g. "Token 123") identifying the specific measurement token head to use.

        Returns
        -------
        torch.distributions.Normal
            A normal distribution parameterized by (mean, std), each of shape
            (batch_size, sequence_length).

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
