# This module implements the TTE and regression generative emission layers used in the model.
# adapted from https://github.com/mmcdermott/EventStreamGPT/blob/main/EventStream/transformer/generative_layers.py
import torch
from typing import Optional
import logging
# from pytorch_lognormal_mixture import LogNormalMixtureDistribution


class TTELayerBase(torch.nn.Module):

    def __init__(self):
        logging.warning("refactor classes to share common predict methods, such as preparing ages for next time prediction")
        raise NotImplementedError

    def forward(self):

        pass


class GeometricTTELayer(torch.nn.Module):
    """A class that outputs an geometric distribution for the number of failures before the first success.

    The geometric distribution is the discrete analogue of the exponential distribution.

    This module is used to predict time to event in the ForCausalSequenceModelling and CausalTimeSeriesModelling sets of heads. 
    The input tensor is projected to get the implied geometric distribution.
    
    Args:
        in_dim: The dimensionality of the input.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, 1)
        # Because we are looking at very long tailed geometric distributions, initialise the bias so small logits are predicted at init
        if self.proj.bias is not None:
            self.proj.bias.data.uniform_(-20, -3)
            
        # self._loss_scalar = 1
        logging.info("Using GeometricTTELayer. This module predicts the time until next event as a geometric distribution, supported on the set {0,1,...}")

    def predict(self,
                hidden_states: torch.tensor,                    # shape: torch.Size([bsz, seq_len, n_embd])
                target_ages: Optional[torch.tensor] = None,             
                attention_mask: Optional[torch.tensor] = None,
                is_generation: bool = False
                ):
        r"""
        """

        if not is_generation:
            assert target_ages is not None
            
            if attention_mask is None:
                raise NotImplementedError
                
            tte_dist = self(hidden_states[:, :-1, :])           # Exponential(rate: torch.Size([bsz, seq_len - 1]))

            # We are predicting the delta of time, but each element in the seq_len just has the time of event. 
            # This means the output mask requires both the time at the event to be available, and the time of 
            # the next event.
            tte_obs_mask = attention_mask[:, :-1] & attention_mask[:, 1:]   # shape: torch.Size([bsz, seq_len - 1])
            
            # Get time to event, excluding first in sequence as we do not know what time the one pre-dating it occurred
            tte_deltas = target_ages[:, 1:] - target_ages[:, :-1]                         # shape: torch.Size([bsz, seq_len - 1])
            tte_deltas = torch.where(tte_obs_mask == 1, tte_deltas, torch.ones_like(tte_deltas)) 
            assert torch.all(tte_deltas >= 0), f"events must be given in time order, {tte_deltas[tte_deltas<0]}"

            # Calculate loss, including on masked values which are set to one just to avoid errors
            log_prob = tte_dist.log_prob(tte_deltas)                        # shape: torch.Size([bsz, seq_len - 1])

            # Mask and sum across sequence (so log likelihood factorises as a product along the sequence)
            #  As we do not filter to ensure that sequences have at least two points, we also add a small positive constant to 
            #  the denominator to avoid division by zero for sequences with only one event, and so no observed transitions as
            #  in those case the numerator is zero due to all transitions being masked.
            tte_ll_per_patient = (log_prob * tte_obs_mask.float()).sum(-1) / (tte_obs_mask.float().sum(-1) + 1e-5)  # shape: torch.Size([bsz])
            
            if torch.isnan(tte_ll_per_patient).any():
                print(f"tte deltas {tte_deltas}")
                print(f"log prob {log_prob}")
                print(f"mask {tte_obs_mask.float()}")
                print(f"numerator is then {(log_prob * tte_obs_mask.float()).sum(-1)}")
                print(f"denominator is {tte_obs_mask.float().sum(-1) + 1e-5}")
                print(f"raised by {tte_ll_per_patient}")
                raise NotImplementedError
            
            # average across batch
            loss = - tte_ll_per_patient.mean() 

        else:        
            # inference-time mini-optimization: only forward the head on the very last position
            tte_dist = self(hidden_states[:, [-1], :])       # Exponential(rate: torch.Size([bsz, 1]))
                                                             #    note: using list [-1] to preserve the seq_len dim
            tte_dist = tte_dist.sample() 
            loss = None

        return tte_dist, loss

    
    def forward(self, hidden_states: torch.Tensor) -> torch.distributions.exponential.Exponential:
        """Forward pass.

        Args:
            hidden_states: The input tensor.

        Returns:
            An `Exponential` distribution with parameters specified by `self.proj(hidden_states)` which has output shape
            `(batch_size, sequence_length, 1)`.
        """
        # The projection has shape (batch_size, sequence_length, 1). We want to squeeze that last dimension.
        logits = self.proj(hidden_states).squeeze(dim=-1)
        return torch.distributions.geometric.Geometric(logits=logits)
        

class ExponentialTTELayer(torch.nn.Module):
    """A class that outputs an exponential distribution for time-to-event.

    The exponential distribution is the probability distribution of the time between events in a Poisson point process,
    i.e., a process in which events occur continuously and independently at a constant average rate. Implemented within
    our framework this is technically no longer a point process as this markov assumption is lost as the rate parameter
    is dependent on the hidden states, which depend on the entire block. The exponential distribution is the continuous 
    analogue of the geometric distribution.

    This module is used to predict time to event in the ForCausalSequenceModelling and CausalTimeSeriesModelling sets of 
    heads. The input tensor is projected to get the implied exponential distribution.

    Args:
        in_dim: The dimensionality of the input.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, 1)
        self._normalising_scaling_constant = 1000
        logging.info("Using ExponentialTTELayer. This module predicts the time until next event as an exponential distribution")


    def predict(self,
                hidden_states: torch.tensor,                    # shape: torch.Size([bsz, seq_len, n_embd])
                target_ages: Optional[torch.tensor] = None,             
                attention_mask: Optional[torch.tensor] = None,
                is_generation: bool = False
                ):
        r"""
        """

        if not is_generation:
            assert target_ages is not None
            
            if attention_mask is None:
                raise NotImplementedError
            
            tte_dist = self(hidden_states[:, :-1, :])           # Exponential(rate: torch.Size([bsz, seq_len - 1]))

            # We are predicting the delta of time, but each element in the seq_len just has the time of event. 
            # This means the output mask requires both the time at the event to be available, and the time of 
            # the next event.
            tte_obs_mask = attention_mask[:, :-1] & attention_mask[:, 1:]   # shape: torch.Size([bsz, seq_len - 1])
            
            # Get time to event, excluding first in sequence as we do not know what time the one pre-dating it occurred
            tte_deltas = target_ages[:, 1:] - target_ages[:, :-1]                         # shape: torch.Size([bsz, seq_len - 1])
            tte_deltas = tte_deltas / self._normalising_scaling_constant  
            tte_deltas = torch.where(tte_obs_mask == 1, tte_deltas, torch.ones_like(tte_deltas)) 
            assert torch.all(tte_deltas >= 0), f"events must be given in time order, {tte_deltas[tte_deltas<0]}"

            # Calculate loss, including on masked values which are set to one just to avoid errors
            log_prob = tte_dist.log_prob(tte_deltas)                        # shape: torch.Size([bsz, seq_len - 1])

            # Mask and sum across sequence (so log likelihood factorises as a product along the sequence)
            #  As we do not filter to ensure that sequences have at least two points, we also add a small positive constant to 
            #  the denominator to avoid division by zero for sequences with only one event, and so no observed transitions as
            #  in those case the numerator is zero due to all transitions being masked.
            tte_ll_per_patient = (log_prob * tte_obs_mask.float()).sum(-1) / (tte_obs_mask.float().sum(-1) + 1e-5)  # shape: torch.Size([bsz])
            
            if torch.isnan(tte_ll_per_patient).any():
                print(f"tte deltas {tte_deltas}")
                print(f"log prob {log_prob}")
                print(f"mask {tte_obs_mask.float()}")
                print(f"numerator is then {(log_prob * tte_obs_mask.float()).sum(-1)}")
                print(f"denominator is {tte_obs_mask.float().sum(-1) + 1e-5}")
                print(f"raised by {tte_ll_per_patient}")
                raise NotImplementedError
            
            # average across batch
            loss = - tte_ll_per_patient.mean()

        else:        
            # inference-time mini-optimization: only forward the head on the very last position
            tte_dist = self(hidden_states[:, [-1], :])       # Exponential(rate: torch.Size([bsz, 1]))
                                                             #    note: using list [-1] to preserve the seq_len dim
            tte_dist = tte_dist.rsample() * self._normalising_scaling_constant
            loss = None

        return tte_dist, loss
        
    def forward(self, hidden_states: torch.Tensor) -> torch.distributions.exponential.Exponential:
        """Forward pass.

        Args:
            hidden_states: The input tensor.

        Returns:
            An `Exponential` distribution with parameters specified by `self.proj(hidden_states)` which has output shape
            `(batch_size, sequence_length, 1)`.
        """

        # print(hidden_states)
        hidden = self.proj(hidden_states)
        # print(hidden)
        
        # To ensure validity given numerical imprecision, we also add a buffer given
        # by the smallest possible positive value permissible given the type of `hidden_states`.
        rate = torch.nn.functional.elu(hidden) + 1 + torch.finfo(hidden_states.dtype).tiny

        # The rate currently has shape (batch_size, sequence_length, 1). We want to squeeze that last
        # dimension.
        rate = rate.squeeze(dim=-1)

        return torch.distributions.exponential.Exponential(rate=rate)



# class LogNormalMixtureTTELayer(torch.nn.Module):
#     """A class that outputs a mixture-of-lognormal distribution for time-to-event.

#     This class is used to initialize a module and project the input tensor to get a specific
#     LogNormal Mixture distribution.

#     Args:
#         in_dim: The dimension of the input tensor.
#         num_components: The number of lognormal components in the mixture distribution.
#         mean_log_inter_time: The mean of the log of the inter-event times. Used to initialize the mean
#                              of the log of the output distribution. Defaults to 0.0.
#         std_log_inter_time: The standard deviation of the log of the inter-event times. Used to initialize
#                             the standard deviation of the logs of the output distributions. Defaults to 1.0.
#     """

#     def __init__(
#         self,
#         in_dim: int,
#         num_components: int,
#         mean_log_inter_time: float = 0.0,
#         std_log_inter_time: float = 1.0,
#     ):
#         super().__init__()

#         # We multiply by 3 in the projections as we need to get the locs, log_scales, and weights for each
#         # component.
#         self.proj = torch.nn.Linear(in_dim, 3 * num_components)

#         self.mean_log_inter_time = mean_log_inter_time
#         self.std_log_inter_time = std_log_inter_time

#     def forward(self, T: torch.Tensor) -> LogNormalMixtureDistribution:
#         """Forward pass.

#         Args:
#             T: The input tensor.

#         Returns:
#             A `LogNormalMixtureDistribution` with parameters specified by `self.proj(T)` which has output
#             shape `(batch_size, sequence_length, 1)`.
#         """
#         params = self.proj(T)

#         locs = params[..., 0::3]
#         log_scales = params[..., 1::3]
#         log_weights = params[..., 2::3]

#         return LogNormalMixtureDistribution(
#             locs=locs,
#             log_scales=log_scales,
#             log_weights=log_weights,
#             mean_log_inter_time=self.mean_log_inter_time,
#             std_log_inter_time=self.std_log_inter_time,
#         )