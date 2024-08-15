import numpy as np
import torch
import torch.nn as nn
import logging
from CPRD.src.modules.head_layers.survival.desurv import ODESurvSingle
from typing import Optional

class ODESurvSingleRiskLayer(nn.Module):
    """ Wrapper around single-risk version of DeSurv
    """
    
    def __init__(self, in_dim, hidden_dim, num_risks, device="cpu", n=15):

        super().__init__()
        self.sr_ode = [ODESurvSingle(cov_dim=in_dim,
                                     hidden_dim=hidden_dim,
                                     device=device,
                                     n=n) 
                       for _ in range(num_risks)]             # do not include pad token as an event 

        # this is the maximum period considered in generation, and also used as normalising constant in DeSurv
        self._time_scale = 365*10
        # the time grid which we generate over
        self.t_eval = np.linspace(0, self._time_scale,  int(self._time_scale*12/365) + 1)    # eval at roughly 1 month increments for 10 years
        self.device = device

        logging.info(f"Using Single-Risk DeSurvival head. This module predicts a separate survival curve for each possible future event")
        logging.info(f"Internally scaling time in survival head by {self._time_scale} days")
        logging.info(f"In generation forwarding DeSurv on the grid between [{self.t_eval.min()}, {self.t_eval.max()}]")
        logging.info(f"with {len(self.t_eval)} intervals of delta={self.t_eval[1]-self.t_eval[0]}")

    def predict(self,
                hidden_states: torch.tensor,                    # shape: torch.Size([bsz, seq_len, n_embd])
                target_tokens: Optional[torch.tensor] = None,   # shape: torch.Size([bsz, seq_len])
                target_ages: Optional[torch.tensor] = None,     # shape: torch.Size([bsz, seq_len])        
                attention_mask: Optional[torch.tensor] = None,  # shape: torch.Size([bsz, seq_len])
                is_causal: bool = True,                         # Whether we forward every step (True) of seq_len, or just the final step (False)
                return_cdf: bool = False,
                return_loss: bool = True,
                ):
        r"""
        """
        
        if is_causal:

            if return_loss:

                assert target_tokens is not None
                assert target_ages is not None
                assert attention_mask is not None
    
                # Get 1 vs. all event types. A list of len vocab_size-1 where each element of the list is an event
                #       The 1st element of list corresponds to 2nd vocab element (vocab index == 0 is the PAD token which is excluded)
                #       k \in {0,1} with 1 if the seq target is the same as the single risk ode's index (position in list), and 0
                #       otherwise
                k = [torch.where(target_tokens[:, 1:] == event + 1, 1, 0) for event, _ in enumerate(self.sr_ode)]
                # shape: [torch.Size([bsz, seq_len - 1]), ...]
                
                # We are considering the delta of time, but each element in the seq_len just has the time of event. 
                # This means the output mask requires both the time at the event, and the time of the next event to be available.
                tte_obs_mask = attention_mask[:, :-1] & attention_mask[:, 1:]   
                # shape: torch.Size([bsz, seq_len - 1])
                
                # Get time to event, excluding first in sequence as we do not know what time the one pre-dating it occurred
                tte_deltas = target_ages[:, 1:] - target_ages[:, :-1]                         
                tte_deltas = tte_deltas / self._time_scale  
                tte_deltas = torch.where(tte_obs_mask == 1, tte_deltas, torch.ones_like(tte_deltas)) 
                assert torch.all(tte_deltas >= 0), f"events must be given in time order, {tte_deltas[tte_deltas<0]}"
                # shape: torch.Size([bsz, seq_len - 1])
    
                # Vectorise
                in_hidden_state = hidden_states[:, :-1, :].reshape((-1, hidden_states.shape[-1]))        # torch.Size([bsz * (seq_len-1), hidden_size])
                tte_deltas = tte_deltas.reshape(-1)                                                      # torch.Size([bsz * (seq_len-1)])
                tte_obs_mask = tte_obs_mask.reshape(-1)                                                  # torch.Size([bsz * (seq_len-1)])
    
                # and apply the observation mask
                in_hidden_state = in_hidden_state[tte_obs_mask == 1]
                tte_deltas = tte_deltas[tte_obs_mask == 1]
                k = [_k.flatten()[tte_obs_mask == 1] for _k in k]
    
                # At this point we have a 1vAll single-risk for each type of event, where tte_deltas are the times to each next
                #  event, whether this occurred or not within this single-risk model. k indicates whether event occurred, where
                #  1=yes, and 0=another event occurred. 
                
                # Calculate losses, excluding masked values. Each sr_ode returns the sum over observed events
                #    to be consistent with other heads, we scale by number of observed values to obtain per SR-model mean
                #    and we sum across the mixture of survival ODEs
                surv_losses = [_sr_ode.loss(in_hidden_state, tte_deltas, _k) / _k.shape[0] for _k, _sr_ode in zip(k, self.sr_ode)]           
                # obtained a list of losses, for each event type
    
                # In generation mode we will return a cumulative density curve which can be used to generate sequences of events.
                surv = {"k": k,
                        "tte_deltas": tte_deltas,
                        }
    
                # TODO: NEEDED HERE?
                # Mask and sum across sequence (so log likelihood factorises as a product along the sequence)
                #  As we do not filter to ensure that sequences have at least two points, we also add a small positive constant to 
                #  the denominator to avoid division by zero for sequences with only one event, and so no observed transitions as
                #  in those case the numerator is zero due to all transitions being masked.
                # tte_ll_per_patient = (log_prob * tte_obs_mask.float()).sum(-1) / (tte_obs_mask.float().sum(-1) + 1e-5)  # shape: torch.Size([bsz])

            else:
                surv_losses = None
                surv = {"k": None,
                        "tte_deltas": None,
                       }
            
            # In generation mode we will return a cumulative density curve which can be used to generate sequences of events.
            surv ={**surv, 
                   "surv_CDF":  self._predict_cdf(in_hidden_state.reshape((-1,in_hidden_state.shape[-1]))) if return_cdf else None}
        
        
        else:        
            # inference-time mini-optimization: only forward the head on the very last position
            in_hidden_state = hidden_states[:, -1, :]

            if return_loss:

                assert target_tokens is not None
                assert target_ages is not None
                assert attention_mask is not None
                
                # Forward the last (non-padded?) state. This will be used for fine-tuning a clinical prediction model, 
                # but another use case for is_causal = False is that we are simply generating future trajectories. 
                # In this case we want to just forward the last hidden state, irrespective of any potential padding
                raise NotImplementedError

            else:
                surv_losses = None
                surv = {"k": None,
                        "tte_deltas": None,
                       }

            # In generation mode we will return a cumulative density curve which can be used to generate sequences of events.
            surv ={**surv, 
                   "surv_CDF":  self._predict_cdf(in_hidden_state) if return_cdf else None}
            
        return surv, surv_losses

    def _predict_cdf(self,
                    hidden_states: torch.tensor,                    # shape: torch.Size([*, n_embd])
                   ):
        """
        Predict survival curves from the hidden states
        """
        # The normalised grid over which to predict
        t_test = torch.tensor(np.concatenate([self.t_eval] * hidden_states.shape[0], 0), dtype=torch.float32, device=self.device)
        t_test /= self._time_scale
        H_test = hidden_states.repeat_interleave(self.t_eval.size, 0).to(self.device, torch.float32)

        # Batched predict: Cannot make all predictions at once due to memory constraints
        preds = []
        for _k, _sr_ode in enumerate(self.sr_ode):
            pred_bsz = 256                                                        # Predict in batches of pred_bsz
            pred = []
            for H_test_batched, t_test_batched in zip(torch.split(H_test, pred_bsz), torch.split(t_test, pred_bsz)):
                pred.append(_sr_ode(H_test_batched, t_test_batched))
            pred = torch.concat(pred)
            preds.append(pred.reshape((hidden_states.shape[0], self.t_eval.size)).cpu().detach().numpy())
            
        return preds
        

    def sample_surv(self, surv):

        assert surv[0].shape[0] == 1, "TODO: not implemented for batches"

        # Sample which event occurs next by sampling with probability proportional to the AUC
        AUCs = [np.sum(_s[0, :]) for _s in surv]          
        weights = torch.tensor(AUCs, dtype=torch.float)
        next_index = torch.multinomial(weights, 1) 
        logging.debug(f"Sampled token {next_index + 1} using area under curve")

        # And then sample at what time this event occurs
        try:
            rsample = np.random.uniform(0, surv[next_index][0,-1])                    # Randomly sample between 0 and the maximum cumulative prob
        except:
            print(next_index)
            raise NotImplementedError
        logging.debug(f"competing-risk generation inverse tranform random sample: {rsample}~U(0,{surv[next_index][0,-1]})")
        time_index = np.sum(surv[next_index] <= rsample) - 1
        delta_age = self.t_eval[time_index]

        next_token_index = next_index.reshape(-1, 1).to(self.device) + 1   # add one as the survival curves do not include the PAD token, which has token index 0
        
        return next_token_index, delta_age
