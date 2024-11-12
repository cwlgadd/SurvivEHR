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

        # the time grid which we generate over - assuming time scales are standardised
        self.t_eval = np.linspace(0, 1, 1000)
        self.device = device

        logging.info(f"Using Single-Risk DeSurvival head. This module predicts an independent survival curve for each possible future event.")
        logging.info(f"In generation forwarding DeSurv on the grid between [{self.t_eval.min()}, {self.t_eval.max()}] with {len(self.t_eval)} intervals")

    def predict(self,
                hidden_states: torch.tensor,                    # shape: torch.Size([bsz, seq_len, n_embd])
                target_tokens: torch.tensor,                    # if is_generation==False: torch.Size([bsz, seq_len]), else torch.Size([bsz, 1])
                target_ages: torch.tensor,                      # if is_generation==False: torch.Size([bsz, seq_len]), else torch.Size([bsz, 1])
                attention_mask: Optional[torch.tensor] = None,  # if is_generation == False: torch.Size([bsz, seq_len]), else None
                is_generation: bool = False,                    # Whether we forward every step (True) of seq_len, or just the final step (False)
                return_cdf: bool = False,
                return_loss: bool = True,
                ):
        r"""

        A 1vAll single-risk for each type of event, where tte_deltas are the times to each next
         event, whether this occurred or not within this single-risk model. k indicates whether event occurred, where
         1=yes, and 0=another event occurred. 
        """

        if not is_generation:

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

            if return_loss:                
                # Calculate losses, excluding masked values. Each sr_ode returns the sum over observed events
                #    to be consistent with other heads, we scale by number of observed values to obtain per SR-model mean
                #    and we sum across the mixture of survival ODEs
                surv_losses = [_sr_ode.loss(in_hidden_state, tte_deltas, _k) / _k.shape[0] for _k, _sr_ode in zip(k, self.sr_ode)] 
    
            else:
                surv_losses = None

            # In generation mode we will return a cumulative density curve which can be used to generate sequences of events.
            if return_cdf:
                preds, pis = self._predict_cdf(in_hidden_state.reshape((-1,in_hidden_state.shape[-1]))) 
            else:
                preds, pis = None, None
            surv ={"k": [k],
                   "tte_deltas": tte_deltas,
                   "surv_CDF": preds,
                   "surv_pi": pis}
        
        else:        
            # inference-time mini-optimization: only forward the head on the very last position
            in_hidden_state = hidden_states[:, -1, :]

            if return_loss:
                # Forward the last state. This will be used for few-shot training a clinical prediction mo+del.
                # Note: Padding doesn't matter as all the padded hidden_state values share the same value as the last observation's hidden state
                assert target_tokens is not None
                assert target_ages is not None
                
                surv_losses = []
                for _idx_ode, _sr_ode in enumerate(self.sr_ode):

                    k = torch.where(target_tokens == _idx_ode + 1, 1, 0)
                    ode_loss = _sr_ode.loss(in_hidden_state, target_ages.reshape(-1), k.reshape(-1)) / target_tokens.shape[0]
                    surv_losses.append(ode_loss)
                    
                    # if _idx_ode in [129 - 1]:
                    #     print(f"Hypertension supervised target token")
                    #     print(k.shape)
                    #     print(target_ages.shape)
                    #     print(torch.hstack((k, target_tokens, target_ages)))
                    #     assert 1 == 0 

                # k = [torch.where(target_tokens == event + 1, 1, 0) for event, _ in enumerate(self.sr_ode)]
                # surv_losses = [_sr_ode.loss(in_hidden_state, target_ages.reshape(-1), _k.reshape(-1)) / _k.shape[0] for _k, _sr_ode in zip(k, self.sr_ode)] 

            else:
                surv_losses = None

            # In generation mode we will return a cumulative density curve which can be used to generate sequences of events.
            if return_cdf:
                preds, pis = self._predict_cdf(in_hidden_state)
            else:
                preds, pis = None, None
            surv ={"k": target_tokens,
                   "tte_deltas": target_ages, 
                   "surv_CDF":  preds,
                   "surv_pi": pis}
            
        return surv, surv_losses

    def _predict_cdf(self,
                    hidden_states: torch.tensor,                    # shape: torch.Size([*, n_embd])
                   ):
        """
        Predict survival curves from the hidden states
        """

        assert hidden_states.dim() == 2, hidden_states.shape
        
        # The normalised grid over which to predict
        t_test = torch.tensor(np.concatenate([self.t_eval] * hidden_states.shape[0], 0), dtype=torch.float32, device=self.device)
        H_test = hidden_states.repeat_interleave(self.t_eval.size, 0).to(self.device, torch.float32)

        # Batched predict: Cannot make all predictions at once due to memory constraints
        preds = []
        for _sr_ode in self.sr_ode:
            pred_bsz = 512                                                        # Predict in batches
            pred = []
            pi = []
            for H_test_batched, t_test_batched in zip(torch.split(H_test, pred_bsz), torch.split(t_test, pred_bsz)):
                _pred = _sr_ode(H_test_batched, t_test_batched)
                pred.append(_pred)
            pred = torch.concat(pred)
            preds.append(pred.reshape(hidden_states.shape[0], self.t_eval.size).cpu().detach().numpy())

        return preds, None

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
