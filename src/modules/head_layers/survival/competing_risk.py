import numpy as np
import torch
import torch.nn as nn
import logging
from CPRD.src.modules.head_layers.survival.desurv import ODESurvMultiple, ODESurvMultipleWithZeroTime

from typing import Optional

class ODESurvCompetingRiskLayer(nn.Module):
    """ Wrapper around competing risk version of DeSurv 
    """

    def __init__(self, in_dim, hidden_dim, num_risks, n=15, concurrent_strategy=None, device="cpu"):
        
        super().__init__()

        self.concurrent_strategy = concurrent_strategy
        match self.concurrent_strategy:
            case "zero_time":
                logging.info(f"Using DeSurv model with non-zero instant risk.")
                self.sr_ode = ODESurvMultipleWithZeroTime(cov_dim=in_dim,
                                                          hidden_dim=hidden_dim,
                                                          num_risks=num_risks,        # do not include pad token as an event 
                                                          device=device,
                                                          n=n) 
            case _: 
                logging.info(f"Using normal DeSurv model.")
                self.sr_ode = ODESurvMultiple(cov_dim=in_dim,
                                              hidden_dim=hidden_dim,
                                              num_risks=num_risks,        # do not include pad token as an event 
                                              device=device,
                                              n=n)                 
                                                                                                                       
        # the time grid which we generate over - assuming time scales are standardised
        self.t_eval = np.linspace(0, 1, 1000)    
        self.device = device

        logging.info(f"Using Competing-Risk DeSurv head.")
        logging.info(f"Evaluating DeSurv-CR on the grid between [{self.t_eval.min()}, {self.t_eval.max()}] with {len(self.t_eval)} intervals")
            

    def predict(self,
                hidden_states: torch.tensor,                    # shape: torch.Size([bsz, seq_len, n_embd])
                target_tokens: Optional[torch.tensor] = None,   # if is_generation==False: torch.Size([bsz, seq_len]), else torch.Size([bsz, 1])
                target_ages: Optional[torch.tensor] = None,     # if is_generation==False: torch.Size([bsz, seq_len]), else torch.Size([bsz, 1])
                attention_mask: Optional[torch.tensor] = None,  # if is_generation == False: torch.Size([bsz, seq_len]), else None
                is_generation: bool = False,                    # Whether we forward every step (True) of seq_len, or just the final step (False)
                return_cdf: bool = False,
                return_loss: bool = True,
                ):
        r"""

        Competing-risk for each type of event, where tte_deltas are the times to each next
         event. Censored events (such as GP visit with no diagnosis/measurement/test. I.e. k=0 (but not padding) are
         not in the currently considered dataset.
        """

        if not is_generation:

            assert target_tokens is not None
            assert target_ages is not None
            assert attention_mask is not None
        
            # Get the competing risk event types. A list of len vocab_size-1 where each element of the list is an event
            #       The 1st element of list corresponds to 2nd vocab element (vocab index == 0 is the PAD token which is excluded)
            #       k \in {0,1} with 1 if the seq target is the same as the single risk ode's index (position in list), and 0
            #       otherwise
            k = target_tokens[:, 1:]                                                                # torch.Size([bsz, seq_len - 1])
            
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
            k = k.flatten()[tte_obs_mask == 1]

            if self.concurrent_strategy == "add_noise":
                exp_dist = torch.distributions.exponential.Exponential(1000)
                tte_deltas[tte_deltas == 0] += exp_dist.sample(tte_deltas[tte_deltas == 0].shape).to(tte_deltas.device)

            if return_loss:
                # Calculate losses, excluding masked values. Each sr_ode returns the sum over observed events
                #    to be consistent with other heads, we scale by number of observed values to obtain per SR-model mean
                #    and we sum across the mixture of survival ODEs
                surv_loss = [self.sr_ode.loss(in_hidden_state, tte_deltas, k) / k.shape[0]]
            else:
                surv_loss = None

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
            in_hidden_state = hidden_states[:, -1, :]                      # torch.Size([bsz, hid_dim])
            
            if return_loss:
                # Forward the last state. This will be used for few-shot training a clinical prediction model.
                # Note: Padding doesn't matter as all the padded hidden_state values share the same value as the last observation's hidden state
                assert target_tokens is not None
                assert target_ages is not None
                assert attention_mask is None

                surv_loss = [self.sr_ode.loss(in_hidden_state, target_ages.reshape(-1), target_tokens.reshape(-1)) / target_tokens.shape[0]]
                
            else:
                # Another use case for is_generation = True is that we are simply generating future trajectories. 
                # In this case we do not have targets, and do not need to calculate the loss
                surv_loss = None

            # In generation mode we will return a cumulative density curve which can be used to generate sequences of events.
            if return_cdf:
                preds, pis = self._predict_cdf(in_hidden_state)
            else:
                preds, pis = None, None
            surv ={"k": target_tokens,
                   "tte_deltas": target_ages, 
                   "surv_CDF":  preds,
                   "surv_pi": pis}
                
        return surv, surv_loss

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
        pred_bsz = 512                                                        # Predict in batches
        pred = []
        pi = []
        for H_test_batched, t_test_batched in zip(torch.split(H_test, pred_bsz), torch.split(t_test, pred_bsz)):
            _pred, _pi = self.sr_ode(H_test_batched, t_test_batched)
            pred.append(_pred)
            pi.append(_pi)

        pred = torch.concat(pred)
        pi = torch.concat(pi)
        pred = pred.reshape((hidden_states.shape[0], self.t_eval.size, -1)).cpu().detach().numpy()
        pi = pi.reshape((hidden_states.shape[0], self.t_eval.size, -1)).cpu().detach().numpy()
        preds = [pred[:, :, _i] for _i in range(pred.shape[-1])]
        pis = [pi[:, :, _i] for _i in range(pi.shape[-1])]

        return preds, pis

    def sample_surv(self, surv):
        """ Generate samples from survival curves using inverse sampling
        """
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
            
        logging.debug(f"competing-risk generation inverse transform random sample: {rsample}~U(0,{surv[next_index][0,-1]})")
        time_index = np.sum(surv[next_index] <= rsample) - 1

        # import matplotlib.pyplot as plt
        # plt.plot(self.t_eval[:50], surv[next_index][0,:50]); 
        # plt.savefig("/rds/homes/g/gaddcz/Projects/CPRD/examples/modelling/SurvivEHR/notebooks/CompetingRisk/0_pretraining/fig_test_generation_curves.png")
        
        delta_age = self.t_eval[time_index]

        next_token_index = next_index.reshape(-1, 1).to(self.device) + 1   # add one as the survival curves do not include the PAD token, which has token index 0
        
        return next_token_index, delta_age
