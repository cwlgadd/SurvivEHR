# Create custom callbacks for our pytorch-lightning model

import numpy as np
from pytorch_lightning import Callback
import torch
from sklearn.manifold import TSNE
import umap
import wandb
import matplotlib.pyplot as plt
from CPRD.src.models.base_callback import BaseCallback
# from CPRD.data.foundational_loader import convert_batch_to_none_causal
from pycox.evaluation import EvalSurv
import pandas as pd
import seaborn as sns
import logging
import copy
        

class PerformanceMetrics(Callback):
    """
    Record metrics for a causal survival model.
    """

    def __init__(self, 
                 ordered_prevalence,
                 log_concordance=True, 
                 ):
        """
        ARGS: ordered_prevalence:       The list of k, ordered by their frequency/prevalence in the dataset.
        """
        
        Callback.__init__(self)
        self.ordered_prevalence = ordered_prevalence
        self.log_concordance = log_concordance
        logging.info(f"Created Performance metric callback for causal self-supervised tasks.")

    def get_metrics(self, 
                    all_cdf, 
                    observed_k,
                    _trainer, 
                    _pl_module, 
                    log_name, 
                    suppress_warnings=False, 
                    k_ahead=None):

        if k_ahead is None:
            suffix = ""
        else:
            suffix = "+" + str(k_ahead)
            
        metric_dict = {}
        try:
            # Calculate which time index to take the CIF at - the evaluation index which we use to rank predictions
            #    Note: if observed_target_age is greater than _pl_module.model.surv_layer.t_eval.max(), then this will take the last value
            #          and may be inaccurate. I.e. if observed_target_age is 4,but we evaluate only on 0 to 1, then this will take 1
            # eval_index = sum(observed_target_age.cpu().numpy() >= _pl_module.model.surv_layer.t_eval) - 1
            # risk_scores = [_cdf[eval_index] for _cdf in all_cdf]

            # Get the risk for each event type k evaluated at eval_index
            risk_scores = [sum(_cdf) for _cdf in all_cdf]

            # Rank by increasing risk, and calculate normalised concordance based on position in ranked risk.
            risk_scores = np.argsort(risk_scores) + 1
            event_concordance = np.where(risk_scores == observed_k.cpu().numpy())[0][0] / (len(risk_scores) - 1)
    
            # Log the score in a way which aggregates across all different outcome types. This will artificially inflate the concordance. 
            #      For example, if we always predict the most prevalent token then the global picture produced here will look artificially good.
            metric_dict = {**metric_dict, log_name+"Cinter" + suffix: event_concordance}
            # Log the score dependent upon what the outcome was. This will help identify if concordance is artificially inflated by high prevalence
            #      For example, a rare event will very likely always have low relative risk vs. a highly prevalent event.
            if k_ahead is None:
                # Only calculate the event specific scores if we are looking at next-event prediction, otherwise we will log too many values
                metric_dict = {**metric_dict, log_name+f"Cintra{observed_k}" + suffix: event_concordance}

            self.log_dict(metric_dict)
        except:
            if suppress_warnings is False:
                logging.warning("Unable to calculate causal metrics, this sample will be skipped - this will bias metrics.")
            raise NotImplementedError

    def get_prevelance_based_metrics(self, 
                                     observed_k,
                                     _trainer, 
                                     _pl_module, 
                                     log_name, 
                                     suppress_warnings=False, 
                                     k_ahead=None):

        if k_ahead is None:
            suffix = ""
        else:
            suffix = "+" + str(k_ahead)

        metric_dict = {}
        try:
            # The risk for each event type k is simply their prevalence
            risk_scores = self.ordered_prevalence
            
            # Calculate normalised concordance based on position in ranked risk.
            # event_concordance = np.where(risk_scores == observed_k.cpu().numpy())[0][0] / (len(risk_scores) - 1)
            risk_scores = np.argsort(risk_scores) + 1
            event_concordance = np.where(risk_scores == observed_k.cpu().numpy())[0][0] / (len(risk_scores) - 1)
    
            # Log the score in a way which aggregates across all different outcome types
            metric_dict = {**metric_dict, log_name+"base_Cinter" + suffix: event_concordance}

            if k_ahead is None:
                # Only calculate the event specific scores if we are looking at next-event prediction, otherwise we will log too many values
                metric_dict = {**metric_dict, log_name+f"base_Cintra{observed_k}" + suffix: event_concordance}

            self.log_dict(metric_dict)
        except:
            if suppress_warnings is False:
                logging.warning("Unable to calculate baseline causal metrics, this sample will be skipped - this will bias metrics.")
            raise NotImplementedError
                
    def run_callback(self,
                     _trainer,
                     _pl_module,
                     batch,
                     log_name:               str='CausalMetrics',
                     plot_outcome_curves:    bool= False,
                     suppress_warnings:      bool=False
                    ):
        """

        Take only one patient per batch (and look at every transition), whilst preserving dimension.
         -This is to reduce computational overhead of looking at every transition across every validation/test patient,
          whilst not introducing bias towards patients with shorter, or longer context lengths.
         -This is still a deterministic reduction as our validation and test sets are not shuffled.

         
            Note: if removing this reduction, you will face memory issues in trying to forward a generation curve for every
                  transition of every patient in the batch - this would need to be replaced by a loop over patients in batch.
                  This will be more accurate, but also significantly increase the computational demands of an already
                  computationally demanding callback.

        """
        # Take only the first patient in the batch
        for key in batch.keys():
            batch[key] = batch[key][[0]]

        # Get the number of transitions within the context window for this patient
        number_of_transitions = torch.sum(batch["tokens"] != 0) - 1

        # and if there are any to evaluate, then continue
        if number_of_transitions > 0:
            
            # Push through the model in a causal fashion, whilst returning the generation curves
            all_outputs, _, _ = _pl_module(batch, is_generation=False, return_loss=False, return_generation=True)

            # Unpack outputs
            pred_surv_CDFs = all_outputs["surv"]["surv_CDF"]                       # [(K,1000) for _ in range(vocab_size)]
            k = all_outputs["surv"]["k"]                                           # [torch.Size([K])]
            
            # print(f"number_of_transitions {number_of_transitions}, {len(pred_surv_CDFs)}, pred_surv_CDFs shape {[_k.shape for _k in pred_surv_CDFs]}")         
            # print(f"number_of_transitions {number_of_transitions}, k shape {[_k.shape for _k in k]}")

            # For each observed outcome in the context window (so for a context of A,B,C,D, for each of B,C and D)
            for transition in range(number_of_transitions):

                # for each of the vocab_size outcomes, get the survival curve specific to this time in the context length
                _outcome_cdfs = [_outcome_cdf[transition, :] for _outcome_cdf in pred_surv_CDFs]         # [(1,1000) for _ in range(vocab_size)]
                _observed_outcome_k = k[0][transition]                                                   # integer index

                # Get the SurvivEHR concordance metric for each transition
                self.get_metrics(_outcome_cdfs,
                                 _observed_outcome_k,
                                 _trainer,
                                 _pl_module, 
                                 log_name=log_name,
                                 suppress_warnings=suppress_warnings
                                )

                # Get the baseline prevalence based concordance metric for each transition
                self.get_prevelance_based_metrics(_observed_outcome_k,
                                                  _trainer,
                                                  _pl_module,
                                                  log_name=log_name,
                                                  suppress_warnings=suppress_warnings
                                                 )

            # Repeat the above, but looking further ahead than next event prediction
            max_look_ahead = np.min((20, number_of_transitions.cpu()))
            look_ahead_steps = [0,1,2,3] + [_i for _i in range(4, max_look_ahead, 3)]
            look_ahead_steps = [_i for _i in look_ahead_steps if _i <= max_look_ahead]
            # For each step looking ahead possible in the context window, (so for a context A in above, for each {A->B,...}, {A->C....}, {A->D,...} etc)
            for look_ahead_by in look_ahead_steps:

                # Choose how we want to sample along the context of the sample 
                # all_transitions = range(1, number_of_transitions - look_ahead_by)            # For each `look_ahead_by' transition possible (for each {A->B,...}, {A,B->C,...}, {A,B,C->D,...} etc)
                last_transition = [number_of_transitions - look_ahead_by - 1]                # For the last transition for which we are able to look `look_ahead_by` steps ahead

                for context in last_transition:

                    # for each of the vocab_size outcomes, get the survival curve specific to this time in the context length
                    _outcome_cdfs = [_outcome_cdf[context, :] for _outcome_cdf in pred_surv_CDFs]         # [(1,1000) for _ in range(vocab_size)]
                    _observed_outcome_kahead = k[0][context + look_ahead_by]                                                   # integer index
    
                    # Get the SurvivEHR concordance metric for each transition
                    self.get_metrics(_outcome_cdfs,
                                     _observed_outcome_kahead,
                                     _trainer,
                                     _pl_module, 
                                     log_name=log_name,
                                     suppress_warnings=suppress_warnings,
                                     k_ahead=look_ahead_by
                                    )
    
                    # Get the baseline prevalence based concordance metric for each transition
                    self.get_prevelance_based_metrics(_observed_outcome_kahead,
                                                      _trainer,
                                                      _pl_module,
                                                      log_name=log_name,
                                                      suppress_warnings=suppress_warnings,
                                                      k_ahead=look_ahead_by
                                                     )

           
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Run callback
        self.run_callback(_trainer=trainer, 
                          _pl_module = pl_module,
                          batch=batch,
                          log_name = "Test:", 
                          )
