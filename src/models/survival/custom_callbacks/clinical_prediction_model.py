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
    Record metrics for survival model.
    """

    def __init__(self, 
                 outcome_token_to_desurv_output_index,
                 log_individual=True,
                 log_combined=False,
                 log_ctd=True, 
                 log_ibs=True, 
                 log_inbll=True,
                ):
        r"""

        ARGS:
            outcome_token_to_desurv_output_index:
                              A dictionary hash map where keys are the outcome tokens of interest, and the values are the index of the DeSurv 
                                output index they belong to. 
                              In the pre-training (self-supervised/causal) and few-shot (supervised with unmodified architecture) cases,
                                the map is from outcome to one of `vocab_size` indices, as the DeSurv heads (in both SR and CR) predict
                                the risk of every token.
                              In the fine-tuning (supervised) case, the DeSurv head is replaced with one that only predicts the relevant 
                                outcomes. In the Single-Risk case, this gives a many-to-1 map, combining all the outcomes into a single-risk 
                                and so every token included in this umbrella maps to the zero-index (as there is only one predicted survival 
                                curve). In the Competing-Risk case, each will map 1-to-1 to the K considered competing-risk outcome indicies.
                                
        KWARGS:
            log_individual:   Whether or not to log for the individual CIFs/CDFs. 
                                For few-shot single-risk, this is the only appropriate choice. For few-shot competing-risk this may highlight which 
                                outcomes the model is better able to predict.
                                For fine-tune single risk, there is only one outcome CIF/CDF and so this is not appropriate.
                              Note:  
                                If there are low prevalence outcome tokens then there are likeliy to be batches with no individuals  
                                experiencing that outcome, leading to warnings. In this case, the batch's metric for that individual 
                                CIF will not be logged which may bias the aggregated reported values.
            log_combined:     Whether or not to log the marginal CIFs/CDFs. 
                                This is only an appropriate choice for the Competing Risk case. In the SR setting, this will raise an
                                exception.
                                
        """
        
        Callback.__init__(self)
        self.outcome_token_to_desurv_output_index = outcome_token_to_desurv_output_index
        self.outcome_tokens = outcome_token_to_desurv_output_index.keys()
        self.log_individual = log_individual
        self.log_combined = log_combined
        self.log_ctd = log_ctd
        self.log_ibs = log_ibs
        self.log_inbll = log_inbll
        logging.info(f"Created Performance metric callback. Calculating metrics for {self.outcome_tokens} with map {self.outcome_token_to_desurv_output_index}")

    def plot_outcome_curve(self, cdf, lbls, _trainer, log_name="outcome_split_curve", ylabel=None):
        
        plt.close()
        cdf_unce = cdf[lbls==1, :]
        cdf_cens = cdf[lbls==0, :]
        
        wandb_images = []
        fig, ax = plt.subplots(1, 1)
        for i in range(cdf_unce.shape[0]):
            plt.plot(np.linspace(0,1,1000), cdf_unce[i,:], c="r", label="event" if i == 0 else None, alpha=1)
        for i in range(cdf_cens.shape[0]):
            plt.plot(np.linspace(0,1,1000), cdf_cens[i,:], c="k", label="censored" if i == 0 else None, alpha=0.1)
        
        plt.legend(loc=2)
        plt.xlabel("t (scaled time)")
        if ylabel is not None:
            plt.ylabel(ylabel)

        _trainer.logger.experiment.log({
                log_name + "-Survival": wandb.Image(fig)
            })

    def get_metrics(self, cdf, lbls, target_ages, _trainer, _pl_module, log_name, suppress_warnings=False):
        
        if np.sum(lbls) == 0 and suppress_warnings is False:
            logging.warning(f"Only censored events in batch. Evaluating metrics will be unstable.")

        metric_dict = {}
        try:
            # Evaluate concordance. Scale using the head layers internal scaling.
            surv = pd.DataFrame(np.transpose((1 - cdf)), index=_pl_module.model.surv_layer.t_eval)
            ev = EvalSurv(surv, target_ages, lbls, censor_surv='km')
    
            # Calculate and log desired metrics
            time_grid = np.linspace(start=0, stop=_pl_module.model.surv_layer.t_eval.max() , num=300)
            if self.log_ctd:
                ctd = ev.concordance_td()                           # Time-dependent Concordance Index
                metric_dict = {**metric_dict, log_name+"ctd": ctd}
            if self.log_ibs:
                ibs = ev.integrated_brier_score(time_grid)          # Integrated Brier Score
                metric_dict = {**metric_dict, log_name+"ibs": ibs}
            if self.log_inbll:
                inbll = ev.integrated_nbll(time_grid)               # Integrated Negative Binomial LogLikelihood
                metric_dict = {**metric_dict, log_name+"inbll": inbll}

            self.log_dict(metric_dict)
        except:
            if suppress_warnings is False:
                logging.warning("Unable to calculate metrics, this batch will be skipped - this will bias metrics.")
                logging.warning(f"{lbls}, {target_ages}")
            else:
                pass
        
    def run_callback(self,
                     _trainer,
                     _pl_module,
                     batch,
                     log_name:               str='Metrics',
                     plot_outcome_curves:    bool=False,
                    ):

        # Make prediction of each survival curve
        all_outputs, _, _ = _pl_module(batch, return_loss=False, return_generation=True)
        pred_surv_CDFs = all_outputs["surv"]["surv_CDF"]
        pred_surv_pis = all_outputs["surv"]["surv_pi"]
        
        target_tokens = batch['target_token'].cpu().numpy()
        target_ages = batch['target_age_delta'].cpu().numpy()

        # Log records for individual outcomes
        ######################################
        if self.log_combined:
            
            # Combine for labels, 1 if any of the outcomes, 0 otherwise
            cdf = np.zeros_like(pred_surv_CDFs[0])
            lbls = np.zeros_like(target_tokens)
            surv_indices_included = []               
            for _outcome_token in self.outcome_tokens:
                # print(f"Adding contributions for outcome token {_outcome_token}")
                _outcome_labels = (target_tokens == _outcome_token)
                lbls += _outcome_labels
                
                _outcome_surv_index = self.outcome_token_to_desurv_output_index[_outcome_token]
                _outcome_cdf = pred_surv_CDFs[_outcome_surv_index]
                # print(f"These can be found in the DeSurv prediction index  {_outcome_surv_index}")
    
                # When different outcomes map to the same CIF/CDF curve, then we do not duplicate
                #    This is only relevant in the supervised case, as the few-shot case will always have 1-to-1 _outcome_token to 
                #    _outcome_surv_index maps
                if _outcome_surv_index not in surv_indices_included:
                    cdf += _outcome_cdf
                    surv_indices_included.append(_outcome_surv_index)

            # Plot the outcome curves
            if plot_outcome_curves:
                ylabel = r"$\sum_{k\in{" + f"{','.join([str(i) for i in self.outcome_tokens])}" + r"}} F_k(t)$"
                self.plot_outcome_curve(cdf, lbls, _trainer, log_name=log_name, ylabel=ylabel)
            # Get metrics
            self.get_metrics(cdf, lbls, target_ages, _trainer, _pl_module, log_name=log_name)
            
        # Log records for individual outcomes
        ######################################
        #  Note: unless this is single risk causal model, then this may not be appropriate beyond diagnostics.
        # For each of the DeSurv output curves, which we want to evaluate separately
        #   1) collect the targets which are relevant to that curve. 
        if self.log_individual:
            assert pred_surv_pis is None, "Individual logging is not supported for Competing-Risk models."
            
            for _desurv_index, _outcome_cdf in enumerate(pred_surv_CDFs):
                
                _outcome_labels = np.zeros_like(target_tokens)
                _outcome_tokens = []
                for _outcome_token in self.outcome_token_to_desurv_output_index.keys():
                    # For each outcome, if it belongs to the `_desurv_index` survival curve then update target labels
                    _outcome_desurv_index = self.outcome_token_to_desurv_output_index[_outcome_token]
                    if _outcome_desurv_index == _desurv_index:
                        _outcome_labels += (target_tokens == _outcome_token)             # 1 if == _outcome_token else 0
                        _outcome_tokens.append(_outcome_token)
    
                # Plot the outcome curve
                if plot_outcome_curves:
                    self.plot_outcome_curve(_outcome_cdf, 
                                            _outcome_labels, 
                                            _trainer, 
                                            log_name=log_name+f"_{_outcome_tokens}",
                                            ylabel=r"$F_{" + f"{','.join([str(i) for i in _outcome_tokens])}" + r"}(t)$")
                # Log metrics
                self.get_metrics(_outcome_cdf, #  if pred_surv_pis is None else _outcome_cdf / pred_surv_pis[_desurv_index], 
                                 _outcome_labels,
                                 target_ages, 
                                 _trainer, 
                                 _pl_module, 
                                 log_name+f"_{_outcome_tokens}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Run callback
        self.run_callback(_trainer=trainer, 
                          _pl_module = pl_module,
                          batch=batch,
                          log_name = "Val:OutcomePerformanceMetrics", 
                          plot_outcome_curves = False
                          )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Run callback
        self.run_callback(_trainer=trainer, 
                          _pl_module = pl_module,
                          batch=batch,
                          log_name = "Test:OutcomePerformanceMetrics", 
                          plot_outcome_curves = True
                          )

class PerformanceValueMetrics(Callback):
    """
    Record metrics for survival model.
    """

    def __init__(self, 
                 outcome_token_to_desurv_output_index,
                ):
        r"""

        ARGS:
            outcome_token_to_desurv_output_index:
                              A dictionary hash map where keys are the outcome tokens of interest, and the values are the index of the DeSurv 
                                output index they belong to. 
                              In the pre-training (self-supervised/causal) and few-shot (supervised with unmodified architecture) cases,
                                the map is from outcome to one of `vocab_size` indices, as the DeSurv heads (in both SR and CR) predict
                                the risk of every token.
                              In the fine-tuning (supervised) case, the DeSurv head is replaced with one that only predicts the relevant 
                                outcomes. In the Single-Risk case, this gives a many-to-1 map, combining all the outcomes into a single-risk 
                                and so every token included in this umbrella maps to the zero-index (as there is only one predicted survival 
                                curve). In the Competing-Risk case, each will map 1-to-1 to the K considered competing-risk outcome indicies.
                                
        """
        
        Callback.__init__(self)
        self.outcome_token_to_desurv_output_index = outcome_token_to_desurv_output_index
        self.outcome_tokens = outcome_token_to_desurv_output_index.keys()
        logging.info(f"Created Value Performance metric callback. Calculating metrics for {self.outcome_tokens} with map {self.outcome_token_to_desurv_output_index}")


    def get_metrics(self, _trainer, _pl_module, log_name, suppress_warnings=False):
        
        metric_dict = {}
        try:
            # Evaluate metrics

            metric_dict = {**metric_dict, log_name+"MSE": 0}
            self.log_dict(metric_dict)
            
        except:
            if suppress_warnings is False:
                logging.warning("Unable to calculate metrics, this batch will be skipped - this will bias metrics.")
            else:
                pass
        
    def run_callback(self, _trainer, _pl_module, batch,
                     log_name:               str='Metrics',
                    ):

        # Make prediction of each survival curve
        all_outputs, _, _ = _pl_module(batch, return_loss=False, return_generation=True)
        value_distributions = all_outputs["values_dist"]
        print(value_distributions)
        
        target_tokens = batch['target_token'].cpu().numpy()
        target_ages = batch['target_age_delta'].cpu().numpy()

        # Log records for individual outcome values
        ######################################
        self.get_metrics(_trainer, _pl_module, log_name=log_name)
            

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Run callback
        self.run_callback(_trainer=trainer, 
                          _pl_module = pl_module,
                          batch=batch,
                          log_name = "Val:OutcomeValuePM", 
                          )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Run callback
        self.run_callback(_trainer=trainer, 
                          _pl_module = pl_module,
                          batch=batch,
                          log_name = "Test:OutcomeValuePM", 
                          )