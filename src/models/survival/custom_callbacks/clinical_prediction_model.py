# Create custom callbacks for our pytorch-lightning model

import numpy as np
from pytorch_lightning import Callback
import torch
from sklearn.manifold import TSNE
import umap
import wandb
import matplotlib.pyplot as plt
from CPRD.src.models.base_callback import BaseCallback
from pycox.evaluation import EvalSurv
import pandas as pd
import seaborn as sns
import logging
import copy

# from pysurvival.pysurvival.utils._metrics import _concordance_index, _brier_score

# def evaluate_concordance_index(risk, t, k, include_ties=True, additional_results=False, **kwargs):
# # Ordering risk, T and E in descending order according to T
#     order = np.argsort(-t)
#     risk = risk[order]
#     t = t[order]
#     k = k[order]

#     # Calculating th c-index
#     results = _concordance_index(risk, t, k, include_ties)

#     if not additional_results:
#         return results[0]
#     else:
#         return results

class Template(Callback, BaseCallback):

    def run_callback(self,
                     _trainer,
                     _pl_module,
                     batch, 
                     log_name,
                    ):
        # Push features through the model. The forward method of the model has two modes decided by is_generation. 
        # ... When true we forward only the last step (which saves compute for appliation of the model), whilst 
        # ... when false we forward all, but only compute the loss.
        # ... We need a third option here, as we want to generate, but we want to do this on all steps - i.e., we
        # ... want to evaluate the metrics across all the sequence (just as we do for the loss)

        # ... is_generation = False ensures we forward all hidden states (not just the last)
        # ... whilst return_cdf = True ensures we do calculate the CDF surves, as these are not required to train the model
        
        # do something
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        if self.do_test is True:
            # Run callback
            self.run_callback(_trainer=trainer, 
                              _pl_module = pl_module,
                              batch=self.test_batch,
                              log_name = "Test:Template", 
                              )
        

class PerformanceMetrics(Callback):
    """
    Record metrics for survival model.
    """

    def get_mae_rmse(self):
        # Get Mean Absolute Error and Root Mean Square Error
        raise NotImplementedError

    def __init__(self, 
                 outcome_tokens,
                 log_ctd=True, 
                 log_ibs=True, 
                 log_inbll=True):
        
        Callback.__init__(self)
        self.outcome_tokens = outcome_tokens
        self.log_ctd = log_ctd
        self.log_ibs = log_ibs
        self.log_inbll = log_inbll

    def plot_outcome_curve(self, cdf, lbls, _trainer):
        
        plt.close()
        cdf_true = cdf[lbls==1,:]
        cdf_false = cdf[lbls==0,:]
        
        wandb_images = []
        fig, ax = plt.subplots(1, 1)
        for i in range(cdf_true.shape[0]):
            plt.plot(np.linspace(1,1826,1826), cdf_true[i,:], c="r", label="outcome occurred next" if i == 0 else None, alpha=1)
        for i in range(cdf_false.shape[0]):
            plt.plot(np.linspace(1,1826,1826), cdf_false[i,:], c="k", label="outcome did not occur next" if i == 0 else None, alpha=0.3)
        
        plt.legend(loc=2)
        plt.xlabel("days")
        plt.ylabel(f"P(t>T) - outcome tokens={','.join([str(i) for i in self.outcome_tokens])}")

        _trainer.logger.experiment.log({
                "outcome_split_curve": wandb.Image(fig)
            })
        
    def run_callback(self,
                     _trainer,
                     _pl_module,
                     batch,
                     log_name:               str='Embedding',
                    ):

        #  
        
        target_tokens = batch['target_token']
        target_ages = batch['target_age_delta'].numpy()
        target_values = batch['target_value']

        all_outputs, _, _ = _pl_module(batch, is_causal=False, return_loss=False, return_generation=True)
        pred_surv_CDFs = all_outputs["surv"]["surv_CDF"]

        # Merge (additively) each outcome risk curve into a single CDF, and make a label for if one of the outcomes occurred or not
        cdf = np.zeros_like(pred_surv_CDFs[0])
        lbls = np.zeros_like(target_tokens)     
        for _outcome_token in self.outcome_tokens:
            # print(f"{_outcome_token}: {pred_surv_CDFs[_outcome_token - 1][:4,:]}")
            cdf += pred_surv_CDFs[_outcome_token - 1] 
            lbls += (target_tokens == _outcome_token).long().numpy()

        if np.sum(lbls) == 0:
            logging.warning("No outcome targets in batch. Evaluating metrics will be unstable.")

        # Evaluate concordance. Scale using the head layers internal scaling.
        surv = pd.DataFrame(np.transpose((1 - cdf)), index=_pl_module.model.surv_layer.t_eval)
        ev = EvalSurv(surv, target_ages, lbls, censor_surv='km')

        # Plot the outcome curves
        if True:
            # Internal plot
            self.plot_outcome_curve(cdf, lbls, _trainer)

            # EvSurv's view for debugging
            fig = ev[1:50].plot_surv()
            _trainer.logger.experiment.log({
                    "EvalSurv_curve": wandb.Image(fig)
                })

        time_grid = np.linspace(start=0, stop=_pl_module.model.surv_layer._time_scale , num=300)

        # Calculate and log desired metrics
        metric_dict = {}
        if self.log_ctd:
            ctd = ev.concordance_td()                           # Time-dependent Concordance Index
            metric_dict = {**metric_dict, log_name+"ctd": ctd}
        if self.log_ibs:
            ibs = ev.integrated_brier_score(time_grid)          # Integrated Brier Score
            metric_dict = {**metric_dict, log_name+"ibs": ibs}
        if self.log_inbll:
            inbll = ev.integrated_nbll(time_grid)               # Integrated Negative Binomial LogLikelihood
            metric_dict = {**metric_dict, log_name+"inbll": inbll}
        # if self.mae:
        #     raise NotImplementedError
        # if self.rmse:
        #     raise NotImplementedError
        self.log_dict(metric_dict)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Run callback
        self.run_callback(_trainer=trainer, 
                          _pl_module = pl_module,
                          batch=batch,
                          log_name = "Val:OutcomePerformanceMetrics", 
                          )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Run callback
        self.run_callback(_trainer=trainer, 
                          _pl_module = pl_module,
                          batch=batch,
                          log_name = "Test:OutcomePerformanceMetrics", 
                          )
