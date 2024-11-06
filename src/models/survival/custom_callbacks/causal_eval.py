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
                 log_ctd=True, 
                 log_ibs=True, 
                 log_inbll=True,
                 plot_outcome_curves=False,
                 ):
        
        Callback.__init__(self)
        self.log_ctd = log_ctd
        self.log_ibs = log_ibs
        self.log_inbll = log_inbll
        self.plot_outcome_curves = plot_outcome_curves

        logging.warning("This class produces biased metrics if self-supervised targets in batch do not have the outcome. " + \
                        "Evaluating metrics with no uncensored values is unstable. In these mini-batches, " \
                        "we set the metrics to 0.5 (ctd) and 1 (inbll, ibs) - this will bias the metrics heavily")
    

    def plot_outcome_curve(self, cdf, lbls, _trainer):
        
        plt.close()
        cdf_true = cdf[lbls==1,:]
        cdf_false = cdf[lbls==0,:]
        
        wandb_images = []
        fig, ax = plt.subplots(1, 1)
        for i in range(cdf_true.shape[0]):
            plt.plot(np.linspace(1,1826,1826), cdf_true[i,:], c="r", label="event" if i == 0 else None, alpha=1)
        for i in range(cdf_false.shape[0]):
            plt.plot(np.linspace(1,1826,1826), cdf_false[i,:], c="k", label="censored" if i == 0 else None, alpha=0.1)
        
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

        # Take only one patient per batch (and look at every transition), whilst preserving dimension.
        for key in batch.keys():
            batch[key] = batch[key][[0], :]
        
        all_outputs, _, _ = _pl_module(batch, is_generation=False, return_loss=False, return_generation=True)
        pred_surv_CDFs = all_outputs["surv"]["surv_CDF"]
        k = all_outputs["surv"]["k"]
        tte_deltas = all_outputs["surv"]["tte_deltas"]

        # If this is being used for a competing risk model, convert targets as required
        if len(k) == 1:
            k = [torch.where(k[0] == event + 1, 1, 0) for event, _ in enumerate(range(len(pred_surv_CDFs)))]
            
        # Merge (additively) each outcome risk curve into a single CDF, and make a label for if one of the outcomes occurred or not
        metric_dict = {}
        for idx_token in range(len(pred_surv_CDFs)):
            num_uncensored_events = torch.sum(k[idx_token])
            if num_uncensored_events > 0:
                cdf = pred_surv_CDFs[idx_token]

                # Evaluate concordance. Scale using the head layers internal scaling.
                surv = pd.DataFrame(np.transpose((1 - cdf)), index=_pl_module.model.surv_layer.t_eval)
                ev = EvalSurv(surv, tte_deltas.cpu().numpy(), k[idx_token].cpu().numpy(), censor_surv='km')
             
                time_grid = np.linspace(start=0, stop=_pl_module.model.surv_layer._time_scale , num=300)
        
                # Calculate and log desired metrics
                if self.log_ctd:
                    ctd = ev.concordance_td()                           # Time-dependent Concordance Index
                    metric_dict = {**metric_dict, f"{log_name}_{idx_token}_ctd": ctd}
                if self.log_ibs:
                    ibs = ev.integrated_brier_score(time_grid)          # Integrated Brier Score
                    metric_dict = {**metric_dict, f"{log_name}_{idx_token}_ibs": ibs}
                if self.log_inbll:
                    inbll = ev.integrated_nbll(time_grid)               # Integrated Negative Binomial LogLikelihood
                    metric_dict = {**metric_dict, f"{log_name}_{idx_token}_inbll": inbll}
                # if self.mae:
                #     raise NotImplementedError
                # if self.rmse:
                #     raise NotImplementedError
   
            else:
                # Calculate and log desired metrics
                if self.log_ctd:
                    metric_dict = {**metric_dict, f"{log_name}_{idx_token}_ctd": 0.5}
                if self.log_ibs:
                    metric_dict = {**metric_dict, f"{log_name}_{idx_token}_ibs": 1.0}
                if self.log_inbll:
                    metric_dict = {**metric_dict, f"{log_name}_{idx_token}_inbll": 1.0}
                

        self.log_dict(metric_dict)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Run callback
        self.run_callback(_trainer=trainer, 
                          _pl_module = pl_module,
                          batch=batch,
                          log_name = "Val:CausalPerformanceMetrics", 
                          )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Run callback
        self.run_callback(_trainer=trainer, 
                          _pl_module = pl_module,
                          batch=batch,
                          log_name = "Test:CausalPerformanceMetrics", 
                          )
