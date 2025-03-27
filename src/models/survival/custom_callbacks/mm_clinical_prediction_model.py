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
from scipy.integrate import trapz
import pandas as pd
import seaborn as sns
import logging
import copy


class RestrictedMeanSurvivalTime(Callback):
    """
    Record Restricted Mean Survival time under the survival model for each patient. Optionally, stratify these by subgroup.
    """

    def __init__(self, 
                 outcome_token_to_desurv_output_index,
                 custom_stratification_method=None,
                 log_individual=True,
                 log_combined=False,
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
            custom_stratification_method:
                             A function which takes as an argument the batch, and returns a stratification label
                             For example, if we want to stratify by gender then the batch dictionary will be inputted, and the return will be a list 
                             of length equal to the number of samples, of the form ["male", "female", "male",...] etc. and the unique strings will be used
                             to stratify the RMST logging.
            log_individual:   Whether or not to log for the individual outcome CIFs/CDFs. 
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
        self.custom_stratification_method = custom_stratification_method
        self.log_individual = log_individual
        self.log_combined = log_combined
        
        logging.info(f"Created RestrictedMeanSurvivalTime callback. Calculating RMST for {self.outcome_tokens} with map {self.outcome_token_to_desurv_output_index}")
        aggregation_doc_str = " (aggregated by a custom function)" if self.custom_stratification_method is not None else ""
        if self.log_combined:
            logging.info(f"This will log the RMST{aggregation_doc_str} for any of the outcomes in {self.outcome_tokens} (using the weighted sum of individual CIF)")
        if self.log_individual:
            logging.info(f"This will log the RMST{aggregation_doc_str} for each individual outcome in {self.outcome_tokens}")

    def get_rmst(self, cdf, lbls_outcome, target_ages, lbls_stratification, _trainer, _pl_module, log_name, suppress_warnings=False):
        """
        Calculate and Restricted Mean Survival Time under the _pl_module.model

        Note: This method averages over the samples in the stratification and then logs the average. This is repeated across different batches, and the average of those is
                reported. This introduces some extra variability. If one batch has only one sample with "stratification 1", and another has many - the average of each is weighted equally.
                TODO: is there a get around with wandb logging? This should still lead to an unbiased estimator.
        """
        
        metric_dict = {}
        try:
            # Create a dictionary mapping stratification labels to CIF
            unique_labels = np.unique(lbls_stratification)
    
            for strat_label in unique_labels:
                is_label = [lbl == strat_label for lbl in lbls_stratification]
    
                # Convert to the survival function
                sub_surv = 1 - cdf[is_label, :]
                assert sub_surv.shape[-1] == _pl_module.model.surv_layer.t_eval.shape[0]

                # Get batch average RMST for the stratified subgroup. This introduces a batch effect.
                rmst = 0
                for sample in range(sub_surv.shape[0]):
                    _rmst = np.min((1, trapz(sub_surv[sample, :], _pl_module.model.surv_layer.t_eval)))
                    rmst += _rmst / sub_surv.shape[0]

                # Log the average of this stratified group within this batch.
                metric_dict = {**metric_dict, log_name + "_" + strat_label: rmst}
                
            self.log_dict(metric_dict)
    
        except:
            if suppress_warnings is False:
                logging.warning("Unable to calculate RMST, this batch will be skipped - this will bias metrics.")
            else:
                pass

    def get_ost(self, cdf, lbls_outcome, target_ages, lbls_stratification, _trainer, _pl_module, log_name, suppress_warnings=False):
        """
        Calculate and Restricted Mean Survival Time under the _pl_module.model

        Note: This method averages over the samples in the stratification and then logs the average. This is repeated across different batches, and the average of those is
                reported. This introduces some extra variability. If one batch has only one sample with "stratification 1", and another has many - the average of each is weighted equally.
                TODO: is there a get around with wandb logging? This should still lead to an unbiased estimator.
        """
        
        metric_dict = {}
        try:
            # Create a dictionary mapping stratification labels to CIF
            unique_labels = np.unique(lbls_stratification)
    
            for strat_label in unique_labels:
                is_label = [lbl == strat_label for lbl in lbls_stratification]
    
                # Stratify by the labels
                sub_surv = 1 - cdf[is_label, :]
                sub_lbls_outcome = lbls_outcome[is_label]
                sub_target_ages = target_ages[is_label]
                assert sub_surv.shape[-1] == _pl_module.model.surv_layer.t_eval.shape[0]
    
                # Get the batch average observed restricted survival time within each stratification. This also introduces the same batch effect
                ost = []
                for sample in range(sub_surv.shape[0]):
                    if sub_lbls_outcome[sample] > 0:
                        # If outcome was observed
                        _ost = np.min((1, sub_target_ages[sample])) 
                    else:
                        _ost = 1
                    ost.append(_ost / sub_surv.shape[0])
                    
                ost = np.sum(ost)

                # Log the average of this stratified group within this batch.
                if not np.isnan(ost):
                    metric_dict = {**metric_dict, log_name + "_" + strat_label: ost}
                
            self.log_dict(metric_dict)
    
        except:
            if suppress_warnings is False:
                logging.warning("Unable to calculate Observed Survival Time, this batch will be skipped - this will bias metrics.")
            else:
                pass

    def run_callback(self,
                     _trainer,
                     _pl_module,
                     batch,
                     log_name:               str='RMST',
                    ):

        # Make prediction of each survival curve
        all_outputs, _, _ = _pl_module(batch, return_loss=False, return_generation=True)
        pred_surv_CDFs = all_outputs["surv"]["surv_CDF"]
        pred_surv_pis = all_outputs["surv"]["surv_pi"]
        
        target_tokens = batch['target_token'].cpu().numpy()
        target_ages = batch['target_age_delta'].cpu().numpy()

        # Optionally process the batch using a custom method to label each patient into a different stratification group.
        if self.custom_stratification_method is not None and callable(self.custom_stratification_method):
            lbls_stratification = self.custom_stratification_method(batch)
            assert len(lbls_stratification) == target_tokens.shape[0]
        else:
            lbls_stratification = ["no_stratification" for _ in range(target_tokens.shape[0])]
            
        # Log records for individual outcomes
        ######################################
        if self.log_combined:
            
            # Combine for labels, 1 if any of the outcomes, 0 otherwise
            cdf = np.zeros_like(pred_surv_CDFs[0])
            lbls_outcome = np.zeros_like(target_tokens)
            surv_indices_included = []               
            for _outcome_token in self.outcome_tokens:
                # Add contribution for _outcome_token

                # Get which samples in the batch have the _outcome_token as the true target outcome
                _outcome_labels = (target_tokens == _outcome_token)
                lbls_outcome += _outcome_labels

                # Check the hash map to see which index of the survival output corresponds to the outcome token 
                #    note: this is needed because some fine-tuning models update the target encodings
                _outcome_surv_index = self.outcome_token_to_desurv_output_index[_outcome_token]
                _outcome_cdf = pred_surv_CDFs[_outcome_surv_index]
    
                # When different outcomes map to the same CIF/CDF curve, then we do not duplicate
                #    Note: this is only relevant in the supervised case, as the few-shot case will always have 1-to-1 _outcome_token to 
                #            _outcome_surv_index maps
                #          an example where this is useful: a fine tune model where you grouped Token 1 and Token 2 to the same outcome
                #             On the first pass you will have already included all of the outcomes associated to that group, so we don't
                #             want to duplicate this.
                if _outcome_surv_index not in surv_indices_included:
                    cdf += _outcome_cdf
                    surv_indices_included.append(_outcome_surv_index)

            # Get metrics
            self.get_rmst(cdf, lbls_outcome, target_ages, lbls_stratification, _trainer, _pl_module, log_name=log_name)
            self.get_ost(cdf, lbls_outcome, target_ages, lbls_stratification, _trainer, _pl_module, log_name="observed" + log_name)
            
        # Log records for individual outcomes
        ######################################
        #  Note: unless this is single risk causal model, then this may not be appropriate beyond diagnostics.
        # For each of the DeSurv output curves, which we want to evaluate separately
        #   1) collect the targets which are relevant to that curve. 
        if self.log_individual:
            raise NotImplementedError
            # assert pred_surv_pis is None, "Individual logging is not supported for Competing-Risk models."
            
            # for _desurv_index, _outcome_cdf in enumerate(pred_surv_CDFs):
                
            #     _outcome_labels = np.zeros_like(target_tokens)
            #     _outcome_tokens = []
            #     for _outcome_token in self.outcome_token_to_desurv_output_index.keys():
            #         # For each outcome, if it belongs to the `_desurv_index` survival curve then update target labels
            #         _outcome_desurv_index = self.outcome_token_to_desurv_output_index[_outcome_token]
            #         if _outcome_desurv_index == _desurv_index:
            #             _outcome_labels += (target_tokens == _outcome_token)             # 1 if == _outcome_token else 0
            #             _outcome_tokens.append(_outcome_token)
    
            #     # Plot the outcome curve
            #     if plot_outcome_curves:
            #         self.plot_outcome_curve(_outcome_cdf, 
            #                                 _outcome_labels, 
            #                                 _trainer, 
            #                                 log_name=log_name+f"_{_outcome_tokens}",
            #                                 ylabel=r"$F_{" + f"{','.join([str(i) for i in _outcome_tokens])}" + r"}(t)$")
            #     # Log metrics
            #     self.get_metrics(_outcome_cdf, #  if pred_surv_pis is None else _outcome_cdf / pred_surv_pis[_desurv_index], 
            #                      _outcome_labels,
            #                      target_ages, 
            #                      _trainer, 
            #                      _pl_module, 
            #                      log_name+f"_{_outcome_tokens}")


    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Run callback
        self.run_callback(_trainer=trainer, 
                          _pl_module = pl_module,
                          batch=batch,
                          log_name = "Test:RMST", 
                          )
