# Create custom callbacks for our pytorch-lightning model
import logging

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Callback
import wandb

class PerformanceMetrics(Callback):
    """
    Record metrics for a causal survival model.
    """
    def __init__(
        self, 
        ordered_prevalence=None, 
        log_concordance=True, 
        log_next_event_matrix=True
    ):
        """
        ARGS: ordered_prevalence:       The list of k, ordered by their frequency/prevalence in the dataset.
        """
        
        super().__init__()
        #
        self.ordered_prevalence = ordered_prevalence
        #
        self.log_concordance = log_concordance
        # 
        self.log_next_event_matrix = log_next_event_matrix
        self.all_truth_next_event_matrix = []
        self.all_model_next_event_matrix = []
        
        logging.info(f"Created PerformanceMetrics callback for causal self-supervised tasks.")
        logging.info(f"\t ")
        
    def _compute_and_log_concordance(
        self, 
        risk_scores, 
        observed_k,
        log_name, 
        log_stratified_by_outcome: bool = False, 
        suppress_warnings: bool = True,
        log_prefix: str = "",
    ):

        try:
            # Rank by increasing risk, and calculate normalised concordance based on position in ranked risk.
            risk_scores = np.argsort(risk_scores) + 1
            event_concordance = np.where(risk_scores == observed_k.cpu().numpy())[0][0] / (len(risk_scores) - 1)
    
            # Log the score in a way which aggregates across all different outcome types. This will artificially inflate the concordance. 
            #      For example, if we always predict the most prevalent token then the global picture produced here will look artificially good.
            metric_dict = {
                f"{log_prefix}{log_name}_no_stratify": event_concordance
            }
            
            # Log the score dependent upon what the outcome was. This will help identify if concordance is artificially inflated by high prevalence
            #      For example, a rare event will very likely always have low relative risk vs. a highly prevalent event.
            # Note: Only calculate these event specific scores if we are looking at next-event prediction, otherwise we will log too many values
            if log_stratified_by_outcome:
                metric_dict[f"{log_prefix}{log_name}_stratify_by_{observed_k}"] = event_concordance

            self.log_dict(metric_dict)
            
        except IndexError as e:
            logging.warning("IndexError in concordance calculation: %s", str(e))
            if not suppress_warnings:
                raise

    def _causal_callback(
        self, 
        outputs,
        batch,
        transitions=None,
        log_name: str = 'CausalMetrics',
        **kwargs
    ):
        """
        For each observed outcome in the context window log the concordance when predicting the next step

        Logs the model concordance, and optionally baseline based on fixed risks (e.g. from prevalence) provided at callback init
        
        Example: for a context of {A,B,C,D}, predict concordance of event {B} from {A}, {C} from {A,B} etc
        """
        
        if transitions is None:
            transitions = torch.sum(batch["tokens"] != 0) - 1                      # Number of transitions within context of this patient
        
        for t in range(transitions):
            if "surv" in outputs:
                cdfs = [cdf[t, :] for cdf in outputs["surv"]["surv_CDF"]]          # [(1,1000) for _ in range(vocab_size)]
                k = outputs["surv"]["k"][0][t]

                # Get the risk for each event type k (averaged over eval_index), then log concordance metric for the transition
                risk_scores = [sum(_cdf) for _cdf in cdfs]
                self._compute_and_log_concordance(risk_scores, k, log_name + "surv", log_stratified_by_outcome=True, **kwargs)
                
            elif "clf" in outputs:
                logits = [logit[t] for logit in outputs["clf"]["logits"]]
                k = outputs["clf"]["k"][0][t]
                
                # Get the baseline prevalence based concordance metric for each transition
                self._compute_and_log_concordance(logits, k, log_name + "clf", log_stratified_by_outcome=True, **kwargs)
                
            else:
                raise NotImplementedError

            # Get the baseline prevalence based concordance metric for each transition
            if self.ordered_prevalence is not None:
                self._compute_and_log_concordance(self.ordered_prevalence, k, log_name + "prevalence", log_stratified_by_outcome=True, **kwargs)

    def _look_ahead_callback(
        self, 
        outputs,
        batch, 
        look_ahead, 
        log_name: str = 'LookAheadMetrics',
        **kwargs
    ):
        """
        For the last outcome in the context window log the concordance when predicting from `look_ahead` steps before. This method always predicts 
        concordance for last observation - only the number of number of context records (always starting from the beginning) vary.

        Logs the model concordance, and optionally baseline based on fixed risks (e.g. from prevalence) provided at callback init
        
        Example: for a context of {A,B,C,D,E,F} and look_ahead=3, predict concordance of event {F} from {A,B,C}. 
        """
        # Update log name to log different look-aheads separately, e.g. "+12_LookAheadMetrics"
        log_name = f"+{look_ahead}_" + log_name

        # Ensure we don't exceed maximum number of look-ahead steps
        transitions = torch.sum(batch["tokens"] != 0) - 1                   # The number of transition in the context window
        assert transitions >= look_ahead, f"`look_ahead` must not exceed number of transitions in context. Got {look_ahead} for {transitions} transitions"

        last_valid_transition = transitions - look_ahead - 1
        assert last_valid_transition >= 0, "last_valid_transition must be a valid index. Ensure you have chosen the correct `look_ahead`."
        
        if "surv" in outputs:
            cdfs = [cdf[last_valid_transition, :] for cdf in outputs["surv"]["surv_CDF"]]          # [(1,1000) for _ in range(vocab_size)]
            k = outputs["surv"]["k"][0][-1]

            # Get the risk for each event type k (averaged over eval_index), then log concordance metric for the transition
            risk_scores = [sum(_cdf) for _cdf in cdfs]
            self._compute_and_log_concordance(risk_scores, k, log_name + "surv", log_stratified_by_outcome=False, **kwargs)
            
        elif "clf" in outputs:
            logits = [logit[last_valid_transition] for logit in outputs["clf"]["logits"]]
            k = outputs["clf"]["k"][0][-1]
            
            # Get the baseline prevalence based concordance metric for each transition
            self._compute_and_log_concordance(logits, k, log_name + "clf", log_stratified_by_outcome=False, **kwargs)
            
        else:
            raise NotImplementedError

        # Get the baseline prevalence based concordance metric for each transition
        if self.ordered_prevalence is not None:
            self._compute_and_log_concordance(self.ordered_prevalence, k, log_name + "prevalence", log_stratified_by_outcome=False, **kwargs)
        
    
    def run_concordance_callback(
        self,
        outputs,
        batch,
        **kwargs
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

        # Early stop if no transitions within context of this patient
        transitions = torch.sum(batch["tokens"] != 0) - 1      
        if transitions <= 0:
            return
        
        # Push through the model in a causal fashion, whilst returning the generation curves
        self._causal_callback(outputs, batch, transitions=transitions, **kwargs)

        # Repeat the above, but looking further k steps ahead
        max_look_ahead = min(20, transitions.item())
        for look_ahead_by in [i for i in range(max_look_ahead) if i < 4 or i % 3 == 1]:
            self._look_ahead_callback(outputs, batch, look_ahead_by, **kwargs)

    def run_next_event_callback(
        self,
        outputs, 
        batch,
        **kwargs
    ):
        """
        """

        # Create empty matrices 
        number_of_unique_tokens = len(outputs["surv"]["surv_CDF"]) if "surv" in outputs else len(outputs["clf"]["logits"])
        truth_next_event_matrix = np.zeros((number_of_unique_tokens, number_of_unique_tokens))
        model_next_event_matrix = np.zeros((number_of_unique_tokens, number_of_unique_tokens))
        
        transitions = torch.sum(batch["tokens"] != 0) - 1
        for t in range(transitions - 1):

            if "surv" in outputs:
                true_prior_event = outputs["surv"]["k"][0][t] - 1
                true_next_event = outputs["surv"]["k"][0][t+1] - 1
                
                # Get the risk for each event type k (averaged over eval_index)
                cdfs = [cdf[t, :] for cdf in outputs["surv"]["surv_CDF"]]          # [(1,1000) for _ in range(vocab_size)]
                risk_scores = [sum(_cdf) for _cdf in cdfs]
                model_next_event = np.argmax(risk_scores)

            elif "clf" in outputs:
                true_prior_event = outputs["clf"]["k"][0][t] - 1
                true_next_event = outputs["clf"]["k"][0][t+1] -1 
                
                # Get the risk for each event type k
                logits = [logit[t] for logit in outputs["clf"]["logits"]]
                model_next_event = np.argmax(logits)
                
            else:
                raise NotImplementedError

            # print(f"True prior event {true_prior_event}")
            # print(f"True next event {true_next_event}")
            # print(f"True model next event {model_next_event}")
            
            # Update matrix of counts for ground truth
            assert (true_prior_event >= 0) and (true_next_event >=0), "Index of true events must be positive"
            truth_next_event_matrix[true_prior_event, true_next_event] += 1

            # Update matrix for counts under model
            assert model_next_event >= 0, "Index of model next event must be positive"
            model_next_event_matrix[true_prior_event, model_next_event] += 1

        self.all_truth_next_event_matrix.append(truth_next_event_matrix)
        self.all_model_next_event_matrix.append(model_next_event_matrix)

    def on_test_batch_end(
        self, 
        trainer,
        pl_module,
        outputs, 
        batch, 
        batch_idx
    ):
        """
        """

        # We run the callback on only the first sample within each batch as this is computationally heavy
        batch = {k: v[[0]] for k, v in batch.items()}          # Take first patient in the batch
        
        transitions = torch.sum(batch["tokens"] != 0) - 1      
        if transitions == 0:
            return

        # Push through module
        outputs, _, _ = pl_module(batch, is_generation=False, return_loss=False, return_generation=True)

        if self.log_concordance:
            self.run_concordance_callback(outputs=outputs, batch=batch, log_prefix="Test:")
            
        if self.log_next_event_matrix:
            self.run_next_event_callback(outputs=outputs, batch=batch, log_prefix="Test:")

    def on_test_epoch_end(
        self,
        trainer, 
        pl_module, 
    ):
        """
        """

        # Combine all the next event matrices and log
        truth_matrix = np.sum(self.all_truth_next_event_matrix, axis=0)
        model_matrix = np.sum(self.all_model_next_event_matrix, axis=0)

        # Generate row & column labels
        size = truth_matrix.shape[0]
        row_labels = [f"prior_event_{i}" for i in range(size)]
        col_labels = [f"next_event_{j}" for j in range(size)]

        # Log each matrix as a table
        for matrix, label in zip([truth_matrix, model_matrix], ["Truth", "Model"]):
            df = pd.DataFrame(matrix, columns=col_labels)
            table = wandb.Table(dataframe=df)
            pl_module.logger.experiment.log({f"Test:NextEventMatrix{label}": table})

        # Reset lists
        self.all_truth_next_event_matrix = []
        self.all_model_next_event_matrix = []
