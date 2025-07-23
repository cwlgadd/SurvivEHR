# Callbacks for embedding labels which can be used for abritrary tokenizers

import torch
import numpy as np
from typing import Dict, List, Optional

def Token_count(batch: Dict[str, torch.Tensor],
                unique_only: bool = False,
                bin_size: int = 25,
                **kwargs
                ) -> List[str]:
    """
    Extracts the number of previous records each patient has experienced.
    """
    
    if "attention_mask" not in batch:
        raise ValueError("The input dictionary must contain a 'attention_mask' key.")
    if "tokens" not in batch:
        raise ValueError("The input dictionary must contain a 'tokens' key.")
    
    labels = []
    for patient_tokens, patient_mask in zip(batch["tokens"], batch["attention_mask"]):
        patient_tokens = patient_tokens[patient_mask == 1]
        tokens_to_check = torch.unique(patient_tokens) if unique_only else patient_tokens
        count = tokens_to_check.shape[0]                     # Count occurrences per patient
        bin_min = (count // bin_size) * bin_size
        labels.append(float(bin_min))          # f"{bin_min:3}-{bin_min+bin_size:3}"
        
    return labels, batch

def log_token_count(batch: Dict[str, torch.Tensor],
                    **kwargs
                    ) -> List[str]:
    labels, batch = Token_count(batch, **kwargs)
    log_labels = [np.log(_label) for _label in labels]
    
    return log_labels, batch

def number_of_preexisting_by_token(batch: Dict[str, torch.Tensor],
                                   tokens: List[int],
                                   unique_only: bool = True,
                                   max_number_of_conditions: int = 5,
                                   **kwargs
                                   ) -> List[str]:
    """
    Extracts the number of previous records each patient has experienced from a list of `tokens`.

    :param batch: A dictionary containing a key `"tokens"` with a tensor of tokenized patient records.
    :type batch: dict
    :param tokens: A list of tokens representing specific conditions of interest.
    :type tokens: list[int]
    :param unique_only: Whether to count only unique matching tokens per patient. Defaults to True.
    :type unique_only: bool

    :return: A list of stratification labels indicating the count of matching records per patient.
    :rtype: list[str]

    Example:

    .. code-block:: python

        batch = {
            "tokens": torch.tensor([[1, 2, 3, 1, 0],
                                    [1, 5, 4, 3, 2]])
        }
        token_list = [1, 4, 5]
        labels = get_existing_counts_stratification_labels(batch, token_list)
        print(labels)  # Output: ['1 current diagnosis', '3 current diagnoses']
    """
    
    if "tokens" not in batch:
        raise ValueError("The input dictionary must contain a 'tokens' key.")
        
    if not all(isinstance(i, int) for i in tokens):
        raise TypeError(f"tokens must be a list of integers. Got {tokens}")

    # Convert tokens to a tensor (ensure it matches dtype and device of input)
    tokens = torch.tensor(tokens, dtype=batch["tokens"].dtype, device=batch["tokens"].device)

    labels = []
    for patient_tokens in batch["tokens"]:
        tokens_to_check = torch.unique(patient_tokens) if unique_only else patient_tokens
        match_mask = torch.isin(tokens_to_check, tokens)    # Create a boolean mask where tokens match any target condition
        count = match_mask.sum().item()                     # Count occurrences per patient
        count = np.min((count, max_number_of_conditions))        
        labels.append(count)
        
    return labels

def static(batch: Dict[str, torch.Tensor],
           dm,
           key: str,
           **kwargs
           ) -> List[str]:
    """
    Extracts labels based on static covariates of each patient.

    :param batch: A dictionary containing a key `"tokens"` with a tensor of tokenized patient records.
    :type batch: dict
    :param dm:    The FastEHR datamodule
    :type dm:    FastEHR.dataloader.FoundationalDataModule
    :param key:   Which static_covariate key to plot
    :type key:   string

    :return: A list of stratification labels indicating the count of matching records per patient.
    :rtype: list[str]

    Example:

    .. code-block:: python

    """
    
    if "static_covariates" not in batch:
        raise ValueError("The input dictionary must contain a 'static_covariates' key.")
        
    # Decode static_covariates
    static_covariates = dm.train_set._decode_covariates(batch["static_covariates"])

    if key not in static_covariates:
        raise ValueError(f"The static_covariates do not contain the {key} key.")

    # Get the values for the variale of interest
    key_static_covariates = static_covariates[key]

    labels = [patient_lbl for patient_lbl in key_static_covariates]
        
    return labels
