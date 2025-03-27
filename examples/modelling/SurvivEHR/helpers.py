import torch
import numpy as np
from typing import Dict, List


def count_prior_tokens(batch: Dict[str, torch.Tensor],
                       tokens: List[int],
                       **kwargs
                      ) -> List[str]:
    """
    Extracts the number of previous records each patient has experienced from a list of `tokens`.

    :param batch: A dictionary containing a key `"tokens"` with a tensor of tokenized patient records.
    :type batch: dict
    :param tokens: A list of tokens representing specific conditions of interest.
    :type tokens: list[int]

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
        print(labels)  # Output: ['2 current diagnoses', '3 current diagnoses']
    """
    
    if "tokens" not in batch:
        raise ValueError("The input dictionary must contain a 'tokens' key.")
        
    if not all(isinstance(i, int) for i in tokens):
        raise TypeError(f"tokens must be a list of integers. Got {tokens}")

    # Convert tokens to a tensor (ensure it matches dtype and device of input)
    tokens = torch.tensor(tokens, dtype=batch["tokens"].dtype, device=batch["tokens"].device)

    # Create a boolean mask where tokens match any target condition
    mask = torch.isin(batch["tokens"], tokens)

    # Count occurrences per patient
    counts = mask.sum(dim=1)

    # Format output with correct singular/plural handling
    return [f"{i} current diagnosis" if i == 1 else f"{i} current diagnoses" for i in counts.tolist()]
    