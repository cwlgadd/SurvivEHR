import polars as pl
from typing import Dict, List, Optional
import torch

def is_interactive():
    """
    Determine if a job is in an interactive shell.
    
    Used to ensure trainer strategy can be set to ``auto`` in interactive sessions. This is because
    ddp is not compatible - when spawning child processes in ddp, PyTorch needs to re-import the 
    entry point module, but there is no script to import in an interactive session.
    """
    try:
        # Works in Jupyter/IPython
        return get_ipython().__class__.__name__ != 'TerminalInteractiveShell'
    except NameError:
        # Not running inside IPython
        return False

def custom_mm_outcomes(dm):
    """
    Extracts a list of outcome event codes from the datamodule's tokenizer.

    This function filters the `_event_counts` DataFrame in the tokenizer to select
    events that meet both of the following criteria:
    - Have a count greater than 0
    - Match the regex pattern `^[A-Z0-9_]+$` (i.e., consist of uppercase letters, digits, or underscores)

    Args:
        dm: A datamodule object with a `tokenizer._event_counts` attribute (assumed to be a Polars DataFrame).

    Returns:
        List[str]: A list of event code strings satisfying the above conditions.
    """
    conditions = (
        dm.tokenizer._event_counts.filter((pl.col("COUNT") > 0) &
            (pl.col("EVENT").str.contains(r'^[A-Z0-9_]+$')))
          .select("EVENT")
          .to_series()
          .to_list()
    )
    return conditions
    
def expand_batch_to_context_on_tokens(batch: Dict[str, torch.Tensor],
                                      target_tokens,
                                     ):
    """
    Expands a batch by generating multiple truncated sequences for each sample, where each 
    end on specified target tokens (e.g., diagnoses of interest). 
    
    For example, given a batch with first row tokens [1,2,3,4,5,6,7,8,9,10], and target tokens
    [3, 5, 9], that row is expanded to three rows: [1,2,3], [1,2,3,4,5] and [1,2,3,4,5,6,7,8,9]

    Parameters
    ----------
    batch : Dict[str, torch.Tensor]
        A dictionary containing batched input data with keys:
            - 'tokens': Tensor of shape (B, L), tokenized clinical codes.
            - 'ages': Tensor of shape (B, L), age at each event.
            - 'values': Tensor of shape (B, L), associated event values (e.g., lab results).
            - 'attention_mask': Tensor of shape (B, L), attention mask (1 if valid, 0 if padded).
            - 'static_covariates': Tensor of shape (B, D), static patient features.

    target_tokens : Iterable[int]
        List or set of token IDs to expand the batch on. Each occurrence of a target token
        results in a new sequence ending at that token.

    Returns
    -------
    Dict[str, torch.Tensor]
        A new batch dictionary with expanded sequences:
            - 'tokens': (N, L_max), padded token sequences.
            - 'ages': (N, L_max), padded age sequences.
            - 'values': (N, L_max), padded value sequences.
            - 'attention_mask': (N, L_max), padded attention masks.
            - 'static_covariates': (N, D), repeated static covariates per new sample.
        
        If no target tokens are found in the batch, returns dictionary of empty tensors of shape (0, ...).
    """
        
    target_tokens = set(target_tokens)
    
    expanded_batch = {
        'tokens': [],
        'ages': [],
        'values': [],
        'attention_mask': [],
        'static_covariates': []
    }

    B, L = batch['tokens'].shape
    for i in range(B):
        tokens = batch['tokens'][i]
        ages = batch['ages'][i]
        values = batch['values'][i]
        mask = batch['attention_mask'][i]
        statics = batch['static_covariates'][i]

        for j in range(L):
            if tokens[j].item() in target_tokens:
                # Create a new sample up to and including index j
                expanded_batch['tokens'].append(tokens[:j+1])
                expanded_batch['ages'].append(ages[:j+1])
                expanded_batch['values'].append(values[:j+1])
                expanded_batch['attention_mask'].append(mask[:j+1])
                expanded_batch['static_covariates'].append(statics)  # same statics

    # Handle edge case: no matching tokens
    if not expanded_batch['tokens']:
        return {
            'tokens': torch.empty((0, L), dtype=torch.long, device=batch['tokens'].device),
            'ages': torch.empty((0, L), dtype=torch.float32, device=batch['ages'].device),
            'values': torch.empty((0, L), dtype=torch.float32, device=batch['values'].device).fill_(float('nan')),
            'attention_mask': torch.empty((0, L), dtype=torch.long, device=batch['attention_mask'].device),
            'static_covariates': torch.empty((0, batch['static_covariates'].shape[1]), dtype=torch.float32, device=batch['static_covariates'].device)
        }
        
    # Pad to maximum length
    max_len = max(t.size(0) for t in expanded_batch['tokens'])
    def pad(tensor_list, pad_value=0.0, dtype=torch.float32):
        return torch.stack([
            torch.cat([t, torch.full((max_len - t.size(0),), pad_value, dtype=dtype, device=t.device)])
            if t.dim() == 1 else
            torch.cat([t, torch.full((max_len - t.size(0),), pad_value, dtype=t.dtype, device=t.device)])
            for t in tensor_list
        ])

    tokens = pad(expanded_batch['tokens'], pad_value=0, dtype=torch.long)
    ages = pad(expanded_batch['ages'], pad_value=0.0, dtype=torch.float32)
    values = pad(expanded_batch['values'], pad_value=float('nan'), dtype=torch.float32)
    attention_mask = pad(expanded_batch['attention_mask'], pad_value=0, dtype=torch.long)
    static_covariates = torch.stack(expanded_batch['static_covariates'])

    return {
        'tokens': tokens,
        'ages': ages,
        'values': values,
        'attention_mask': attention_mask,
        'static_covariates': static_covariates
    }


def filter_batch_by_context_length(batch: Dict[str, torch.Tensor],
                                   min_context_length: int,
                                   max_context_length: int,
                                   ) -> Dict[str, torch.Tensor]:
    """
    Filters a batch by removing entire samples where the number of valid (non-padding)
    tokens exceeds a specified maximum context length.

    Parameters
    ----------
    batch : Dict[str, torch.Tensor]
        A dictionary containing batched input data with keys:
            - 'tokens': Tensor of shape (B, L), tokenized clinical codes.
            - 'ages': Tensor of shape (B, L), age at each event.
            - 'values': Tensor of shape (B, L), associated event values (e.g., lab results).
            - 'attention_mask': Tensor of shape (B, L), attention mask (1 if valid, 0 if padded).
            - 'static_covariates': Tensor of shape (B, D), static patient features.

    min_context_length : int
        Minimum number of valid tokens (according to the attention mask) allowed in each sample.
        Any sample below this threshold is removed.
    max_context_length : int
        Maximum number of valid tokens (according to the attention mask) allowed in each sample.
        Any sample exceeding this threshold is removed.

    Returns
    -------
    Dict[str, torch.Tensor]
        A filtered batch dictionary where all sequences meet the context length constraint.
        If no samples meet the criteria, returns empty tensors of appropriate shapes.
    """
    attention_mask = batch['attention_mask']
    valid_lengths = attention_mask.sum(dim=1)
    keep_mask = (valid_lengths >= min_context_length) & (valid_lengths <= max_context_length)
    keep_indices = keep_mask.nonzero(as_tuple=True)[0]

    if keep_indices.numel() == 0:
        B, L = batch['tokens'].shape
        D = batch['static_covariates'].shape[1]
        device = batch['tokens'].device
        return {
            'tokens': torch.empty((0, L), dtype=torch.long, device=device),
            'ages': torch.empty((0, L), dtype=torch.float32, device=device),
            'values': torch.empty((0, L), dtype=torch.float32, device=device).fill_(float('nan')),
            'attention_mask': torch.empty((0, L), dtype=torch.long, device=device),
            'static_covariates': torch.empty((0, D), dtype=torch.float32, device=device)
        }

    return {
        key: tensor[keep_indices]
        for key, tensor in batch.items()
    }

