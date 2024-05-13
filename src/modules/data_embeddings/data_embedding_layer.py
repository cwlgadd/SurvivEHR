import torch
from torch import nn
import math
from typing import Optional
import logging
from CPRD.src.modules.data_embeddings.dynamic_embedding_layer import JointDynamicEmbeddingLayer, SplitDynamicEmbeddingLayer

class DataEmbeddingLayer(torch.nn.Module):
    r""" This class embeds a PyTorch Batch into a fixed size embedding.
    """
    def __init__(
        self,
        vocab_size:             int,
        embed_dim:              int,
        static_weight:          float= 1 / 2,
        dynamic_weight:         float= 1 / 2,
        **kwargs
    ):

        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # 
        self.static_proj = nn.Linear(16, self.embed_dim)

        
        # self.dynamic_embedding_layer = JointDynamicEmbeddingLayer(vocab_size=vocab_size,
        #                                                           embed_dim=embed_dim,
        #                                                           **kwargs)
        self.dynamic_embedding_layer = SplitDynamicEmbeddingLayer(vocab_size=vocab_size,
                                                                  embed_dim=embed_dim,
                                                                  cat_event_embed_dim=embed_dim,
                                                                  num_value_embed_dim=embed_dim,
                                                                  **kwargs)

    def _static_embedding(
        self,
        covariates: torch.Tensor                     # bsz, num_covariates
    ):
        batch_size, num_covariates = covariates.shape

        return self.static_proj(covariates)
    
    def _dynamic_embedding(
        self, 
        tokens: torch.Tensor,                        # bsz, seq_len
        values: Optional[torch.Tensor] = None        # bsz, seq_len
    ):
        """ Return an embedding of the token indices, weighted by values if present.

            Masked values are indicated by np.nan or torch.nan elements
        """
        assert tokens.shape == values.shape
        batch_size, sequence_length = tokens.shape

        # Flatten sequence
        tokens = tokens.reshape(-1)
        values = values if values is None else values.reshape(-1)

        # Return embedding
        return self.dynamic_embedding_layer(tokens, values).view(batch_size, sequence_length, self.embed_dim)

    def forward(
        self,
        tokens:                      torch.Tensor,                          # bsz, seq_len
        values:                      Optional[torch.Tensor] = None,         # bsz, seq_len
        covariates:                  Optional[torch.Tensor] = None,         # bsz, num_covariates
    ):
        
        embedded = self._dynamic_embedding(tokens=tokens, values=values)      # shape: (batch_size, sequence_length, embed_dim)

        if covariates is not None:
            static = self._static_embedding(covariates=covariates)
            print(static.shape)
            embedded += static
        
        return embedded