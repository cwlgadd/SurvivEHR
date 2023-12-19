import torch
from torch import nn
import math
from typing import Optional
import logging


class DataEmbeddingLayer(torch.nn.Module):
    r"""
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        # value_embed_dim: Optional[int] = None
    ):

        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # self.v_embed_dim = value_embed_dim if value_embed_dim is not None else embed_dim

        # Split case        
        self.token_embed_layer = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        #  note, we don't need embedding bag as we do not need to model groups of events. 
        #        However, it is useful to use for the numerical embedding for the per_sample_weights 
        #        in order to relatively weight different values. Similarly mode is only to allow for
        #        using sample weights functionality.
        self.value_embed_layer =  nn.EmbeddingBag(self.vocab_size, self.embed_dim, mode="sum", padding_idx=0)
        

    def _split_embed(
        self, 
        tokens: torch.Tensor,                        # bsz, seq_len
        values: Optional[torch.Tensor] = None        # bsz, seq_len
    ):
        """
        """
        tok_emb = self.token_embed_layer(tokens)              # shape: (batch_size, sequence_length, embed_dim)
        
        if values is None:
            # logging.info(f"X returning without values {values}")
            return tok_emb

        # For tokens with no accompanying value, set to padding indx, and so they do not contribute to the gradient
        valued_tokens = torch.where(torch.isnan(values), 0, tokens)
        values = torch.where(torch.isnan(values), 0, values)

        val_emb = self.value_embed_layer(valued_tokens.reshape((-1,1)), per_sample_weights=values.reshape((-1,1)))
        val_emb = val_emb.reshape(tok_emb.shape)

        return tok_emb + val_emb  

    def forward(
        self,
        tokens: torch.Tensor,                   # bsz, seq_len
        values: Optional[torch.Tensor] = None,         # bsz, seq_len
    ):
        """
        """

        embedded = self._split_embed(tokens=tokens, values=values)      # shape: (batch_size, sequence_length, embed_dim)
        
        return embedded