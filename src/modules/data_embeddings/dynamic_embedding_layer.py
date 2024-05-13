import torch
from torch import nn
import math
from typing import Optional
import logging


class JointDynamicEmbeddingLayer(torch.nn.Module):
    r"""
    """

    def __init__(self,
                 vocab_size:             int,
                 embed_dim:              int,
                ):
        """
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.embed_layer = nn.Embedding(num_embeddings=self.vocab_size, 
                                        embedding_dim=self.embed_dim,
                                        padding_idx=0)


    def forward(self,
                tokens:    torch.Tensor,
                values:    torch.Tensor,
                ) -> torch.Tensor:
        """ Return an embedding of the input tokens, weighted by values, if present 

        Args:
            tokens: A tensor of shape ``(batch_size,)`` that contains the indices of the observations in the batch. Zero indicates padding
            values: A tensor of shape ``(batch_size,)`` that contains the continuous values associated with the observations in the batch,
                    If values are not present for an observation the value of this tensor will be torch.nan

        Returns: 
            A tensor for shape ``(batch_size, out_dim)`` that contains the embedding of the token indices weighted by their associated 
            observed values, if present. 
        """
        
        # For tokens with no accompanying value, set to the average value of one. This
        values = torch.where(torch.isnan(values), 0, values)

        emb = self.embed_layer(tokens)  # , per_sample_weights=values.reshape((-1,1))
        scaled_emb = emb * values.reshape((-1,1))
        return emb


class SplitDynamicEmbeddingLayer(torch.nn.Module):
    r"""
    """

    def __init__(self,
                 vocab_size:             int,
                 embed_dim:              int,
                 cat_event_embed_dim:    int,
                 num_value_embed_dim:    int,
                 cat_event_weight:       float = 1 / 2,
                 num_value_weight:       float = 1 / 2,
                 ):
        """
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.cat_event_embed_dim = cat_event_embed_dim
        self.num_value_embed_dim = num_value_embed_dim
        self.cat_event_weight = cat_event_weight
        self.num_value_weight = num_value_weight

        # 
        self.cat_event_embed_layer = nn.Embedding(num_embeddings=self.vocab_size, 
                                                  embedding_dim=self.cat_event_embed_dim,
                                                  padding_idx=0)
        self.cat_event_proj = nn.Linear(self.cat_event_embed_dim, self.embed_dim)

        # 
        self.num_value_embed_layer = nn.EmbeddingBag(num_embeddings=self.vocab_size, 
                                                     embedding_dim=self.num_value_embed_dim,
                                                     padding_idx=0,
                                                     mode="sum")
        self.num_value_proj = nn.Linear(self.num_value_embed_dim, self.embed_dim)
        
    def forward(self,
                tokens:    torch.Tensor,
                values:    Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """ Return an embedding of the input tokens, added to another embedding of valued tokens, weighted by values

        Args:
            tokens: A tensor of shape ``(batch_size,)`` that contains the indices of the observations in the batch. Zero indicates padding
            values: A tensor of shape ``(batch_size,)`` that contains the continuous values associated with the observations in the batch,
                    If values are not present for an observation the value of this tensor will be torch.nan

        Returns: 
            A tensor for shape ``(batch_size, out_dim)`` that contains the embedding of the token indices weighted by their associated 
            observed values, if present. 
        """

        # Token embedding
        tok_proj_emb = self.cat_event_proj(self.cat_event_embed_layer(tokens))    # shape: batch_size * sequence_length, embed_dim)

        if values is None:
            # logging.debug("Returning embedding layer without values")
            return tok_proj_emb
         
        numeric_values = torch.where(torch.isnan(values), 0, values)              # For tokens with no accompanying value, set to zero so they have no weighting
        tokens_with_a_value = torch.where(torch.isnan(values), 0, tokens)         # zero if padded token, or a token with no value (e.g. diagnosis, or test with missing value)

        value_proj_emb = self.num_value_proj(self.num_value_embed_layer(tokens_with_a_value.reshape(-1,1), per_sample_weights=numeric_values.reshape(-1,1)))
         
        return ( self.cat_event_weight * tok_proj_emb ) + ( self.num_value_weight * value_proj_emb )
