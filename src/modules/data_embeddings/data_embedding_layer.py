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
        value_embed_dim: Optional[int] = None
    ):

        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.v_embed_dim = value_embed_dim if value_embed_dim is not None else embed_dim

        # Split case
        #  note, we don't need embedding bag as we do not need to model groups of events. 
        #        However, it is useful to use for the numerical embedding for the per_sample_weights 
        #        in order to relatively weight different values
        self.token_embed_layer = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        
        self.value_embed_layer = nn.ModuleList(
            [nn.EmbeddingBag(self.vocab_size, self.v_embed_dim, mode="sum"),     # mode is only to allow sample weights, we arent actually doing a reduce map
             nn.Linear(self.v_embed_dim, self.embed_dim)
            ]
        )
        

    def _split_embed(
        self, 
        tokens: torch.Tensor,                        # bsz, seq_len
        # values: Optional[torch.Tensor] = None        # bsz, seq_len
    ):
        """
        """
        # logging.debug(f"tokens {tokens.shape}: {tokens}")
        # logging.debug(f"values {values.shape}: {values}")
                
        assert not torch.isnan(tokens).any(), f"tokens {tokens.shape}, {tokens}"

        tok_emb = self.token_embed_layer(tokens)              # shape: (batch_size, sequence_length, embed_dim)
        assert not torch.isnan(tok_emb).any(), f"tok_emb {tok_emb.shape}, {tok_emb}"

        return tok_emb

        # # if no values passed, or all if all are masked then return token embedding
        # if values is None:     
        #     logging.info(f"values none")
        # else:
        #     logging.info(f"using value embedding")
        #     pass
            
        # # Log mask of values (test if this is a good idea)
        # # values = torch.where(values != np.nan, torch.log(values), -np.inf)
        # values = torch.where(torch.isnan(values), 0, values)

        # val_emb = self.value_embed_layer[0](tokens.reshape((-1,1)), per_sample_weights=values.reshape((-1,1)))
        # val_emb = self.value_embed_layer[1](val_emb).reshape(tok_emb.shape)

        # # logging.debug(f"values mask {values.shape}: {values}")
        # # logging.debug(f"token embedding {tok_emb.shape}: {tok_emb}")
        # # logging.debug(f"value embedding {val_emb.shape}: {val_emb}")

        return tok_emb + val_emb  

    def forward(
        self,
        tokens: torch.Tensor,         # bsz, seq_len
        # values: torch.Tensor,         # bsz, seq_len
    ):
        """
        """

        embedded = self._split_embed(tokens) # , values)      # shape: (batch_size, sequence_length, embed_dim)
        
        return embedded