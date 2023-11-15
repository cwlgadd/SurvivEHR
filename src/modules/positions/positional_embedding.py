import torch
from torch import nn
import math
from typing import Optional
import logging

class PositionalEmbedding(torch.nn.Module):
    r"""
        A module for applying index-based position embeddings through one hot encoding

    ARGS:
        block_size: The largest context block supported
        encoding_dim: The desired size of the output embedding.
        
    KWARGS:
    """

    def __init__(self,
                 block_size:int, 
                 embed_dim: int,
                ):
        """
        """
        super().__init__()
        self.wpe = nn.Embedding(block_size, embed_dim)
        
        logging.info("Using Positional Embedding. This module uses the index position of an event within the block of events.")

    def forward(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        
        ARGS: 
            tokens: tokens
                Tensor, shape ``[bsz, seq_len]`` 
            
        Returns
                Tensor, shape ``[1, seq_len, embed_dim]``

        """
        bsz, seq_len = tokens.size()
        positional_info = torch.arange(seq_len, device=tokens.device).tile((bsz, 1))
        return self.wpe(positional_info)


def test(bsz=1, seq_len=10, embed_dim=6):
    pass
    

if __name__ == "__main__":
    
    test()
    