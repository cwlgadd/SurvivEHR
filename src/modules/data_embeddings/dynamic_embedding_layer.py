import torch
from torch import nn
import math
from typing import Optional
import logging

class DynamicEmbeddingLayer(torch.nn.Module):
    r"""
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 embed_mode: str = "split",
                 padding_idx: int = 0
                ):

        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_mode = embed_mode
        
        if self.embed_mode.lower() == "joint":
            raise NotImplementedError
        elif self.embed_mode.lower() == "split"::
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self,
                tokens: torch.Tensor,
                values: torch.Tensor):
        pass
        

    