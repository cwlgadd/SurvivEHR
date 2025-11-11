# Adapted from  https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import torch
import math
from typing import Optional
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import logging

class PositionalEncoding(torch.nn.Module):
    r"""
        A module for applying index-based position encodings

    .. math::
        P\left(k, 2i\right) = \sin \left(\frac{k}{n**{2i/d}}\right) 
        P\left(k, 2i + 1\right) = \cos \left(\frac{k}{n**{2i/d}}\right)

    ARGS:
        encoding_dim: (d) The desired size of the output embedding.
        
    KWARGS:
        n_scalar: (n) The scalar used to initialize the frequency space. Defaults to 10,000 following "Attention Is All You Need".
        max_length: The maximum sequence length, for precomputing positional encoding.
    """

    def __init__(self, 
                 encoding_dim: int,
                 n_scalar: float = 10000.0,
                 max_length: int = 5000, 
                ):
        """
        """
        assert encoding_dim % 2 == 0, "PositionalEncoding: encoding_dim must be even"
        
        super().__init__()
        self.encoding_dim = encoding_dim

        # pre-compute positional encoding matrix        
        position = torch.arange(max_length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, encoding_dim, 2, device=device) * (-math.log(n_scalar) / encoding_dim))
        div_term = torch.nn.Parameter(div_term, requires_grad=False)
        self.pe = torch.zeros(1, max_length, encoding_dim, device=device)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)
        
        logging.info("Using Positional Encoding. This module uses the index position of an event within the block of events.")

    def forward(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        
        ARGS: 
            tokens: 
                Tensor, shape ``[bsz, seq_len]`` 
            
        Returns
                Tensor, shape ``[1, seq_len, encoding_dim]``

        """
        return self.pe[:, :tokens.size(1), :]

    
class TemporalPositionalEncoding(torch.nn.Module):
    """A module for applying time-based position encodings


    .. math::
        P\left(k, 2i\right) = \sin \left(\frac{k}{n**{2i/d}}\right) 
        P\left(k, 2i + 1\right) = \cos \left(\frac{k}{n**{2i/d}}\right)

    ARGS:
        embedding_dim: (d) The desired size of the output embedding.
        
    KWARGS:
        n_scalar: (n) The maximum observed timepoint, used to initialize the frequency space. Defaults to 10,000 following "Attention Is All You Need".
    """
    
    def __init__(self, 
                 encoding_dim: int,
                 n_scalar: float = 10000.0,
                ):
        """
        """
        assert encoding_dim % 2 == 0, "TemporalPositionalEncoding: encoding_dim must be even"
        
        super().__init__()
        self.encoding_dim = encoding_dim

        # pre-compute positional encoding matrix. 
        div_term = torch.exp(torch.arange(0, encoding_dim, 2) * (-math.log(n_scalar) / encoding_dim))
        # We assume that time positions are scaled by year (in our CPRD example this is a 5 year scale). Unscale here to an order of days
        div_term *= 5*365
        self.div_term = torch.nn.Parameter(div_term, requires_grad=False)

        logging.info("Using Temporal Positional Encoding. This module uses the patient's age at an event within their time series.")

    def forward(self, ages: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            ages: Time points for token observation
                Tensor, shape ``[bsz, seq_len]``

        Returns:
                Tensor, shape ``[bsz, seq_len, encoding_dim]``
        """
        assert ages is not None, "If using a temporal positional encoder you must supply ages at tokenized events"

        bsz, seq_len = ages.shape
        ages = ages.unsqueeze(-1)                    # Unsqueeze for broadcasting through the encoding dim
        
        temporal_encodings = torch.zeros(bsz, seq_len, self.encoding_dim, device=ages.device)
        temporal_encodings[:, :, 0::2] = torch.sin(ages * self.div_term.unsqueeze(0).unsqueeze(0))    # [bsz, seq_len, 1] * [1, 1, encoding_dim / 2]
        temporal_encodings[:, :, 1::2] = torch.cos(ages * self.div_term.unsqueeze(0).unsqueeze(0))

        return temporal_encodings
