import math
import os
from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transformers.modeling_utils import ModuleUtilsMixin
import logging
from typing import Optional

from SurvivEHR.src.modules.positions.positional_encoding import PositionalEncoding
from SurvivEHR.src.modules.positions.positional_embedding import PositionalEmbedding
from SurvivEHR.src.modules.data_embeddings.data_embedding_layer import DataEmbeddingLayer
from SurvivEHR.src.modules.block import Block

            
class Transformer(nn.Module, ModuleUtilsMixin):
    r"""The bare GPT Model transformer outputting raw hidden-states without any specific head on top.
    
    TODO: ModuleUtilsMixin can be inherited from PreTrainedModel instead later
    
    Encoder only example: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Simplified version of this decoder only model: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L355
    """
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.config.is_decoder = True             # For transformers module internals
        self.embed_dim = config.n_embd    # 512
        layer_norm_epsilon = 1e-5
        
        if config.learn_positional_embedding:
            self.wpe = PositionalEmbedding(config.block_size, self.embed_dim)
        else:
            self.wpe = PositionalEncoding(encoding_dim=self.embed_dim, max_length=config.block_size)
        self.wte = nn.Embedding(vocab_size, self.embed_dim, padding_idx=0)
        self.drop = torch.nn.Dropout(p=config.dropout) if config.dropout is not None else None      # embed dropout
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=layer_norm_epsilon)

        # init all weights  
        self.apply(self._init_weights)       # replace with post_init inherited from PreTrainedModel
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, 
                tokens: torch.tensor, 
                attention_mask: Optional[torch.tensor] = None,
                **kwargs
               ):
        """

        ARGS:
            tokens: 
                Tensor, shape ``[bsz, seq_len]``
        
        KWARGS:
            attention_mask:
                Optional[torch.tensor], shape ``[bsz, seq_len]``

        RETURNS:
            hidden_states:
            Tensor, shape ``[bsz, seq_len, embed_dim]``
        
        
        return:
        """
        bsz, seq_len = tokens.size()
        assert seq_len <= self.config.block_size, f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"

        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(attention_mask, tokens.shape)

        # Get token embeddings
        tok_emb = self.wte(tokens)                         # token embeddings of shape (bsz, seq_len, embed_dim)
        # Get positional embeddings/encodings
        pos_emb = self.wpe(tokens=tokens)                  # positional embeddings of shape (bsz or 1, seq_len, embed_dim)
        # Combine (broadcasts in some choices of encodings)
        x = tok_emb + pos_emb
        
        if self.drop is not None:
            x = self.drop(x)
            
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        
        return x
