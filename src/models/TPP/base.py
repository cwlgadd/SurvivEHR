import math
import os
from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
# from transformers import PreTrainedModel
from transformers.modeling_utils import ModuleUtilsMixin               
from CPRD.src.modules.positions.positional_encoding import TemporalPositionalEncoding
from CPRD.src.modules.data_embeddings.data_embedding_layer import DataEmbeddingLayer
from CPRD.src.modules.block import Block

import logging
from typing import Optional

# class GPTPreTrainedModel(PreTrainedModel):
#     """
#     An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
#     models.
#     """

#     # config_class = GPTNeoConfig
#     # load_tf_weights = load_tf_weights_in_gpt_neo
#     # base_model_prefix = "transformer"
#     # supports_gradient_checkpointing = True
#     # _no_split_modules = ["GPTNeoBlock"]
#     # _skip_keys_device_placement = "past_key_values"
    
#     def __init__(self, *inputs, **kwargs):
#         super().__init__(*inputs, **kwargs)

#     def _init_weights(self, module, init_std=0.1):
#         """Initialize the weights."""
#         if isinstance(module, (nn.Linear,)):
#             # Slightly different from TF versions which use truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=init_std)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=init_std)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def _set_gradient_checkpointing(self, module, gradient_checkpointing_func=None):
#         if isinstance(module, GPTModel):
#             module.gradient_checkpointing_func = gradient_checkpointing_func
#             module.gradient_checkpointing = gradient_checkpointing_func is not None

            
class TPPTransformer(nn.Module, ModuleUtilsMixin):
    r"""The bare GPT Model transformer for modelling time-to-event, outputting raw hidden-states without any specific head on top.
    
    TODO: ModuleUtilsMixin can be inherited from PreTrainedModel instead later
    """
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.config.is_decoder = True             # For transformers module internals
        self.embed_dim = config.n_embd    
        layer_norm_epsilon = 1e-5

        # Data and positional encodings
        self.wpe = TemporalPositionalEncoding(encoding_dim=self.embed_dim)                
        # self.wte = DataEmbeddingLayer(vocab_size, self.embed_dim)
        self.wte = nn.Embedding(vocab_size, self.embed_dim)

        # Define transformer
        self.drop = torch.nn.Dropout(p=config.dropout) if config.dropout is not None else None      # embed dropout
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=layer_norm_epsilon)

        # init all weights  
        # self.apply(self._init_weights)   #  (TODO: does this need to be done here if its called inside headed modules)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, 
                tokens: torch.tensor, 
                ages: torch.tensor,
                attention_mask: Optional[torch.tensor] = None
               ):
        """
        
        tokens: 
            Tensor, shape ``[bsz, seq_len]``
        ages: 
        
        attention_mask:
            Optional[torch.tensor], shape ``[bsz, seq_len]``

        
        targets:
        
        
        return:
        """
        bsz, seq_len = tokens.size()
        assert seq_len <= self.config.block_size, f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"

        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(attention_mask, tokens.shape)
            
        # Get token embeddings
        tok_emb = self.wte(tokens)                         #  shape (bsz, seq_len, embed_dim)
        assert not torch.isnan(tok_emb).any(), f"tok_emb {tok_emb.shape},\n {tokens}, {tok_emb}, {self.wte.weight}"

        # Get positional embeddings/encodings
        pos_emb = self.wpe(tokens=tokens, ages=ages)       # positional embeddings of shape (bsz or 1, seq_len, embed_dim)

        # Combine (broadcasts in some choices of encodings)
        x = tok_emb + pos_emb
        
        if self.drop is not None:
            x = self.drop(x)
            
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        
        return x
