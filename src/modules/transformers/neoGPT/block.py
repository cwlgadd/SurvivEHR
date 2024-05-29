# Following architecture from https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py
import torch
from torch import nn
from typing import Optional
import logging 
from CPRD.src.modules.transformers.neoGPT.self_attention import MultiHeadedSelfAttention


class MLP(nn.Module):
    """
    architecture from: https://github.com/karpathy/nanoGPT/blob/master/model.py
    """
    def __init__(self, cfg):
        """
        intermediate_size: default to 4 * hidden_size
        """
        super().__init__()
        hidden_size = cfg.transformer.n_embd * 4
        
        self.c_fc    = nn.Linear(cfg.transformer.n_embd, hidden_size, bias=cfg.transformer.bias)
        self.acti    = nn.ReLU()   # GELU
        self.c_proj  = nn.Linear(hidden_size, cfg.transformer.n_embd, bias=cfg.transformer.bias)
        self.dropout = nn.Dropout(float(cfg.transformer.dropout))

    def forward(self, x):
        x = self.c_fc(x)
        x = self.acti(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
        
class Block(nn.Module):
    """
    architecture from: https://github.com/karpathy/nanoGPT/blob/master/model.py
    """
    def __init__(self, cfg):
        """
        """
        super().__init__()

        layer_norm_epsilon = 1e-5       # The epsilon used by the layer normalization layers.
        
        self.ln_1 = nn.LayerNorm(cfg.transformer.n_embd, eps=layer_norm_epsilon)
        self.attn = MultiHeadedSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.transformer.n_embd, eps=layer_norm_epsilon)
        self.mlp = MLP(cfg)
    
    def forward(self,
                hidden_states,
                layer_past=None,
                attention_mask=None,
                head_mask=None,
                use_cache=False,
                output_attentions=False,
               ):
        """
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]     # hidden_states, present, (attentions, cross_attentions)

        return hidden_states  
    