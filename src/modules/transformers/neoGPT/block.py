# Following architecture from https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py
import torch
from torch import nn
from typing import Optional
import logging 
from CPRD.src.modules.transformers.neoGPT.self_attention import MultiHeadedSelfAttention

class Adapter(nn.Module):
    """
    Fine-tuning adapter based on: "Parameter-Efficient Transfer Learning for NLP" - https://arxiv.org/pdf/1902.00751
    """
    def __init__(self, input_dim, hidden_dim):
        super(Adapter, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self._init_identity()

    def forward(self, x):
        # X shape: (bsz, seq_len, n_embd)
        assert len(x.shape) == 3
        input_x = x.clone()
        
        x_flat = x.view(x.shape[0]*x.shape[1], self.input_dim)
        x_flat = self.proj(x_flat)
        
        return input_x + x_flat.view(x.shape[0], x.shape[1], self.input_dim)   # Skip connection

    def _init_identity(self):
        """
        Make proj(x) = 0 for all x by initializing
        all weights and biases to zero.
        """
        # First linear layer
        self.proj[0].weight.data.zero_()
        self.proj[0].bias.data.zero_()
        # Second linear layer
        self.proj[2].weight.data.zero_()
        self.proj[2].bias.data.zero_()
        

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
    def __init__(self, cfg, use_adapter=False):
        """
        use_adapter: Bool or integer type
        """
        super().__init__()

        layer_norm_epsilon = 1e-5       # The epsilon used by the layer normalization layers.
        
        self.ln_1 = nn.LayerNorm(cfg.transformer.n_embd, eps=layer_norm_epsilon)
        self.attn = MultiHeadedSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.transformer.n_embd, eps=layer_norm_epsilon)
        self.mlp = MLP(cfg)

        # Adapter
        self.use_adapter = use_adapter
        if self.use_adapter:
            adapter_dim = 128 if self.use_adapter is True else self.use_adapter
            self.adapter_1 = Adapter(input_dim=cfg.transformer.n_embd, hidden_dim=adapter_dim)
            self.adapter_2 = Adapter(input_dim=cfg.transformer.n_embd, hidden_dim=adapter_dim)
    
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

        if self.use_adapter:
            attn_output = self.adapter_1(attn_output)
        
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)

        if self.use_adapter:
            feed_forward_hidden_states = self.adapter_2(feed_forward_hidden_states)
        
        # residual connection
        hidden_states = feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]     # hidden_states, present, (attentions, cross_attentions)

        return hidden_states  
    