# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py
import torch
from torch import nn
import math
import logging

class MultiHeadedSelfAttention(nn.Module):
    r"""
    Causal multi-headed self attention block
    
    Batching heads for efficiency
    """
    def __init__(self,
                 cfg
                ):
        
        super().__init__()
        
        assert cfg.transformer.n_embd % cfg.transformer.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(cfg.transformer.n_embd, 3 * cfg.transformer.n_embd, bias=cfg.transformer.bias)
        # output projection
        self.c_proj = nn.Linear(cfg.transformer.n_embd, cfg.transformer.n_embd, bias=cfg.transformer.bias)
        # regularization
        self.attn_dropout = nn.Dropout(cfg.transformer.attention_dropout)
        self.resid_dropout = nn.Dropout(cfg.transformer.resid_dropout)
        self.n_head = cfg.transformer.n_head
        self.n_embd = cfg.transformer.n_embd
        self.dropout = cfg.transformer.dropout

        # max_positions = cfg.transformer.max_positions   #  The maximum sequence length that this model might ever be used with (block size), typically set this large
        # bias = torch.tril(torch.ones((max_positions, max_positions), dtype=bool)).view(1, 1, max_positions, max_positions)
        # # local causal self attention is a sliding window where each token can only attend to the previous
        # # window_size tokens. This is implemented by updating the causal mask such that for each token
        # # all other tokens are masked except the previous window_size tokens.
        if cfg.transformer.attention_type == "local":
            raise NotImplementedError(f"{cfg.transformer.attention_type} attention not supported in NanoGPT implementation of self-attention")
        #     bias = torch.bitwise_xor(bias, torch.tril(bias, -cfg.transformer.window_size))
        # # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("bias", bias, persistent=False)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x, attention_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q.to(torch.float16), 
                                                                 k.to(torch.float16), 
                                                                 v.to(torch.float16), 
                                                                 attn_mask=attention_mask.bool(), 
                                                                 dropout_p=self.dropout if self.training else 0
                                                                ).to(torch.float32)
            
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(attention_mask == 0, float('-inf'))
            att = torch.nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y