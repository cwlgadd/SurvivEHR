import math
import os
from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transformers import PreTrainedModel
from CPRD.src.modules.positional_encodings import PositionalEncoding as PositionalEncoding
from CPRD.src.modules.block import Block


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


            
class GPTModel(nn.Module):
    r"""The bare GPT Model transformer outputting raw hidden-states without any specific head on top.
    """
    # Encoder only example: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    # Simplified version of this decoder only model: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L355

    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        
        # config (add to hydra later)
        self.embed_dim = config.n_embd    # 512
        nhead = config.n_head
        # embedding_type = "temporal"
        layer_norm_epsilon = 1e-5
        self.learn_position_encoding = config.learn_position_encoding
        
        if self.learn_position_encoding:
            wpe = nn.Embedding(config.block_size, self.embed_dim)
        else:
            wpe = PositionalEncoding(embedding_dim=self.embed_dim, dropout=0.0)
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, self.embed_dim),
            wpe = wpe,
            blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(self.embed_dim, eps=layer_norm_epsilon),
        ))

        self.lm_head = nn.Linear(config.n_embd, vocab_size)
        # self.init_weights()

    def forward(self, idx, targets=None, t=None):
        """
        
        idx:
        
        t:
        
        targets:
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        
        if self.learn_position_encoding:
            tok_emb = self.transformer.wte(idx) # (B,T,C)
            pos_emb = self.transformer.wpe(torch.arange(t, device=device)) # (T,C)
            x = tok_emb + pos_emb
        else:
            tok_emb = self.transformer.wte(idx)     # token embeddings of shape (b, t, n_embd)
            x = self.transformer.wpe(tok_emb, t) # position embeddings of shape (t, n_embd)
        
        x = self.transformer.blocks(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            # if we are given some desired targets also calculate the loss
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -16:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def test():
    from CPRD.src.models.gpt_neo.config import GPTCNeoConfig
    import torch

    cfg = GPTCNeoConfig()
    model = GPTModel(cfg)
    print(model)
    
    token_indices = torch.arange(128).reshape((2,-1))
    print(token_indices.shape)
    logits, loss = model(token_indices)
    print(logits.shape)
    print(loss)
    
if __name__ == "__main__":
    
    test()