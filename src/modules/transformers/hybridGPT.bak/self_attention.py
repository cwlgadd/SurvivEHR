# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py
import torch
from torch import nn
import logging

class MultiHeadedSelfAttention(nn.Module):
    r"""
    Causal multi-headed self attention block
    
    Batching heads for efficiency
    """
    def __init__(self, 
                 cfg,
                 attention_type: str = "global"
                ):
        
        super().__init__()
        
        # config (add to hydra later)
        max_positions = 512   #  The maximum sequence length that this model might ever be used with (block size)
                              #    Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        window_size = 256     #  The size of the sliding window for local attention
        # attention_dropout = 0.0   # The dropout ratio for the attention probabilities.
        # resid_dropout = config.resid_dropout    # Residual dropout used in the attention pattern.
        dropout = cfg.transformer.dropout
        self.embed_dim = cfg.transformer.n_embd          #  Dimensionality of the encoder layers and the pooler layer.
        self.num_heads = cfg.transformer.n_head          # Number of attention heads for each attention layer in the Transformer encoder.
        
        # TODO: flash attention improves speed, but support is only in PyTorch >= 2.0: see https://github.com/karpathy/nanoGPT/blob/master/model.py
        # However, modules supported on bear currently doesn't allow so I'm not supporting yet.
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if self.flash:
            logging.warning("TODO: Flash attention is available on this system but not implemented.")
        
        bias = torch.tril(torch.ones((max_positions, max_positions), dtype=bool)).view(1, 1, max_positions, max_positions)
        # local causal self attention is a sliding window where each token can only attend to the previous
        # window_size tokens. This is implemented by updating the causal mask such that for each token
        # all other tokens are masked except the previous window_size tokens.
        if attention_type == "local":
            bias = torch.bitwise_xor(bias, torch.tril(bias, -window_size))
        elif attention_type == "global":
            # Just calculate attention over the full block
            pass
        elif attention_type == "sparse":
            # TODO: add sparse attention. This will be the last record of each token,
            #       and longer ranges with specified tokens (e.g. diagnoses)
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        self.register_buffer("bias", bias, persistent=False)
        # idk what this does in the hugging face 
        # self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)   
        
        self.attn_dropout = nn.Dropout(float(dropout))
        self.resid_dropout = nn.Dropout(float(dropout))
        
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} " +
                             f"and `num_heads`: {self.num_heads}).")

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits embed_dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into embed_dim
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(self,
                hidden_states,
                attention_mask=None,
                layer_past=None,
                head_mask=None,
                use_cache=False,
                output_attentions=False,
               ):
        """
        use_cache:   GPT Cache is a system that enhances the performance and efficiency of language models by incorporating caching mechanisms. 
                     It aims to optimize the retrieval process of relevant information by storing precomputed embeddings and their corresponding similar vectors.
        """
        
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
            raise NotImplementedError # Not tested

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

def test(bsz=1, seq_len=10, embed_dim=2048):
    attention_head = MultiHeadedSelfAttention()
    print(attention_head)
    
    test_hidden = torch.rand((bsz, seq_len, embed_dim))
    a, present, atns = attention_head(test_hidden, use_cache=False, output_attentions=True)
    print(a)
    print(present)
    print(atns)
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test()