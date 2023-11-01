from torch import nn
from typing import Optional
import logging 
from CPRD.src.modules.self_attention import MultiHeadedSelfAttention
from CPRD.src.modules.mlp import MLP

class Block(nn.Module):
    """
    architecture from: https://github.com/karpathy/nanoGPT/blob/master/model.py
    """
    def __init__(self, config):
        """
        """
        super().__init__()

        # to add to config
        layer_norm_epsilon = 1e-5       # The epsilon used by the layer normalization layers.
        hidden_size = config.n_embd
        attention_type = config.attention_type
        
        self.ln_1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = MultiHeadedSelfAttention(config, attention_type=attention_type)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.mlp = MLP(config)
    
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
            outputs = (hidden_states,) + outputs[1:]

        return hidden_states  # outputs  # hidden_states, present, (attentions, cross_attentions)
    
def test(bsz=1, seq_len=10):
    from CPRD.src.models.gpt_neo.config import GPTCNeoConfig
    import torch

    config = GPTCNeoConfig()
    DecoderBlock = Block(config)
    
    h = torch.rand((bsz, seq_len, config.n_embd))
    h = DecoderBlock(h)
    print(h)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
        
    test()