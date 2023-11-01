from torch import nn
from typing import Optional
import logging 


class MLP(nn.Module):
    """
    architecture from: https://github.com/karpathy/nanoGPT/blob/master/model.py
    """
    def __init__(self, config):
        """
        intermediate_size: default to 4 * hidden_size
        """
        super().__init__()
        hidden_size = config.n_embd * 4
        
        self.c_fc    = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
        self.acti    = nn.ReLU()   # GELU
        self.c_proj  = nn.Linear(hidden_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(self, x):
        x = self.c_fc(x)
        x = self.acti(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
    
def test_nano():
    from CPRD.src.models.gpt_neo.config import GPTCNeoConfig
    import torch

    cfg = GPTCNeoConfig()
    MLP_layer = MLP(cfg)
    
    h = torch.rand((64, cfg.n_embd))
    h = MLP_layer(h)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
        
    test_nano()