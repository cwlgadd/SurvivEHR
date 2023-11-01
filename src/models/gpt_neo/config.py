from dataclasses import dataclass

# @dataclass
# class GPTCNeoConfig:
#     block_size: int = 1024
#     n_layer: int = 12
#     n_head: int = 12
#     n_embd: int = 768
#     resid_dropout: float = 0.0
#     # bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
#     attention_type: str = "global"
    
@dataclass
class GPTCNeoConfig:
    block_size: int = 32
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    resid_dropout: float = 0.0
    bias: bool = True
    attention_type: str = "global"
