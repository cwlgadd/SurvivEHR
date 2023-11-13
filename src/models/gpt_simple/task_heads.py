import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from CPRD.src.models.gpt_simple.transformer import GPTModel

from typing import Optional
import logging

class GPTModelForCausalLM(nn.Module):
    r"""    
    The GPT Neo Model transformer with a large language modeling head on top
    """
    
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        
        self.transformer = GPTModel(config, vocab_size)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

        # weight tying on embedding and softmax layer. See https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # initialise all the weights
        self.apply(self.transformer._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
         
        # report number of parameters
        # print("Number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    
    def forward(self, 
                tokens: torch.tensor, 
                positions: Optional[torch.tensor] = None,
                ages: Optional[torch.tensor] = None,
                attention_mask: Optional[torch.tensor] = None,
                targets: Optional[torch.tensor] = None):
        
        x = self.transformer(tokens=tokens, positions=positions, ages=ages, attention_mask=attention_mask)
        
        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.contiguous().view(B*T, C)
            targets = targets.contiguous().view(B*T)
            loss = F.cross_entropy(logits, targets)
            # if we are given some desired targets also calculate the loss
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def generate(self, 
                 tokens: torch.tensor,
                 eos_token: Optional[int] = None,               # add this later
                 positions: Optional[torch.tensor] = None,
                 ages: Optional[torch.tensor] = None,
                 max_new_tokens: int = 50, 
                 default_age_interval: int = 50):
        """ Generate future samples.
        
            if using age at event in the positional encoder, we are sampling at an interval of one year.
        """
        
        if np.any([_pos_config in self.config.pos_encoding.lower().split("-") for _pos_config in ["temporal"]]):
            logging.warning(f"""Using positional {self.config.pos_encoding} requires ages, 
                                but this head has no way of sampling age at next event.
                                Using {default_age_interval} days as intervals""")
        
        
        # tokens is (B, T) array of indices in the current context
        if positions is None:
            positions = torch.arange(tokens.shape[1]).tile((tokens.shape[0], 1)).to(tokens.device)              # [bsz, seq_len]
        if ages is None:
            default_age_interval = 28
            ages = torch.arange(tokens.shape[1]).tile((tokens.shape[0], 1)).to(tokens.device) * default_age_interval   # [bsz, seq_len]
            ages += 20*365
        
        for _ in range(max_new_tokens):
            # crop tokens to the last block_size tokens
            tokens_window = tokens[:, -self.config.block_size:]
            positions_window = positions[:, -self.config.block_size:] 
            ages_window = ages[:, -self.config.block_size:] 
            # get the predictions
            logits, loss = self(tokens_window, positions=positions_window, ages=ages_window)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            token_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            tokens = torch.cat((tokens, token_next), dim=1) # (B, T+1)
            positions = torch.cat((positions, positions[:, [-1]]+1), dim=1) 
            ages = torch.cat((ages, ages[:, [-1]]+default_age_interval), dim=1) 
            
            # if token_next == eos_token:
            #     raise NotImplementedError
            #     break
            
        return tokens, positions, ages
    

# class GPTModelForCausalSurv(GPTPreTrainedModel):
#     r"""    
#     The GPT Neo Model transformer with a competing risk survival modeling head on top 
#     """
    
#     def __init__(self, config):
#         super().__init__(config)
#         self.transformer = GPTNeoModel(config)

#         raise NotImplementedError

        
def test_clm():
    """ Test model on a simple language generation task
    
    note: Would be nice to also test temporal positional encoding at this stage? Is there a dataset for simple language modelling where time is included. E.g. accounting for pauses in speech.
          Could also just model a time series dataset to test it
    """
    raise NotImplementedError
    

def test_slm():
    """ Test model with survival head
    """
    raise NotImplementedError
    

    
if __name__ == "__main__":
    
    test_llm()
    test_surv()