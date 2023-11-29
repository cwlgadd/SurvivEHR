import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from CPRD.src.models.transformer.base import Transformer

from typing import Optional
import logging

class TransformerForCausalLM(nn.Module):
    r"""    
    The GPT Neo Model transformer with a large language modeling head on top
    """
    
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        
        self.transformer = Transformer(config, vocab_size)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

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
                attention_mask: Optional[torch.tensor] = None,
                is_generation: bool = False
                ):
        r"""

        ARGS:
            tokens (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Tokens for language modeling. Indices are selected in `[0, ..., config.vocab_size]`, including the padding index
                which defaults to 0 in the accompanying data module. These are not ignored (masked) by default and you should also 
                pass the `attention_mask`. With the attention mask the loss is only computed for labels in `[0, ..., config.vocab_size]`

        KWARGS:
            attention_mask:

            is_generation:

        Note:
        We have no way of computing the loss for the final element of the sequence as we have no subsequent target.
          In the original versions we passed a shifted set of targets, but now we internally shift the block  
          for generative modelling tasks. Therefore, we must remove the final sequence element's hidden state
          (as this has no target). There are two approaches here
        We could (1) just predict the final seq_len - 1 tokens. In this case, the first element is not included as a target in the loss. 
          e.g. see https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L981
        Or we can (2) prepend the hidden states with the zero vector. In this case the decoding head learns to predict the first token from the zero vector. 
          e.g. see https://github.com/mmcdermott/EventStreamGPT/blob/0008ef47141f86eb496bc8db5a42701728e3f662/EventStream/transformer/conditionally_independent_model.py#L84
        """
        hidden_states = self.transformer(tokens=tokens, attention_mask=attention_mask)       # shape: (bsz, seq_len, n_embd)
        
        if not is_generation:
            # if we are not generating samples, we calculate the loss

            # TODO: Test which form is better
            if True: 
                logits = self.lm_head(hidden_states)                                 # shape: (bsz, seq_len, n_embd)
                logits = logits[:, :-1, :].contiguous()                              # shape: (bsz, seq_len - 1, n_embd)
                targets = torch.where(attention_mask[:, 1:] == 1, tokens[:, 1:], -100) if attention_mask is not None else tokens[:, 1:]
            else:
                hidden_states = torch.cat((
                    torch.zeros_like(hidden_states[:, 0, :]).unsqueeze(1),
                    hidden_states[:, :-1, :]
                    ), dim=1)
                logits = self.lm_head(hidden_states)                                 # shape: (bsz, seq_len, n_embd)
                targets = torch.where(attention_mask == 1, tokens, -100) if attention_mask is not None else tokens
            # setting masked target tokens to -100 which is the default index to be ignored in F.cross_entropy

            B, T, C = logits.shape
            logits = logits.contiguous().view(B*T, C)
            targets = targets.contiguous().view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=-100)
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(hidden_states[:, [-1], :])    # note: using list [-1] to preserve the seq_len dim
            loss = None

        return logits, loss
    
    def generate(self, 
                 tokens: torch.tensor,
                 # eos_token: Optional[int] = None,               # add this later
                 max_new_tokens: int = 50, 
                 ):
        """ Generate future samples.
        
            if using age at event in the positional encoder, we are sampling at an interval of one year.
        """
        
        for _ in range(max_new_tokens):
            # crop tokens to the last block_size tokens
            tokens_window = tokens[:, -self.config.block_size:]

            # get the predictions
            logits, _ = self(tokens=tokens_window, is_generation=True)     # shape: (bsz, 1, n_embd)
            
            # focus only on the last time step (though we only forwarded last step through head anyway)
            logits = logits[:, -1, :] 
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # sample from the distribution
            token_next = torch.multinomial(probs, num_samples=1)           # shape: (bsz, 1)
            
            # append generated samples to the running sequence
            tokens = torch.cat((tokens, token_next), dim=1)                # shape: (bsz, num_tokens + 1)
            
            # if token_next == eos_token:
            #     raise NotImplementedError
            #     break
            
        return tokens
        
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