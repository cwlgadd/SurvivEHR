from collections import defaultdict
from tqdm import tqdm

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"


def setup(causal_language_model, from_trained=True):
    
    tokenizer = AutoTokenizer.from_pretrained(causal_language_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print(tokenizer)
    
    if from_trained:
        model = AutoModelForCausalLM.from_pretrained(causal_language_model).to(torch_device)
    else:
         # Building the config
        config = AutoConfig.from_pretrained(causal_language_model)
        print(config)

        # Building the model from the config
        model = AutoModelForCausalLM.from_config(config).to(torch_device)
    print(model)
    
    # GPT2 was not trained with padding tokens, this has to be added but padded values will not be used due to attention mask
    
    return model, tokenizer


if __name__ == "__main__":
    
    # CLM = "sshleifer/tiny-gpt2"
    CLM = "gpt2"
    
    model, tokenizer = setup(CLM)
    
    
    batch_sequences = ["But what about second breakfast?",
                       "Don't think he knows about second breakfast, Pip.",
                       "What about elevensies?",
                      ]
    for sequence in batch_sequences:
        model_inputs = tokenizer(sequence, return_tensors='pt').to(torch_device)

        print(model_inputs)

        # generate 40 new tokens
        greedy_output = model.generate(**model_inputs, max_new_tokens=40)

        print("Output:\n" + 100 * '-')
        print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
