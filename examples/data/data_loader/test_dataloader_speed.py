import pytorch_lightning 
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import sqlite3
from dataclasses import dataclass
import logging
from CPRD.data.foundational_loader import FoundationalDataModule
from CPRD.src.models.benchmarks.karpathy_gpt.transformer import GPTLanguageModel
from CPRD.src.models.transformer.task_heads.causal_lm import TransformerForCausalLM
from tqdm import tqdm
import time

if __name__ == "__main__":
    
    torch.manual_seed(1337)
    logging.basicConfig(level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    # Get a list of patients which fit a reduced set of criterion
    path_to_db = "/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/archive/Version2/"
    
    # Build 
    dm = FoundationalDataModule(path_to_db=path_to_db,
                                load=False,
                                tokenizer="non-tabular",
                                batch_size=64,
                                max_seq_length=256,
                                unk_freq_threshold=0,
                                include_measurements=True,
                                # drop_missing_data=True,
                                include_diagnoses=True,
                                # drop_empty_dynamic=True,
                                min_workers=4
                               )
    
    vocab_size = dm.train_set.tokenizer.vocab_size
    
    print(f"{len(dm.train_set)} training patients")
    print(f"{len(dm.val_set)} validation patients")
    print(f"{len(dm.test_set)} test patients")
    print(f"{vocab_size} vocab elements")

    start = time.time()   # starting time
    for row_idx, row in enumerate(dm.train_set):
        print(time.time() - start)
        start = time.time()
        if row_idx > opt.batch_size - 1:
            break
    print(f"{row}")