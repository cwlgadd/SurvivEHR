import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
import logging
import time
import pickle as pkl
from tqdm import tqdm
import pandas as pd

from FastEHR.dataloader import FoundationalDataModule
from FastEHR.adapters.BEHRT import BehrtDFBuilder


if __name__ == "__main__":
    
    logging.disable(logging.CRITICAL)
    torch.manual_seed(1337)
    
    logging.basicConfig(level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}.")

    # load the configuration file, override any settings 
    with initialize(version_base=None, config_path="../../../modelling/SurvivEHR/confs", job_name="dataset_creation_notebook"):
        cfg = compose(config_name="config_CompetingRisk11M", overrides=[])
    
    # Create new dataset 
    cfg.data.path_to_ds = "/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/FineTune_MultiMorbidity50+/"
    # Removing windowing applied to SurvivEHR by default so BEHRT can set it's own windowing
    cfg.transformer.block_size = 1e6
    
    print(OmegaConf.to_yaml(cfg))

    # Build 
    dm = FoundationalDataModule(
        path_to_db=cfg.data.path_to_db,
        path_to_ds=cfg.data.path_to_ds,
        load=True,
        overwrite_practice_ids = "/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/PreTrain/practice_id_splits.pickle",
        overwrite_meta_information=cfg.data.meta_information_path,
        num_threads=12,
        supervised=True,
        adapter="BEHRT",
        subsample_training=20000,
    )
    
    vocab_size = dm.train_set.tokenizer.vocab_size
    print(f"{len(dm.train_set)} training patients")
    print(f"{len(dm.val_set)} validation patients")
    print(f"{len(dm.test_set)} test patients")
    print(f"{vocab_size} vocab elements")
    print(dm.adapter.tokenizer)

    # Save built tokenizer to file
    bert_vocab = {'token2idx': dm.adapter.tokenizer}
    with open(cfg.data.path_to_ds + "BEHRT/token2idx.pkl", "wb") as f:
        pkl.dump(bert_vocab, f)

    # Create pandas DataFrame datasets
    for dataloader, split in zip([dm.train_dataloader(), dm.test_dataloader(), dm.val_dataloader()],
                                 ["train", "test", "val"],
                                 ):
        builder = BehrtDFBuilder(
            token_map=dm.adapter.tokenizer,
            pad_token="PAD",
            class_token="CLS",
            sep_token="SEP",
            id_prefix="P",
            zfill=7,
            min_seq_len=2,
        )
        
        chunks = []
        for idx_batch, batch in tqdm(
            enumerate(dataloader),
            desc=f"Creating BEHRT {split}-dataset",
            total=len(dataloader)
        ):
        
            builder.add_batch(batch["tokens"], batch["ages"])
        
            if idx_batch % 10 == 0:
                df_chunk = builder.flush()
        
                if not df_chunk.empty:
                    chunks.append(df_chunk)
        
        # Final flush
        final_chunk = builder.flush()
        if not final_chunk.empty:
            chunks.append(final_chunk)
        
        # Concatenate all chunks (or return empty df)
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            df.to_parquet(cfg.data.path_to_ds + f"BEHRT/{split}_dataset.parquet", index=False)
        
            print(len(df))
            print(df["patid"][0])
            print(df["caliber_id"][0])
            print(df["age"][0])
        else:
            logging.warning("No valid data")
