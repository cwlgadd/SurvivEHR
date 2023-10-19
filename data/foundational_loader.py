from sklearn.model_selection import train_test_split as sk_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import polars as plr
from abc import ABC
import sqlite3
from typing import Optional
from collections import defaultdict
import os
from CPRD.data.dataset.dataset_polars import EventStreamDataset
from CPRD.data.utils.tokenizers.discrete import DiscreteTokenizer
import random


class FoundationalDataModule(pl.LightningDataModule, ABC):
    r"""
    """
    
    def __init__(self, 
                 identifiers:list,
                 batch_size:int = 512,
                 min_workers:int = 2,
                 weighted_sampler:bool = False,
                 load_event_stream:Optional[str] = None,
                 save_event_stream:Optional[str] = None
                ):
        """
        PyTorch-Lightning datamodule for foundational models
        
        ARGS:
            practice_patient_id (list[str])
                List of practice patient identifiers which satisfy study criteria.
                
        KWARGS:
            batch_size (int): 
                
            min_workers (int):
                
            weighted_sampler (bool):
                NotImplemented. 
            load_event_stream (optional, str):
            
            save_event_stream (optional, str):
            
        """
        
        super(FoundationalDataModule, self).__init__()
        
        self.batch_size = batch_size
        self.min_workers = min_workers
        
        # Get DL friendly representation, either by loading or building from scratch.
        if load_event_stream is not None:
            event_stream = EventStreamDataset()
            event_stream.load(load_event_stream)
        else:
            event_stream = EventStreamDataset()
            event_stream.fit(identifiers, empty_dynamic_strategy="drop", indexing_strategy=None)  
            if save_event_stream is not None:
                event_stream.save(save_event_stream)
                
        # Train/test/validation split cohort
        ##############
        (self.train_ids, self.test_ids, self.val_ids), weight_dict = self.train_test_split(event_stream.identifiers)
        
        # Training, testing, and validation sets
        ##############
        splits = [event_stream.DL_frame.filter(plr.col("PRACTICE_PATIENT_ID").is_in(labels)) for labels in [self.train_ids, self.test_ids, self.val_ids]]
        self.train_set = FoundationalDataset(splits[0], freq_threshold=1e-8)
        self.test_set = FoundationalDataset(splits[1], tokenizer=self.train_set.tokenizer)
        self.val_set = FoundationalDataset(splits[2], tokenizer=self.train_set.tokenizer)
        
        # TODO Weighted random sampler for training set. Naive approach to account for different event sequence lengths. TODO: better way without pre-slicing all the dataset rows
        #    Up-sample larger sequences 
        #############
        if (weight_dict is not None) and weighted_sampler:        
            raise NotImplementedError
        else:        
            self.train_sampler = None
            self.train_shuffle = True
            
    def train_test_split(self, identifiers):
        # Split frame into training, validation, and test
        train_ids, test_ids = sk_split(identifiers, test_size=0.1)
        test_ids, val_ids = sk_split(test_ids, test_size=0.5)

        # Random sampler weights
        # TODO: Add weighted sampler if we later choose to aggregate samples?
        # weight_dict = {}
        # ntrain_unique_samples = len(train_df.index.unique())
        # for cancer_id, group in train_df.groupby('cancer_type'):
        #     unique_samples = len(group.index.unique()) / ntrain_unique_samples
        #     if unique_samples > 0:
        #         weight_dict[cancer_id] = 1 / unique_samples
        weight_dict = None

        return (train_ids, test_ids, val_ids), weight_dict
    
    @staticmethod
    def collate_fn(data:list[dict]):     
        r""" 
        Collect and collate separate dictionaries.
        
        During this operation, pad the sequence lengths to the maximum length seen within the batch and tokenize
        
        # TODO: tokenization could also be pre-processed rather than done on the fly.
        """
        
        # Combine individual dictionaries into one
        #     For dynamic rows (events, values, ages at event, and event types) these become a ragged list of lists.
        allkeys = set().union(*data)
        batch_dict = {k: [d[k] for d in data if k in d] for k in allkeys}
        
        # Pad dymamic columns
        # for k in ["EVENT", "VALUE", "AGE_AT_EVENT", "EVENT_TYPE"]:
        #     batch_dict[k] =  pad_sequence(batch_dict[k], batch_first=True)
        
        print(batch_dict.keys())
        print(type(batch_dict["EVENT"]))
        print(batch_dict["EVENT"][0])
        
        raise NotImplementedError
        
        worker_batch = {"labels": [None for i in range(random.randint(10,15))]}
        # print(worker_batch)
        return worker_batch
    
    def train_dataloader(self):
        return DataLoader(            
            dataset=self.train_set,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            num_workers=np.min((self.min_workers,os.cpu_count())),
            collate_fn=self.collate_fn,
            shuffle=self.train_shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_set,
            batch_size=self.batch_size,
            num_workers=np.min((self.min_workers,os.cpu_count())),
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=np.min((self.min_workers,os.cpu_count())),
            collate_fn=self.collate_fn,
            shuffle=False
        )

    
class FoundationalDataset(Dataset):
    r"""
    """
    
    @property
    def event_frequency(self) -> plr.DataFrame:
        r"""
        Get polars dataframe with three columns: event, count and relative frequencies
        
        Returns 
        ┌──────────────────────────┬─────────┬──────────┐
        │ EVENT                    ┆ counts  ┆ freq     │
        │ ---                      ┆ ---     ┆ ---      │
        │ str                      ┆ u32     ┆ f64      │
        ╞══════════════════════════╪═════════╪══════════╡
        │ <event name 1>           ┆ n1      ┆ p1       │
        │ <event name 2>           ┆ n2      ┆ p2       │
        │ …                        ┆ …       ┆ …        │
        └──────────────────────────┴─────────┴──────────┘
        """
        event_freq = (self.event_stream
                      .select(plr.col("EVENT").explode())
                      .to_series(index=0)
                      .value_counts(sort=True)
                     )                        
        event_freq = event_freq.with_columns((plr.col('counts') / event_freq.select(plr.sum("counts"))).alias('freq'))
        return event_freq
    
    def __init__(self,
                 event_stream,
                 tokenizer:Optional[DiscreteTokenizer] = None,
                 **kwargs
                ):
        super().__init__()
        
        self.event_stream = event_stream        
        
        # Build vocabulary from training set. Vocabularly begins with UNK (unknown) token, and is then ordered by event frequency
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = DiscreteTokenizer()
            self.tokenizer.fit_vocabulary(self.event_frequency, **kwargs)
        print(f"vocab size {self.tokenizer.vocab_size}")
        
        # Pre-process tokenization
        self.tokenize_stream()
        
    def __len__(self):
        # Number of patients after potentially removing some
        return self.event_stream.select(plr.count())["count"][0]
    
    def __getitem__(self, idx):
        # Return event stream and static variables
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(self.event_stream.row(idx, named=True))
        return self.event_stream.row(idx, named=True)
    
    def tokenize_stream(self):
        # tokenize even stream and convert to PyTorch tensors
        
        
        return 
        
if __name__ == "__main__":

    # Report what is in the db
    ###########################
    PATH_TO_DB = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"
    conn = sqlite3.connect(PATH_TO_DB)
    cursor = conn.cursor()
    # Check what tables were built into DB
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Loaded tables {[table[0] for table in tables]}")
    # Report how many entries in each table
    cursor.execute("SELECT COUNT(*) FROM static_table")
    print('\t static_table has', cursor.fetchone()[0], 'records.')
    cursor.execute("SELECT COUNT(*) FROM diagnosis_table")
    print('\t diagnosis_table has', cursor.fetchone()[0], 'records.')
    cursor.execute("SELECT COUNT(*) FROM measurement_table")
    print('\t measurement_table has', cursor.fetchone()[0], 'records.')
    
    # Example of cohort filtering
    if False:
        # Get the list of patients which fit our criterion
        identifiers1 = queries.query_measurement(["bmi", "hydroxyvitamin2", "hydroxyvitamin3"], cursor)
        identifiers2 = queries.query_diagnosis(["HF", "FIBROMYALGIA"], cursor)
        identifiers = list(set(identifiers1).intersection(identifiers2))    # Turn smaller list into the set
    else: 
        cursor.execute("SELECT practice_patient_id FROM static_table")
        identifiers = [ppid[0] for ppid in cursor.fetchall()]
        
    # "SELECT * FROM static_table WHERE PRACTICE_PATIENT_ID=:PRACTICE_PATIENT_ID", {'PRACTICE_PATIENT_ID': identifier}
    foundational_dm = FoundationalDataModule(identifiers=identifiers, batch_size=256)
    # print(foundational_dm.identifiers[:20])
    # print(len(foundational_dm.identifiers))
    # print(len(foundational_dm.train_set))
    # print(len(foundational_dm.test_set))
    # print(len(foundational_dm.val_set))
    
    for idx, batch in enumerate(foundational_dm.train_dataloader()):
        print(f"Batch {idx}")
        print(batch)
