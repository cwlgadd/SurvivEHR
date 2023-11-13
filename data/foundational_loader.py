from sklearn.model_selection import train_test_split as sk_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import polars as plr
from abc import ABC
from textwrap import wrap
from typing import Optional
from collections import defaultdict
import os
from CPRD.data.dataset.dataset_polars import EventStreamDataset
from CPRD.data.utils.tokenizers.tokenizer import TokenizerBase as Tokenizer
import random

# Testing modules
from CPRD.data.database import queries
import sqlite3


class FoundationalDataModule(pl.LightningDataModule, ABC):
    r"""
    """
    
    def __init__(self, 
                 identifiers:list,
                 max_seq_length:int = 256,
                 batch_size:int = 512,
                 unk_freq_threshold=1e-2,
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
            
            unk_freq_threshold (float). 
                Value between 0 and 1, controlling at what level of frequency rare tokens (equiv. conditions/measurements 
                with this tokenizer) are mapped to the UNK token. Used to reduce vocabulary size
                
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
        self.train_set = FoundationalDataset(splits[0], 
                                             max_seq_length=max_seq_length,
                                             freq_threshold=unk_freq_threshold)
        self.test_set = FoundationalDataset(splits[1],
                                            max_seq_length=max_seq_length, 
                                            tokenizer=self.train_set.tokenizer)
        self.val_set = FoundationalDataset(splits[2],
                                           max_seq_length=max_seq_length, 
                                           tokenizer=self.train_set.tokenizer)
                
        # TODO Weighted random sampler for training set. Naive approach to account for different event sequence lengths.
        # TODO: better way without pre-slicing all the dataset rows?
        #    Up-sample larger sequences?
        #############
        if (weight_dict is not None) and weighted_sampler:        
            raise NotImplementedError
        else:        
            self.train_sampler = None
            self.train_shuffle = True
        
        # print(f"Tokenizer description: \n {self.train_set.tokenizer.fit_description}")

            
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
    
    def decode(self, sequence:list[int]):
        return self.train_set.tokenizer.decode(sequence)
    
    def encode(self, sequence:list[str]):
        return self.train_set.tokenizer.encode(sequence)
    
    @staticmethod
    def collate_fn(data:list[dict]):     
        r""" 
        Collect and collate separate dictionaries.
        
        During this operation, pad the sequence lengths to the maximum length seen within the batch and tokenize
        
        """
        # Combine individual dictionaries into one
        #     For dynamic rows (events, values, ages at event, and event types) these become a ragged list of lists.
        allkeys = set().union(*data)
        
        batch_dict = {k: [d[k] for d in data if k in d] for k in allkeys}
        
        batch_dict["attention_mask"] = [torch.ones_like(d) for d in batch_dict["input_ids"]]

#         print(batch_dict["input_ids"][0])
#         print(batch_dict["input_positions"][0].shape)
#         print(batch_dict["attention_mask"][0])
#         print(f"New: \n IDs:{batch_dict['input_ids'][0]} \nMask:{batch_dict['attention_mask'][0]}")
        
        worker_batch = {"input_ids": pad_sequence(batch_dict["input_ids"]).T,
                        "target_ids": pad_sequence(batch_dict["target_ids"]).T,
                        "input_pos": pad_sequence(batch_dict["input_pos"]).T,
                        "target_pos": pad_sequence(batch_dict["target_pos"]).T,
                        "input_ages": pad_sequence(batch_dict["input_ages"]).T,
                        "target_ages": pad_sequence(batch_dict["target_ages"]).T,
                        "attention_mask": pad_sequence(batch_dict["attention_mask"]).T
                       }
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
            dataset=self.val_set,
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
                 max_seq_length:int = 256,
                 tokenizer:Optional[Tokenizer] = None,
                 **kwargs
                ):
        super().__init__()
        
        self.event_stream = event_stream    
        self.max_seq_length = max_seq_length
        
        # Build vocabulary from training set. Vocabularly begins with UNK (unknown) token, and is then ordered by event frequency
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer()
            self.tokenizer.fit(self.event_frequency, **kwargs)
        
        # TODO: Pre-process tokenization
        # self.tokenize_stream()
        
    def __len__(self):
        # Number of patients after potentially removing some
        return self.event_stream.select(plr.count())["count"][0]
    
    def __getitem__(self, idx):
        r"""
        Return tokenized and padded event stream and static variables
        
        # TODO: tokenization could also be pre-processed rather than done on the fly.        
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get row from the DL representation frame
        row_dict = self.event_stream.row(idx, named=True)
        
        # Zip together events and their values. When no value (None) do not include it. 
        #    Inside of this, split value by character
        #    Delimeter everything with a comma
        sequence_tokens = ",".join([f"{_e},{','.join(wrap(str(_v),1))}"  if _v is not None else str(_e) for _e, _v in zip(row_dict["EVENT"], row_dict["VALUE"])])
        sequence_ages = ",".join([",".join([str(_a)]*(1+len(str(_v)))) if _v is not None else str(_a) for _a, _v in zip(row_dict["AGE_AT_EVENT"], row_dict["VALUE"])])
        
        # Then, turn sequence into a list, splitting via delimeter, and encode        
        encoded_sequence = self.tokenizer.encode(sequence_tokens.split(","))
        sequence_pos = np.arange(len(encoded_sequence))
        sequence_ages = [int(_a) for _a in sequence_ages.split(",")]
        
        # Sub-sample a maximum number of tokens
        if len(encoded_sequence) > self.max_seq_length + 1:
            start_pos = np.random.randint(low=0, high=len(encoded_sequence)-self.max_seq_length-1, size=1)[0]
            sequence_pos = np.arange(start_pos, start_pos+self.max_seq_length+1)
            encoded_sequence = encoded_sequence[start_pos:start_pos+self.max_seq_length+1]            
            sequence_ages = sequence_ages[start_pos:start_pos+self.max_seq_length+1]

        # stagger and split for input and targets
        encoded_sequence_in = encoded_sequence[:-1]
        encoded_sequence_out = encoded_sequence[1:]

        sequence_pos_in = sequence_pos[:-1]
        sequence_pos_out = sequence_pos[1:]
        
        sequence_ages_in = sequence_ages[:-1]
        sequence_ages_out = sequence_ages[1:]
            
        # print(len(sequence_tokens.split(",")))
        # print(len(sequence_ages.split(",")))
        # print(encoded_sequence)
        
        return {"identifier": row_dict["PRACTICE_PATIENT_ID"],
                "sex": row_dict["SEX"],
                "ethnicity": row_dict["ETHNICITY"],
                "year_of_birth": row_dict["YEAR_OF_BIRTH"],
                "input_ids": torch.tensor(encoded_sequence_in),
                "input_pos": torch.tensor(sequence_pos_in),                
                "input_ages": torch.tensor(sequence_ages_in),
                "target_ids": torch.tensor(encoded_sequence_out),
                "target_pos": torch.tensor(sequence_pos_out),                
                "target_ages": torch.tensor(sequence_ages_out),
               }
    
    
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
        
    foundational_dm = FoundationalDataModule(identifiers=identifiers, batch_size=256)
    
    for idx, batch in enumerate(foundational_dm.train_dataloader()):
        break
    print(batch["input_positions"])
    print(batch["attention_mask"])
    model_inputs = batch["input_ids"].numpy()
    print(model_inputs)
    print(type(model_inputs))

    decoded_sequences = [foundational_dm.decode([model_inputs[j, i] for j in range(model_inputs.shape[0])]) 
                         for i in range(model_inputs.shape[1])]
    print(decoded_sequences[0])
