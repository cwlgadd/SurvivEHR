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
from CPRD.data.tokenizers.base import TokenizerBase
from CPRD.data.tokenizers.tokenizers import NonTabular, Tabular
import random
import logging

# Testing modules
from CPRD.data.database import queries
import sqlite3


class FoundationalDataModule(pl.LightningDataModule, ABC):
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

        load_event_stream (optional, str):
        
        save_event_stream (optional, str):
        
    """
    
    def __init__(self, 
                 identifiers:list,
                 tokenizer:str = "tabular",
                 max_seq_length:int = 256,
                 batch_size:int = 512,
                 unk_freq_threshold=1e-2,
                 min_workers:int = 2,
                 load_event_stream:Optional[str] = None,
                 save_event_stream:Optional[str] = None,
                 include_diagnoses: bool = True,
                 include_measurements: bool = True,        
                 preprocess_measurements: bool = True
                ):
       
        
        super(FoundationalDataModule, self).__init__()
        
        self.batch_size = batch_size
        self.min_workers = min_workers 
        use_weighted_sampler = False     # Not implemented
        empty_dynamic_strategy = "drop"
        indexing_strategy = None
        
        # Get the DL friendly representation, either by loading or building from scratch.
        event_stream = EventStreamDataset()
        if load_event_stream is not None:
            event_stream.load(load_event_stream)
        else:
            event_stream.fit(identifiers, 
                             empty_dynamic_strategy=empty_dynamic_strategy,
                             indexing_strategy=indexing_strategy,
                             include_diagnoses=include_diagnoses,
                             include_measurements=include_measurements,
                             preprocess_measurements=preprocess_measurements)
            if save_event_stream is not None:
                event_stream.save(save_event_stream)
        self.standardisation_dict = event_stream.standardisation_dict
                
        # Train/test/validation splits on cohort
        (train_split, test_split, val_split), weight_dict = self._train_test_val_split(event_stream)        

        # Create tokenizer
        match tokenizer:
            case "tabular":
                logging.info("Using tabular tokenizer")
                self.tokenizer = Tabular()
            case "non-tabular":
                logging.info("Using non-tabular tokenizer")
                self.tokenizer = NonTabular()
            case _:
                logging.warning(f"Tokenizer {tokenizer} doesn't exist")
        # and build based on vocabulary from training set. 
        #   Vocabularly begins with the PAD (padding) token, then the UNK (unknown) token, and is then ordered by token frequency
        self.tokenizer.fit(train_split, freq_threshold=unk_freq_threshold)
        logging.debug(self.tokenizer._stoi)
        logging.debug(self.tokenizer._itos)
        
        # Train/test/validation GenerativeDatasets
        [self.train_set, self.test_set, self.val_set] = [FoundationalDataset(split, 
                                                                             max_seq_length=max_seq_length,
                                                                             tokenizer=self.tokenizer)
                                                         for split in [train_split, test_split, val_split]]
                
        # TODO Weighted random sampler for training set. Naive approach to account for different event sequence lengths.
        # TODO: better way without pre-slicing all the dataset rows?
        #    Up-sample larger sequences?
        #############
        if (weight_dict is not None) and use_weighted_sampler:        
            raise NotImplementedError
        else:        
            self.train_sampler = None
            self.train_shuffle = True
        
        # print(f"Tokenizer description: \n {self.train_set.tokenizer.fit_description}")

    def _train_test_val_split(self, event_stream):
        # Split identifiers into training, validation, and test
        train_ids, test_ids = sk_split(event_stream.identifiers, test_size=0.1)
        test_ids, val_ids = sk_split(test_ids, test_size=0.5)
        # Split frame into training, validation, and test
        splits = [event_stream.DL_frame.filter(plr.col("PRACTICE_PATIENT_ID").is_in(labels)) 
                  for labels in [train_ids, test_ids, val_ids]]
        weight_dict = None
        return (splits[0], splits[1], splits[2]), weight_dict
    
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
        
        batch_dict["attention_mask"] = [torch.ones_like(d) for d in batch_dict["tokens"]]
        
        worker_batch = {
            "tokens": pad_sequence(batch_dict["tokens"], padding_value=0).T,
            "ages": pad_sequence(batch_dict["ages"]).T,
            "values": pad_sequence(batch_dict["values"], padding_value=torch.nan).T,
            # "target_tokens": pad_sequence(batch_dict["target_tokens"], padding_value=0).T,
            # "target_ages": pad_sequence(batch_dict["target_ages"]).T,
            # "target_values": pad_sequence(batch_dict["target_values"], padding_value=torch.nan).T,
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
    def __init__(self,
                 event_stream,
                 tokenizer:TokenizerBase,
                 max_seq_length:int = 256,
                 **kwargs
                ):
        super().__init__()
        
        self.event_stream = event_stream    
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        # self.tokenize_stream()         # TODO: Pre-process tokenization
        
    def __len__(self):
        # Number of samples in dataset
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

        sequence_tokens = row_dict["EVENT"]                                                        # e.g. ["bmi", "DEPRESSION", "bmi"] 
        sequence_ages = [int(_a) for _a in row_dict["AGE_AT_EVENT"]]                               #      [6661, 7569, 7866]
        sequence_values = [float(_v) if _v is not None else np.nan for _v in row_dict["VALUE"]]    #      [23.3, np.nan, 27.7]
        
        if not self.tokenizer.is_tabular:
            # Merge together events and their values. When value is None, do not include it. 
            # E.g. events = ["bmi", "DEPRESSION", "bmi"] and values = [23.3, None, 27.7] --> merge to --> "bmi", "2", "3", ".", "3", "DEPRESSION", "bmi", "2", "7", ".", "7""
            #      ages   = [6661, 7569, 7866] --> merge to --> [6661,6661,6661,6661,6661,7569, ...]
            sequence_tokens = [[_event] + wrap(str(_value), 1) if _value is not np.nan else [_event] for _event, _value in zip(sequence_tokens, sequence_values)]
            sequence_tokens = sum(sequence_tokens, [])        # concat list of lists            
            sequence_ages = [[_age] + [_age for _ in range(len(str(_value)))] if _value is not np.nan else [_age] for _age, _value in zip(sequence_ages, sequence_values)]
            sequence_ages = sum(sequence_ages, [])            # concat list of lists
            sequence_values = [np.nan for _ in range(len(sequence_tokens))]
            
        # Then encode the sequence
        encoded_tokens = self.tokenizer.encode(sequence_tokens)
        
        # Sub-sample if number of tokens exceeds requested block size
        if len(encoded_tokens) > self.max_seq_length:
            start_pos = np.random.randint(low=0, high=len(encoded_tokens)-self.max_seq_length, size=1)[0]
            encoded_tokens = encoded_tokens[start_pos:start_pos+self.max_seq_length]            
            sequence_ages = sequence_ages[start_pos:start_pos+self.max_seq_length]
            sequence_values = sequence_values[start_pos:start_pos+self.max_seq_length]
            # sequence_tokens = sequence_tokens[start_pos:start_pos+self.max_seq_length]
        # print(f"{sequence_tokens} \n\t-> {encoded_tokens} \n\t-> {self.tokenizer.decode(encoded_tokens)}")
        
        # stagger and split for input and targets
        encoded_tokens = torch.tensor(encoded_tokens)    # [:-1]
        sequence_ages = torch.tensor(sequence_ages)
        sequence_values = torch.tensor(sequence_values)

        return {"identifier": row_dict["PRACTICE_PATIENT_ID"],
                #"sex": row_dict["SEX"],
                #"ethnicity": row_dict["ETHNICITY"],
                #"year_of_birth": row_dict["YEAR_OF_BIRTH"],
                "tokens": encoded_tokens,
                "ages": sequence_ages,
                "values": sequence_values,
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
