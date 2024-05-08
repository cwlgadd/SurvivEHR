from sklearn.model_selection import train_test_split as sk_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import polars as plr
import pyarrow.parquet as pq
from abc import ABC
from textwrap import wrap
from typing import Optional
from collections import defaultdict
import os
from CPRD.data.dataset.dataset_polars import PolarsDataset
from CPRD.data.tokenizers.base import TokenizerBase
from CPRD.data.tokenizers.tokenizers import NonTabular, Tabular
import random
import logging
from pathlib import Path
from tqdm import tqdm
import pickle

# Testing modules
from CPRD.data.database import queries
import sqlite3

import time

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
                 path_to_db: str,
                 load: bool,
                 tokenizer: str = "tabular",
                 max_seq_length: int = 256,
                 batch_size: int = 512,
                 unk_freq_threshold=1e-2,
                 min_workers:int = 1,
                 **kwargs   
                ):
       
        
        super().__init__()
        
        self.batch_size = batch_size
        self.min_workers = min_workers 
        
        # Get the DL friendly representation, either by loading or building from scratch.
        polars_dataset = PolarsDataset(path_to_db=path_to_db)
        meta_information = polars_dataset.fit(path=path_to_db + "polars/", load=load, **kwargs)
        self.meta_information = meta_information
        logging.debug(meta_information)
    
        # Create tokenizer, and build based on vocabulary from training set. 
        #   Vocabularly begins with the PAD (padding) token, then the UNK (unknown) token, and is then ordered by token frequency        
        logging.info(f"Using tokenizer {tokenizer}")
        self.tokenizer = Tabular() if tokenizer == "tabular" else NonTabular()
        self.tokenizer.fit(meta_information, freq_threshold=unk_freq_threshold, **kwargs)
        
        # Train/test/validation GenerativeDatasets
        parquet_path = path_to_db + "polars/"
        dataset_kwargs = {"max_seq_length": max_seq_length, "standardise_values": True, "meta_information": meta_information, "load": load}
        self.train_set = FoundationalDataset(parquet_path, "train", tokenizer=self.tokenizer, **dataset_kwargs)
        self.test_set = FoundationalDataset(parquet_path, "test", tokenizer=self.tokenizer, **dataset_kwargs)
        self.val_set = FoundationalDataset(parquet_path, "val", tokenizer=self.tokenizer, **dataset_kwargs)

    def standardise(self, event, value):
        _row = self.meta_information["measurement_tables"][self.meta_information["measurement_tables"].event==event]
        _lqr = _row["approx_lqr"].to_numpy()[0]
        _uqr = _row["approx_uqr"].to_numpy()[0]
        return float(( value - _lqr ) /  (_uqr - _lqr))
        
    def unstandardise(self, event, value):
        _row = self.meta_information["measurement_tables"][self.meta_information["measurement_tables"].event==event]
        _lqr = _row["approx_lqr"].to_numpy()[0]
        _uqr = _row["approx_uqr"].to_numpy()[0]
        return float(( value * (_uqr - _lqr) ) + _lqr)
        
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
            "attention_mask": pad_sequence(batch_dict["attention_mask"]).T
            }
        return worker_batch

    def train_dataloader(self):
        return DataLoader(            
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=np.min((self.min_workers, os.cpu_count())),
            pin_memory=True,
            collate_fn=self.collate_fn,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=np.min((self.min_workers, os.cpu_count())),
            pin_memory=True,
            collate_fn=self.collate_fn,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=np.min((self.min_workers, os.cpu_count())),
            pin_memory=True,
            collate_fn=self.collate_fn,
            shuffle=False
        )

    
class FoundationalDataset(Dataset):
    r"""
    """
    def __init__(self,
                 parquet_path: str, 
                 split: str,
                 tokenizer:TokenizerBase,
                 load: bool=False, 
                 max_seq_length:int = 256,
                 standardise_values:bool = True,
                 meta_information:Optional[dict] = None
                ):
        super().__init__()
        
        self.parquet_path = parquet_path
        self.sub_dir = f"split={split}/"
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.standardise_values = standardise_values
        self.meta_information = meta_information
        self.meta_measurement = meta_information["measurement_tables"] if meta_information is not None else None
        if self.standardise_values is True:
            assert self.meta_information is not None

        logging.info(f"Creating {self.sub_dir} dataset")
        logging.debug(f"\t Using root {self.parquet_path}")
        
        self.dataset = pq.ParquetDataset(parquet_path + self.sub_dir, validate_schema=False)

        # create a hash map that lets us look up the correct parquet in O(1)
        #    TODO: update to use fragments API        
        if load is True:
            try:
                logging.info(f"\t Loading {self.sub_dir} hash map for parquet")
                with open(self.parquet_path + f'lookup_hashmap_{split}.pickle', 'rb') as handle:
                   self.file_row_count_dict = pickle.load(handle)
            except OSError as e:
                raise FileNotFoundError
        else:
            logging.info(f"\t Creating {self.sub_dir} hash map from parquet")
            logging.debug(f"\t\t saving to {self.parquet_path}")     
            self.file_row_count_dict = self._get_file_row_counts(self.parquet_path + f'split={split}/')
            with open(self.parquet_path + f'lookup_hashmap_{split}.pickle', 'wb') as handle:
                pickle.dump(self.file_row_count_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Pre-calculate total number of samples (patients) on initialisation        
        self.total_samples = sum(self.file_row_count_dict.values())
        logging.info(f"\t Hash map created for {self.sub_dir} with {self.total_samples:,} samples")
        
    def __len__(self):
        """ 
            Get the total number of rows across all Parquet files.
        """
        return self.total_samples
    
    def _get_file_row_counts(self, parquet_path):

        # Get all files at specified path, and calculate how many samples are in each file for faster reading during __getitem__
        file_row_counts = {}
        for file in tqdm(Path(parquet_path).rglob('*.parquet'), 
                         desc="Getting file row counts. This allows the creation of an index to file map, increasing read efficiency"):
            file_row_counts[file] = pq.ParquetFile(file).metadata.num_rows   # update hash map
    
        return file_row_counts
    
    def __getitem__(self, idx):
        r"""
            Get a single item from the dataset, 
            
            tokenized and padded event stream and static variables
        """
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Determine which file and row the idx corresponds to.
        #    TODO: replace with a better sorting algorithm. Can use the fact that most files have a maximum row number to estimate where to start search. Regardless, this won't be a bottleneck 
        for file in self.file_row_count_dict.keys():
            if idx >= self.file_row_count_dict[file]:
                idx -= self.file_row_count_dict[file]
            else:
                break
            
        # Read the corresponding row from the Parquet file        
        row_df = pq.read_table(file, filters=[('row_nr','=', idx)]).to_pandas().loc[0]
        
        # Unpack rows, and optionally standardise values
        sequence_tokens, sequence_values, sequence_ages = [], [], []
        for next_event, next_value, next_age in zip(row_df["EVENT"], row_df["VALUE"], row_df["DAYS_SINCE_BIRTH"]):

            ## TOKENS
            ##########
            # e.g. ["bmi", "DEPRESSION", "bmi", ...] 
            sequence_tokens.append(next_event)

            ## VALUES
            #########
            # Manual removal of some quirky values from DExtER
            if next_event in ["Ex_smoker_84", "Never_smoked_tobacco_85"]:
                next_value = np.nan
            
            # e.g.  [23.3, np.nan, 27.7, ...] if not standardising, else [0.12, np.nan, 0.23, ...]
            if next_value is None:
                # If the next token does not have a value then we just set it as np.nan
                next_value = np.nan
            else:
                # Else we can continue to process it
                event_meta = self.meta_measurement[self.meta_measurement.event == next_event]   
                if next_event in self.meta_measurement.event.tolist():
                    lqr, uqr = event_meta.approx_lqr.to_numpy()[0], event_meta.approx_uqr.to_numpy()[0]
                else:
                    lqr, uqr = -np.inf, np.inf
                    
                # Define outlier boundaries based on quantiles
                if next_value < lqr or next_value > uqr:
                    # If it does have a value, but this is an outlier, we set it as np.nan which is the same way we record missing values above
                    next_value = np.nan
                else:
                    if self.standardise_values and next_event in self.meta_measurement.event.tolist():
                        # Else we can optionally continue by standardising it
                        next_value = (next_value -lqr) / (uqr - lqr)

                        # if lqr < 300 or upr > 300:
                        #     print(f"{next_event}: {lqr} - {uqr}: \t\t{next_value} -> {new_value}")

            # # Testing for tabular models
            # if next_value is np.nan:
            #     print(f"{next_event}: \t\t{next_value}")
                            
            sequence_values.append(float(next_value))

            ## AGES
            ########
            # e.g. [6661, 7569, 7866, ...]
            sequence_ages.append(next_age)

            # print(f"\n\n {next_event}, {next_value}, {next_age}")
        
        if not self.tokenizer.is_tabular:
            # Merge together events and their values (and when value is None, do not include it)
            #     E.g. events = ["bmi", "DEPRESSION", "bmi"] and values = [23.3, None, 27.7] --> merge to --> "bmi", "2", "3", ".", "3", "DEPRESSION", "bmi", "2", "7", ".", "7""
            #          ages   = [6661, 7569, 7866] --> merge to --> [6661,6661,6661,6661,6661,7569, ...]
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

        # print([f"{_event} {_value}" for _event, _value in zip(sequence_tokens, sequence_values)])
        
        return {"tokens": torch.tensor(encoded_tokens),
                "ages": torch.tensor(sequence_ages),
                "values": torch.tensor(sequence_values),
               }
