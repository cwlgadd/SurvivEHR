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
       
        
        super(FoundationalDataModule, self).__init__()
        
        self.batch_size = batch_size
        self.min_workers = min_workers 
        
        # Get the DL friendly representation, either by loading or building from scratch.
        polars_dataset = PolarsDataset(path_to_db=path_to_db)
        meta_information = polars_dataset.fit(path=path_to_db + "polars/", load=load, **kwargs)
        logging.debug(meta_information)
    
        # Create tokenizer, and build based on vocabulary from training set. 
        #   Vocabularly begins with the PAD (padding) token, then the UNK (unknown) token, and is then ordered by token frequency        
        logging.info(f"Using tokenizer {tokenizer}")
        self.tokenizer = Tabular() if tokenizer == "tabular" else NonTabular()
        self.tokenizer.fit(meta_information, freq_threshold=unk_freq_threshold, **kwargs)
        
        # Train/test/validation GenerativeDatasets
        parquet_path = path_to_db + "polars/"
        dataset_kwargs = {"max_seq_length": max_seq_length, "tokenizer": self.tokenizer, "meta_information": meta_information}
        self.train_set = FoundationalDataset(parquet_path + "split=train/", **dataset_kwargs)
        self.test_set = FoundationalDataset(parquet_path + "split=test/", **dataset_kwargs)
        self.val_set = FoundationalDataset(parquet_path + "split=val/", **dataset_kwargs)
    
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
                 tokenizer:TokenizerBase,
                 max_seq_length:int = 256,
                 standardise_values:bool = True,
                 meta_information:Optional[dict] = None
                ):
        super().__init__()

        logging.info("Creating dataset")
        
        self.parquet_path = parquet_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.standardise_values = standardise_values
        self.meta_information = meta_information

        # create a hash map that lets us look up the correct parquet in O(1)
        # logging.info("Creating hash map")
        # self.file_row_map = self._create_file_row_map()

        # Get all files at specified path,
        #    calculate how many samples are in each file for faster reading
        pathlist = Path(parquet_path).rglob('*.parquet')
        self.parquet_files = []
        self.row_group_counts = []
        for file in tqdm(pathlist, desc="Calculating chunk index splits "):
            df = pq.read_table(file).to_pandas()
            self.parquet_files.append(file)            
            self.row_group_counts.append(df.shape[0])
        
    def _create_file_row_map(self):
        """
        Create a hash map of file paths and row group offsets.

        Returns:
            dict: A hash map where keys are file paths and values are lists of row group offsets.
        """
        pathlist = Path(self.parquet_path).rglob('*.parquet')
        file_row_map = {}                                                         # Create an empty hash map to store file paths and row group offsets
        total_rows = 0                                                            # Initialize total_rows to track cumulative row counts across all files
        
         # Iterate over each file in the root directory
        for file_path in pathlist:
            
            parquet_file = pq.ParquetFile(file_path)                              # Open the Parquet file
            num_row_groups = parquet_file.metadata.num_row_groups                 # Get the number of row groups in the Parquet file
            row_group_offsets = [total_rows] * num_row_groups                     # Create a list of row group offsets for the current file    
            file_row_map[file_path] = row_group_offsets                           # Store the file path and row group offsets in the hash map
            total_rows += parquet_file.metadata.num_rows                          # Update total_rows by adding the number of rows in the current file
        
        return file_row_map                                                       # return populated hash map
        
    def __len__(self):
        """ 
            Get the total number of rows across all Parquet files.
        """
        return sum(self.row_group_counts)

    def _get_file_and_row_group(self, idx):
        """
        Get the file path and row group index corresponding to the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the file path and row group index.
        """    
        # Iterate over the hash map entries (file paths and row group offsets)
        for file_path, row_group_offsets in self.file_row_map.items():
             # If the index is less than the cumulative row count for the current file
            if idx < row_group_offsets[-1]:
                # Find the row group index corresponding to the index within the current file
                row_group_idx = next(i for i, offset in enumerate(row_group_offsets) if idx < offset)

                # Open the Parquet file
                parquet_file = pq.ParquetFile(file_path)
                
                # Read the specified row group from the Parquet file
                table = parquet_file.read_row_group(row_group_idx)
                
                # Convert the row group to a Pandas DataFrame and get the first row
                row_df = table.to_pandas().loc[0]

                return row_df

        # If the index exceeds the total number of samples, raise an IndexError
        raise IndexError(f"Index {idx} out of range")
    
    def __getitem__(self, idx):
        r"""
            Get a single item from the dataset, 
            
            tokenized and padded event stream and static variables
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Determine which file and row to read based on the index
        file_idx = 0
        while idx >= self.row_group_counts[file_idx]:
            idx -= self.row_group_counts[file_idx]
            file_idx += 1
        # Read the corresponding row from the Parquet file        
        row_df = pq.read_table(self.parquet_files[file_idx], filters=[('row_nr','=', idx)]).to_pandas().loc[0]
        
        # Unpack rows, and optionally standardise values
        sequence_tokens, sequence_values, sequence_ages = [], [], []
        standardisation_keys = self.meta_information["measurement_tables"].event.tolist() if self.meta_information is not None else None
        for next_event, next_value, next_age in zip(row_df["EVENT"], row_df["VALUE"], row_df["DAYS_SINCE_BIRTH"]):

            # e.g. ["bmi", "DEPRESSION", "bmi", ...] 
            sequence_tokens.append(next_event)

            # e.g.  [23.3, np.nan, 27.7, ...] if not standardising, else [0.12, np.nan, 0.23, ...]
            if next_value is not None:
                if self.standardise_values:
                    assert self.meta_information is not None
                    if next_event in standardisation_keys:
                        event_meta = self.meta_information["measurement_tables"][self.meta_information["measurement_tables"]["event"] == next_event]
                        next_value = (next_value - event_meta["bias"]) / event_meta["scale"]
                next_value = float(next_value)
            else:
                next_value = np.nan
            sequence_values.append(next_value)

            # e.g. [6661, 7569, 7866, ...]
            sequence_ages.append(int(next_age))

            # print(f"{next_event}, {next_value}, {next_age}")
        
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

        return {"identifier": row_df["PRACTICE_PATIENT_ID"],
                #"sex": row_df["SEX"],
                #"ethnicity": row_df["ETHNICITY"],
                #"year_of_birth": row_df["YEAR_OF_BIRTH"],
                "tokens": encoded_tokens,
                "ages": sequence_ages,
                "values": sequence_values,
               }
