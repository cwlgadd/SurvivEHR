from sklearn.model_selection import train_test_split as sk_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
import polars as plr
import pyarrow.parquet as pq
import pyarrow.dataset as ds
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
            
        
    """
    
    def __init__(self, 
                 path_to_db: str,
                 path_to_ds: str,
                 load: bool,
                 tokenizer: str = "tabular",
                 batch_size: int = 64,
                 min_workers:int = 1,
                 overwrite_meta_information: Optional[str] = None,                 
                 **kwargs   
                ):
        """
        
        ARGS:
            path_to_db:
                Complete path to the SQL database folder

            path_to_ds:
                Complete path to the pre-processed dataset folder, either to load from or save to
            
            load:  
                True: load directly from previously processed parquet files; or False: create the parquet files again and save to `path`.  
        KWARGS:
            tokenizer:

            batch_size:

            min_workers:

            overwrite_meta_information:
                If you want to overwrite the meta_information, for example using quantile bounds for some measurements, then there is no need
                to pre-process it again. In this case, pass in the path to an existing meta_information pickled file. 

        **KWARGS 
            freq_threshold:
                Value between 0 and 1, controlling at what level of frequency rare tokens are mapped to the UNK token. Used to reduce vocabulary size
            practice_inclusion_conditions:
                The set of inclusion conditions to query against the collector. For example, only include patients from practices where ["COUNTRY = 'E'"]
            include_static:
                Whether to include static information in the meta_information
            include_diagnoses:
                Whether to include diagnoses in the meta_information, and in the parquet dataset
            include_measurements
                Whether to include measurements in the meta_information, and in the parquet dataset
            drop_empty_dynamic: bool = True,

            drop_missing_data: bool = True,
            
            exclude_pre_index_age: bool = False,

            max_seq_length:

            standardise_values:

            global_diagnoses:

        """
        super().__init__()
        
        self.batch_size = batch_size
        self.min_workers = min_workers 
        
        # Get the DL friendly representation, either by loading or building from scratch.
        if load is False:
            polars_dataset = PolarsDataset(path_to_db=path_to_db)
            polars_dataset.fit(path=path_to_ds,
                               overwrite_meta_information=overwrite_meta_information,
                               **kwargs)

        # Load meta information
        meta_path = path_to_ds + "meta_information.pickle" if overwrite_meta_information is None else overwrite_meta_information
        with open(meta_path, 'rb') as f:
            self.meta_information = pickle.load(f)
            logging.info(f"Using meta information from {meta_path}")
        # Load the file_row_count_dicts for each split
        file_row_count_dicts = {}
        for _key in ["train", "test", "val"]:
            file_row_path = path_to_ds + f"file_row_count_dict_{_key}.pickle"
            with open(file_row_path, 'rb') as f:
                file_row_count_dicts[_key] = pickle.load(f)
                logging.info(f"Using {_key} file-row count dictionary from {file_row_path}")
    
        # Create tokenizer, and build based on vocabulary from training set. 
        #   Vocabularly begins with the PAD (padding) token, then the UNK (unknown) token, and is then ordered by token frequency        
        self.tokenizer = Tabular() if tokenizer.lower() == "tabular" else NonTabular()
        self.tokenizer.fit(self.meta_information, **kwargs)
        logging.info(f"Using {tokenizer} tokenizer, created from meta information and containing {self.tokenizer.vocab_size} tokens")
        
        # Train/test/validation GenerativeDatasets
        dataset_args = {"tokenizer": self.tokenizer, "meta_information": self.meta_information}
        self.train_set = FoundationalDataset(path_to_ds, "train", **dataset_args, file_row_count_dict=file_row_count_dicts["train"], **kwargs)
        self.test_set = FoundationalDataset(path_to_ds, "test", **dataset_args, file_row_count_dict=file_row_count_dicts["test"], **kwargs)
        self.val_set = FoundationalDataset(path_to_ds, "val", **dataset_args, file_row_count_dict=file_row_count_dicts["val"], **kwargs)

    def standardise(self, event, value):
        # Standardise a single value and event
        if event in list(self.meta_information["measurement_tables"].event):
            _row = self.meta_information["measurement_tables"][self.meta_information["measurement_tables"].event==event]
            _lqr = _row["approx_lqr"].to_numpy()[0]
            _uqr = _row["approx_uqr"].to_numpy()[0]
            return float(( value - _lqr ) /  (_uqr - _lqr)) - 0.5
        else:
            return value
        
    def unstandardise(self, event, value):
        if event in list(self.meta_information["measurement_tables"].event):
            _row = self.meta_information["measurement_tables"][self.meta_information["measurement_tables"].event==event]
            _lqr = _row["approx_lqr"].to_numpy()[0]
            _uqr = _row["approx_uqr"].to_numpy()[0]
            return float(( (value + 0.5) * (_uqr - _lqr) ) + _lqr)    
        else:
            return value
    
    def encode(self, sequence:list[str]):
        return self.train_set.tokenizer.encode(sequence)

    def decode(self, sequence:list[int]):
        return self.train_set.tokenizer.decode(sequence)
    
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
            "static_covariates": pad_sequence(batch_dict["static_covariates"], padding_value=0).T,   # not actually padded, will only ever see fixed length sequences
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
            # pin_memory=True,
            collate_fn=self.collate_fn,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=np.min((self.min_workers, os.cpu_count())),
            # pin_memory=True,
            collate_fn=self.collate_fn,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=np.min((self.min_workers, os.cpu_count())),
            # pin_memory=True,
            collate_fn=self.collate_fn,
            shuffle=False
        )

    
class FoundationalDataset(Dataset):
    r"""
    """

    def view_sample(self, idx, max_dynamic_events=None, report_time=False):
        """ Wrapper around __getitem__ to print a sample in a read-friendly format """
        # Get row
        start_time = time.time()
        batch = self.__getitem__(idx)
        if report_time:
            print(f"Time to retrieve sample index {idx} was {time.time() - start_time} seconds\n")
        # static
        static = self._decode_covariates(batch["static_covariates"])
        for _key in static:
            print(f"{_key}".ljust(20) + f"| {static[_key][0]}")
        # dynamic
        print("\nToken".ljust(76) + "| Age".ljust(20) + "| Standardised value".ljust(20) + "\n" + "="*115)
        for idx_event, (token, age, value) in enumerate(zip(self.tokenizer.decode(batch["tokens"].tolist()).split(" "), batch["ages"], batch["values"])):
            print(f"{token}".ljust(75) + f"| {age}".ljust(20) + f"| {value:.2f}".ljust(20))
            if max_dynamic_events is not None and idx_event >= max_dynamic_events - 1:
                break
        
    def __init__(self,
                 parquet_path:                 str, 
                 split:                        str,
                 tokenizer:                    TokenizerBase,
                 meta_information:             dict,
                 file_row_count_dict:          dict,
                 max_seq_length:               int = 256,
                 standardise_values:           bool = True,
                 global_diagnoses:             bool = False,
                 **kwargs
                ):
        """

        ARGS:
            parquet_path:

            split:

            tokenizer:

            meta_information:

            file_row_count_dict:

        KWARGS:
            max_seq_length:

            standardise_values

            global_diagnoses:

        **KWARGS:
            None
        """        
        super().__init__()
        
        self.parquet_path = parquet_path
        self.sub_dir = f"split={split}/"
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.standardise_values = standardise_values
        self.global_diagnoses = global_diagnoses 
        self.meta_information = meta_information

        # Create a PyArrow dataset directly from the PolarsDataset saved hive partitioned dataset
        # NOTE:   This can take some time to initialise, but using the API is cleaner
        # self.dataset = pq.ParquetDataset(parquet_path + self.sub_dir, validate_schema=False)
        # self.dataset = ds.dataset(parquet_path + self.sub_dir, format="parquet", partitioning="hive")
        # These are really slow options, creation takes a ds.dataset: 70 secs, vs.  pq.Parquet: 133 secs.
        # Then the APIs for taking items is also really slow:
        #    e.g. `self.dataset.take([idx]).to_pandas().loc[0]` for `ds.dataset`                                
        #     or   filtering predicates `filter=(ds.field('CHUNK') == chunk ) & (ds.field('row_nr') == index)`  -> 0.5 sec/read
        #    TODO: is there a way to simplify code using PyArrow that retains speed?
        self.file_row_count_dict = file_row_count_dict
        # Pre-calculate total number of samples (patients) on initialisation        
        # NOTE:   Updating to the ds.dataset API (instead of previously using custom a hash map) now means the dataset 
        #         length is being calculated every initialisation - this is very slow (~13 mins) as it needs to touch every file.
        #         Fortunately pyarrow saves total row count of each file in the footer, so we dont need to load the data.
        #         Consider caching this sum somewhere when building dataset. E.g. in meta information? Currently I can't
        #         extract it from here as this information isn't aggregated across train/test/val splits.
        # self.total_samples = self.dataset.count_rows()   
        # self.total_samples = sum(_frag.count_rows() for _frag in self.dataset.get_fragments())
        self.total_samples = sum(self.file_row_count_dict.values())
        logging.info(f"Loaded {self.parquet_path + self.sub_dir} dataset, with {self.total_samples:,} samples")

        # Create one-hot encoder map for static categorical variables.
        #   Note we use one hot even when the data is ordinal (e.g. with IMD deprivation score) so we can include the predict with missing data at inference time
        self.static_1hot = {}
        for _key in self.meta_information["static_table"].keys():
            encoder = OneHotEncoder(handle_unknown='error')
            encoder.fit([[cat] for cat in self.meta_information["static_table"][_key]["category"]])
            self.static_1hot[_key] = encoder
        
    def __len__(self):
        """ 
            Get the total number of rows across all Parquet files.
        """
        return self.total_samples
    
    def __getitem__(self, idx):
        r"""
            Get a single item from the dataset, 
            
            tokenized and padded event stream and static variables
        """
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Determine which file and row the idx corresponds to.
        #    TODO: replace with a better sorting algorithm.
        #          Can use the fact that most files have a maximum number of rows to 
        #          estimate where to start search.
        for file in self.file_row_count_dict.keys():
            if idx >= self.file_row_count_dict[file]:
                idx -= self.file_row_count_dict[file]
            else:
                break
                
        # Read the corresponding row from the Parquet dataset
        try:
            # row_df = self.dataset.take([idx]).to_pandas().loc[0]
            row_df = pq.read_table(file).to_pandas().loc[idx]
            # row_df = pq.read_table(file, filters=[('row_nr','=', idx)]).to_pandas().loc[0]
        except:
            raise ValueError(f"No data found for index {idx}")

        # Static variables
        ##################
        static_covariates = self._parquet_row_to_static_covariates(row_df)

        # Dynamic variables
        ##################
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
                event_meta = self.meta_information["measurement_tables"][self.meta_information["measurement_tables"].event == next_event]   
                if next_event in self.meta_information["measurement_tables"].event.tolist():
                    lqr, uqr = event_meta.approx_lqr.to_numpy()[0], event_meta.approx_uqr.to_numpy()[0]
                else:
                    lqr, uqr = -np.inf, np.inf
                    
                # Define outlier boundaries based on quantiles
                if next_value < lqr or next_value > uqr:
                    # If it does have a value, but this is an outlier, we set it as np.nan which is the same way we record missing values above
                    next_value = np.nan
                else:
                    if self.standardise_values and next_event in self.meta_information["measurement_tables"].event.tolist():
                        # Else we can optionally continue by standardising it to [0,1]
                        next_value = (next_value -lqr) / (uqr - lqr)

                        # But if we use a joint data embedding we will be scaling token embeddings by the value, and so we 
                        # instead scale to [-0.5,0.5], so the scaling is symetric around the average value of the token.
                        # this still leads to strong assumptions with this embedding choice (that average values are not meaningful)
                        next_value = next_value - 0.5 

            sequence_values.append(float(next_value))

            ## AGES
            ########
            # e.g. [6661, 7569, 7866, ...]
            sequence_ages.append(next_age)

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
        ##########################
        encoded_tokens = self.tokenizer.encode(sequence_tokens)
        # Get a windowed sub-block from the patient's history if context length exceeds block size
        start_pos = np.random.randint(low=0, high=len(encoded_tokens)-self.max_seq_length, size=1)[0] if len(encoded_tokens) > self.max_seq_length else 0
        end_pos = start_pos + self.max_seq_length
        # Get the diagnoses that we will (optionally) not be dropping, as these have life long implications        
        if self.global_diagnoses:
            # Replace the first X events with the diagnoses that occurred before this sampled context block
            earlier_global_events = self.tokenizer.encode([_event for _event in sequence_tokens[:start_pos]
                                                           if _event in self.meta_information["diagnosis_table"].event.tolist()])
            earlier_global_ages = [ _age for _event, _age in zip(sequence_tokens[:start_pos], sequence_ages[:start_pos]) 
                                   if _event in self.meta_information["diagnosis_table"].event.tolist()]
            earlier_global_values = [_value for _event, _value in zip(sequence_tokens[:start_pos], sequence_values[:start_pos]) 
                                     if _event in self.meta_information["diagnosis_table"].event.tolist()]
            start_pos += len(earlier_global_events)

            # TODO: this does not check if the pushed back events were global themselves
        else:
            earlier_global_events, earlier_global_ages, earlier_global_values = [], [], []
            
        # combine        
        encoded_tokens = earlier_global_events + encoded_tokens[start_pos:end_pos]            
        sequence_ages = earlier_global_ages + sequence_ages[start_pos:end_pos]
        sequence_values = earlier_global_values + sequence_values[start_pos:end_pos]
                
        return {"static_covariates": torch.tensor(static_covariates, dtype=torch.float),
                "tokens": torch.tensor(encoded_tokens),
                "ages": torch.tensor(sequence_ages),
                "values": torch.tensor(sequence_values),
               }

    def _parquet_row_to_static_covariates(self, row_df):
        """ From the row loaded from a parquet file, get the encoded static variables. For example, one hot encodings where applicable
        """

        covariates = []

        # one-hot covariates
        for _key in self.meta_information["static_table"].keys():
            # print(_key)
            # Dummy variable warning 
            #    Currently we include a third category for gender (I), which (TODO: double check this..) as an intersex gender. 
            #    If we drop this we need to be careful of dummy variables as SEX becomes binary.
            X_c = np.array([[row_df[_key]]], dtype=object)
            Xt_c =  self.static_1hot[_key].transform(X_c).toarray()
            covariates.append(Xt_c)

        # continuous
        # Year-of-birth
        yob_lower, yob_upper = 1900, 2024
        yob_standardised = (row_df["YEAR_OF_BIRTH"].year - yob_lower ) / (yob_upper - yob_lower)
        covariates.append(np.asarray(yob_standardised).reshape((1,-1)))

        covariates = np.hstack(covariates).squeeze()
        # print(covariates)
        # print(covariates.shape)

        return covariates

    def _encode_covariates(self, 
                           sex,
                           deprivation,
                           ethnicity,
                           year_of_birth
                           ):
        covariates = []
        for _key, _covariate in zip(["SEX", "IMD", "ETHNICITY"], [sex, deprivation, ethnicity]):
            X_c = np.array([[_covariate]], dtype=object)
            Xt_c =  self.static_1hot[_key].transform(X_c).toarray()
            # print(self.static_1hot[_key].categories_)
            covariates.append(Xt_c)
            
        yob_lower, yob_upper = 1900, 2024
        yob_standardised = (year_of_birth - yob_lower ) / (yob_upper - yob_lower)
        covariates.append(np.asarray(yob_standardised).reshape((1,-1)))

        covariates = np.hstack(covariates).squeeze()
        
        return torch.tensor(covariates, dtype=torch.float)

    def _decode_covariates(self,
                           covariates:       torch.tensor         # bsz, nun_covariates
                          ):

        if len(covariates.shape) == 1:
            covariates = covariates.unsqueeze(0)

        features = {}
        for _key in self.meta_information["static_table"].keys():
            # print(_key)
            num_covariates = len(self.static_1hot[_key].categories_[0])
            covariates_key = covariates[:, :num_covariates]
            covariates = covariates[:, num_covariates:]
            features[_key] = self.static_1hot[_key].inverse_transform(covariates_key)[:, 0]

        yob_lower, yob_upper = 1900, 2024
        yob = ( covariates * (yob_upper - yob_lower) ) + yob_lower
        features["birth_year"] = yob[:, 0]

        return features
            
