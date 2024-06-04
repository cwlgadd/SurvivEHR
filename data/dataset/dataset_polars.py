# Build deep-Learning friendly representations from each stream of input data (static, diagnosis, measurements) from the reformatted SQL database
from typing import Optional, Any, Union
from collections.abc import Sequence
import itertools
import pathlib
import sqlite3
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pickle
import numpy as np
from CPRD.data.dataset.collector import SQLiteDataCollector
from sklearn.model_selection import train_test_split as sk_split
import logging
from tqdm import tqdm
import psutil
import os

class PolarsDataset:
    
    def __init__(self, path_to_db, db_name="cprd.db"):
        """
        """
        super().__init__()
        self.save_path = None
        self.path_to_db = path_to_db
        self.db_name = db_name
        self.collector = SQLiteDataCollector(self.path_to_db + self.db_name)
        self.collector.connect()
        
    def fit(self,
            path:                       str,
            inclusion_conditions:       Optional[list[str]] = None,
            include_static:             bool = True,
            include_diagnoses:          bool = True,
            include_measurements:       bool = True,
            overwrite_meta_information: Optional[str] = None,                 
            **kwargs
           ):
        r"""
        Create Deep-Learning friendly dataset

         Load information from SQL tables into polars frames for each table in chunks. For each chunk
           * Then combine, align frames, and put into a DL friendly lazy Polars representation
           * iteratively find normalisation statistics, counts, or any other meta information 
           * Save polars frames to parquets
           * Create a hashmap dictionary which allows us to do faster lookups than native PyArrow solutions
           
        ARGS:
            path:  
                Full path to folder where parquet files containing the Polars dataset, meta information, and file-look up pickles
            
        KWARGS:
            inclusion_conditions:
                The set of inclusion conditions to query against the collector. For example, only include patients from ["COUNTRY = 'E'"]
            include_static:
                Whether to include static information in the meta_information
            include_diagnoses:
                Whether to include diagnoses in the meta_information, and in the parquet dataset
            include_measurements
                Whether to include measurements in the meta_information, and in the parquet dataset
            overwrite_meta_information:
                If you want to overwrite the meta_information, for example using quantile bounds for some measurements, then there is no need
                to pre-process it again.

        **KWARGS:
            drop_empty_dynamic: bool = True,

            drop_missing_data: bool = True,
            
            exclude_pre_index_age: bool = False,

            

        TODO: pickle this class rather than separately saving the different attributes. However the SQLite collector cannot be pickled
            
        """
       
        self.save_path = path
        logging.info(f"Building Polars datasets and saving to {path}")     
    
        # Train, test, validation split
        self.train_practice_ids, self.val_practice_ids, self.test_practice_ids = self._train_test_val_split(inclusion_conditions=inclusion_conditions)

        # Collect meta information. 
        #    These are pre-calculations for torch loader len(), tokenization, and optionally for standardisation        
        if overwrite_meta_information is None:
            meta_information = self.collector.get_meta_information(practice_ids = None, # all_train = list(itertools.chain.from_iterable(self.train_practice_ids))
                                                                   static       = include_static,
                                                                   diagnoses    = include_diagnoses,
                                                                   measurement  = include_measurements
                                                                   )
            with open(self.save_path + f'meta_information.pickle', 'wb') as handle:
                pickle.dump(meta_information, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #  Create train/test/val DL Polars datasets
        # Loop over practice IDs
        #    * create the generator that performs a lookup on the database for each practice ID and returns a lazy frame for each table's rows        
        #    * collating the lazy tables into a lazy DL-friendly representation,
        for split_name, split_ids in zip(["train", "test", "val"], [self.train_practice_ids,  self.test_practice_ids,  self.val_practice_ids]):

            # for debugging just create dataset on only first
            # split_ids = split_ids[:1]
            
            path: pathlib.Path = self.save_path + f"split={split_name}"    # /{chunk_name}.parquet

            # Check directory is currently empty
            assert not any(path.iterdir())
            
            logging.info(f"Writing {split_name} split into a DL friendly .parquet dataset.")
            self._write_parquet_dl_dataset(save_path=path, 
                                           split_ids=split_ids, 
                                           include_diagnoses = include_diagnoses,
                                           include_measurements = include_measurements,
                                           **kwargs)
            
            logging.info(f"Creating file_row_count_dicts for file-index look-ups")
            hashmap = self._get_file_row_counts(path)
            with open(self.save_path + f'file_row_count_dict_{split_name}.pickle', 'wb') as handle:
                pickle.dump(hashmap, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def _train_test_val_split(self, inclusion_conditions=None):
        
        # get a list of practice IDs which are used to chunk the database
        #    We can optionally subset which practice IDs should be included in the study based on some criteria.
        #    For example, asserting that we only want practices in England can be achieved by adding an inclusion
        #    that applies to the static table
        logging.info(f"Chunking by unique practice ID with {'no' if inclusion_conditions is None else inclusion_conditions} inclusion conditions")
        practice_ids = self.collector._extract_distinct(table_names=["static_table"],
                                                        identifier_column="PRACTICE_ID",
                                                        inclusion_conditions=inclusion_conditions
                                                       )
        
        # Create train/test/val splits, each is list of practice_id
        logging.info(f"Creating train/test/val splits using practice_ids")
        train_practice_ids, test_practice_ids = sk_split(practice_ids, test_size=0.1)       
        test_practice_ids, val_practice_ids = sk_split(test_practice_ids, test_size=0.5)
        return train_practice_ids, val_practice_ids, test_practice_ids
    
    def _write_parquet_dl_dataset(self,
                                  save_path:               str,
                                  split_ids:               list[str],
                                  include_diagnoses:       bool = True,
                                  include_measurements:    bool = True,
                                  **kwargs,
                                 ) -> pl.LazyFrame:
        r"""
        Build the DL-friendly representation in polars given the list of `practice_patient_id`s which fit study criteria
                
        ARGS:
            
        KWARGS:
        
        """
        
        # Create the generator which we will chunk over
        # Can process entire list of IDs at once by changing to `distinct_values=[split_ids]`
        logging.debug(f"Generating over practices IDs")
        generator = self.collector._generate_lazy_by_distinct(distinct_values=split_ids,                            
                                                              identifier_column="PRACTICE_ID",
                                                              include_diagnoses=include_diagnoses,
                                                              include_measurements=include_measurements)

        total_samples = 0
        for _idx, (chunk_name, lazy_table_frames_dict) in enumerate(tqdm(generator, total=len(split_ids))):
            
            # Merge the lazy polars tables provided by the generator into one lazy polars frame
            lazy_batch = self.collector._collate_lazy_tables(lazy_table_frames_dict, **kwargs)

            if save_path is not None:
                # save splits `hive` style partitioning using parquet which is a columnal format, but gives efficient compression
                # TODO: make directories if they dont already exist

                # include row count so we can filter when reading from file
                df = lazy_batch.collect().with_row_count(offset=total_samples).to_pandas()   

                # convert row count to a lower cardinality bin which can be used in the hive partitioning for faster reading
                # ... the smaller the window the more files created, storage space used, and the longer this takes to run, but 
                # ... the faster the read efficiency.
                df = df.assign(CHUNK = [int(_v / 500) for _v in df['row_nr']])                  
                
                total_samples += len(df.index)
                
                table = pa.Table.from_pandas(df)
                pq.write_to_dataset(table, root_path=save_path, 
                                    partition_cols=['COUNTRY', 'HEALTH_AUTH', 'PRACTICE_ID', 'CHUNK'],
                                   )
                
                logging.debug(f'{df.iloc[0]["COUNTRY"]}, {df.iloc[0]["HEALTH_AUTH"]}, {df.iloc[0]["PRACTICE_ID"]}')

        logging.info(f"Created dataset at {save_path} with {total_samples:,} samples")
        
        return 

    def _get_file_row_counts(self, parquet_path):

        # Get all files at specified path, and calculate how many samples are in each file for faster reading during __getitem__
        file_row_counts = {}
        desc = "Getting file row counts. This allows the creation of an index to file map, increasing read efficiency"
        for file in tqdm(pathlib.Path(parquet_path).rglob('*.parquet'), desc=desc):
            file_row_counts[file] = pq.ParquetFile(file).metadata.num_rows   # update hash map
    
        return file_row_counts