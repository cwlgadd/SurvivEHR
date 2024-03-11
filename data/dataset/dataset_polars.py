# Build deep-Learning friendly representations from each stream of input data (static, diagnosis, measurements) from the reformatted SQL database
from typing import Optional, Any, Union
from collections.abc import Sequence
import pathlib
import sqlite3
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pickle
import numpy as np
from CPRD.data.dataset.collector import SQLiteDataCollector
from sklearn.model_selection import train_test_split as sk_split
import logging
from tqdm import tqdm
import psutil
import os
            
class PolarsDataset:
    r"""
        Combine data streams
    """
    
    def __init__(self, path_to_db, db_name="cprd.db"):
        """
        """
        super().__init__()
        self.save_path = None
        self.path_to_db = path_to_db
        self.db_name = db_name
        self.collector = SQLiteDataCollector(self.path_to_db + self.db_name)
        
    def fit(self,
            path: str,
            load: bool = False,
            **kwargs
           ) -> pl.LazyFrame:
        r"""
        Create Deep-Learning friendly dataset

         Load information from SQL tables into polars frames for each table in chunks. For each chunk
           * Then combine, align frames, and put into a DL friendly lazy Polars representation
           * iteratively find normalisation statistics, counts, or any other meta information 
           * Save polars frames to parquets, and pickle meta information
           
        ARGS:
            path:  full to to folder where parquet files containing the Polars dataset and meta information
            load:  True: load directly from previously processed parquet files; or False: create the parquet files again and save to `path`.        
        """
       
        self.save_path = path
        if load is True:
            try:
                logging.info(f"Loading Polars dataset from {path}")
                with open(path + 'meta_information.pickle', 'rb') as handle:
                   self.meta_information = pickle.load(handle)
            except OSError as e:
                raise FileNotFoundError
        else:
            logging.info(f"Building Polars dataset and saving to {path}")        
            self.meta_information = self._build_DL_representation(save_path=path, **kwargs)            
            with open(path + 'meta_information.pickle', 'wb') as handle:
                pickle.dump(self.meta_information, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        return self.meta_information
            
    def _build_DL_representation(self,
                                 save_path: str,
                                 pratice_inclusion_conditions: Optional[list] = None,                                 
                                 include_measurements: bool = True,
                                 include_diagnoses: bool = True,
                                 preprocess_measurements: bool = False,
                                 **kwargs,
                                ) -> pl.LazyFrame:
        r"""
        Build the DL-friendly representation in polars given the list of `practice_patient_id`s which fit study criteria
                
        ARGS:
            
        KWARGS:
        
        RETURNS:
            Polars lazy frame, of the (anonymized) form:
            ┌──────────────────────┬─────┬─────────────┬───────────────┬──────────────────────┬─────────────────────────┬─────────────────────┐
            │ PRACTICE_PATIENT_ID  ┆ SEX ┆ cov 2 ...   ┆ YEAR_OF_BIRTH ┆ VALUE                ┆ EVENT                   ┆ AGE_AT_EVENT        │
            │ ---                  ┆ --- ┆ ---         ┆ ---           ┆ ---                  ┆ ---                     ┆ ---                 │
            │ str                  ┆ str ┆ <->         ┆ str           ┆ list[f64]            ┆ list[str]               ┆ list[i64]           │
            ╞══════════════════════╪═════╪═════════════╪═══════════════╪══════════════════════╪═════════════════════════╪═════════════════════╡
            │ <anonymous 1>        ┆ M   ┆ ...         ┆ yyy-mm-dd     ┆ [null, 21.92]        ┆ ["diagnosis name", ...] ┆ [age 1, age 2]      │
            │ <anonymous 2>        ┆ F   ┆ ...         ┆ yyy-mm-dd     ┆ [27.1, 75.0, … 91.0] ┆ ["record name", ...]    ┆ [age 1, age 2, … ]  │
            │ …                    ┆ …   ┆ ...         ┆ …             ┆ …                    ┆ …                       ┆ …                   │
            │ <anonymous N>        ┆ F   ┆ ...         ┆ yyy-mm-dd     ┆ [70.0, 0.1, … 80.0]  ┆ ["record name", ...]    ┆ [age 1, age 2, … ]  │
            └──────────────────────┴─────┴─────────────┴───────────────┴──────────────────────┴─────────────────────────┴─────────────────────┘
            with index cols: (age at index, age at start, age at end)

            Baseline covariates include
                * SEX            ("M", "F")
                * ETHNICITY      ()
                * YEAR_OF_BIRTH  (yy-mm-dd format)
                * 
        """
        self.collector.connect()
        
        # Specify which tables to use. 
        #    Static table must always be used as this contains the YEAR_OF_BIRTH used to calculate AGE_AT_EVENT positions
        tables_to_use = ["static_table"]
        if include_measurements:
            tables_to_use.append("measurement_table")            
        if include_diagnoses:            
            tables_to_use.append("diagnosis_table")

        # get a list of practice IDs which are used to chunk the database
        #    We can optionally pass a filtering kwarg through .fit() method to constrain which practice IDs should be included in the study 
        #    based on some criteria. For example, asserting that we only want practices in England can be achieved by adding an inclusion
        #    that applies to the static table
        logging.info(f"Chunking by unique practice ID: {'no' if pratice_inclusion_conditions is None else pratice_inclusion_conditions} inclusion conditions")
        practice_ids = self.collector.extract_practice_ids(["static_table"], "PRACTICE_PATIENT_ID", conditions=pratice_inclusion_conditions)
        
        # Create train/test/val splits 
        logging.info(f"Creating train/test/val splits using practice_patient_ids")
        train_practice_ids, test_practice_ids = sk_split(practice_ids, test_size=0.1)       
        test_practice_ids, val_practice_ids = sk_split(test_practice_ids, test_size=0.5)

        # Collect meta information that is used for tokenization, but also optionally for standardisation
        # Collect meta information
        #    * create the data container on the first pass, and update it after
        #    * count all event occurrences for tokenizer,
        #    * where values are included also record the number observed for standardisation
        #    * and calculating standardisation statistics for the training set
        meta_information = None
        logging.info(f"Collecting meta information of training split for tokenization/standardisation")
        generator = self.collector._lazy_generate_by_practice_id(train_practice_ids, tables_to_use, "PRACTICE_PATIENT_ID")            
        for chunk_name, lazy_table_frames_dict in tqdm(generator, total=len(train_practice_ids)):
            logging.debug(chunk_name)
            meta_information = self.collector._online_standardisation(meta_information, **lazy_table_frames_dict)
        logging.debug(f"Collected meta information \n\n {meta_information}")
        
        #  Create train/test/val DL Polars datasets
        # Loop over practice IDs
        #    * create the generator that performs a lookup on the database for each practice ID and returns a lazy frame for each table's rows        
        #    * collating the lazy tables into a lazy DL-friendly representation,
        for split_name, split_ids in zip(["train", "test", "val"], [train_practice_ids, test_practice_ids, val_practice_ids]):
            
            logging.info(f"Collating {split_name} split into a DL friendly format")
            generator = self.collector._lazy_generate_by_practice_id(split_ids, tables_to_use, "PRACTICE_PATIENT_ID")            
            for chunk_name, lazy_table_frames_dict in tqdm(generator, total=len(split_ids)):

                # Merge the lazy polars tables provided by the generator into one lazy polars frame
                lazy_batch = self.collector._collate_lazy_tables(**lazy_table_frames_dict, **kwargs)

                if save_path is not None:
                    # save splits `hive` style partitioning                # TODO: make directories if they dont already exist
                    # use parquet which is a columnal format               # TODO: is this the best for fast loading? Row format would be faster
                    path: pathlib.Path = f"{save_path}/split={split_name}"    # /{chunk_name}.parquet

                    df = lazy_batch.with_row_count().collect().to_pandas() # include row count so we can lazily filter the in reads
                    table = pa.Table.from_pandas(df)
                    
                    pq.write_to_dataset(table, root_path=path, 
                                        partition_cols=['COUNTRY', 'HEALTH_AUTH', 'PRACTICE_ID'],
                                       )

        self.collector.disconnect()

        return meta_information