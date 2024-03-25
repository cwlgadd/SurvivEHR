# Build deep-Learning friendly representations from each stream of input data (static, diagnosis, measurements) from the reformatted SQL database
from typing import Optional, Any, Union
from collections.abc import Sequence
import itertools
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

    def _train_test_val_split(self):
        
        # get a list of practice IDs which are used to chunk the database
        #    We can optionally pass a filtering kwarg through .fit() method to constrain which practice IDs should be included in the study 
        #    based on some criteria. For example, asserting that we only want practices in England can be achieved by adding an inclusion
        #    that applies to the static table
        logging.info(f"Chunking by unique practice ID with no inclusion conditions")
        practice_ids = self.collector._extract_distinct(table_names=["static_table"],
                                                        identifier_column="PRACTICE_ID",
                                                        # condition= e.g. WHERE 'pass this here'
                                                       )
        
        # Create train/test/val splits 
        logging.info(f"Creating train/test/val splits using practice_ids")
        train_practice_ids, test_practice_ids = sk_split(practice_ids, test_size=0.1)       
        test_practice_ids, val_practice_ids = sk_split(test_practice_ids, test_size=0.5)

        # Get a list of practice_patient_ids for every practice_id used in the training set
        logging.info(f"Extracting practice_patient_ids for each practice")
        # Train
        train_practice_patient_ids = []
        for practice in tqdm(train_practice_ids, desc="Train", total=len(train_practice_ids)):
            train_practice_patient_ids.append(self.collector._extract_distinct(["static_table"], "PRACTICE_PATIENT_ID", conditions=[f"PRACTICE_ID = '{practice}'"]))
        # Test
        test_practice_patient_ids = []
        for practice in tqdm(test_practice_ids, desc="Test", total=len(test_practice_ids)):
            test_practice_patient_ids.append(self.collector._extract_distinct(["static_table"], "PRACTICE_PATIENT_ID", conditions=[f"PRACTICE_ID = '{practice}'"]))
        # Validation
        val_practice_patient_ids = []
        for practice in tqdm(val_practice_ids, desc="Validation", total=len(val_practice_ids)):
            val_practice_patient_ids.append(self.collector._extract_distinct(["static_table"], "PRACTICE_PATIENT_ID", conditions=[f"PRACTICE_ID = '{practice}'"]))

        self.train_practice_patient_ids = train_practice_patient_ids
        self.test_practice_patient_ids = test_practice_patient_ids
        self.val_practice_patient_ids = val_practice_patient_ids
    
    def _build_DL_representation(self,
                                 save_path: str,
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
        
        # Create train/test/val splits
        self._train_test_val_split()
        
        # Collect meta information that is used for tokenization, but also optionally for standardisation        
        all_train = list(itertools.chain.from_iterable(self.train_practice_patient_ids))
        meta_information = self.collector.get_meta_information(practice_patient_ids=all_train,
                                                               diagnoses   = include_diagnoses,
                                                               measurement = include_measurements)
        
        #  Create train/test/val DL Polars datasets
        # Loop over practice IDs
        #    * create the generator that performs a lookup on the database for each practice ID and returns a lazy frame for each table's rows        
        #    * collating the lazy tables into a lazy DL-friendly representation,
        for split_name, split_ids in zip(["train", "test", "val"], [self.train_practice_patient_ids, 
                                                                    self.test_practice_patient_ids, 
                                                                    self.val_practice_patient_ids]):
            
            logging.info(f"Collating {split_name} split into a DL friendly format. Generating over practices IDs")
            generator = self.collector._lazy_generate_by_distinct(distinct_values=split_ids, 
                                                                  identifier_column="PRACTICE_PATIENT_ID",
                                                                  include_diagnoses=include_diagnoses,
                                                                  include_measurements=include_measurements)

            for chunk_name, lazy_table_frames_dict in tqdm(generator, total=len(split_ids)):
                
                # Merge the lazy polars tables provided by the generator into one lazy polars frame
                lazy_batch = self.collector._collate_lazy_tables(lazy_table_frames_dict, **kwargs)

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