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
                try:
                    with open(path + 'meta_information_edited.pickle', 'rb') as handle:
                       self.meta_information = pickle.load(handle)
                    logging.info(f"Loaded Polars dataset from {path}. Using edited version of meta_information")
                except:
                    with open(path + 'meta_information.pickle', 'rb') as handle:
                       self.meta_information = pickle.load(handle)
                    logging.info(f"Loaded Polars dataset from {path}.")
            except OSError as e:
                raise FileNotFoundError
        else:
            logging.info(f"Building Polars dataset and saving to {path}")     
            self.meta_information = self._build_DL_representation(save_path=path, **kwargs)            
            with open(path + 'meta_information.pickle', 'wb') as handle:
                pickle.dump(self.meta_information, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        return self.meta_information

    def _train_test_val_split(self, inclusion_conditions=None):
        
        # get a list of practice IDs which are used to chunk the database
        #    We can optionally pass a filtering kwarg through .fit() method to constrain which practice IDs should be included in the study 
        #    based on some criteria. For example, asserting that we only want practices in England can be achieved by adding an inclusion
        #    that applies to the static table
        logging.info(f"Chunking by unique practice ID with {'no' if inclusion_conditions is None else inclusion_conditions} inclusion conditions")
        practice_ids = self.collector._extract_distinct(table_names=["static_table"],
                                                        identifier_column="PRACTICE_ID",
                                                        inclusion_conditions=inclusion_conditions
                                                       )
        
        # Create train/test/val splits, each is list of practice_id
        logging.info(f"Creating train/test/val splits using practice_ids")
        self.train_practice_ids, test_practice_ids = sk_split(practice_ids, test_size=0.1)       
        self.test_practice_ids, self.val_practice_ids = sk_split(test_practice_ids, test_size=0.5)
    
    def _build_DL_representation(self,
                                 save_path:               str,
                                 include_static:          bool = True,
                                 include_measurements:    bool = True,
                                 include_diagnoses:       bool = True,
                                 preprocess_measurements: bool = False,
                                 inclusion_conditions:    Optional[str] = False,
                                 **kwargs,
                                ) -> pl.LazyFrame:
        r"""
        Build the DL-friendly representation in polars given the list of `practice_patient_id`s which fit study criteria
                
        ARGS:
            
        KWARGS:
        
        """
        self.collector.connect()
        
        # Create train/test/val splits
        self._train_test_val_split(inclusion_conditions=inclusion_conditions)
        
        # Collect meta information that is used for tokenization, but also optionally for standardisation        
        # all_train = list(itertools.chain.from_iterable(self.train_practice_ids))
        meta_information = self.collector.get_meta_information(practice_ids = None, #self.train_practice_ids,
                                                               static       = include_static,
                                                               diagnoses    = include_diagnoses,
                                                               measurement  = include_measurements
                                                              )
        logging.debug(meta_information)
        
        #  Create train/test/val DL Polars datasets
        # Loop over practice IDs
        #    * create the generator that performs a lookup on the database for each practice ID and returns a lazy frame for each table's rows        
        #    * collating the lazy tables into a lazy DL-friendly representation,
        for split_name, split_ids in zip(["train", "test", "val"], [self.train_practice_ids,  self.test_practice_ids,  self.val_practice_ids]):

            # Create the generator which we will chunk over
            logging.info(f"Collating {split_name} split into a DL friendly format. Generating over practices IDs")
            generator = self.collector._generate_lazy_by_distinct(distinct_values=split_ids,
                                                                  identifier_column="PRACTICE_ID",
                                                                  include_diagnoses=include_diagnoses,
                                                                  include_measurements=include_measurements)
            
            for chunk_name, lazy_table_frames_dict in tqdm(generator, total=len(split_ids)):
                
                # Merge the lazy polars tables provided by the generator into one lazy polars frame
                lazy_batch = self.collector._collate_lazy_tables(lazy_table_frames_dict, **kwargs)

                if save_path is not None:
                    # save splits `hive` style partitioning                # TODO: make directories if they dont already exist
                    # use parquet which is a columnal format               # TODO: is this the best for fast loading? Row format would be faster
                    path: pathlib.Path = f"{save_path}/split={split_name}"    # /{chunk_name}.parquet

                    df = lazy_batch.collect().to_pandas() # include row count so we can lazily filter the in reads  .with_row_count()
                    
                    logging.debug(f'{df.iloc[0]["COUNTRY"]}, {df.iloc[0]["HEALTH_AUTH"]}, {df.iloc[0]["PRACTICE_ID"]}')

                    for df_chunk in [df[i:i+500] for i in range(0, len(df), 500)]:

                        df_chunk = df_chunk.copy()
                        df_chunk['row_nr'] = np.arange(len(df_chunk))

                        table = pa.Table.from_pandas(df_chunk)
                        pq.write_to_dataset(table, root_path=path, 
                                            partition_cols=['COUNTRY', 'HEALTH_AUTH', 'PRACTICE_ID'],
                                           )
                        # ds.write_dataset(table, path, 
                        #                  format="parquet",
                        #                  partitioning=['COUNTRY', 'HEALTH_AUTH', 'PRACTICE_ID'],
                        #                  max_rows_per_file=500,
                        #                  max_rows_per_group=500,
                        #                    )

        self.collector.disconnect()

        return meta_information