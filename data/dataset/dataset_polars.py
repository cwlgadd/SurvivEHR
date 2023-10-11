# Build deep-Learning friendly representations from each stream of input data (static, diagnosis, measurements) from the reformatted SQL database
from typing import Optional, Any, Union
from collections.abc import Sequence
import sqlite3
import pandas as pd
import polars as pl
import numpy as np
from CPRD.data.database import queries


class DatasetBase:
    r"""
    """
    
    def __init__(self):
        
        self.path_to_db = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"        
        #    Polars connection
        self.connection_token = 'sqlite://' + self.path_to_db  
      
            
class EventStreamDataset(DatasetBase):
    r"""
        Combine data streams
        
        will perform:
            - Combining of time ordered stream of events
            - Tokenization
    """
    
    def __init__(self):
        
        super().__init__()
        

    def _load_static(self, practice_patient_ids):
        r"""
        
        Anonymized example:
        ┌───────────────────────┬─────┬─────────────┬───────────────┐
        │ PRACTICE_PATIENT_ID   ┆ SEX ┆ ETHNICITY   ┆ YEAR_OF_BIRTH │
        │ ---                   ┆ --- ┆ ---         ┆ ---           │
        │ str                   ┆ str ┆ str         ┆ str           │
        ╞═══════════════════════╪═════╪═════════════╪═══════════════╡
        │ <anonymous 1>         ┆ M   ┆ MISSING     ┆ yyy-mm-dd     │
        │ <anonymous 2>         ┆ F   ┆ WHITE       ┆ yyy-mm-dd     │
        │ …                     ┆ …   ┆ …           ┆ …             │
        │ <anonymous N>         ┆ F   ┆ SOUTH_ASIAN ┆ yyy-mm-dd     │
        └───────────────────────┴─────┴─────────────┴───────────────┘
        """
        # TODO: doing this filtering in SQL, before polars, would be better.
        query = "SELECT * FROM static_table"
        static = (
            pl.read_database(query=query, connection_uri=self.connection_token).lazy()
            .filter(pl.col("PRACTICE_PATIENT_ID").is_in(practice_patient_ids))
        )
        
        return static

    
    def _load_dynamic(self, practice_patient_ids):
        r"""
        
        Anonymized example:
        ┌──────────────────────┬───────────────────────┬───────────────────────────────────┬─────────────────────────┬───────────────────────────────────┐
        │ PRACTICE_PATIENT_ID  ┆ VALUE                 ┆ EVENT                             ┆ AGE_AT_EVENT            ┆ EVENT_TYPE                        │
        │ ---                  ┆ ---                   ┆ ---                               ┆ ---                     ┆ ---                               │
        │ str                  ┆ list[f64]             ┆ list[str]                         ┆ list[i64]   (in days)   ┆ list[str]                         │
        ╞══════════════════════╪═══════════════════════╪═══════════════════════════════════╪═════════════════════════╪═══════════════════════════════════╡
        │ <anonymous 1>        ┆ [null, 21.92]         ┆ ["diagnosis name", "record name"  ┆ [age 1, age 2]          ┆ ["multi_label_classification", "un│
        │ <anonymous 2>        ┆ [27.1, 75.0, … 91.0]  ┆ ["record name", ...]              ┆ [age 1, age 2, … ]      ┆ ["univariate_regression", "univa… │
        │ …                    ┆ …                     ┆ …                                 ┆ …                       ┆ …                                 │
        │ <anonymous N>        ┆ [70.0, 0.1, … 80.0]   ┆ ["record name", ...]              ┆ [age 1, age 2, … ]      ┆ ["univariate_regression", "univa… │
        └──────────────────────┴───────────────────────┴───────────────────────────────────┴─────────────────────────┴───────────────────────────────────┘
        """
        # TODO: can these reads be replaced with a lazy read, or be streamable (i.e. don't load entire tables before converting to lazyframe)
        # TODO: filtering patients in SQL, before polars, would be better.

        query = "SELECT * FROM measurement_table"
        measurement_lazy_frame = pl.read_database(query=query, connection_uri=self.connection_token).lazy()
        
        query = "SELECT * FROM diagnosis_table"
        diagnosis_lazy_frame = pl.read_database(query=query, connection_uri=self.connection_token).lazy()
        
        combined_frame = pl.concat([measurement_lazy_frame, diagnosis_lazy_frame], how="diagonal")
        
        event_stream = (
            combined_frame
            .sort("AGE_AT_EVENT")
            .filter(pl.col("PRACTICE_PATIENT_ID").is_in(practice_patient_ids))
            .groupby("PRACTICE_PATIENT_ID")     
            .agg(["VALUE", "EVENT", "AGE_AT_EVENT", "EVENT_TYPE"])                  # Turn into lists
        )
        
        return event_stream
        
    def load_cache(self):
        raise NotImplementedError
        
    def save_cache(self):
        raise NotImplementedError
        
    def tokenizer(self):
        return
    
    def build_DL_cached_representation(self,
                                       practice_patient_id: list,
                                       remove_empty_events:bool = False) -> pl.LazyFrame:
        r"""
        Build the DL-friendly representation in polars given the list of `practice_patient_id`s that fits study criteria
        
        TODO: 
            allow caching with saving/loading
            replace upstream data preprocessing (across Dexter -> R -> sqlite), to a single package

        Anonymized example:
        ┌──────────────────────┬─────┬─────────────┬───────────────┬──────────────────────┬─────────────────────────┬─────────────────────┬────────────────────────┐
        │ PRACTICE_PATIENT_ID  ┆ SEX ┆ ETHNICITY   ┆ YEAR_OF_BIRTH ┆ VALUE                ┆ EVENT                   ┆ AGE_AT_EVENT        ┆ EVENT_TYPE             │
        │ ---                  ┆ --- ┆ ---         ┆ ---           ┆ ---                  ┆ ---                     ┆ ---                 ┆ ---                    │
        │ str                  ┆ str ┆ str         ┆ str           ┆ list[f64]            ┆ list[str]               ┆ list[i64]           ┆ list[str]              │
        ╞══════════════════════╪═════╪═════════════╪═══════════════╪══════════════════════╪═════════════════════════╪═════════════════════╪════════════════════════╡
        │ <anonymous 1>        ┆ M   ┆ MISSING     ┆ yyy-mm-dd     │ [null, 21.92]        ┆ ["diagnosis name", ...] ┆ [age 1, age 2]      ┆ ["multi_label_cl...", ]│
        │ <anonymous 2>        ┆ F   ┆ WHITE       ┆ yyy-mm-dd     │ [27.1, 75.0, … 91.0] ┆ ["record name", ...]    ┆ [age 1, age 2, … ]  ┆ ["univariate_reg...", ]│
        │ …                    ┆ …                 ┆ …             ┆ …                    ┆ …                       ┆ …                   ┆ …                      │
        │ <anonymous N>        ┆ F   ┆ SOUTH_ASIAN ┆ yyy-mm-dd     │ [70.0, 0.1, … 80.0]  ┆ ["record name", ...]    ┆ [age 1, age 2, … ]  ┆ ["univariate_reg...", ]│
        └──────────────────────┴─────┴─────────────┴───────────────┴──────────────────────┴─────────────────────────┴─────────────────────┴────────────────────────┘
        
        ARGS:
            
        
        KWARGS:
            remove_empty_events (bool): True: remove patients which do not have any recorded
            dynamic (event stream) events. False: remove nulls with empty list.
        """
        
        print("Building DL-friendly representation")
        static = self._load_static(practice_patient_id)
        dynamic = self._load_dynamic(practice_patient_id)
        print(static.collect())
        print(dynamic.collect())
        
        static, dynamic = pl.align_frames(static, dynamic, on="PRACTICE_PATIENT_ID")
        # print(static.collect())
        # print(dynamic.collect())
        
        combined_frame = (
            pl.concat([static.collect(), 
                       dynamic.drop("PRACTICE_PATIENT_ID").collect()], how="horizontal")
        )
        # print(combined_frame)
        
        # For cases with no dynamic values, replace with empty list, or drop
        if remove_empty_events:
            combined_frame = combined_frame.drop_nulls()
        else:
            for cols in ["VALUE", "EVENT", "AGE_AT_EVENT", "EVENT_TYPE"]:
                combined_frame = (
                    combined_frame.with_columns(pl.col('VALUE').fill_null(list()))
                )

        return combined_frame

        
if __name__ == "__main__":

    # Connect to SQL db on file 
    #    SQLite
    path_to_db = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()
    
    # Example of filtering
    if False:
        # Get the list of patients which fit our criterion
        identifiers1 = queries.query_measurement(["bmi", "hydroxyvitamin2", "hydroxyvitamin3"], cursor)
        identifiers2 = queries.query_diagnosis(["HF", "FIBROMYALGIA"], cursor)
        identifiers = list(set(identifiers1).intersection(identifiers2))    # Turn smaller list into the set
    else: 
        cursor.execute("SELECT practice_patient_id FROM static_table")
        identifiers = [ppid[0] for ppid in cursor.fetchall()]

    # Create event stream - this interfaces with the SQL database to create a DL friendly representation of the data
    estream = EventStreamDataset()
    
    # Example measurement filter (acting on SQL table via sqlite)
    estream.build_DL_cached_representation(identifiers)

