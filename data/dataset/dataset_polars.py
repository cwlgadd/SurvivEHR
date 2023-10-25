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
    """
    @property
    def DL_frame(self) -> pl.LazyFrame:
        assert self._DL_frame is not None, "DL representation still needs to be built"
        return self._DL_frame
    
    @property
    def identifiers(self) -> list[str]:
        return self.DL_frame["PRACTICE_PATIENT_ID"].to_list()
    
    def __init__(self):
        """
        """
        super().__init__()
        self._DL_frame = None
        
    def __str__(self):
        return str(self._DL_frame)
        
    def load(self):
        raise NotImplementedError
        
    def save(self):
        raise NotImplementedError

    def _load_static(self, practice_patient_ids
                    ) -> pl.LazyFrame:
        r"""
        Load static table from SQL
        
         ARGS:
            practice_patient_id (list[str])
                List of practice patient identifiers which satisfy study criteria.
            
        KWARGS:
        
        
        RETURNS:
            Polars lazy frame, of the (anonymized) form:
            ┌──────────────────────┬─────┬───────────┬───────────────┬─────────────┬─────────────┬───────────────┐
            │ PRACTICE_PATIENT_ID  ┆ SEX ┆ ETHNICITY ┆ YEAR_OF_BIRTH ┆ INDEX_AGE   ┆ START_AGE   ┆ END_AGE       │
            │ ---                  ┆ --- ┆ ---       ┆ ---           ┆ ---         ┆ ---         ┆ ---           │
            │ str                  ┆ str ┆ str       ┆ str           ┆ i64 (days)  ┆ i64 (days)  ┆ i64 (days)    │
            ╞══════════════════════╪═════╪═══════════╪═══════════════╪═════════════╪═════════════╪═══════════════╡
            │ <anonymous 1>        ┆ M   ┆ WHITE     ┆ yyyy--mm-dd   ┆ dd          ┆ dd          ┆ dd            │
            │ <anonymous 2>        ┆ F   ┆ MISSING   ┆ yyyy--mm-dd   ┆ dd          ┆ dd          ┆ dd            │
            │ …                    ┆ …   ┆ …         ┆ …             ┆             ┆             ┆               │
            │ <anonymous N>        ┆ M   ┆ WHITE     ┆ yyyy--mm-dd   ┆ dd          ┆ dd          ┆ dd            │
            └──────────────────────┴─────┴───────────┴───────────────┴─────────────┴─────────────┴───────────────┘
        """
        # TODO: doing any filtering in SQL, before polars, would be better.
        query = "SELECT * FROM static_table"
        static = (
            pl.read_database(query=query, connection_uri=self.connection_token).lazy()
            .filter(pl.col("PRACTICE_PATIENT_ID").is_in(practice_patient_ids))
        )
        return static

    
    def _load_dynamic(self, practice_patient_ids
                     ) -> pl.LazyFrame:
        r"""    
        Load and merge dynamic tables from SQL
        
        ARGS:
            practice_patient_id (list[str])
                List of practice patient identifiers which satisfy study criteria.
            
        KWARGS:
        
        
        RETURNS:
            Polars lazy frame, of the (anonymized) form:
            ┌──────────────────────┬───────────────────────┬───────────────────────────────────┬─────────────────────────┬───────────────────────────────────┐
            │ PRACTICE_PATIENT_ID  ┆ VALUE                 ┆ EVENT                             ┆ AGE_AT_EVENT            ┆ EVENT_TYPE                        │
            │ ---                  ┆ ---                   ┆ ---                               ┆ ---                     ┆ ---                               │
            │ str                  ┆ list[f64]             ┆ list[str]                         ┆ list[i64]   (in days)   ┆ list[str]                         │
            ╞══════════════════════╪═══════════════════════╪═══════════════════════════════════╪═════════════════════════╪═══════════════════════════════════╡
            │ <anonymous 1>        ┆ [null, 21.92]         ┆ ["diagnosis name", "record name"  ┆ [age 1, age 2]          ┆ ["categorical", "univariate_regre │
            │ <anonymous 2>        ┆ [27.1, 75.0, … 91.0]  ┆ ["record name", ...]              ┆ [age 1, age 2, … ]      ┆ ["univariate_regression", "univa… │
            │ …                    ┆ …                     ┆ …                                 ┆ …                       ┆ …                                 │
            │ <anonymous N2>       ┆ [70.0, 0.1, … 80.0]   ┆ ["record name", ...]              ┆ [age 1, age 2, … ]      ┆ ["univariate_regression", "univa… │
            └──────────────────────┴───────────────────────┴───────────────────────────────────┴─────────────────────────┴───────────────────────────────────┘
        """
        # TODO: can these reads be replaced with a lazy read, or be streamable (i.e. don't load entire tables before converting to lazyframe)
        # TODO: filtering patients in SQL, before polars, would be better.

        query = "SELECT * FROM measurement_table"
        measurement_lazy_frame = pl.read_database(query=query, connection_uri=self.connection_token).lazy()

        query = "SELECT * FROM diagnosis_table"
        diagnosis_lazy_frame = pl.read_database(query=query, connection_uri=self.connection_token).lazy()

        # TODO: Any filtering of what subset of events (both diagnosis and records) we are interesting in could be done here, and/or in a downstream tokenizer
        
        combined_frame = pl.concat([measurement_lazy_frame, diagnosis_lazy_frame], how="diagonal")
        
        event_stream = (
            combined_frame
            .sort("AGE_AT_EVENT")
            .filter(pl.col("AGE_AT_EVENT") > -300)                                  # Remove entries before conception - include pregnancy period due to genetic conditions
            .filter(pl.col("PRACTICE_PATIENT_ID").is_in(practice_patient_ids))
            .groupby("PRACTICE_PATIENT_ID")     
            .agg(["VALUE", "EVENT", "AGE_AT_EVENT", "EVENT_TYPE"])                  # Turn into lists
        )
        
        # TODO: Example of an outlier with negative age of blood pressure reading (apparently taken in 19th century) which hasn't been removed. 
        #       Need to perform more outlier detections.
        # measurement_print = (measurement_lazy_frame.collect()
        #             .filter(pl.col("EVENT").is_in(["diastolic_blood_pressure"]))
        #             .filter(pl.col("PRACTICE_PATIENT_ID").is_in(["p20485_2548264020485"]))
        #            )
        # print(measurement_print)
                
        return event_stream
        
    def _build_DL_representation(self,
                                 practice_patient_id: list
                                ) -> pl.LazyFrame:
        r"""
        Build the DL-friendly representation in polars given the list of `practice_patient_id`s which fit study criteria
                
        ARGS:
            practice_patient_id (list[str])
                List of practice patient identifiers which satisfy study criteria.
            
        KWARGS:
        
        
        RETURNS:
            Polars lazy frame, of the (anonymized) form:
            ┌──────────────────────┬─────┬─────────────┬───────────────┬───────────┬──────────────────────┬─────────────────────────┬─────────────────────┬────────────────────────┐
            │ PRACTICE_PATIENT_ID  ┆ SEX ┆ ETHNICITY   ┆ YEAR_OF_BIRTH ┆ index cols┆ VALUE                ┆ EVENT                   ┆ AGE_AT_EVENT        ┆ EVENT_TYPE             │
            │ ---                  ┆ --- ┆ ---         ┆ ---           ┆  ---      ┆ ---                  ┆ ---                     ┆ ---                 ┆ ---                    │
            │ str                  ┆ str ┆ str         ┆ str           ┆int/int/int┆ list[f64]            ┆ list[str]               ┆ list[i64]           ┆ list[str]              │
            ╞══════════════════════╪═════╪═════════════╪═══════════════╪═══┬═══┬═══╪══════════════════════╪═════════════════════════╪═════════════════════╪════════════════════════╡
            │ <anonymous 1>        ┆ M   ┆ MISSING     ┆ yyy-mm-dd     │dd │dd │dd ┆ [null, 21.92]        ┆ ["diagnosis name", ...] ┆ [age 1, age 2]      ┆ ["multi_label_cl...", ]│
            │ <anonymous 2>        ┆ F   ┆ WHITE       ┆ yyy-mm-dd     │dd │dd │dd ┆ [27.1, 75.0, … 91.0] ┆ ["record name", ...]    ┆ [age 1, age 2, … ]  ┆ ["univariate_reg...", ]│
            │ …                    ┆ …                 ┆ …             ┆dd │dd │dd ┆ …                    ┆ …                       ┆ …                   ┆ …                      │
            │ <anonymous N>        ┆ F   ┆ SOUTH_ASIAN ┆ yyy-mm-dd     │dd │dd │dd ┆ [70.0, 0.1, … 80.0]  ┆ ["record name", ...]    ┆ [age 1, age 2, … ]  ┆ ["univariate_reg...", ]│
            └──────────────────────┴─────┴─────────────┴───────────────┴───┴───┴───┴──────────────────────┴─────────────────────────┴─────────────────────┴────────────────────────┘
            with index cols: (age at index, age at start, age at end)

        """
        
        static = self._load_static(practice_patient_id)
        dynamic = self._load_dynamic(practice_patient_id)
        
        static, dynamic = pl.align_frames(static, dynamic, on="PRACTICE_PATIENT_ID")
        
        combined_frame = (
            pl.concat([static.collect(), 
                       dynamic.drop("PRACTICE_PATIENT_ID").collect()], how="horizontal")
        )
        
        self._DL_frame = combined_frame
    
    def _remove_empty(self, drop:bool = True):
        """
        Handle strategies for subjects with no temporal data.
        
        KWARGS:
            drop (bool): 
                True: remove patients which do not have any recorded. False: replace with empty list
        """        
        assert self._DL_frame is not None
        
        # Catch cases with no dynamic values,
        if drop:
            # drop
            self._DL_frame = self._DL_frame.drop_nulls()
        else:
            # replace with empty list
            for col in ["VALUE", "EVENT", "AGE_AT_EVENT", "EVENT_TYPE"]:
                self._DL_frame = self._DL_frame.with_columns(pl.col(col).fill_null(list()))
            raise NotImplementedError #TODO: check this isn't deprecated
            
    def _exclude_events_before_index_age(self):
        """
        Handle starting point for temporal sequences
        """
        assert self._DL_frame is not None

        raise NotImplementedError
    
    def fit(self,
            practice_patient_id: list,
            empty_dynamic_strategy:Optional[str] = None,
            indexing_strategy:Optional[str] = None
           ) -> pl.LazyFrame:
        r"""
        Create Deep-Learning friendly dataset
        
        ARGS:
            practice_patient_id (list[str])
                List of practice patient identifiers which satisfy study criteria.
        
        KWARGS:
            empty_dynamic_strategy (str): 
                Strategy used to remove patients which do not have any recorded temporal events.
            indexing_strategy (str): 
                TODO: Strategy used to dictate at what cut-off do we begin including events. E.g. Remove entries in pre-index CPRD history
        """
            
        print("Building DL-friendly representation")            
        self._build_DL_representation(practice_patient_id=practice_patient_id)
        
        if empty_dynamic_strategy is not None:
            match empty_dynamic_strategy:
                case "drop":
                    print("Dropping samples with no temporal events")
                    self._remove_empty(drop=True)
                case _:
                    self._remove_empty(drop=False)
                    
        if indexing_strategy is not None:            
            print("Removing entries before index date")
            self._exclude_events_before_index_age()
    
        
if __name__ == "__main__":

    # Connect to SQL db on file 
    #    SQLite
    path_to_db = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()
    
    # Example of filtering study population
    if True:
        # Get the list of patients which fit our criterion
        identifiers1 = queries.query_measurement(["bmi", "hydroxyvitamin2", "hydroxyvitamin3"], cursor)
        identifiers2 = queries.query_diagnosis(["HF", "FIBROMYALGIA"], cursor)
        identifiers = list(set(identifiers1).intersection(identifiers2))    # note: try to turn smaller list into the set
        # TODO: turn these into a wrapped query
    else: 
        # Use all unique patients (static table has one row per data owner)
        cursor.execute("SELECT practice_patient_id FROM static_table")
        identifiers = [ppid[0] for ppid in cursor.fetchall()]

    # Create the event stream dataset - this reads the SQL database tables to create a DL friendly representation of the data
    estream = EventStreamDataset()
    estream.fit(identifiers, empty_dynamic_strategy="drop", indexing_strategy=None)
    print(estream)
