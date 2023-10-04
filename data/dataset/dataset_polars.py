# Build deep-Learning friendly representations from each stream of input data (static, diagnosis, measurements) from the reformatted SQL database
from typing import Optional, Any, Union
from collections.abc import Sequence
import sqlite3
import pandas as pd
import polars as pl
import numpy as np
from CPRD.data.database import queries

class DatasetBase:

    
    def __init__(self):
        
        self.path_to_db = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"        
        #    Polars connection
        self.connection_token = 'sqlite://' + self.path_to_db  
        
        
    def _filter_col_inclusion(self,
                              df: pl.LazyFrame, 
                              col_inclusion_targets: dict[str, bool | Sequence[Any]]) -> pl.LazyFrame:
        """ Filter polars lazy frame by column
        """
        filter_exprs = []
        for col, incl_targets in col_inclusion_targets.items():
            match incl_targets:
                case True:
                    filter_exprs.append(pl.col(col).is_not_null())
                case False:
                    filter_exprs.append(pl.col(col).is_null())
                case _:
                    filter_exprs.append(pl.col(col).is_in(list(incl_targets)))

        return df.filter(pl.all(filter_exprs))
        
            
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
        query = "SELECT * FROM static_table"
        q1 = (
            pl.read_database(query=query, connection_uri=self.connection_token).lazy()
            .filter(pl.col("PRACTICE_PATIENT_ID").is_in(practice_patient_ids))
            .collect(streaming=True)
        )
        print(f"static: {type(q1)}: {q1}")

    
    def _load_dynamic(self, practice_patient_ids):
        
        # Can these reads be replaced with a lazy read (i.e. don't load entire tables before converting to lazyframe)
        query = "SELECT * FROM measurement_table"
        measurement_lazy_frame = pl.read_database(query=query, connection_uri=self.connection_token).lazy()
        
        query = "SELECT * FROM diagnosis_table"
        diagnosis_lazy_frame = pl.read_database(query=query, connection_uri=self.connection_token).lazy()
        
        combined_frame = pl.concat([measurement_lazy_frame, diagnosis_lazy_frame])
        
        q1 = (
            combined_frame
            # .join(diagnosis_lazy_frame, left_on=True, how="left")
            .sort("AGE_AT_EVENT")
            .filter(pl.col("PRACTICE_PATIENT_ID").is_in(practice_patient_ids))
            .groupby("PRACTICE_PATIENT_ID")
            .agg(["VALUE", "EVENT", "AGE_AT_EVENT", "EVENT_TYPE"])
            .collect(streaming=True)
        )
        print(f"measurements: {type(q1)}: {q1}")
        

    def build_DL_cached_representation(self,
                                       practice_patient_id: list) -> pl.LazyFrame:

        
        self._load_static(practice_patient_id)
        self._load_dynamic(practice_patient_id)
        

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

