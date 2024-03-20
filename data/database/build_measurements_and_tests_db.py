import sqlite3
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import glob
import zipfile
import multiprocessing
import logging
sqlite3.register_adapter(np.int32, lambda val: int(val))
import time

class Measurements():

    @staticmethod
    def extract_measurement_name(fname):
        # Measurement/test name is contained in the file names following version dependent prefixes. Remove them.
        mname = fname.split("/")[-1]
        prefixes = ["AVF2_masterDataOptimal_v3_fullDB20231112045951_",
                    "AVF2_masterDataOptimal_v220230327110229_"]
        for prefix in prefixes:
            if mname.startswith(prefix):
                return mname[len(prefix):-4]
                
        return mname[:-4]
    
    def __init__(self, db_path, path_to_data, load=False):
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        self.connection_token = 'sqlite://' + self.db_path 
        self.path_to_data = path_to_data

        if load is False:
            # Create table                     
            self.connect()
            logging.info(f"Creating measurement_table")
            self.cursor.execute("""CREATE TABLE measurement_table ( PRACTICE_PATIENT_ID str,
                                                                    EVENT text,
                                                                    VALUE real,                                                                 
                                                                    DATE text )""")
            
            self.disconnect()

    def __str__(self):
        self.connect()
        self.cursor.execute("SELECT COUNT(*) FROM measurement_table")
        s = f'Measurement table with {self.cursor.fetchone()[0]/1e6:.2f}M records.'
        return s
        
    def connect(self):
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            logging.debug("Connected to SQLite database")
        except sqlite3.Error as e:
            logging.warning(f"Error connecting to SQLite database: {e}")

    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            self.cursor = None
            logging.debug("Disconnected from SQLite database")

    def build_table(self, unzip=False, verbose=1, **kwargs):
        r""" 
        Build measurements and tests table in database
    
        Example of produced table (this is not real data):
        ┌──────────────────────┬───────┬──────────────────┬──────────────┐
        │ PRACTICE_PATIENT_ID  ┆ VALUE ┆ EVENT            ┆  ┆
        │ ---                  ┆ ---   ┆ ---              ┆ ---          ┆
        │ str                  ┆ f64   ┆ str              ┆ i64 (days)   ┆
        ╞══════════════════════╪═══════╪══════════════════╪══════════════╡
        │ <anonymous 1>        ┆ 23.3  ┆ bmi              ┆ 10254        ┆
        │ <anonymous 1>        ┆ 24.1  ┆ bmi              ┆ 11829        ┆
        │ …                    ┆ …     ┆ …                ┆ …            ┆
        │ <anonymous N>        ┆ 0.17  ┆ eosinophil_count ┆ 12016        ┆
        └──────────────────────┴───────┴──────────────────┴──────────────┴
        """
    
        self.connect()

        self.cursor.execute( """PRAGMA temp_store = 1""")
        
        # Fill table
        path = self.path_to_data + "*.csv" if unzip is False else path_to_data + "*.zip"
        for fname in sorted(glob.glob(path)):
            self._add_file_to_table(fname, verbose=verbose, **kwargs)
            self.connection.commit()
            
        self._make_index()

        self.disconnect()
    
    def _make_index(self):
        # Create index
        logging.info("Creating indexes on measurement_table")
        for index in ["PRACTICE_PATIENT_ID", "EVENT"]:
            logging.info(index)
            query = f"CREATE INDEX IF NOT EXISTS measurement_{index}_idx ON measurement_table ({index});"
            logging.debug(query)
            self.cursor.execute(query)
            self.connection.commit()

            # query = f"PRAGMA index_list('measurement_table');"
            # logging.debug(query)
            # self.cursor.execute(query)
            # result = self.cursor.fetchall()
            # print(result)

    def _delete_index(self):
        pass

    def _add_file_to_table(self, fname, chunksize=200000, verbose=0):
        
        mt_name = self.extract_measurement_name(fname)
    
        if verbose > 0:
            print(f'Inserting {mt_name} into table from \n\t {fname}.')
    
        generator = pd.read_csv(fname, chunksize=chunksize, iterator=True, low_memory=False, on_bad_lines='skip')
        # low_memory=False just silences an error, TODO: add dtypes
        # on_bad_lines='skip', some lines have extra delimeters from DEXTER bug, handle this by skipping them. This maintains backwards compat
        for chunk_idx, df in enumerate(tqdm(generator, desc=f"Adding {mt_name}".ljust(70))):

            # Start counting indices from 1
            df.index += 1
            
            # DEXTER changed output column header for event date, to keep compatibility use the ordering
            if chunk_idx == 0 and verbose > 2:
                print(df.columns.tolist())
                
            event_date_col, event_value_col = None, None
            for colname in df.columns:
                # get the column name which contains the value, checking across all the column names used by DEXTER
                if colname.lower().endswith("value"):
                    event_value_col = colname
                elif colname.lower().endswith(mt_name.lower()):
                    event_value_col = colname
    
                # get the column name which contains the date, checking across all the column names used by DEXTER
                if colname.lower().endswith("event_date"):
                    event_date_col = colname
                if colname.lower().endswith("event_date)"):
                    event_date_col = colname

            if chunk_idx == 0 and verbose > 2:
                print(f"Using event_date_col {event_date_col}, and event_value_col {event_value_col}")
    
            # Subset to the ID and event details
            df = df[["PRACTICE_PATIENT_ID", event_value_col, event_date_col]].copy()

            df.insert(1, 'EVENT', mt_name)
    
            # Pull records from df to update SQLite .db with
            #   records or rows in a list of tuples [(ID, MEASUREMENT NAME, MEASUREMENT VALUE, AGE AT MEASUREMENT, EVENT TYPE),]
            records = df.to_records(index=False,
                                    column_dtypes={
                                        event_value_col: np.float64,
                                        }
                                    )
            if chunk_idx == 0 and verbose > 2:
                print(records)
                          
            # Add rows to database....... (practice_id, patient_id, value, event, date) 
            self.cursor.executemany('INSERT INTO measurement_table VALUES(?,?,?,?);', records);           # Add rows to database

            if verbose > 1:
                print('Inserted', self.cursor.rowcount, 'records to the table.')    

    
    
