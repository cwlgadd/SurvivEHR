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

        # we can remove the file extension (either .csv or .zip)
        mname = mname[:-4]
        
        # depending on DEXTER output version there are prefixes on filenames which we can remove
        prefixes = ["AVF2_masterDataOptimal_v3_fullDB20231112045951_",
                    "AVF2_masterDataOptimal_v220230327110229_",
                    "AVF1_masterDataOptimal_v3_fullDB20231112044822_"]
        for prefix in prefixes:
            if mname.startswith(prefix):
                mname = mname[len(prefix):]

        # and we remove characters which will confuse SQL commands
        mname = mname.replace("-", "_")
        mname = mname.replace(".", "")
                
        return mname

    @property
    def measurement_table_names(self):
        self.cursor.execute("""SELECT * FROM sqlite_master;""")
        names = [_a[1]  for _a in self.cursor.fetchall() if _a[0] == "table" and _a[1].startswith("measurement")]
        return names

    @property
    def query_measurement_aggregations(self):
        pass
    
    def __init__(self, db_path, path_to_data, load=False):
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        self.connection_token = 'sqlite://' + self.db_path 
        self.path_to_data = path_to_data

        if load is False:
            self.build_table()

    def __str__(self):
        self.connect()
        s = "Measurement table:"
        s += "\nMeasurement & Count (thousands)"
        total_count = 0
        for table in self.measurement_table_names:
            self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
            measurement = table[12:]
            count = self.cursor.fetchone()[0]
            s += f'\n{measurement}:'.ljust(40) + f'& {count:,}'.rjust(15) 
            total_count += count
        s += "\nTotal".ljust(40) + f"& {total_count}"
        
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

    def build_table(self, unzip=True, verbose=1, **kwargs):
        r""" 
        Build measurements and tests table in database
    
        """
    
        self.connect()

        # Fill table
        # Each file is a table which partitions measurements
        path = self.path_to_data + "*.csv" if unzip is False else self.path_to_data + "*.zip"
        for filename in sorted(glob.glob(path)):
        
            measurement_name = self.extract_measurement_name(filename)

            self._create_measurement_partition(measurement_name)
            self._file_to_measurement_table(filename, measurement_name, verbose=verbose, **kwargs)
            
            self.connection.commit()
            
        self.disconnect()
    
    def _create_measurement_partition(self, measurement_name):
        
        self.cursor.execute("""DROP TABLE IF EXISTS measurement_""" + measurement_name)
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS measurement_""" + measurement_name + """ ( 
                                        PRACTICE_ID int,
                                        PATIENT_ID int,
                                        EVENT text,
                                        VALUE real,                                                                 
                                        DATE text )""")

        # Create index
        logging.debug(f"Creating PRACTICE_PATIENT_ID index on measurement_{measurement_name}")
        for index in ["PRACTICE_ID"]:
            query = f"CREATE INDEX IF NOT EXISTS '{measurement_name}_{index}_idx' ON measurement_{measurement_name} ({index});"
            logging.debug(query)
            self.cursor.execute(query)

    def _file_to_measurement_table(self, filename, measurement_name, chunksize=200000, verbose=0):
        """
        """
    
        logging.debug(f'Inserting {measurement_name} into table from \n\t {filename}.')
    
        generator = pd.read_csv(filename, chunksize=chunksize, iterator=True, low_memory=False, on_bad_lines='skip',
                                dtype={'PRACTICE_PATIENT_ID': 'str'})
        # low_memory=False just silences an error, TODO: add dtypes
        # on_bad_lines='skip', some lines have extra delimeters from DEXTER bug, handle this by skipping them. This maintains backwards compat
        for chunk_idx, df in enumerate(tqdm(generator, desc=f"Adding {measurement_name}".ljust(70))):

            # DEXTER gives multiple different file formats in the measurement files
            file_columns = df.columns
            event_date_col, event_value_col = None, None
            for colname in file_columns:
                # get the column name which contains the value, checking across all the column names used by DEXTER
                if colname.lower().endswith("value"):
                    event_value_col = colname
                # elif colname.lower().endswith(measurement_name.lower()):
                #     event_value_col = colname
    
                # get the column name which contains the date, checking across all the column names used by DEXTER
                if colname.lower().endswith("event_date"):
                    event_date_col = colname
                if colname.lower().endswith("event_date)"):
                    event_date_col = colname
            assert event_date_col is not None #and event_value_col is not None

            # if there is no value column then create one and fill with np.nans
            if event_value_col is None:
                event_value_col = "value"
                df.insert(1, event_value_col, None)

            # Add practice ID column
            split = df['PRACTICE_PATIENT_ID'].str.split('_')
            df['PRACTICE_ID'] = split.str[0].str.lstrip('p')
            df['PATIENT_ID'] = split.str[1]
            
            # Subset to the ID and event details
            df = df[["PRACTICE_ID", "PATIENT_ID", event_value_col, event_date_col]].copy()
            df.insert(2, 'EVENT', measurement_name)

            # Pull records from df to update SQLite .db with
            #   records or rows in a list of tuples [(ID, MEASUREMENT NAME, MEASUREMENT VALUE, AGE AT MEASUREMENT, EVENT TYPE),]
            records = df.to_records(index=False,
                                    # column_dtypes={
                                    #     event_value_col: np.float64,                                        
                                    #     "PRACTICE_ID": "int64",
                                    #     "PATIENT_ID": "int64",
                                    #     }
                                    )
            if chunk_idx == 0:
                logging.info(f"Used event_date_col {event_date_col}, and event_value_col {event_value_col}")
                logging.info(f"Selected from available columns {file_columns.tolist()}")
                # logging.debug(records)
                          
            self._records_to_table_measurement(records, measurement_name)

    def _records_to_table_measurement(self, records, measurement_name, **kwargs):

            
        # Add rows to database....... (practice_id, patient_id, value, event, date) 
        self.cursor.executemany('INSERT INTO measurement_' + measurement_name + ' VALUES(?,?,?,?,?);', records);           # Add rows to database

        logging.debug(f'Inserted {self.cursor.rowcount} records to the table.')
