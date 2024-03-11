import sqlite3
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import glob
import zipfile
sqlite3.register_adapter(np.int32, lambda val: int(val))


# def get_diagnoses_by_PPID(identifier, cursor):
#     cursor.execute("SELECT * FROM static_information WHERE PRACTICE_PATIENT_ID=:PRACTICE_PATIENT_ID", {'PRACTICE_PATIENT_ID': identifier})
#     return cursor.fetchall()


# def insert_diagnosis(patient, cursor):
#     cursor.execute("INSERT INTO diagnosis_table VALUES (:PRACTICE_PATIENT_ID, :CONDITION, :AGE_AT_DIAGNOSIS)", 
#                    {'PRACTICE_PATIENT_ID': patient.identifier,
#                     'CONDITION': patient.condition, 
#                     'AGE_AT_DIAGNOSIS': patient.condition_age})


def extract_measurement_name(fname):
    # Measurement/test name is contained in the file names following a fixed pattern. Best option is to extract using this
    mname = fname.split("/")[-1]
    return mname[47:-4]


def build_measurements_table(connector, path_to_data, chunksize=200000, unzip=False, verbose=0):
    r""" 
    Build measurements and tests table in database

    Example of produced table (this is not real data):
    ┌──────────────────────┬───────┬──────────────────┬──────────────┐
    │ PRACTICE_PATIENT_ID  ┆ VALUE ┆ EVENT            ┆ AGE_AT_EVENT ┆
    │ ---                  ┆ ---   ┆ ---              ┆ ---          ┆
    │ str                  ┆ f64   ┆ str              ┆ i64 (days)   ┆
    ╞══════════════════════╪═══════╪══════════════════╪══════════════╡
    │ <anonymous 1>        ┆ 23.3  ┆ bmi              ┆ 10254        ┆
    │ <anonymous 1>        ┆ 24.1  ┆ bmi              ┆ 11829        ┆
    │ …                    ┆ …     ┆ …                ┆ …            ┆
    │ <anonymous N>        ┆ 0.17  ┆ eosinophil_count ┆ 12016        ┆
    └──────────────────────┴───────┴──────────────────┴──────────────┴
    """

    c = connector.cursor()
    
    c.execute("""CREATE TABLE measurement_table (
                 PRACTICE_PATIENT_ID text,
                 VALUE real, 
                 EVENT text,
                 DATE text
                 )""")

    path = path_to_data + "*.csv" if unzip is False else path_to_data + "*.zip"
    for fname in glob.glob(path):
        
        mt_name = extract_measurement_name(fname)

        if verbose > 0:
            print(f'Building {mt_name} to the measurements/tests table from \n\t {fname}.')

        generator = pd.read_csv(fname, chunksize=chunksize, iterator=True, low_memory=False, on_bad_lines='skip')
        # low_memory=False just silences an error, TODO: add dtypes
        # on_bad_lines='skip', some lines have extra delimeters from DEXTER bug, handle this by skipping them. This maintains backwards compat
        
        for chunk_idx, df in enumerate(tqdm(generator, desc=f"Adding {mt_name} measurements to table")):
            
            # Start counting indices from 1
            df.index += 1
            
            # DEXTER changed output column header for event date, to keep compatibility use the ordering
            if chunk_idx == 0:
                print(df.columns)
                
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

            if chunk_idx == 0:
                print(f"Using event_date_col {event_date_col}, and event_value_col {event_value_col}")

                
                # event_date_col = df.columns[3]         # TOTAL_ALKALINE_PHOSPHATASE_48EVENT_DATE',       
                # event_value_col = df.columns[4]        # 'TOTAL_ALKALINE_PHOSPHATASE_48Value
                
                
            
            # Subset to the ID and event details
            df = df[["PRACTICE_PATIENT_ID", event_value_col, event_date_col]]

            df.insert(2, 'EVENT', mt_name)

            # print(df.head())

            # Pull records from df to update SQLite .db with
            #   records or rows in a list of tuples [(ID, MEASUREMENT NAME, MEASUREMENT VALUE, AGE AT MEASUREMENT, EVENT TYPE),]
            records = df.to_records(index=False, 
                                        column_dtypes={event_value_col: np.float64,
                                                      # "age_at_event": np.float64,
                                                       }
                                        )
                          
            # Add rows to database
            c.executemany("""INSERT INTO measurement_table  
                             (practice_patient_id, value, event, date) 
                             VALUES
                              (?,?,?,?);
                          """, records);
           
            if verbose > 1:
                print('Inserted', c.rowcount, 'records to the table.')    
            
    c.execute("SELECT COUNT(*) FROM measurement_table")
    print('\t Measurement and test table built with', c.fetchone()[0], 'records.')
    
    #commit the changes to db			
    connector.commit()
    
    
