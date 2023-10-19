import sqlite3
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
sqlite3.register_adapter(np.int32, lambda val: int(val))


# def get_diagnoses_by_PPID(identifier, cursor):
#     cursor.execute("SELECT * FROM static_information WHERE PRACTICE_PATIENT_ID=:PRACTICE_PATIENT_ID", {'PRACTICE_PATIENT_ID': identifier})
#     return cursor.fetchall()


# def insert_diagnosis(patient, cursor):
#     cursor.execute("INSERT INTO diagnosis_table VALUES (:PRACTICE_PATIENT_ID, :CONDITION, :AGE_AT_DIAGNOSIS)", 
#                    {'PRACTICE_PATIENT_ID': patient.identifier,
#                     'CONDITION': patient.condition, 
#                     'AGE_AT_DIAGNOSIS': patient.condition_age})


def build_measurements_table(connector, path_to_data, chunksize=20000, verbose=0):
    r""" 
    Build measurements and tests table in database

    Produced anonymized table:
    ┌──────────────────────┬───────┬──────────────────┬──────────────┬───────────────────────┐
    │ PRACTICE_PATIENT_ID  ┆ VALUE ┆ EVENT            ┆ AGE_AT_EVENT ┆ EVENT_TYPE            │
    │ ---                  ┆ ---   ┆ ---              ┆ ---          ┆ ---                   │
    │ str                  ┆ f64   ┆ str              ┆ i64 (days)   ┆ str                   │
    ╞══════════════════════╪═══════╪══════════════════╪══════════════╪═══════════════════════╡
    │ <anonymous 1>        ┆ 23.3  ┆ bmi              ┆ 10254        ┆ univariate_regression │
    │ <anonymous 1>        ┆ 24.1  ┆ bmi              ┆ 11829        ┆ univariate_regression │
    │ …                    ┆ …     ┆ …                ┆ …            ┆ …                     │
    │ <anonymous N>        ┆ 0.17  ┆ eosinophil_count ┆ 12016        ┆ univariate_regression │
    └──────────────────────┴───────┴──────────────────┴──────────────┴───────────────────────┘
    """

    c = connector.cursor()
    
    c.execute("""CREATE TABLE measurement_table (
                 PRACTICE_PATIENT_ID text,
                 VALUE real, 
                 EVENT text,
                 AGE_AT_EVENT integer,
                 EVENT_TYPE text
                 )""")
    
    index_start = 1
    for df in tqdm(pd.read_csv(path_to_data, chunksize=chunksize, iterator=True), desc="Building measurements table"):
        # df = df.rename(columns={col: col.replace(' ', '') for col in df.columns}) # Remove spaces from columns (not used: dded for consistency)
        
        # Start counting indices from 1
        df.index += index_start
        
        # Pull records from df to update SQLite .db with
        #   records or rows in a list of tuples [(ID, MEASUREMENT NAME, MEASUREMENT VALUE, AGE AT MEASUREMENT, EVENT TYPE),]
        records = df.to_records(index=False, column_dtypes={"value": np.float64,
                                                            "age_at_event": np.float64,
                                                           })
                      
        # Add rows to database
        c.executemany("""INSERT INTO measurement_table  
                         (practice_patient_id, value, event, age_at_event, event_type) 
                         VALUES
                          (?,?,?,?,?);
                      """, records);
       
        if verbose > 0:
            print('Inserted', c.rowcount, 'records to the table.')    
            
    c.execute("SELECT COUNT(*) FROM measurement_table")
    print('\t Measurement and test table built with', c.fetchone()[0], 'records.')
    
    #commit the changes to db			
    connector.commit()
    
    
if __name__ == "__main__":
    """ build database of diagnoses """
    
    PATH_TO_DB = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"
    conn = sqlite3.connect(PATH_TO_DB)
    # conn = sqlite3.connect(':memory:')              # For debugging
        
    
    PATH_TO_DATA = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/measurements.csv"
    build_measurements_table(conn, PATH_TO_DATA)

   
