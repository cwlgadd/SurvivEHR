import sqlite3
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

def get_diagnoses_by_PPID(identifier, cursor):
    cursor.execute("SELECT * FROM static_information WHERE PRACTICE_PATIENT_ID=:PRACTICE_PATIENT_ID", {'PRACTICE_PATIENT_ID': identifier})
    return cursor.fetchall()


# def insert_diagnosis(patient, cursor):
#     with conn:
#         cursor.execute("INSERT INTO diagnosis_table VALUES (:PRACTICE_PATIENT_ID, :EVENT, :AGE_AT_EVENT)", 
#                        {'PRACTICE_PATIENT_ID': patient.identifier,
#                         'EVENT': patient.condition, 
#                         'AGE_AT_EVENT': patient.condition_age,
#                         'EVENT_TYPE': "multi_label_classification"
#                        })


def build_diagnosis_table(connector, path_to_data, chunksize=20000, verbose=0):
    r"""
    Build measurements and tests table in database

    Produced anonymized table:
    ┌──────────────────────┬───────┬──────────────┬──────────────┬────────────────────────────┐
    │ PRACTICE_PATIENT_ID  ┆ VALUE ┆ EVENT        ┆ AGE_AT_EVENT ┆ EVENT_TYPE                 │
    │ ---                  ┆ ---   ┆ ---          ┆ ---          ┆ ---                        │
    │ str                  ┆ f64   ┆ str          ┆ i64 (days)   ┆ str                        │
    ╞══════════════════════╪═══════╪══════════════╪══════════════╪════════════════════════════╡
    │ <anonymous 1>        ┆ null  ┆ HF           ┆ 11632        ┆ categorical                │
    │ <anonymous 2>        ┆ null  ┆ HF           ┆ 25635        ┆ categorical                │
    │ …                    ┆ …     ┆ …            ┆ …            ┆ …                          │
    │ <anonymous N>        ┆ null  ┆ FIBROMYALGIA ┆ 8546         ┆ categorical                │
    └──────────────────────┴───────┴──────────────┴──────────────┴────────────────────────────┘
    """

    c = connector.cursor()
    
    c.execute("""CREATE TABLE diagnosis_table (
                 PRACTICE_PATIENT_ID text,
                 VALUE real, 
                 EVENT text,
                 AGE_AT_EVENT integer,
                 EVENT_TYPE text
                 )""")
    
    index_start = 1
    for df in tqdm(pd.read_csv(path_to_data, chunksize=chunksize, iterator=True, encoding='utf-8'), desc="Building diagnosis table"):
        
        # df = df.rename(columns={col: col.replace(' ', '') for col in df.columns}) # Remove spaces from columns 
        
        # Start counting indices from 1
        df.index += index_start
        
        # Remove any un-interesting or censored conditions
        # columns_to_drop = ['']
        # for col in df.columns:
        #     if col in columns_to_drop:
        #         df = df.drop(col, axis=1)
        
        for condition in df.columns[1:]:
            
            # Subset to the ID and condition
            df_condition = df[["PRACTICE_PATIENT_ID", condition]].dropna()
            
            # Add condition as new column
            df_condition["condition"] = condition
            
            # Add empty value for merging tables later
            df_condition["value"] = np.nan
            
            # and rename the condition column 
            df_condition = df_condition.rename(columns={condition: "age_at_diagnosis"}) 
            
            # and order them as we want to see them in the table
            df_condition = df_condition[["PRACTICE_PATIENT_ID", "value", "condition", "age_at_diagnosis"]]
            
            # Add data type
            df_condition["event_type"] = "categorical"

            # Pull records from df to update SQLite .db with
            #   records or rows in a list of tuples [(ID, CONDITION, AGE_AT_DIAGNOSIS),]
            records = df_condition.to_records(column_dtypes={"age_at_diagnosis": "int32"}, index=False)
            # print(records)

            # Add rows to database
            c.executemany('INSERT INTO diagnosis_table VALUES(?,?,?,?,?);', records);
        
            if verbose > 0:
                print('Inserted', c.rowcount, 'records to the table.')

    c.execute("SELECT COUNT(*) FROM diagnosis_table")
    print('\t Diagnosis table built with', c.fetchone()[0], 'records.')
    
    #commit the changes to db			
    connector.commit()
    
    
if __name__ == "__main__":
    """ build database of diagnoses """
    
    PATH_TO_DB = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"
    conn = sqlite3.connect(PATH_TO_DB)
    # conn = sqlite3.connect(':memory:')              # For debugging
        
    
    PATH_TO_DATA = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/diagnosis_history.csv"
    build_diagnosis_table(conn, PATH_TO_DATA)
