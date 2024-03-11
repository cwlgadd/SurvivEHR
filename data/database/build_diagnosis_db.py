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


def build_diagnosis_table(connector, path_to_data, chunksize=200000, verbose=0):
    r"""
    Build measurements and tests table in database

    Produced anonymized table:
    ┌──────────────────────┬──────────────┬──────────────┐
    │ PRACTICE_PATIENT_ID  ┆ EVENT        ┆ AGE_AT_EVENT │
    │ ---                  ┆ ---          ┆ ---          │
    │ str                  ┆ str          ┆ i64 (days)   │
    ╞══════════════════════╪══════════════╪══════════════╡
    │ <anonymous 1>        ┆ HF           ┆ 11632        │
    │ <anonymous 1>        ┆ HF           ┆ 25635        │
    │ …                    ┆ …            ┆ …            │
    │ <anonymous N>        ┆              ┆ 10546        │
    │ <anonymous N>        ┆ DEATH        ┆ 27546        │
    └──────────────────────┴──────────────┴──────────────┘

    TODO: I'm sure much of this can be optimised
    
    """

    c = connector.cursor()
    
    c.execute("""CREATE TABLE diagnosis_table (
                 PRACTICE_PATIENT_ID text,
                 EVENT text,
                 DATE text
                 )""")
    
    generator = pd.read_csv(path_to_data, chunksize=chunksize, iterator=True, encoding='utf-8', low_memory=False)
    # low_memory=False just silences an error, TODO: add dtypes
    for df in tqdm(generator, desc="Building diagnosis table"):
        
        # Start indices from 1
        df.index += 1

        #####################
        # Conditions
        #####################
        # Rename the column headers: Get diagnosis columns and a mapping to re-name them to something more appropriate
        date_columns = df.columns[list(range(19,164,2))]                                                           # Take only the columns with diagnosis dates
        condition_names = [_condition.removeprefix('BD_MEDI:') for _condition in date_columns]                     # Remove pre-fix
        for replace in ["_BHAM_CAM", "_FINAL", "_BIRM_CAM", "_MM", "_11_3_21", "_20092020", "_120421"]:            #   and a number of polluting values in titles
            condition_names = [_condition.replace(replace, '') for _condition in condition_names]
        condition_names = [ _condition.split(":", 1)[0] for _condition in condition_names]                         # and finally strip the condition number
        rename_dict = dict(zip(date_columns, condition_names))
        
        # Convert to days since birth: Get dates of diagnosis and year of birth so we can calculate the time difference
        # date_format = '%Y-%m-%d'
        # for _dcondition in date_columns:
            # df[_dcondition] = (pd.to_datetime(df[_dcondition], format=date_format) - pd.to_datetime(df["YEAR_OF_BIRTH"],format=date_format)).dt.days
            # df[_dcondition] = pd.to_datetime(df[_dcondition], format=date_format).dt.strftime(date_format)

        # Rename and subset dataframe
        df = df.rename(columns=rename_dict)
        df_conditions = df[["PRACTICE_PATIENT_ID"] + condition_names]
        
        # Remove any un-interesting or censored conditions
        # columns_to_drop = ['']
        # for col in df.columns:
        #     if col in columns_to_drop:
        #         df = df.drop(col, axis=1)

        for condition in df_conditions.columns[1:]:
            
            # Subset to the ID and condition
            df_one_condition = df_conditions[["PRACTICE_PATIENT_ID", condition]].dropna()
            
            # Add condition as new column
            df_one_condition["condition"] = condition
            
            # and rename the condition column 
            df_one_condition = df_one_condition.rename(columns={condition: "age_at_diagnosis"})   
            
            # and order them as we want to see them in the table
            df_one_condition = df_one_condition[["PRACTICE_PATIENT_ID", "condition", "age_at_diagnosis"]]
            
            # Pull records from df to update SQLite .db with
            #   records or rows in a list of tuples [(ID, CONDITION, AGE_AT_DIAGNOSIS),]
            records = df_one_condition.to_records(index=False,
                                                  # column_dtypes={"age_at_diagnosis": "int32"}, 
                                                 )
            c.executemany('INSERT INTO diagnosis_table VALUES(?,?,?);', records);           # Add rows to database
        
            if verbose > 1:
                print(f'Inserted {c.rowcount} {condition} records to the table.')

        #####################
        # For death
        #####################
        # Subset to the ID and death
        df_death = df[["PRACTICE_PATIENT_ID", "DEATH_DATE", "YEAR_OF_BIRTH"]].dropna()
        # df_death["age_at_diagnosis"] = (pd.to_datetime(df_death["DEATH_DATE"], format=date_format) - pd.to_datetime(df_death["YEAR_OF_BIRTH"],format=date_format)).dt.days
        df_death["condition"] = "DEATH"
        df_death = df_death[["PRACTICE_PATIENT_ID", "condition", "DEATH_DATE"]]
        # Pull records from df to update SQLite .db with
        #   records or rows in a list of tuples [(ID, CONDITION, AGE_AT_DIAGNOSIS),]
        records = df_death.to_records(index=False,
                                      # column_dtypes={"age_at_diagnosis": "int32"},
                                      )
        c.executemany('INSERT INTO diagnosis_table VALUES(?,?,?);', records);           # Add rows to database
        if verbose > 1:
            print(f'Inserted {c.rowcount} DEATH records to the table.')

    c.execute("SELECT COUNT(*) FROM diagnosis_table")
    print('\t Diagnosis table built with', c.fetchone()[0], 'records.')
    
    #commit the changes to db			
    connector.commit()

