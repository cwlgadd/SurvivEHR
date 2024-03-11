import sqlite3
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from dateutil import relativedelta

# class StaticDB:
#     """A sample static observations class"""

#     def __init__(self, identifier, sex, ethnicity, year_of_birth):
#         self.identifier = identifier
#         self.sex = sex
#         self.ethnicity = ethnicity
#         self.age = year_of_birth

#     # def __repr__(self):
#     #     return "Employee('{}', '{}', {})".format(self.first, self.last, self.pay)
    

# def get_static_by_PPID(identifier, cursor):
#     cursor.execute("SELECT * FROM static_table WHERE PRACTICE_PATIENT_ID=:PRACTICE_PATIENT_ID", {'PRACTICE_PATIENT_ID': identifier})
#     return cursor.fetchall()


# def insert_patient(patient, cursor):
#     with conn:
#         cursor.execute("INSERT INTO static_table VALUES (:PRACTICE_PATIENT_ID, :SEX, :ETHNICITY, :YEAR_OF_BIRTH)", 
#                        {'PRACTICE_PATIENT_ID': patient.identifier,
#                         'SEX': patient.sex, 
#                         'ETHNICITY': patient.ethnicity, 
#                         'YEAR_OF_BIRTH': patient.age})

def convert_to_datetime(input):
    # function that reformats input string to datetime type
    return datetime.strptime(input, "%Y-%m-%dT%H:%M:%S.%f%z")            
            
def build_static_table(connector, path_to_data, chunksize=200000, verbose=1):
    r"""
    
    Produced anonymized table:
    ┌──────────────────────┬─────┬───────────┬───────────────┬─────────────┬─────────────┬───────────────┐
    │ PRACTICE_PATIENT_ID  ┆ SEX ┆ etc..     ┆ YEAR_OF_BIRTH ┆ INDEX_AGE   ┆ START_AGE   ┆ END_AGE       │
    │ ---                  ┆ --- ┆ ---       ┆ ---           ┆ ---         ┆ ---         ┆ ---           │
    │ str                  ┆ str ┆           ┆ str           ┆ i64 (days)  ┆ i64 (days)  ┆ i64 (days)    │
    ╞══════════════════════╪═════╪═══════════╪═══════════════╪═════════════╪═════════════╪═══════════════╡
    │ <anonymous 1>        ┆ M   ┆           ┆ yyyy--mm-dd   ┆ dd          ┆ dd          ┆ dd            │
    │ <anonymous 2>        ┆ F   ┆           ┆ yyyy--mm-dd   ┆ dd          ┆ dd          ┆ dd            │
    │ …                    ┆ …   ┆ …         ┆ …             ┆             ┆             ┆               │
    │ <anonymous N>        ┆ M   ┆           ┆ yyyy--mm-dd   ┆ dd          ┆ dd          ┆ dd            │
    └──────────────────────┴─────┴───────────┴───────────────┴─────────────┴─────────────┴───────────────┘

    Baseline covariates include
        * ETHNICITY
        * SEX
        * COUNTRY
        * HEALTH_AUTH
        
    """
    c = connector.cursor()
    
    c.execute("""CREATE TABLE static_table (
                 PRACTICE_PATIENT_ID text,
                 ETHNICITY text,      
                 YEAR_OF_BIRTH text,    
                 SEX text,  
                 COUNTRY text,
                 HEALTH_AUTH text,
                 INDEX_DATE text,
                 START_DATE text,
                 END_DATE text
                 )""")
    
    index_start = 1
    generator = pd.read_csv(path_to_data, chunksize=chunksize, iterator=True, encoding='utf-8', low_memory=False)
    # low_memory=False just silences an error, TODO: add dtypes
    for df in tqdm(generator, desc="Building static table"):

        # Start counting indices from 1
        df.index += index_start
        
        # Convert index dates to days since birth
        date_format = '%Y-%m-%d'
        # df[["INDEX_DATE", "START_DATE", "END_DATE"]] = df[["INDEX_DATE", "START_DATE", "END_DATE"]].apply(pd.to_datetime, format=date_format)
        # yob_datetime = pd.to_datetime(df['YEAR_OF_BIRTH'], format=date_format)
        # df['AGE_AT_INDEX'] = (df['INDEX_DATE'] - yob_datetime).dt.days
        # df['AGE_AT_START'] = (df['START_DATE'] - yob_datetime).dt.days
        # df['AGE_AT_END'] = (df['END_DATE'] - yob_datetime).dt.days

        # df['INDEX_DATE'] = df['INDEX_DATE'].dt.strftime(date_format)
        # df['START_DATE'] = df['START_DATE'].dt.strftime(date_format)
        # df['END_DATE'] = df['END_DATE'].dt.strftime(date_format)
        
        # Keep only some interesting columns. Can add more later if needed
        columns = ['PRACTICE_PATIENT_ID', 
                   'ETHNICITY', 'YEAR_OF_BIRTH', 
                   'SEX', 'COUNTRY',
                   'HEALTH_AUTH',
                   # 'AGE_AT_INDEX','AGE_AT_START','AGE_AT_END',
                   'INDEX_DATE','START_DATE','END_DATE',
                  ]
        df = df[columns]

        # for col in df.columns:
        #     if col not in columns:
        #         df = df.drop(col, axis=1)

        # Pull records from df to update SQLite .db with records or rows in a list
        records = df.to_records(index=False,
                                # column_dtypes={"AGE_AT_INDEX": "int32",
                                #                "AGE_AT_START": "int32",
                                #                "AGE_AT_END": "int32",
                                #                }
                               )
        
        # Add rows to database
        c.executemany('INSERT INTO static_table VALUES(?,?,?,?,?,?,?,?,?);', records);
        
        if verbose > 1:
            print('Inserted', c.rowcount, 'data owners to the table.')

    c.execute("SELECT COUNT(*) FROM static_table")
    print('\t Static table built with', c.fetchone()[0], 'records.')
    
    #commit the changes to db			
    connector.commit()
    
