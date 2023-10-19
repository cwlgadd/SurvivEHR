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
            
            
def build_static_table(connector, path_to_data, chunksize=20000, verbose=1):
    r"""
    
    Produced anonymized table:
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
    c = connector.cursor()
    
    c.execute("""CREATE TABLE static_table (
                 PRACTICE_PATIENT_ID text,
                 SEX text,
                 ETHNICITY text,      
                 YEAR_OF_BIRTH text,                               
                 INDEX_AGE integer,
                 START_AGE integer,
                 END_AGE integer
                 )""")
    
    index_start = 1
    for df in tqdm(pd.read_csv(path_to_data, chunksize=chunksize, iterator=True, encoding='utf-8'), desc="Building static table"):
        
        df = df.rename(columns={col: col.replace(' ', '') for col in df.columns}) # Remove spaces from columns (not used: dded for consistency)
        
        # Convert index dates to days since birth
        date_format = '%Y-%m-%d'
        df[["INDEX_DATE", "START_DATE", "END_DATE"]] = df[["INDEX_DATE", "START_DATE", "END_DATE"]].apply(pd.to_datetime, format=date_format)
        yob_datetime = pd.to_datetime(df['YEAR_OF_BIRTH'], format=date_format)
        df['AGE_AT_INDEX'] = (df['INDEX_DATE'] - yob_datetime).dt.days
        df['AGE_AT_START'] = (df['START_DATE'] - yob_datetime).dt.days
        df['AGE_AT_END'] = (df['END_DATE'] - yob_datetime).dt.days

        # Start counting indices from 1
        df.index += index_start
        
        # Remove the un-interesting columns. Can add later if needed
        columns = ['PRACTICE_PATIENT_ID', 'SEX', 'ETHNICITY', 'AGE_AT_INDEX','AGE_AT_START','AGE_AT_END', 'YEAR_OF_BIRTH']
        for col in df.columns:
            if col not in columns:
                df = df.drop(col, axis=1)
        
        # Pull records from df to update SQLite .db with records or rows in a list
        records = df.to_records(index=False, column_dtypes={"AGE_AT_INDEX": "int32",
                                                            "AGE_AT_START": "int32",
                                                            "AGE_AT_END": "int32",
                                                           })
        
        # Add rows to database
        c.executemany('INSERT INTO static_table VALUES(?,?,?,?,?,?,?);', records);
        
        if verbose > 1:
            print('Inserted', c.rowcount, 'data owners to the table.')

    if verbose > 0:
        c.execute("SELECT COUNT(*) FROM static_table")
        print('\t Static table built with', c.fetchone()[0], 'records.')
    
    #commit the changes to db			
    connector.commit()
    
    
if __name__ == "__main__":
    """ build database of static covariates """
    
    PATH_TO_DB = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"
    conn = sqlite3.connect(PATH_TO_DB)
    # conn = sqlite3.connect(':memory:')              # For debugging
        
    
    PATH_TO_DATA = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/static.csv"
    build_static_table(conn, PATH_TO_DATA)
