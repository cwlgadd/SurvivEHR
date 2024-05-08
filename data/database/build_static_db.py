import sqlite3
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from dateutil import relativedelta
import logging

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

class Static():

    # @staticmethod
    # def convert_to_datetime(input):
    #     # function that reformats input string to datetime type
    #     return datetime.strptime(input, "%Y-%m-%dT%H:%M:%S.%f%z")    
    
    def __init__(self, db_path, path_to_data, load=False):
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        self.connection_token = 'sqlite://' + self.db_path 
        self.path_to_data = path_to_data

        if load is False:
            # Create table        
            self.connect()
            
            logging.info(f"Removing previous static table (if exists)")
            self.cursor.execute("""DROP TABLE IF EXISTS static_table;""")
            
            logging.info(f"Creating static_table")
            self.cursor.execute("""CREATE TABLE static_table ( PRACTICE_ID integer,
                                                               PATIENT_ID integer,
                                                               ETHNICITY text,      
                                                               YEAR_OF_BIRTH text,    
                                                               SEX text,  
                                                               COUNTRY text,
                                                               IMD integer,
                                                               HEALTH_AUTH text,
                                                               INDEX_DATE text,
                                                               START_DATE text,
                                                               END_DATE text
                                                            )""")
            self.build_table()
            self.disconnect()

    
    def __str__(self):
        self.connect()
        self.cursor.execute("SELECT COUNT(*) FROM static_table")
        s = f'Static table with {self.cursor.fetchone()[0]/1e6:.2f}M records.'
        self.disconnect()
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

    def build_table(self, verbose=1, **kwargs):
        """
        """
        self.connect()
        self._add_file_to_table(self.path_to_data, verbose=verbose, **kwargs)
        self._make_index()
        self.disconnect()
        
    def _make_index(self):
        # Create index
        logging.info("Creating indexes on static_table")
        query = "CREATE INDEX IF NOT EXISTS static_index ON static_table (PRACTICE_ID, PATIENT_ID, HEALTH_AUTH, COUNTRY, SEX, ETHNICITY); "
        logging.debug(query)
        self.cursor.execute(query)
        self.connection.commit()

    def _add_file_to_table(self, fname, chunksize=200000, verbose=0, **kwargs):

        generator = pd.read_csv(fname, chunksize=chunksize, iterator=True, encoding='utf-8', low_memory=False,
                               dtype={'PATIENT_ID': 'str'}
                               )
        # low_memory=False just silences an error, TODO: add dtypes
        for df in tqdm(generator, desc="Building static table"):

            # Start counting indices from 1
            df.index += 1
            
            # Keep only some interesting columns. Can add more later if needed
            columns = ['PRACTICE_ID',  'PATIENT_ID',
                       'ETHNICITY', 'YEAR_OF_BIRTH', 
                       'SEX', 'COUNTRY',
                       'IMD',
                       'HEALTH_AUTH',
                       'INDEX_DATE','START_DATE','END_DATE',
                      ]
            df = df[columns].copy()

            # remove p at the start so we can store as int
            df['PRACTICE_ID'] = df['PRACTICE_ID'].apply(lambda x: x.replace('p', ''))
            # df['PATIENT_ID'] = df['PATIENT_ID'].astype(int)

            # for col in df.columns:
            #     if col not in columns:
            #         df = df.drop(col, axis=1)
    
            # Pull records from df to update SQLite .db with records or rows in a list
            records = df.to_records(index=False,
                                    # column_dtypes={
                                    #     "PRACTICE_ID": int,
                                    #     "PATIENT_ID": int,
                                    #     }
                                   )
            # Add rows to database
            self.cursor.executemany('INSERT INTO static_table VALUES(?,?,?,?,?,?,?,?,?,?,?);', records);
            
            if verbose > 1:
                print('Inserted', self.cursor.rowcount, 'data owners to the table.')

