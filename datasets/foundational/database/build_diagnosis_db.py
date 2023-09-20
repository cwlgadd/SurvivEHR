import sqlite3
import pandas as pd
# from sqlalchemy import create_engine # database connection
import os


# class StaticDB:
#     """A sample static observations class"""

#     def __init__(self, identifier, sex, ethnicity, year_of_birth):
#         self.identifier = identifier
#         self.sex = sex
#         self.ethnicity = ethnicity
#         self.age = year_of_birth

#     # def __repr__(self):
#     #     return "Employee('{}', '{}', {})".format(self.first, self.last, self.pay)


def get_diagnoses_by_PPID(identifier, cursor):
    cursor.execute("SELECT * FROM static_information WHERE PRACTICE_PATIENT_ID=:PRACTICE_PATIENT_ID", {'PRACTICE_PATIENT_ID': identifier})
    return cursor.fetchall()


def insert_diagnosis(patient, cursor):
    with conn:
        cursor.execute("INSERT INTO diagnosis_table VALUES (:PRACTICE_PATIENT_ID, :CONDITION, :AGE_AT_DIAGNOSIS)", 
                       {'PRACTICE_PATIENT_ID': patient.identifier,
                        'CONDITION': patient.condition, 
                        'AGE_AT_DIAGNOSIS': patient.condition_age})


def build_diagnosis_table(connector, path_to_data, chunksize=20000, verbose=0):
    """
    """

    c = connector.cursor()
    
    c.execute("""CREATE TABLE diagnosis_table (
                 PRACTICE_PATIENT_ID text,
                 CONDITION text,
                 AGE_AT_DIAGNOSIS text                 
                 )""")
    
    index_start = 1
    for df in pd.read_csv(path_to_data, chunksize=chunksize, iterator=True, encoding='utf-8'):
        
        df = df.rename(columns={col: col.replace(' ', '') for col in df.columns}) # Remove spaces from columns (not used: dded for consistency)
        
        # Convert to datetimes
        # df['YEAR_OF_BIRTH'] = pd.to_datetime(df['YEAR_OF_BIRTH']) 
    
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
            
            # and rename the condition column 
            df_condition = df_condition.rename(columns={condition: "age_at_diagnosis"}) 
            
            # and order them as we want to see them in the table
            df_condition = df_condition[["PRACTICE_PATIENT_ID", "condition", "age_at_diagnosis"]]

            # Pull records from df to update SQLite .db with
            #   records or rows in a list of tuples [(ID, CONDITION, AGE_AT_DIAGNOSIS),]
            records = df_condition.to_records(index=False)

            # Add rows to database
            c.executemany('INSERT INTO diagnosis_table VALUES(?,?,?);', records);
        
            if verbose > 0:
                print('Inserted', c.rowcount, 'records to the table.')

    c.execute("SELECT COUNT(*) FROM diagnosis_table")
    print('Diagnosis table built with', c.fetchone()[0], 'records.')
    
    #commit the changes to db			
    connector.commit()
    
    
if __name__ == "__main__":
    """ build database of diagnoses """
    
    PATH_TO_DB = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"
    conn = sqlite3.connect(PATH_TO_DB)
    # conn = sqlite3.connect(':memory:')              # For debugging
        
    
    PATH_TO_DATA = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/diagnosis_history.csv"
    build_diagnosis_table(conn, PATH_TO_DATA)
