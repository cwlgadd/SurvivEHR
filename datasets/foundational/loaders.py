from sklearn.model_selection import train_test_split as sk_split
# from sklearn import preprocessing

import random

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from abc import ABC
import sqlite3
from typing import Optional
        
        
class FoundationalDataModule(pl.LightningDataModule, ABC):
    
    PATH_TO_DB = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"
    
    def __init__(self, 
                 sql_filter_strings:Optional[list] =None):
        """
        """
        
        super(FoundationalDataModule, self).__init__()

        self.conn = sqlite3.connect(self.PATH_TO_DB)
        self.cursor = self.conn.cursor()
        
        # Check what tables were built into DB
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()
        print(f"Loaded tables {[table[0] for table in tables]}")

        # Report how many entries in each table
        self.cursor.execute("SELECT COUNT(*) FROM static_table")
        print('static_table has', self.cursor.fetchone()[0], 'records.')
        self.cursor.execute("SELECT COUNT(*) FROM diagnosis_table")
        print('Loaded diagnosis_table has', self.cursor.fetchone()[0], 'records.')
    
        # Get all unique patient IDs
        self.cursor.execute("SELECT PRACTICE_PATIENT_ID FROM static_table;")
        self.identifiers = [ppid[0] for ppid in self.cursor.fetchall()]
        
        # Filter based on study criteria
        if sql_filter_strings is not None:
            for sql_query in sql_filter_strings:
                self.filter_query(sql_query)

        # Train/test/validation split cohort
        (self.train_set, self.test_set, self.val_set) = self.train_test_split()
        

    def filter_query(self, sql_filter_string):
        r""" in-place SQL query which will update the subset of patients of interest. Repeated calls reduces this subset further
        
        #TODO this is obviously very open to sql injection - can fix later but I just want first version working for now to start ML devel
        # In fixing, it wont be easy to abstract this code across tables and I dont really want to write separate filter definitions for each 
        #   table as the number will increase.
        
        ARGS:
            sql_filter_string (str): 
                sql query string which should return a list of the PRACTICE_PATIENT_IDs which satisfy criteria
                
        KWARGS:
            
        """
        assert sql_filter_string[:26] == "SELECT PRACTICE_PATIENT_ID"        # quick fix to TODO in docstring
        
        self.cursor.execute(sql_filter_string)
        valid_identifiers = [ppid[0] for ppid in self.cursor.fetchall()]
        self.identifiers = list(set(valid_identifiers).intersection(self.identifiers))    # Turn smaller list into the set
        
        assert len(self.identifiers) > 0, f"No remaining samples fit query \n {sql_filter_string}"

        
    def train_test_split(self):
        # Split frame into training, validation, and test
        print(type(self.identifiers))
        train_ids, test_ids = sk_split(self.identifiers, test_size=0.2)
        test_ids, val_ids = sk_split(test_ids, test_size=0.5)

        # Random sampler weights
        # TODO: Add weighted sampler if we later choose to aggregate samples?
        # weight_dict = {}
        # ntrain_unique_samples = len(train_df.index.unique())
        # for cancer_id, group in train_df.groupby('cancer_type'):
        #     unique_samples = len(group.index.unique()) / ntrain_unique_samples
        #     if unique_samples > 0:
        #         weight_dict[cancer_id] = 1 / unique_samples

        return (train_ids, test_ids, val_ids)#, weight_dict
        
if __name__ == "__main__":
    
    # Example static_table query
    #   two filters. First we reduce the cohort to white males by querying the static_table
    #                Next we reduce the cohort further to those with depression by querying the diagnosis_table
    sql_filter_strings = ["""SELECT PRACTICE_PATIENT_ID
                            FROM static_table
                            WHERE SEX = 'M' AND ETHNICITY = 'WHITE'""",
                          """SELECT PRACTICE_PATIENT_ID
                            FROM diagnosis_table
                            WHERE CONDITION = 'DEPRESSION' """
                        ]
    
    # "SELECT * FROM static_table WHERE PRACTICE_PATIENT_ID=:PRACTICE_PATIENT_ID", {'PRACTICE_PATIENT_ID': identifier}
    foundational_dm = FoundationalDataModule(sql_filter_strings)
    print(foundational_dm.identifiers[:20])
    print(len(foundational_dm.identifiers))
    print(len(foundational_dm.train_set))
    print(len(foundational_dm.test_set))
    print(len(foundational_dm.val_set))