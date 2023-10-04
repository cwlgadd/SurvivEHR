from sklearn.model_selection import train_test_split as sk_split
# from sklearn import preprocessing
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from abc import ABC
import sqlite3
from typing import Optional
import os


class FoundationalDataModule(pl.LightningDataModule, ABC):
    
    PATH_TO_DB = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"
    
    def __init__(self, 
                 batch_size:int,
                 sql_filter_strings:Optional[list] = None,
                 weighted_sampler:bool = False
                ):
        """
        """
        
        super(FoundationalDataModule, self).__init__()
        
        self.batch_size = batch_size

        self.conn = sqlite3.connect(self.PATH_TO_DB)
        self.cursor = self.conn.cursor()
        
        # Check what tables were built into DB
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()
        print(f"Loaded tables {[table[0] for table in tables]}")
        # Report how many entries in each table
        self.cursor.execute("SELECT COUNT(*) FROM static_table")
        print('\t static_table has', self.cursor.fetchone()[0], 'records.')
        self.cursor.execute("SELECT COUNT(*) FROM diagnosis_table")
        print('\t diagnosis_table has', self.cursor.fetchone()[0], 'records.')
    
        # Get all unique patient IDs (we assume no missing static variables)
        ##############
        self.cursor.execute("""SELECT 
                                   practice_patient_id 
                               FROM
                                   static_table;
                            """)
        self.identifiers = [ppid[0] for ppid in self.cursor.fetchall()]
        
        # Filter identifiers based on study criteria
        ##############
        if sql_filter_strings is not None:
            for sql_query in sql_filter_strings:
                self.filter_query(sql_query)

        # Train/test/validation split cohort
        ##############
        (self.train_ids, self.test_ids, self.val_ids), weight_dict = self.train_test_split()
        
        # Training set
        ##############
        self.train_set = FoundationalDataset(self.train_ids, self.cursor)
        
        # Weighted random sampler for training set
        if (weight_dict is not None) and weighted_sampler:
            raise NotImplementedError
        else:        
            self.train_sampler = None
            self.train_shuffle = True
            
        # Test and validation set
        ##############
        self.test_set = FoundationalDataset(self.test_ids, self.cursor)
        self.val_set = FoundationalDataset(self.val_ids, self.cursor)

        

    def filter_query(self, sql_filter_string):
        r""" in-place SQL query which will update the subset of patients of interest. Repeated calls reduces this subset further
        
        #TODO this is obviously very open to sql injection - can fix later but I just want first version working for now to start ML devel
        # In fixing, it wont be easy to abstract this code across tables and I dont really want to write separate filter definitions for each 
        #   table as the number will increase.
        #
        # Goal will be something functioning like:
        # cursor.execute("SELECT 
        #                     practice_patient_id
        #                 FROM 
        #                     '%s'
        #                 WHERE
        #                     '%s',
        #                 (<table>, <query string>, )
        #                )
        #            
        # However, you can't pass a table name as a string, as when doing it properly as above then
        # table_name is expected to be a table object and not a string.
        
        ARGS:
            sql_filter_string (str): 
                sql query string which should return a list of the PRACTICE_PATIENT_IDs which satisfy criteria
                
        KWARGS:
            
        """
        assert sql_filter_string[:26] == "SELECT PRACTICE_PATIENT_ID"        # quick fix to TODO in docstring above 
        
        self.cursor.execute(sql_filter_string)
        valid_identifiers = [ppid[0] for ppid in self.cursor.fetchall()]
        self.identifiers = list(set(valid_identifiers).intersection(self.identifiers))    # Turn smaller list into the set
        
        assert len(self.identifiers) > 0, f"No remaining samples fit query \n {sql_filter_string}"

        
    def train_test_split(self):
        # Split frame into training, validation, and test
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
        weight_dict = None

        return (train_ids, test_ids, val_ids), weight_dict
    
    # TODO: all of the below dataloaders are set to use a single worker. 
    #      This was because I'll need to create separate connections to the db for each thread. 
    #      To get proper use out of GPUs I should fix this
    def train_dataloader(self):
        return DataLoader(
            sampler=self.train_sampler,
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=1, # np.min((8,os.cpu_count())),
            shuffle=self.train_shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_set,
            batch_size=self.batch_size,
            num_workers=1 # np.min((8,os.cpu_count())),
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=1, # np.min((8,os.cpu_count())),
            shuffle=False
        )

    
class FoundationalDataset(Dataset):
    r"""
    """
    
    def __init__(self, pp_ids, sql_cursor):
        self.pp_ids = pp_ids
        self.cursor = sql_cursor
        
    def build_cache(self):
        return 
       
    def __len__(self):
        return len(self.pp_ids)
    
    def __getitem__(self, idx):
        # Return event stream and static variables
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        static_covariates = self.static_events(idx)
        print(static_covariates)
        
        diagnosis_events, diagnosis_times = self.diagnosis_events(idx)
        # print(diagnosis_events)
        
        return {"static": static_covariates,
                "events": diagnosis_events,
                "event_time_stamps": diagnosis_times
               }
    
    def static_events(self, idx):
        self.cursor.execute("""SELECT 
                                   sex, 
                                   ethnicity,
                                   year_of_birth
                               FROM 
                                   static_table 
                               WHERE practice_patient_id = ?
                            """, 
                            (self.pp_ids[idx],))
        return self.cursor.fetchall()

    
    def diagnosis_events(self, idx):
        self.cursor.execute("""SELECT 
                                   condition,
                                   age_at_diagnosis
                               FROM 
                                   diagnosis_table 
                               WHERE practice_patient_id = ?
                            """, 
                            (self.pp_ids[idx],))
        diagnosis_info = self.cursor.fetchall()
        print(f"diagnosis info {diagnosis_info}")
        diagnosis_events = [event[0] for event in diagnosis_info]
        diagnosis_times = [event[1] for event in diagnosis_info]
        print(diagnosis_events)
        print(diagnosis_times)
        return diagnosis_events, diagnosis_times
        
            
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
    foundational_dm = FoundationalDataModule(batch_size=256, sql_filter_strings=sql_filter_strings)
    # print(foundational_dm.identifiers[:20])
    # print(len(foundational_dm.identifiers))
    # print(len(foundational_dm.train_set))
    # print(len(foundational_dm.test_set))
    # print(len(foundational_dm.val_set))
    
    for batch in foundational_dm.train_dataloader():
        # print(len(batch["static"]))
        pass