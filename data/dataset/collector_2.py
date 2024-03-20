import sqlite3
import polars as pl
import pandas as pd
import numpy as np
from typing import Optional, Any, Union
import logging


class SQLiteDataCollector:
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        # self.db_path = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"        
        self.connection_token = 'sqlite://' + self.db_path 

    def connect(self):
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            logging.info("Connected to SQLite database")
        except sqlite3.Error as e:
            logging.warning(f"Error connecting to SQLite database: {e}")

    def disconnect(self):
        if self.connection:
            self.connection.close()
            logging.info("Disconnected from SQLite database")

    def get_distinct_rows(self,
                          table_name: str,
                          column_name: str,
                          condition: Optional[str] = None
                          ) -> list[str]:
        """
        Get a list of unique values in a table's column, with optional condition filter.
        """
        try:
            query = f"SELECT DISTINCT {column_name} FROM {table_name};"

            if condition is not None:
                #  conditions to add to the search
                h