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
            logging.debug("Connected to SQLite database")
        except sqlite3.Error as e:
            logging.warning(f"Error connecting to SQLite database: {e}")

    def disconnect(self):
        if self.connection:
            self.connection.close()
            logging.debug("Disconnected from SQLite database")

    def _extract_AGG(self,
                     table_name:          str,
                     identifier_column:   str,
                     aggregations:        str = "COUNT(*)",
                     condition:           Optional[str] = None,
                     ):
        # Construct the SQL query to extract unique prefixes for each table
        query = f"SELECT {identifier_column}, {aggregations} FROM {table_name}"

        # If we want to add a condition
        if condition is not None:
            query += f" WHERE {condition}"

        query += f" GROUP BY {identifier_column}"
        
        # Execute the query
        logging.info(f"Query: {query[:100] if len(query) > 100 else query}")
        self.cursor.execute(query)

        # Fetch unique prefixes for the current table and update the set
        result = self.cursor.fetchall()

        return result
    
    def _extract_distinct(self,
                          table_names:         list[str],
                          identifier_column:   str,
                          conditions:          Optional[list] = None,
                          ) -> list[str]:
        """
        Get a list of unique practice IDs contained in a collection of tables
        """
        # Initialize an empty set to store unique prefixes
        unique_prefixes = set()
        
        # Iterate over each table
        for idx_table, table in enumerate(table_names):
            # Construct the SQL query to extract unique prefixes for each table
            query = f"SELECT DISTINCT {identifier_column} FROM {table}"

            # If we want to add a condition
            if conditions is not None:
                if conditions[idx_table] is not None:
                    query += f" WHERE {conditions[idx_table]}"
            
            # Execute the query
            logging.debug(f"Query: {query[:100] if len(query) > 100 else query}")
            self.cursor.execute(query)
            
            # Fetch unique prefixes for the current table and update the set
            prefixes = self.cursor.fetchall()
            unique_prefixes.update([prefix[0] for prefix in prefixes])

        return list(unique_prefixes)
            
    def _lazy_generate_by_distinct(self,
                                   distinct_values: list,
                                   table_names: list[str], 
                                   identifier_column: str,
                                   conditions: Optional[list[str]] = None
                                   ) -> list[pl.LazyFrame]:
        """
        practice_ids: is a list of practice identifiers that prefix the practice_patient_id column in every table
        table_names:  is a list of table names from which you want to fetch data.
        columns:      is a list of column names you want to retrieve from each table.
        conditions:   is a list of conditions to filter the data, where each condition applies to a specific table.
        """
        # Iterate over each
        for distinct_value in distinct_values:
            rows_by_table = {}
            for idx_table, table in enumerate(table_names):
                # Construct query for fetching rows with the current prefix for the current table
                if type(distinct_value) == list:
                    _dv = [f"'{dv}'" for dv in distinct_value]
                    sep_list = ",".join(_dv)
                    query = f"SELECT * FROM {table} WHERE {identifier_column} IN ({sep_list});"  # LIMIT {chunk_size}
                else:
                    query = f"SELECT * FROM {table} WHERE {identifier_column} = '{distinct_value}'"  # LIMIT {chunk_size}
                    
                if conditions is not None:
                    if conditions[idx_table] is not None:
                        #  conditions for each table, e.g. a query asking for only certain diagnoses or measurements to be 
                        #  included in the generator
                        query += f" WHERE {conditions[idx_table]}"
                                  
                logging.debug(f"Query: {query[:100] if len(query) > 100 else query}")
                rows_by_table["lazy_" + table.split("_")[0]] = pl.read_database(query=query, connection_uri=self.connection_token).lazy()
                    
            # Yield the fetched rows as a chunk
            yield distinct_value, rows_by_table
                
    def _collate_lazy_tables(self,
                             lazy_static: pl.LazyFrame,
                             lazy_diagnosis: Optional[pl.LazyFrame] = None,
                             lazy_measurement: Optional[pl.LazyFrame] = None,
                             drop_empty_dynamic: bool = True,
                             drop_missing_data: bool = False,
                             exclude_pre_index_age: bool = False,
                             ) -> pl.LazyFrame:
        """
        Notes:  The collection is applied after this function
                We do not sort within the lazy operation, so the row order will not be deterministic
        """
        #################
        # STATIC EVENTS #
        #################
        # Convert all dates to datetime format
        lazy_static = (
            lazy_static
            .with_columns(pl.col("INDEX_DATE").str.to_datetime("%Y-%m-%d"))
            .with_columns(pl.col("START_DATE").str.to_datetime("%Y-%m-%d"))
            .with_columns(pl.col("END_DATE").str.to_datetime("%Y-%m-%d"))
            .with_columns(pl.col("YEAR_OF_BIRTH").str.to_datetime("%Y-%m-%d"))                
        )
        
        ##################
        # DYNAMIC EVENTS #
        ##################
        # Optionally drop missing measurements
        if drop_missing_data:
            if lazy_measurement is not None:
                logging.debug("Dropping missing measurements")
                lazy_measurement = lazy_measurement.drop_nulls()
    
        # Optionally drop events occurring before indexing
        if exclude_pre_index_age:
            logging.debug("Removing observations before index age")
            raise NotImplementedError

        # Merge all frames containing time series data
        if lazy_diagnosis is not None and lazy_measurement is not None:
            # Stacking the frames vertically. Value column in diagnostic is filled with null
            lazy_combined_frame = pl.concat([lazy_measurement, lazy_diagnosis], how="diagonal")
        elif lazy_measurement is not None:
            lazy_combined_frame = lazy_measurement            
        elif lazy_diagnosis is not None:
            # Note: if we load diagnoses with no measurements, we are not concating values, so add this missing column
            lazy_combined_frame = lazy_diagnosis.with_columns(pl.lit(None).cast(pl.Utf8).alias('VALUE'))
        else:
            raise NotImplementedError

        # Convert event date to time since birth by linking the dynamic diagnosis and measurement frames to the static one
        # Subtract the dates and create a new column for the result            
        lazy_combined_frame = (
            lazy_combined_frame
            .with_columns(pl.col("DATE").str.to_datetime("%Y-%m-%d"))
            .join(lazy_static, on="PRACTICE_PATIENT_ID", how="inner")
            .select([
                  (pl.col("DATE") - pl.col("YEAR_OF_BIRTH")).dt.days().alias("DAYS_SINCE_BIRTH"), "*"
                ])
            )

        # Remove entries before conception (negative to include pregnancy period)
        agg_cols = ["VALUE", "EVENT", "DAYS_SINCE_BIRTH"]
        lazy_combined_frame = (
            lazy_combined_frame
            .sort("DAYS_SINCE_BIRTH")
            .filter(pl.col("DAYS_SINCE_BIRTH") > -365)                                  
            .groupby("PRACTICE_PATIENT_ID")     
            .agg(agg_cols)                                                # Turn into lists
            .sort("PRACTICE_PATIENT_ID")                                  # make lazy collection deterministic
        )

        if drop_empty_dynamic:
            logging.debug("Removing patients with no observed events")
            lazy_combined_frame = lazy_combined_frame.drop_nulls()

        ############################
        # MERGE STATIC AND DYNAMIC #
        ############################
        # Align the polars frames, linking on patient idenfitifer, then concatentate into a single frame (dropping repeated identifier)
        #    If identifier exists in one but not the other, default behaviour is to fill with null, these are handled by filtering later
        #    All these operations are performed lazily
        # lazy_static, lazy_combined_frame = pl.align_frames(lazy_static, lazy_combined_frame, on="PRACTICE_PATIENT_ID")            # align on identifiers
        # lazy_combined_frame = pl.concat([lazy_static, lazy_combined_frame], how="diagonal")       
        lazy_combined_frame = lazy_combined_frame.join(lazy_static, on="PRACTICE_PATIENT_ID", how="left")

        # Spit practice and patient ID for hive saving
        lazy_combined_frame = (
            lazy_combined_frame.with_columns(
                [
                    pl.col('PRACTICE_PATIENT_ID').apply(lambda s: s.split('_')[0]).alias('PRACTICE_ID'),
                    pl.col('PRACTICE_PATIENT_ID').apply(lambda s: s.split('_')[1]).alias('PATIENT_ID')
                    ])
        )
            
        return lazy_combined_frame

    def get_meta_information(self,
                             practice_patient_ids:  list,
                             diagnoses:             bool = True,
                             measurement:           bool = True,
                            ) -> dict:

        logging.info("Collecting meta information from database. This will be used for tokenization and standardisation.")
        
        # Initialise meta information 
        meta_information = {}

        # TODO: calculate these only on training splits! Especially if standardisation gets implemented
        sep_list = ",".join([f"'{dv}'" for dv in practice_patient_ids])
        condition = f"PRACTICE_PATIENT_ID IN ({sep_list});" 
        
        if diagnoses is True:
            
            result = self._extract_AGG("diagnosis_table", identifier_column="EVENT", aggregations="COUNT(*)")
            diagnoses, counts = zip(*result)
            diagnosis_meta = pd.DataFrame({"event": diagnoses,
                                           "count": [i for i in counts],
                                           })
            logging.info(f"diagnosis_meta:\n{diagnosis_meta}")
            meta_information["diagnosis_table"] = diagnosis_meta
            
        if measurement is True:
            # Get total number of entries for each measurement
            result = self._extract_AGG("measurement_table", identifier_column="EVENT", aggregations="COUNT(*)")  #  condition=condition
            measurements, counts = zip(*result)
            
            # Get total number of entries with corresponding observed values for each measurement
            result = self._extract_AGG("measurement_table", identifier_column="EVENT", aggregations="COUNT(VALUE)")  # condition="VALUE IS NOT NULL",
            measurements_obs, counts_obs = zip(*result)
            obs_counts = [_c if _m in measurements else 0 for _m, _c in zip(measurements_obs, counts_obs)]
            
            # Get total number of entries with corresponding observed values for each measurement
            result = self._extract_AGG("measurement_table", identifier_column="EVENT", aggregations="MIN(VALUE), MAX(VALUE)")  # condition="VALUE IS NOT NULL",
            measurements_obs, min_obs, max_obs = zip(*result)
            obs_bias = [_min if _meas in measurements else 0 for _meas, _min, _max in zip(measurements_obs, min_obs, max_obs)]
            obs_scale = [_max - _min if _meas in measurements else 1 for _meas, _min, _max in zip(measurements_obs, min_obs, max_obs)]
            
            obs_dict = {k:v for k,v in zip(measurements_obs, counts_obs)}
            # Get variance of observed values for each measurement
            
            measurement_meta = pd.DataFrame({"event": measurements,
                                             "count": counts,
                                             "count_obs": obs_counts,
                                             "bias": obs_bias,
                                             "scale": obs_scale,
                                             })
            logging.info(f"measurement_meta: {measurement_meta}")

            meta_information["measurement_table"] = measurement_meta
            
        logging.info(meta_information)

        return meta_information
                             
    