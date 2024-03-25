import sqlite3
import polars as pl
import pandas as pd
import numpy as np
from typing import Optional, Any, Union
import logging
from CPRD.data.database.build_static_db import Static
from CPRD.data.database.build_diagnosis_db import Diagnoses
from CPRD.data.database.build_measurements_and_tests_db import Measurements


class SQLiteDataCollector(Static, Diagnoses, Measurements):
    
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
        logging.debug(f"Query: {query[:300] if len(query) > 300 else query}")
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
                                   identifier_column: str,
                                   include_diagnoses: bool = True, 
                                   include_measurements: bool = True,
                                   conditions: Optional[list[str]] = None
                                   ) -> list[pl.LazyFrame]:
        """
        practice_ids: is a list of practice identifiers that prefix the practice_patient_id column in every table
        table_names:  is a list of table names from which you want to fetch data.
        columns:      is a list of column names you want to retrieve from each table.
        conditions:   is a list of conditions to filter the data, where each condition applies to a specific table.
        """
        
        table_names = ["static_table"]
        if include_diagnoses:            
            table_names.append("diagnosis_table")
        if include_measurements:
            for measurement_table in self.measurement_table_names:
                table_names.append(measurement_table)   

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
                df = pl.read_database(query=query, connection_uri=self.connection_token)
                if len(df) > 0:
                    rows_by_table["lazy_" + table] = df.lazy()

            # Yield the fetched rows as a chunk
            yield distinct_value, rows_by_table
                
    def _collate_lazy_tables(self,
                             lazy_frames: pl.LazyFrame,
                             drop_empty_dynamic: bool = True,
                             drop_missing_data: bool = False,
                             exclude_pre_index_age: bool = False,
                             ) -> pl.LazyFrame:
        """
        Notes:  The collection is applied after this function
                We do not sort within the lazy operation, so the row order will not be deterministic
        """
        # Static lazy frame
        lazy_static = lazy_frames["lazy_static_table"]

        # Diagnosis lazy frame, optional
        lazy_diagnosis = lazy_frames["lazy_diagnosis_table"] if "lazy_diagnosis_table" in lazy_frames.keys() else None

        # Measurement lazy frames, optional
        measurement_keys = [key for key in lazy_frames if key.startswith("lazy_measurement_")]
        measurement_lazy_frames = [lazy_frames[key] for key in measurement_keys]
        lazy_measurement = pl.concat(measurement_lazy_frames)
        
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

            measurements, counts, obs_counts, obs_biases, obs_scales = [], [], [], [], []
            for table in self.measurement_table_names:
                
                # Get total number of entries for each unique event in measurement table
                result = self._extract_AGG(table, identifier_column="EVENT", aggregations="COUNT(*)")  #  condition=condition
                table_measurements, table_counts = zip(*result)
                
                # Get total number of entries with corresponding observed values for each unique event in measurement table
                result = self._extract_AGG(table, identifier_column="EVENT", aggregations="COUNT(VALUE)")  # condition="VALUE IS NOT NULL",
                table_measurements_obs, table_counts_obs = zip(*result)
                table_obs_counts = [_c if _m in table_measurements else 0 for _m, _c in zip(table_measurements_obs, table_counts_obs)]
                
                # Get total number of entries with corresponding observed values for each unique event in measurement table
                result = self._extract_AGG(table, identifier_column="EVENT", aggregations="MIN(VALUE), MAX(VALUE)")  # condition="VALUE IS NOT NULL",
                table_measurements_obs, table_min_obs, table_max_obs = zip(*result)
                table_obs_biases = [_min if _meas in table_measurements else 0 for _meas, _min, _max in zip(table_measurements_obs, table_min_obs, table_max_obs)]
                table_obs_scales = [_max - _min if _meas in table_measurements else 1 for _meas, _min, _max in zip(table_measurements_obs, table_min_obs, table_max_obs)]

                # Collate each tables summary statistics obtained from aggregations across each table
                #    For example, if one measurement file has height and weight measurements this may be:
                #        table_measurement = (`height`, `weight`)
                #        table_counts      = (10100,     10200  )
                #    Note, if we have measurements split across file this will create repeated entries
                for idx, _ in enumerate(table_measurements):
                    measurements.append(table_measurements[idx])
                    counts.append(table_counts[idx])
                    obs_counts.append(table_obs_counts[idx])
                    obs_biases.append(table_obs_biases[idx])
                    obs_scales.append(table_obs_scales[idx])
                
                
            measurement_meta = pd.DataFrame({"event": measurements,
                                             "count": counts,
                                             "count_obs": obs_counts,
                                             "bias": obs_biases,
                                             "scale": obs_scales,
                                             })
            logging.info(f"measurement_meta: {measurement_meta}")

            meta_information["measurement_tables"] = measurement_meta
            
        logging.info(meta_information)

        return meta_information
                             
    