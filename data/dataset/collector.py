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

    def extract_unique_rows(self,
                            table_name: str,
                            column_name: str
                           ) -> list[str]:
        """
        Get a list of unique values in a tables column
        """
        try:
            query = f"SELECT DISTINCT {column_name} FROM {table_name};"

            # Execute the query
            logging.debug(f"Query: {query}")
            self.cursor.execute(query)

            # Fetch unique prefixes for the current table and update the set
            unique_values = self.cursor.fetchall()
            return [uv[0] for uv in unique_values]
            
        except sqlite3.Error as e:
            print(f"Error extracting unique entries for column {column} of table {table}: {e}")
                       
    def extract_practice_ids(self, 
                             table_names: list[str],
                             identifier_column: str,
                             delimiter: str = '_',
                             conditions: Optional[list] = None,
                            ) -> list[str]:
        """
        Get a list of unique practice IDs contained in a collection of tables
        """
        try:
            # Initialize an empty set to store unique prefixes
            unique_prefixes = set()
            
            # Iterate over each table
            for idx_table, table in enumerate(table_names):
                # Construct the SQL query to extract unique prefixes for each table
                query = f"SELECT DISTINCT SUBSTR({identifier_column}, 1, INSTR({identifier_column}, '{delimiter}') - 1) FROM {table} WHERE {identifier_column} LIKE '%{delimiter}%'"

                # If we want to add a condition
                if conditions is not None:
                    if conditions[idx_table] is not None:
                        query += f" AND {conditions[idx_table]}"
                
                # Execute the query
                logging.debug(f"Query: {query}")
                self.cursor.execute(query)
                
                # Fetch unique prefixes for the current table and update the set
                prefixes = self.cursor.fetchall()
                unique_prefixes.update([prefix[0] for prefix in prefixes])
    
            return list(unique_prefixes)
            
        except sqlite3.Error as e:
            print("Error extracting unique prefixes:", e)

    
    def _lazy_generate_by_practice_id(self, 
                                      practice_ids: list[str],
                                      table_names: list[str], 
                                      identifier_column: str,
                                      conditions: Optional[list[str]] = None
                                      ) -> list[pl.LazyFrame]:
        """
        practice_ids: is a lsit of practice identifiers that prefix the practice_patient_id column in every table
        table_names:  is a list of table names from which you want to fetch data.
        columns:      is a list of column names you want to retrieve from each table.
        conditions:   is a list of conditions to filter the data, where each condition applies to a specific table.
        """
        try:
            # Iterate over each prefix
            for prefix in practice_ids:
                rows_by_table = {}
                for idx_table, table in enumerate(table_names):
                    # Construct query for fetching rows with the current prefix for the current table
                    prefix_query = f"SELECT * FROM {table} WHERE {identifier_column} LIKE '{prefix}%'"  # LIMIT {chunk_size}

                    if conditions is not None:
                        if conditions[idx_table] is not None:
                            #  conditions for each table, e.g. a query asking for only certain diagnoses or measurements to be 
                            #  included in the generator
                            prefix_query += f" WHERE {conditions[idx_table]}"
                            
                    rows_by_table["lazy_" + table.split("_")[0]] = pl.read_database(query=prefix_query, connection_uri=self.connection_token).lazy()
                        
                # Yield the fetched rows as a chunk
                yield prefix, rows_by_table
                
        except sqlite3.Error as e:
            print("Error generating table chunks:", e)

    
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

    def _online_standardisation(self,
                                meta_information: Optional[dict],
                                lazy_static: Optional[pl.LazyFrame] = None,
                                lazy_diagnosis: Optional[pl.LazyFrame] = None,
                                lazy_measurement: Optional[pl.LazyFrame] = None,
                                ) -> dict:
        """

        TODO: this can be optimised
         #    Standardisation initialisation nested dictionaries with keys of all events with values:
        #   `    We standardise regardless of whether it will be used. This ensures any parquet files always match to the meta data.
        # TODO: this can be replaced by a proper data structure class
        """
        
        # On first batch we initialise the container
        if meta_information is None:
            meta_information = {}
            if lazy_static is not None:
                pass
            if lazy_diagnosis is not None:
                unique_diagnoses = self.extract_unique_rows("diagnosis_table", "EVENT")
                logging.debug(f"unique_diagnoses: {unique_diagnoses}")
                diagnosis_meta = pd.DataFrame({"event": unique_diagnoses,
                                               "count": [0 for _ in unique_diagnoses],
                                               })
                meta_information["diagnosis_table"] = diagnosis_meta
                
            if lazy_measurement is not None:
                unique_measurements = self.extract_unique_rows("measurement_table", "EVENT")
                logging.debug(f"unique_measurements: {unique_measurements}")
                measurement_meta = pd.DataFrame({"event": unique_measurements,
                                                 "count": [0 for _ in unique_measurements],
                                                 "count_obs": [0 for _ in unique_measurements],
                                                 "mean":  [0 for _ in unique_measurements],              # initialisation at zero works for batch mean/std calculations
                                                 "std":  [0 for _ in unique_measurements],               # but may not work if you implement other types
                                                 })
                meta_information["measurement_table"] = measurement_meta
            logging.debug(meta_information)
            
        # On all batches we perform batch updates on all meta information gathered
        for _key in meta_information.keys():
            if _key == "static_table":
                assert lazy_static is not None
                pass
                
            elif _key == "diagnosis_table":
                assert lazy_diagnosis is not None

                # Get counts
                lazy_count = lazy_diagnosis.groupby("EVENT").agg(pl.col('DATE').count().alias("COUNT")).collect()
                batch_diagnoses = lazy_count["EVENT"].to_list()

                for index, diagnosis_row in meta_information[_key].iterrows():
                    # for all diagnoses
                    diagnosis = diagnosis_row["event"]

                    if diagnosis in batch_diagnoses:
                        # Get old and new values
                        old_count = diagnosis_row["count"]
                        batch_count = lazy_count.filter(pl.col("EVENT") == diagnosis)["COUNT"][0]
                        new_count = old_count + batch_count
                        # update values
                        meta_information[_key].loc[meta_information[_key]['event'] == diagnosis, ['count',]] = [new_count,]
                
            elif _key == "measurement_table":
                assert lazy_measurement is not None

                # get count including missing value entries
                lazy_count = lazy_measurement.groupby("EVENT").agg(pl.col('VALUE').count().alias("COUNT")).collect()
                
                # Remove missing value entries and get batch statistics required for standardisation
                lazy_obs_measurement = lazy_measurement.drop_nulls()
                lazy_count_obs = lazy_obs_measurement.groupby("EVENT").agg(pl.col('VALUE').count().alias("COUNT")).collect()                
                lazy_mean = lazy_obs_measurement.groupby("EVENT").agg(pl.col('VALUE').mean().alias("MEAN")).collect()
                lazy_std = lazy_obs_measurement.groupby("EVENT").agg(pl.col('VALUE').std().alias("STD")).collect()
                obs_batch_measurements = lazy_count_obs["EVENT"].to_list()
               
                for index, measurement_row in meta_information[_key].iterrows():
                    # for all measurements
                    measurement = measurement_row["event"]

                    if measurement in obs_batch_measurements:
                        print(measurement)
                        # Get old and new values
                        old_count = measurement_row["count"]
                        old_count_obs = measurement_row["count_obs"]
                        old_mean = measurement_row["mean"]
                        old_std = measurement_row["std"]

                        batch_count = lazy_count.filter(pl.col("EVENT") == measurement)["COUNT"][0]
                        batch_count_obs = lazy_count_obs.filter(pl.col("EVENT") == measurement)["COUNT"][0]
                        batch_mean = lazy_mean.filter(pl.col("EVENT") == measurement)["MEAN"][0]
                        batch_std = lazy_std.filter(pl.col("EVENT") == measurement)["STD"][0]

                        new_count = old_count + batch_count
                        new_count_obs = old_count_obs + batch_count_obs
                        new_mean = self._online_mean_update(mu_m = old_mean,   m = old_count_obs,
                                                            mu_n = batch_mean, n = batch_count_obs)
                        new_std = self._online_std_update(sigma_m = old_std,   mu_m = old_mean,   m = old_count_obs,
                                                          sigma_n = batch_std, mu_n = batch_mean, n = batch_count_obs)

                        # update values
                        meta_information[_key].loc[meta_information[_key]['event'] == measurement, ['count', "count_obs", "mean", "std"]] = [new_count, new_count_obs, new_mean, new_std]
                        
                    else:
                        # Measurement was not in batch and so statistics do not need to be updated
                        pass
                        
            else:
                raise NotImplementedError
            
        return meta_information

    def _online_mean_update(self,
                            mu_m: float, m: int,
                            mu_n: float, n: int
                           ) -> float:
        
        updated_empirical_mean = ((m / (m + n)) * mu_m) + ((n / (m + n)) * mu_n)
        
        return updated_empirical_mean

    def _online_std_update(self,
                           sigma_m: float, mu_m: float, m: int,
                           sigma_n: float, mu_n: float, n: int
                           ) -> float:
        
        term1 =  ((m / (m + n)) * sigma_m**2) 
        term2 =  ((n / (m + n)) * sigma_n**2) 
        correction_by_means = (((n * m) / (m + n)**2) * (mu_m - mu_n)**2) 
        
        updated_empirical_var = term1 + term2 + correction_by_means
        updated_empirical_std = np.sqrt(updated_empirical_var)
        return updated_empirical_std

