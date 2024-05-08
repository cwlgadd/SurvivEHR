import sqlite3
import polars as pl
import pandas as pd
import numpy as np
from typing import Optional, Any, Union
import logging
from CPRD.data.database.build_static_db import Static
from CPRD.data.database.build_diagnosis_db import Diagnoses
from CPRD.data.database.build_measurements_and_tests_db import Measurements
from tqdm import tqdm
from tdigest import TDigest

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

    def _extract_distinct(self,
                          table_names:         list[str],
                          identifier_column:   str,
                          conditions:          Optional[list] = None,
                          ) -> list[str]:
        """
        Get a list of distinct `identifier_column' values, contained in a collection of tables
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

    def _extract_AGG(self,
                     table_name:          str,
                     identifier_column:   Optional[str] = None,
                     aggregations:        str = "COUNT(*)",
                     condition:           Optional[str] = None,
                     ):
        """
        Perform (optionally grouped) aggregations over tables. 
        
        For example, how many of each diagnosis, total observed values for a certain measurement, etc.
        """
        query = f"SELECT "
        
        if identifier_column is not None:
            query += f"{identifier_column}, "
            
        query += f"{aggregations} FROM {table_name}"

        # If we want to add a condition
        if condition is not None:
            query += f" WHERE {condition}"

        if identifier_column:
            query += f" GROUP BY {identifier_column}"
        
        # Execute the query
        logging.debug(f"Query: {query[:300] if len(query) > 300 else query}")
        self.cursor.execute(query)

        # Fetch unique prefixes for the current table and update the set
        result = self.cursor.fetchall()

        return result

    def _t_digest_values(self,
                         table_name:          str,
                         ):
        """
        Approximate percentiles using Ted Dunning's t-digest algorithm (see https://github.com/tdunning/t-digest)
        
           This is a data structure for online accumulation of rank-based statistics, such as quantiles and trimmed means.
        """

        digest = TDigest()
        
        self.cursor.execute(f"SELECT VALUE FROM {table_name}")

        fetches_count = 0
        while True:
            records = self.cursor.fetchmany(10000)
            
            if not records or fetches_count > 1e5 / 10000:
                # exit loop when no more records to fetch, or we reach some approximating limit 
                break 
            values = np.array([_record[0] for _record in records if _record[0] is not None])
            digest.batch_update(values)
            fetches_count += 1
        
        return digest
    
    # def _lazy_generate_by_join(self,
    #                            distinct_values: list[list],
    #                            identifier_column: str,
    #                            include_diagnoses: bool = True, 
    #                            include_measurements: bool = True,
    #                            conditions: Optional[list[str]] = None
    #                           ) -> list[pl.LazyFrame]:
    #     """
    #     Generator which provides lazy frames by an identifying column and certain values.

    #     TODO: this isn't used yet. In theory should be faster, but currently with higher memory requirements
    #     """
    #     table_names = ["static_table"]
    #     if include_diagnoses:            
    #         table_names.append("diagnosis_table")
    #     if include_measurements:
    #         for measurement_table in self.measurement_table_names:
    #             table_names.append(measurement_table)   

    #     # Iterate over each
    #     for inner_list in distinct_values:

    #         # Create temporary table that we can join against
    #         self.cursor.execute(f"""CREATE TABLE IF NOT EXISTS generate_by_join (PRACTICE_PATIENT_ID TEXT);""")

    #         # Add each identifier to the temporary table
    #         insert_query = f'INSERT INTO generate_by_join ({identifier_column}) VALUES (?)'
    #         self.cursor.executemany(insert_query, [(id,) for id in inner_list])

    #         self.cursor.execute("CREATE INDEX IF NOT EXISTS generate_by_join_index ON generate_by_join (PRACTICE_PATIENT_ID);")

    #         rows_by_table = {}
    #         for idx_table, table in enumerate(table_names):
    #             print(table)

    #             # EXPLAIN QUERY PLAN
    #             join_query = f"""SELECT {table}.*
    #                                 FROM generate_by_join 
    #                                 JOIN {table} 
    #                                     ON generate_by_join.PRACTICE_PATIENT_ID = {table}.PRACTICE_PATIENT_ID;"""
    #             self.cursor.execute(join_query)
    #             # print(self.cursor.fetchall()[0])
    #             result = self.cursor.fetchmany(100)
    #             print(result)
            
    #             # join_query = f"""SELECT * FROM generate_by_join"""
    #             # self.cursor.execute(join_query)
    #             # print(self.cursor.fetchone())
                
    #             # join_query = f"SELECT * FROM {table} WHERE PRACTICE_PATIENT_ID = 'p21505_6499892121505'"
    #             # df = pl.read_database(join_query, connection_uri=self.connection_token)
    #             # df = pl.read_database(join_query, connection=self.connection)
    #             # self.cursor.execute(join_query)
    #             # data = self.cursor.fetchall()
    #             # columns = [description[0] for description in self.cursor.description]
    #             # df = pl.DataFrame(data, schema=columns)

    #             # df = pl.DataFrame(self.cursor.fetchall())
                
    #             # print(df.head())

    #         self.cursor.execute("DROP TABLE generate_by_join;")
                
                
    #             # Construct query for fetching rows with the current prefix for the current table
    
    def _generate_lazy_by_distinct(self,
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
                    sep_list = ",".join([f"'{dv}'" for dv in distinct_value])
                    chunk_size = len(sep_list)
                    query = f"SELECT * FROM {table} WHERE {identifier_column} IN ({sep_list});"  # LIMIT {chunk_size}
                    
                else:
                    query = f"SELECT * FROM {table} WHERE {identifier_column} = '{distinct_value}'"  # LIMIT {chunk_size}
                    chunk_size = 1 
                    
                if conditions is not None:
                    if conditions[idx_table] is not None:
                        #  conditions for each table, e.g. a query asking for only certain diagnoses or measurements to be 
                        #  included in the generator
                        query += f" {conditions[idx_table]}"
                                  
                logging.debug(f"Query: {query[:120] if len(query) > 100 else query}")
                df = pl.read_database(query=query, connection_uri=self.connection_token)
                if len(df) > 0:
                    rows_by_table["lazy_" + table] = df.lazy()

            # Yield the fetched rows as a chunk
            yield distinct_value, rows_by_table
                
    def _collate_lazy_tables(self,
                             lazy_frames: pl.LazyFrame,
                             drop_empty_dynamic: bool = True,
                             drop_missing_data: bool = True,
                             exclude_pre_index_age: bool = False,
                             ) -> pl.LazyFrame:
        """
        Notes:  The collection is applied after this function
                We do not sort within the lazy operation, so the row order will not be deterministic
        """
        # Static lazy frame
        lazy_static = lazy_frames["lazy_static_table"]
        # print(lazy_static.collect().head())

        # Diagnosis lazy frame, optional
        lazy_diagnosis = lazy_frames["lazy_diagnosis_table"] if "lazy_diagnosis_table" in lazy_frames.keys() else None
        # print(lazy_diagnosis.collect().head())
        # print(lazy_diagnosis.collect().tail())

        # Measurement lazy frames, optional
        measurement_keys = [key for key in lazy_frames if key.startswith("lazy_measurement_")]
        measurement_lazy_frames = [lazy_frames[key] for key in measurement_keys]
        lazy_measurement = pl.concat(measurement_lazy_frames)
        # print(lazy_measurement.collect().head())
        
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

        # print(lazy_static.head().collect())
        # print(lazy_combined_frame.head().collect())
        
        # Convert event date to time since birth by linking the dynamic diagnosis and measurement frames to the static one
        # Subtract the dates and create a new column for the result            
        lazy_combined_frame = (
            lazy_combined_frame
            .with_columns(pl.col("DATE").str.to_datetime("%Y-%m-%d"))
            .join(lazy_static, on=["PRACTICE_ID", "PATIENT_ID"], how="inner")
            .select([
                  (pl.col("DATE") - pl.col("YEAR_OF_BIRTH")).dt.days().alias("DAYS_SINCE_BIRTH"), "*"
                ])
            )
        # print(lazy_combined_frame.head().collect())

        # Remove entries before conception (negative to include pregnancy period)
        agg_cols = ["VALUE", "EVENT", "DAYS_SINCE_BIRTH"]
        lazy_combined_frame = (
            lazy_combined_frame
            .sort("DAYS_SINCE_BIRTH")
            .filter(pl.col("DAYS_SINCE_BIRTH") > -365)                                  
            .groupby(["PRACTICE_ID", "PATIENT_ID"])
            .agg(agg_cols)                                                # Turn into lists
            .sort(["PRACTICE_ID", "PATIENT_ID"])                                  # make lazy collection deterministic
        )
        # print(lazy_combined_frame.head().collect())

        if drop_empty_dynamic:
            logging.debug("Removing patients with no observed events")
            lazy_combined_frame = lazy_combined_frame.drop_nulls()

        ############################
        # MERGE STATIC AND DYNAMIC #
        ############################
        # Align the polars frames, linking on patient idenfitifer, then concatentate into a single frame (dropping repeated identifier)
        #    If identifier exists in one but not the other, default behaviour is to fill with null, these are handled by filtering later
        #    All these operations are performed lazily
        lazy_combined_frame = lazy_combined_frame.join(lazy_static, on=["PRACTICE_ID", "PATIENT_ID"], how="left")
        # print(lazy_combined_frame.head().collect())

        return lazy_combined_frame

    def get_meta_information(self,
                             practice_ids:          Optional[list] = None,
                             diagnoses:             bool = True,
                             measurement:           bool = True,
                            ) -> dict:

        # Standardisation is TODO
        logging.info("Collecting meta information from database. This will be used for tokenization and (optionally) standardisation.")
        
        # Initialise meta information 
        meta_information = {}

        # TODO: calculate these only on training splits! Especially if standardisation gets implemented
        if practice_ids is not None:
            raise NotImplementedError
            # condition = f"PRACTICE_ID IN ({",".join([f"'{dv}'" for dv in practice_ids])});" 
        
        if diagnoses is True:
            logging.info("\t Diagnosis meta information")
            result = self._extract_AGG("diagnosis_table", identifier_column="EVENT", aggregations="COUNT(*)")
            diagnoses, counts = zip(*result)
            diagnosis_meta = pd.DataFrame({"event": diagnoses,
                                           "count": [i for i in counts],
                                           })
            meta_information["diagnosis_table"] = diagnosis_meta
            logging.debug(f"diagnosis_meta: \n{diagnosis_meta}")
            
        if measurement is True:
            logging.info("\t Measurements meta information")
            measurements = []                                                # List of measurements
            counts, obs_counts = [], []                                      # List of measurement counts, and how many of those have observed values
            obs_digest, obs_mins, obs_maxes, obs_means = [], [], [], []      # T-digest data structure for approximate quantiles, and exact statistics for observed values
            cutoff_lower, cutoff_upper = [], []                              # Cut-off values based on approximate quantiles, used for filtering and standardisation
            
            for table in tqdm(self.measurement_table_names, desc="Measurements".rjust(50), total=len(self.measurement_table_names)):

                # Get the measurement name from the table's name
                measurement = table[12:]

                # Online accumulation to approximate quantiles which will then be used for standardisation and outlier removal.
                digest = self._t_digest_values(table)
                # From summary statistics get the standardisation limits
                iqr = digest.percentile(75) - digest.percentile(25)
                cutoff_lower.append(digest.percentile(25) - 1.5*iqr)
                cutoff_upper.append(digest.percentile(75) + 1.5*iqr)
                
                # Get total number of entries, number of observed values, and statistics, for each unique event in measurement table
                result = self._extract_AGG(table, aggregations=f"COUNT(*), COUNT(VALUE), MIN(VALUE), MAX(VALUE), AVG(VALUE)")  #  condition=condition
                table_counts, table_counts_obs, table_min_obs, table_max_obs, table_mean_obs = result[0]

                # Collate each tables summary statistics, here we have assumed that each measurement table contains a unique event
                measurements.append(measurement)
                counts.append(table_counts)
                obs_counts.append(table_counts_obs)
                obs_digest.append(digest)
                obs_mins.append(table_min_obs)
                obs_maxes.append(table_max_obs)
                obs_means.append(table_mean_obs)
                
            measurement_meta = pd.DataFrame({"event": measurements,
                                             "count": counts,
                                             "count_obs": obs_counts,
                                             "digest": obs_digest,
                                             "min": obs_mins,
                                             "max": obs_maxes,
                                             "mean": obs_means,
                                             "approx_lqr": cutoff_lower,
                                             "approx_uqr": cutoff_upper
                                             })

            meta_information["measurement_tables"] = measurement_meta
            logging.debug(f"measurement_meta: \n{measurement_meta}")
            
        return meta_information
                             
    