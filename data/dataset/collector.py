import sqlite3
import polars as pl
import pandas as pd
import numpy as np
from typing import Optional, Any, Union, Callable
import logging
from CPRD.data.database.build_static_db import Static
from CPRD.data.database.build_diagnosis_db import Diagnoses
from CPRD.data.database.build_measurements_and_tests_db import Measurements
from tqdm import tqdm
from tdigest import TDigest

class SQLiteDataCollector(Static, Diagnoses, Measurements):
    """ A class which interfaces with the SQLite database to collect and collate patient records

        Functionality additionally includes collecting meta information from the SQLite database 
    """
    def __init__(self, db_path):
        self.db_path = db_path

        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            self.connection = sqlite3.connect(self.db_path, timeout=20000)
            self.cursor = self.connection.cursor()
            logging.debug("Connected to SQLite database")
        except sqlite3.Error as e:
            logging.warning(f"Error connecting to SQLite database: {e}")

    def disconnect(self):
        if self.connection:
            self.connection.close()
            logging.debug("Disconnected from SQLite database")

    def _extract_distinct(self,
                          table_names:          list[str],
                          identifier_column:    str,
                          inclusion_conditions: Optional[list] = None,
                          combine_approach:     str = "AND"
                          ) -> list[str]:
        """
        Get a list of distinct `identifier_column' values, contained in a collection of tables
        """
        # Initialize an empty set to store distinct values
        unique_distinct = set()
        
        # Iterate over each table
        for idx_table, table in enumerate(table_names):
            
            # Construct the SQL query to extract distinct `identifier_column` entries for each specified table
            query = f"SELECT DISTINCT {identifier_column} FROM {table}"

            # If we want to add a condition
            if inclusion_conditions is not None:
                if inclusion_conditions[idx_table] is not None:
                    query += f" WHERE {inclusion_conditions[idx_table]}"
            
            # Execute the query
            logging.debug(f"Query: {query[:100] if len(query) > 100 else query}")
            self.cursor.execute(query)
            
            # Fetch distinct query values for the current table
            new_distinct_values = [_dv[0] for _dv in self.cursor.fetchall()]

            # and update the set
            if combine_approach == "OR":
                # For example, can condition static table to get `identifier_column` values
                #      based in England  OR  with condition X
                unique_distinct.update(new_distinct_values)
            elif combine_approach == "AND":
                #      based in England  AND  with condition X
                unique_distinct = unique_distinct & set(new_distinct_values) if idx_table != 0 else set(new_distinct_values)
            else:
                raise NotImplementedError

        return list(unique_distinct)

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
            try:
                digest.batch_update(values)
                fetches_count += 1
            except:
                logging.warning(f"Unable to batch update t-digest for values {values} from {table_name}, skipping batch")
                pass 
                
        return digest
    
    def _generate_lazy_by_distinct(self,
                                   distinct_values: list,
                                   identifier_column: str,
                                   include_diagnoses: bool = True, 
                                   include_measurements: bool = True,
                                   conditions: Optional[list[str]] = None
                                   ) -> list[pl.LazyFrame]:
        """
        ARGS:
            distinct_values:
                    is a list of distinct values on which to partition the identifier_column
            identifier_column: 
                    the identifier column to use for partioning (either PRACTICE_ID or PATIENT_ID. Default: PRACTICE_ID).

        KWARGS:
            include_diagnoses:      
                    Whether to include diagnoses table values in the list of returned lazy frames
            include_measurements:
                    Whether to include measurement table values in the list of returned lazy frames
            conditions:  
                    is a list of conditions to filter the data (removing rows from sql tables), where each condition applies to a specific table. 
                    Note, this is probably not required, as this does not remove entire patients, nor practices, based on criteria - but singular events.
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
                    query = f"SELECT * FROM {table} WHERE {identifier_column} IN ({sep_list});"
                    
                else:
                    query = f"SELECT * FROM {table} WHERE {identifier_column} = '{distinct_value}'"
                
                if conditions is not None:
                    if conditions[idx_table] is not None:
                        #  conditions for each table, e.g. a query asking for only certain diagnoses or measurements to be included in the generator
                        query += f"AND {conditions[idx_table]}"
                                  
                logging.debug(f"Query: {query[:120] if len(query) > 120 else query}")

                # Load with polars
                #    This can cause timeout issues
                # df = pl.read_database(query=query, connection_uri='sqlite://' + self.db_path)

                # Load with pandas then convert to polars. 
                #   This lets us use the existing connection from the sqlite3 package and so we can specify longer timeout
                #   We also need to specify 'VALUE' is a float, as pandas will convert this to string (not all queries are on tables with VALUE)
                pandas_df = pd.read_sql_query(query, self.connection)
                if "VALUE" in pandas_df.columns:
                    pandas_df["VALUE"] = pandas_df["VALUE"].astype(float)
                df = pl.from_pandas(pandas_df)

                if len(df) > 0:
                    rows_by_table["lazy_" + table] = df.lazy()

            # Yield the fetched rows as a chunk
            yield distinct_value, rows_by_table
                
    def _collate_lazy_tables(self,
                             lazy_frames,   
                             study_inclusion_method              = None,
                             drop_empty_dynamic:            bool = True,
                             drop_missing_data:             bool = True,
                             **kwargs
                             ) -> pl.LazyFrame:
        """
        Merge each lazy frame from each, applying optional conditions.
        
        ┌───────────┬──────────┬────────────┬────────────┬───┬───────────┬──────────┬──────────┬───────────┐
        │ PRACTICE_ ┆ PATIENT_ ┆ VALUE      ┆ EVENT      ┆ … ┆ HEALTH_AU ┆ INDEX_DA ┆ START_DA ┆ END_DATE  │
        │ ID        ┆ ID       ┆ ---        ┆ ---        ┆   ┆ TH        ┆ TE       ┆ TE       ┆ ---       │
        │ ---       ┆ ---      ┆ list[f64]  ┆ list[str]  ┆   ┆ ---       ┆ ---      ┆ ---      ┆ datetime[ │
        │ i64       ┆ i64      ┆            ┆            ┆   ┆ str       ┆ datetime ┆ datetime ┆ μs]       │
        │           ┆          ┆            ┆            ┆   ┆           ┆ [μs]     ┆ [μs]     ┆           │
        ╞═══════════╪══════════╪════════════╪════════════╪═══╪═══════════╪══════════╪══════════╪═══════════╡
        │ 20429     ┆ 22038164 ┆ [60.0,     ┆ ["Diastoli ┆ … ┆ South     ┆ 2005-01- ┆ 2005-01- ┆ 2022-03-1 │
        │           ┆ 20429    ┆ 120.0, …   ┆ c_blood_pr ┆   ┆ East      ┆ 01       ┆ 01       ┆ 7         │
        │           ┆          ┆ 100.0]     ┆ essure_5", ┆   ┆           ┆ 00:00:00 ┆ 00:00:00 ┆ 00:00:00  │
        │           ┆          ┆            ┆ "…         ┆   ┆           ┆          ┆          ┆           │
        │ 20429     ┆ 22038165 ┆ [20.7,     ┆ ["Body_mas ┆ … ┆ South     ┆ 2018-06- ┆ 2018-06- ┆ 2022-03-1 │
        │           ┆ 20429    ┆ null, …    ┆ s_index_3" ┆   ┆ East      ┆ 27       ┆ 27       ┆ 7         │
        │           ┆          ┆ 144.0]     ┆ , "Body_ma ┆   ┆           ┆ 00:00:00 ┆ 00:00:00 ┆ 00:00:00  │
        │           ┆          ┆            ┆ ss…        ┆   ┆           ┆          ┆          ┆           │
        │ 20429     ┆ 22038168 ┆ [null,     ┆ ["Never_sm ┆ … ┆ South     ┆ 2011-04- ┆ 2011-04- ┆ 2022-03-1 │
        │           ┆ 20429    ┆ 90.0,      ┆ oked_tobac ┆   ┆ East      ┆ 23       ┆ 23       ┆ 7         │
        │           ┆          ┆ 130.0]     ┆ co_85",    ┆   ┆           ┆ 00:00:00 ┆ 00:00:00 ┆ 00:00:00  │
        │           ┆          ┆            ┆ "Dia…      ┆   ┆           ┆          ┆          ┆           │
        │ 20429     ┆ 22038169 ┆ [25.9,     ┆ ["Body_mas ┆ … ┆ South     ┆ 2005-01- ┆ 2005-01- ┆ 2011-11-0 │
        │           ┆ 20429    ┆ 80.0, …    ┆ s_index_3" ┆   ┆ East      ┆ 01       ┆ 01       ┆ 7         │
        │           ┆          ┆ 120.0]     ┆ , "Diastol ┆   ┆           ┆ 00:00:00 ┆ 00:00:00 ┆ 00:00:00  │
        │           ┆          ┆            ┆ ic…        ┆   ┆           ┆          ┆          ┆           │
        │ 20429     ┆ 22038170 ┆ [24.8,     ┆ ["Body_mas ┆ … ┆ South     ┆ 2005-01- ┆ 2005-01- ┆ 2008-06-1 │
        │           ┆ 20429    ┆ 76.0, …    ┆ s_index_3" ┆   ┆ East      ┆ 01       ┆ 01       ┆ 9         │
        │           ┆          ┆ null]      ┆ , "Diastol ┆   ┆           ┆ 00:00:00 ┆ 00:00:00 ┆ 00:00:00  │
        │           ┆          ┆            ┆ ic…        ┆   ┆           ┆          ┆          ┆           │
        └───────────┴──────────┴────────────┴────────────┴───┴───────────┴──────────┴──────────┴───────────┘

        Notes:  The collection is applied after this function
                We do not sort within the lazy operation, so the row order will not be deterministic
        """
        
        ##############################
        # GET THE LAZY POLARS FRAMES #
        ##############################
        
        # Static lazy frame, converting all dates to datetime format
        lazy_static = (
            lazy_frames["lazy_static_table"]
            .with_columns(pl.col("INDEX_DATE").str.to_datetime("%Y-%m-%d"))
            .with_columns(pl.col("START_DATE").str.to_datetime("%Y-%m-%d"))
            .with_columns(pl.col("END_DATE").str.to_datetime("%Y-%m-%d"))
            .with_columns(pl.col("YEAR_OF_BIRTH").str.to_datetime("%Y-%m-%d"))  
        )

        # Diagnosis lazy frame
        lazy_diagnosis = lazy_frames["lazy_diagnosis_table"] if "lazy_diagnosis_table" in lazy_frames.keys() else None

        # Measurement lazy frames
        measurement_keys = [key for key in lazy_frames if key.startswith("lazy_measurement_")]
        measurement_lazy_frames = [lazy_frames[key] for key in measurement_keys]
        lazy_measurement = pl.concat(measurement_lazy_frames) if len(measurement_lazy_frames) > 0 else None
        #    and optionally drop missing measurements
        if drop_missing_data and measurement_lazy_frames is not None:
            logging.debug("Dropping missing measurements")
            lazy_measurement = lazy_measurement.drop_nulls()

        #####################################
        # MERGE SOURCES OF TIME_SERIES DATA #
        #####################################
        
        # Merge all frames containing time series data
        if lazy_diagnosis is not None and lazy_measurement is not None:
            # Stacking the frames vertically. Value column in diagnostic is filled with null
            lazy_combined_frame = pl.concat([lazy_measurement, lazy_diagnosis], how="diagonal")
        elif lazy_measurement is not None:
            #
            lazy_combined_frame = lazy_measurement            
        elif lazy_diagnosis is not None:
            # Note: if we load diagnoses with no measurements, we are not concating values, so add this missing column
            lazy_combined_frame = lazy_diagnosis.with_columns(pl.lit(None).cast(pl.Utf8).alias('VALUE'))
        else:
            raise NotImplementedError

        # Dynamic lazy frame, converting all dates to datetime format
        lazy_combined_frame = (
            lazy_combined_frame
            .with_columns(pl.col("DATE").str.to_datetime("%Y-%m-%d"))
        )

        # Convert event date to time since birth by linking the dynamic diagnosis and measurement frames to the static one
        # Subtract the dates and create a new column for the result            
        lazy_combined_frame = (
            lazy_combined_frame
            .join(lazy_static.select(["PRACTICE_ID", "PATIENT_ID", "YEAR_OF_BIRTH"]),                     # Add birth year information to calculate relative event times
                  on=["PRACTICE_ID", "PATIENT_ID"], how="inner")
            .select([
                  (pl.col("DATE") - pl.col("YEAR_OF_BIRTH")).dt.days().alias("DAYS_SINCE_BIRTH"), "*"
                ])
            .drop("YEAR_OF_BIRTH")
            )

        # Drop events which occur at unrealistic ages
        lazy_combined_frame = (
            lazy_combined_frame
            .filter(pl.col("DAYS_SINCE_BIRTH") > -365)
            .filter(pl.col("DAYS_SINCE_BIRTH") < 125*365)
        )
        
        #################
        # FILTER FRAMES #
        #################

        # Reduce based on study criteria. You may pass your own custom criteria method
        if study_inclusion_method is not None:
            lazy_static, lazy_combined_frame = study_inclusion_method(lazy_static, lazy_combined_frame)

        # Remove patients without multiple events
        lazy_combined_frame = (
            lazy_combined_frame.groupby("PATIENT_ID")
                .agg(pl.count("PATIENT_ID").alias("count"))
                .filter(pl.col("count") > 1)
                .join(lazy_combined_frame, on="PATIENT_ID", how="inner")
                .select(lazy_combined_frame.columns)
        )
        
        #############
        # AGGREGATE #
        #############

        # Remove entries before conception (negative to include pregnancy period, e.g. diagnosed with genetic condition pre-birth)
        agg_cols = ["VALUE", "EVENT", "DAYS_SINCE_BIRTH", "DATE"]
        lazy_combined_frame = (
            lazy_combined_frame
            .sort("DAYS_SINCE_BIRTH")
            .groupby(["PRACTICE_ID", "PATIENT_ID"])
            .agg(agg_cols)                                                # Turn into lists
            .sort(["PRACTICE_ID", "PATIENT_ID"])                          # make lazy collection output deterministic
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
        lazy_combined_frame = lazy_combined_frame.join(lazy_static, on=["PRACTICE_ID", "PATIENT_ID"], how="inner")


        return lazy_combined_frame

    def get_meta_information(self,
                             practice_ids:          Optional[list] = None,
                             static:                bool = True,
                             diagnoses:             bool = True,
                             measurement:           bool = True,
                            ) -> dict:

        # Standardisation is TODO
        logging.info("\n\nCollecting meta information from database. This will be used for tokenization and (optionally) standardisation.")
        
        # Initialise meta information 
        meta_information = {}

        # TODO: calculate these only on training splits! Especially if standardisation gets implemented
        if practice_ids is not None:
            raise NotImplementedError
            # condition = f"PRACTICE_ID IN ({",".join([f"'{dv}'" for dv in practice_ids])});" 

        if static is True:
            logging.info("\t Static meta information")
            static_meta = {}
            for categorical_covariate in ["SEX", "IMD", "ETHNICITY"]:                
                result = self._extract_AGG("static_table", identifier_column=categorical_covariate, aggregations="COUNT(*)")
                category, counts = zip(*result)
                static_meta[categorical_covariate] = pd.DataFrame({"category": category,
                                                                   "count": [i for i in counts],
                                                                   })
            meta_information["static_table"] = static_meta
            logging.debug(f"static_meta: \n{static_meta}")
        
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

                result = self._extract_AGG(table, aggregations=f"COUNT(*), COUNT(VALUE)")  #  condition=condition
                table_counts, table_counts_obs = result[0]

                if table_counts_obs > 0:
                    # Online accumulation to approximate quantiles which will then be used for standardisation and outlier removal.
                    digest = self._t_digest_values(table)
                    # From summary statistics get the standardisation limits
                    iqr = digest.percentile(75) - digest.percentile(25)
                    digest_lower = digest.percentile(25) - 1.5*iqr
                    digest_upper = digest.percentile(75) + 1.5*iqr
                    
                    # Get total number of entries, number of observed values, and statistics, for each unique event in measurement table
                    result = self._extract_AGG(table, aggregations=f"MIN(VALUE), MAX(VALUE), AVG(VALUE)")  #  condition=condition
                    table_min_obs, table_max_obs, table_mean_obs = result[0]
                else:
                    # Catch cases where there are no values for this measurement
                    digest = None
                    digest_lower, digest_upper = None, None
                    # Get total number of entries, number of observed values, and statistics, for each unique event in measurement table
                    table_min_obs, table_max_obs, table_mean_obs = None, None, None
                    
                # Collate each tables summary statistics, here we have assumed that each measurement table contains a unique event
                measurements.append(measurement)
                counts.append(table_counts)
                obs_counts.append(table_counts_obs)
                obs_digest.append(digest)
                obs_mins.append(table_min_obs)
                obs_maxes.append(table_max_obs)
                obs_means.append(table_mean_obs)
                cutoff_lower.append(digest_lower)
                cutoff_upper.append(digest_upper)

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
                             
    