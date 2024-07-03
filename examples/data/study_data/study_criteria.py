import polars as pl

def cvd_inclusion_method(lazy_static,
                         lazy_combined_frame,
                         index_on_events=["TYPE2DIABETES", "SUBSTANCEMISUSE"], 
                         exclude_on_events_prior_to_index=None,   # add statins here   , [""]
                         exclude_on_events=["TYPE1DM"],
                         outcomes=["COPD", "SUBSTANCEMISUSE"],
                         study_period=["1998-01-01", "2019-12-31"],
                         age_at_entry_range=[25, 85],
                         min_registered_years=1,
                         ):
    return index_inclusion_method(lazy_static,
                                  lazy_combined_frame,
                                  index_on_events=index_on_events, 
                                  exclude_on_events_prior_to_index=exclude_on_events_prior_to_index,
                                  exclude_on_events=exclude_on_events,
                                  outcomes=outcomes,
                                  study_period=study_period,
                                  age_at_entry_range=age_at_entry_range,
                                  min_registered_years=min_registered_years
                                 )

def index_inclusion_method(lazy_static,
                           lazy_combined_frame,
                           index_on_events=["TYPE2DIABETES"], 
                           exclude_on_events_prior_to_index=None,   # add statins here   , [""]
                           exclude_on_events=["TYPE1DM"],
                           outcomes=["COPD", "SUBSTANCEMISUSE"],
                           study_period=["1998-01-01", "2019-12-31"],
                           age_at_entry_range=[25, 85],
                           min_registered_years=1,
                           ):
    """
    ARGS:
        lazy_static

        lazy_combined_frame

    KWARGS:
        index_on_events
            The conditions each patient must have ANY (not all) of, to be included in the cohort study
        exclude_on_events_prior_to_index

        exclude_on_events
        
        study_period
            the start and end date of the study period, in the form ["yyyy-mm-dd", "yyyy-mm-dd"] in increasing order
        age_at_entry_range
            the minimum and maximum age at cohort entry, in the form [lower,upper] in years
        min_registered_years
            the minimum number of years a patient must be registered at the practice for at cohort entry

    Note: because this is called on a per practice basis, we dont need to worry about overlapping PATIENT_ID between practices (only the combination is unique)
    """

    #############################################################
    # RETAIN ONLY PATIENTS WITHIN STUDY FOCUS     
    # Only include patients who have a `index_on_events` event
    #############################################################
    
    # Retain only patients with the required events occurring
    patients_with_required_events = (
        lazy_combined_frame
        .filter(pl.col("EVENT").is_in(index_on_events))                                                  # Include only patients who experienced the events                  
    )

    # if not at any time, then we only include patients where the events occur during the study period
    at_any_time = True
    if not at_any_time:
        patients_with_required_events = (
            patients_with_required_events
            .filter(pl.col('DATE') >= pl.lit(study_period[0]).str.strptime(pl.Date, fmt="%F"))               # after study start date
            .filter(pl.col('DATE') <= pl.lit(study_period[1]).str.strptime(pl.Date, fmt="%F"))               # and before study end date
        )

    # Get the patient list
    patients_with_required_events = (
        patients_with_required_events
        .unique("PATIENT_ID")
        .select(pl.col('PATIENT_ID'))
    )

    # and reduce original frames using this list
    lazy_combined_frame = patients_with_required_events.join(lazy_combined_frame, on=["PATIENT_ID"], how="inner")  
    lazy_static = patients_with_required_events.join(lazy_static, on=["PATIENT_ID"], how="inner")  

    #############################################################
    # REMOVE PATIENTS OUTSIDE OF STUDY FOCUS     
    # Exclude patients who have an `exclude_on_events` event
    #############################################################
    
    # TODO: I dont know how to do this with fully lazy execution, currently collecting the list
    if exclude_on_events is not None:
        # Remove patients who have any of these events, regardless of when
        patients_with_excluded_events = (
            lazy_combined_frame
            .filter(pl.col("EVENT").is_in(exclude_on_events))                                                # We want to exclude patients who experienced the events
            .select(pl.col('PATIENT_ID'))                                                                    # get the patient list
            .unique()
        )
        
        # Get the patient list
        patients_with_excluded_events_list = patients_with_excluded_events.collect().to_series().to_list()

        # and reduce original frames using this list
        lazy_combined_frame = lazy_combined_frame.filter(~pl.col("PATIENT_ID").is_in(patients_with_excluded_events_list))
        lazy_static = lazy_static.filter(~pl.col("PATIENT_ID").is_in(patients_with_excluded_events_list))

    ##############################################
    # REMOVE PATIENTS ON REGISTRATION LENGTH     #
    ##############################################
    # Remove all patients who are not registered at the practice for a period of at least `min_registered_years` 
    lazy_static = (
        lazy_static
        .select([
            (pl.col("END_DATE") - pl.col("START_DATE")).dt.days().alias("DAYS_REGISTERED"), "*"
            ])
        .filter(pl.col('DAYS_REGISTERED') >= min_registered_years*365.25)
        .drop("DAYS_REGISTERED")
    )
    patients_within_registration_length = lazy_static.select(pl.col('PATIENT_ID'))
    lazy_combined_frame = patients_within_registration_length.join(lazy_combined_frame, on=["PATIENT_ID"], how="inner")

       
    #############################################################
    # SET INDEX DATE
    # to the first time an `index_on_events` event occurs if in eligible period
    # else to first time meeting age requirement in study period
    #############################################################
    first_index_date = (
        lazy_combined_frame
        .filter(pl.col("EVENT").is_in(index_on_events))                # Look at only the events we can index from
        .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])                   # Sort to ensure date order within patients
        .unique(subset=["PRACTICE_ID", "PATIENT_ID"], keep="first")    # Keep chronologically first required event experienced by patient
        .with_columns((pl.col('DATE').alias('FIRST_INDEX_DATE')))      # This is the earliest possible index date. It would be later if this lies outside of study dates
        .select(["PRACTICE_ID", "PATIENT_ID", "FIRST_INDEX_DATE"])     # take only the columns we need
    )

    earliest_cohort_entry = (
        lazy_static
        .with_columns((pl.col('YEAR_OF_BIRTH') + pl.duration(days=int(age_at_entry_range[0]*365.25))).alias('EARLIEST_AGE_REQUIREMENT'))   # Calculate the earliest point at which age requirement is met
        .with_columns((pl.lit(study_period[0]).str.strptime(pl.Date, fmt="%F")).alias('EARLIEST_COHORT_ENTRY'))                            # The earliest entry date requirement
        .select([
            pl.max("EARLIEST_AGE_REQUIREMENT", "EARLIEST_COHORT_ENTRY").alias("EARLIEST_INDEX_DATE"), "*"
            ])
        .select(["PRACTICE_ID", "PATIENT_ID", "EARLIEST_INDEX_DATE"])                                                             # take only the columns we need
    )

    index_date = (
        first_index_date
        .join(earliest_cohort_entry, on=["PRACTICE_ID", "PATIENT_ID"], how="inner")
        .select([
            pl.max("FIRST_INDEX_DATE", "EARLIEST_INDEX_DATE").alias("INDEX_DATE"), "*"
        ])
        .select(["PRACTICE_ID", "PATIENT_ID", "INDEX_DATE"])
        .filter(pl.col('INDEX_DATE') <= pl.lit(study_period[1]).str.strptime(pl.Date, fmt="%F"))               # and before study end date
    )

    # and inner join this back to the original frames to get all records of those patients
    lazy_combined_frame = index_date.join(lazy_combined_frame, on=["PRACTICE_ID", "PATIENT_ID"], how="inner")  
    lazy_static = index_date.join(lazy_static.drop("INDEX_DATE"), on=["PRACTICE_ID", "PATIENT_ID"], how="inner")  

    #############################################################
    # Remove patients where no events follow the index date 
    # as we want followup predictions, we need to ensure an event occurs
    #############################################################
    patients_with_entries_after_index = (
        lazy_combined_frame
        .filter(pl.col("DATE") > pl.col("INDEX_DATE"))
        .select(pl.col('PATIENT_ID'))                                                                    # get the patient list
        .unique()
    )
    patients_with_entries_after_index_list = patients_with_entries_after_index.collect().to_series().to_list()
    
    lazy_combined_frame = lazy_combined_frame.filter(pl.col("PATIENT_ID").is_in(patients_with_entries_after_index_list))
    lazy_static = lazy_static.filter(pl.col("PATIENT_ID").is_in(patients_with_entries_after_index_list))
    
    #############################################################
    # Remove patients where `exclude_on_events_prior_to_index` 
    # occured before the index event
    #############################################################
    if exclude_on_events_prior_to_index is not None:
        # For example, if we are interested in the possibility of prescribing a medicine then we may want to exclude those already on the medicine
        assert index_on_events is not None
        
        raise NotImplementedError

    #############################################################
    # GET EVENTS WHICH OCCUR BEFORE INDEX (INCLUSIVE)
    #############################################################
    lazy_combined_frame_before_index = (
        lazy_combined_frame
        .filter(pl.col('DATE') <= pl.col('INDEX_DATE'))
    )
        
    #############################################################
    # GET OUTCOMES WHICH OCCUR AFTER INDEX
    # or if no outcome, the last seen event that wasnt an outcome
    #############################################################
        
    # Get outcomes occurring after index, and before study end
    lazy_combined_frame_outcomes = (
        lazy_combined_frame
        .filter(pl.col("EVENT").is_in(outcomes))
        .filter(pl.col("DATE") > pl.col("INDEX_DATE"))
        .filter(pl.col("DATE") <= pl.lit(study_period[1]).str.strptime(pl.Date, fmt="%F"))  # Only look at events within study period
    )

    # Get last observation within study period
    lazy_combined_frame_last_event_in_study = (
        lazy_combined_frame
        .filter(pl.col("DATE") <= pl.lit(study_period[1]).str.strptime(pl.Date, fmt="%F"))  # Only look at events within study period
        .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])                                        # Sort to ensure date order within patients
        .unique(subset=["PRACTICE_ID", "PATIENT_ID"], keep="last")                          # Keep chronologically last event experienced by patient
    )

    # Take first between these - this will the first outcome if one has occurred, otherwise the last event
    outcome = (
        pl.concat([lazy_combined_frame_outcomes, lazy_combined_frame_last_event_in_study])
        .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])                                        # Sort to ensure date order within patients
        .unique(subset=["PRACTICE_ID", "PATIENT_ID"], keep="first")                          # Keep chronologically last event experienced by patient
    )

    # and merge this with the events that occurred up to and including the index from the last section
    new_combined_frame = (
            pl.concat([lazy_combined_frame_before_index, outcome])
            .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])                                        # Sort to ensure date order within patients
        )

    #############################################################
    # For debugging
    #############################################################
    debug = False
    if debug:
        pl.Config.set_tbl_cols(200)
        pl.Config.set_fmt_str_lengths(100)
        
        # All
        print(lazy_static.collect())
        print(lazy_combined_frame_outcomes.collect())
        
        # Look at a certain patient
        patient_id = 567788920185
        print(f"static: {lazy_static.filter(pl.col('PATIENT_ID') == patient_id).collect()}")
        print(f"lazy_combined_frame: {lazy_combined_frame.filter(pl.col('PATIENT_ID') == patient_id).sort('DATE').collect()}")
        
        print(f"lazy_combined_frame_before_index: {lazy_combined_frame_before_index.filter(pl.col('PATIENT_ID') == patient_id).sort('DATE').collect()}")
        print(f"lazy_combined_frame_outcomes: {lazy_combined_frame_outcomes.filter(pl.col('PATIENT_ID') == patient_id).sort('DATE').collect()}")

        print(f"lazy_combined_frame_last_event_in_study: {lazy_combined_frame_last_event_in_study.filter(pl.col('PATIENT_ID') == patient_id).sort('DATE').collect()}")
        print(f"outcome: {outcome.filter(pl.col('PATIENT_ID') == patient_id).sort('DATE').collect()}")
        print(f"new_combined_frame: {new_combined_frame.filter(pl.col('PATIENT_ID') == patient_id).sort('DATE').collect()}")

        raise NotImplementedError

    return lazy_static, new_combined_frame