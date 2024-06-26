import polars as pl

def cvd_inclusion_method(lazy_static,
                         lazy_combined_frame,
                         required_on_events=["TYPE2DIABETES"], 
                         exclude_on_events_prior_to_required=None,   # add statins here   , [""]
                         exclude_on_events=["TYPE1DM"],
                         study_period=["1998-01-01", "2019-12-31"],
                         age_at_entry_range=[25, 85],
                         min_registered_years=1,
                        ):
    """
    ARGS:
        lazy_static

        lazy_combined_frame

    KWARGS:
        required_on_events
            The conditions each patient must have ANY (not all) of, to be included in the cohort study
        exclude_on_events_prior_to_required

        exclude_on_events
        
        study_period
            the start and end date of the study period, in the form ["yyyy-mm-dd", "yyyy-mm-dd"] in increasing order
        age_at_entry_range
            the minimum and maximum age at cohort entry, in the form [lower,upper] in years
        min_registered_years
            the minimum number of years a patient must be registered at the practice for at cohort entry

    Note: because this is called on a per practice basis, we dont need to worry about overlapping PATIENT_ID between practices
    """

    ##############################################
    # REMOVE PATIENTS OUTSIDE OF STUDY FOCUS     #
    ##############################################
    at_any_time = True
    if required_on_events is not None:
        # Retain only patients with the required events occurring
        patients_with_required_events = (
            lazy_combined_frame
            .filter(pl.col("EVENT").is_in(required_on_events))                                                  # Include only patients who experienced the events                  
        )
        if not at_any_time:
            # if not at any time, then we only include patients where the events occur during the study period
            patients_with_required_events = (
                patients_with_required_events
                .filter(pl.col('DATE') >= pl.lit(study_period[0]).str.strptime(pl.Date, fmt="%F"))               # after study start date
                .filter(pl.col('DATE') <= pl.lit(study_period[1]).str.strptime(pl.Date, fmt="%F"))               # and before study end date
            )
        patients_with_required_events = patients_with_required_events.select(pl.col('PATIENT_ID'))               # get the patient list
        
        # and inner join this back to the original frames to get all records of those patients
        lazy_combined_frame = patients_with_required_events.join(lazy_combined_frame, on=["PATIENT_ID"], how="inner")  
        lazy_static = patients_with_required_events.join(lazy_static, on=["PATIENT_ID"], how="inner")  

    if exclude_on_events_prior_to_required is not None:
        # Remove patients who had events that occured before an index event
        assert required_on_events is not None
        
        raise NotImplementedError
    
    index_at_first_required_event = True
    if index_at_first_required_event:
        # Without this, patients are indexed (enter cohort) at their first eligible observation (within time period, sufficient age, and registered long enough)
        # Here we update it so that patients enter with these conditions, and the condition after their first `required_on_events` event occurrs
        earliest_index_date = (
            lazy_combined_frame
            .filter(pl.col("EVENT").is_in(required_on_events))                # Look at only the events we can index from
            .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])                      # Sort to ensure date order within patients
            .unique(subset=["PRACTICE_ID", "PATIENT_ID"], keep="first")       # Keep chronologically first required event experienced by patient
            .with_columns((pl.col('DATE').alias('EARLIEST_INDEX_DATE')))      # This is the earliest possible index date. It would be later if this lies outside of study dates
            .select(["PRACTICE_ID", "PATIENT_ID", "EARLIEST_INDEX_DATE"])     # take only the columns we need
        )

        lazy_combined_frame = (
            lazy_combined_frame
            .join(earliest_index_date.select(["PRACTICE_ID", "PATIENT_ID", "EARLIEST_INDEX_DATE"]), on=["PRACTICE_ID", "PATIENT_ID"], how="inner")
            .filter(pl.col("DATE") >= pl.col("EARLIEST_INDEX_DATE"))                                 # keep only events which occur after index date
            .drop("EARLIEST_INDEX_DATE")
        )

    # TODO: I dont know how to do this with fully lazy execution, currently collecting the list
    if exclude_on_events is not None:
        # Remove patients who have any of these events, regardless of when
        patients_with_excluded_events = (
            lazy_combined_frame
            .filter(pl.col("EVENT").is_in(exclude_on_events))                                                # We want to exclude patients who experienced the events
            .select(pl.col('PATIENT_ID'))                                                                    # get the patient list
            .unique()
        )
        patients_with_excluded_events_list = patients_with_excluded_events.collect().to_series().to_list()
        lazy_combined_frame = lazy_combined_frame.filter(~pl.col("PATIENT_ID").is_in(patients_with_excluded_events_list))
       
    # print(lazy_combined_frame.sort(["PRACTICE_ID", "PATIENT_ID", "DATE"]).collect())
    # print(lazy_static.sort(["PRACTICE_ID", "PATIENT_ID"]).collect())
        
    ##############################################
    # REMOVE PATIENTS ON REGISTRATION LENGTH     #
    ##############################################
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

    ##############################################
    # REMOVE RECORDS OUTSIDE OF STUDY DATES      #
    ##############################################
    # If this leaves any patients with no events, but still in the static frame that is ok - as we filter empty dynamic events later
    lazy_combined_frame = (
        lazy_combined_frame
        .filter(pl.col('DATE') >= pl.lit(study_period[0]).str.strptime(pl.Date, fmt="%F"))               # after study start date
        .filter(pl.col('DATE') <= pl.lit(study_period[1]).str.strptime(pl.Date, fmt="%F"))               # and before study end date
    )

    ##############################################
    # FILTER BY AGE AT COHORT ENTRY              #
    ##############################################
    #    TODO: currently calculating in days, but could be easier to just calculate time difference directly in years if supported? TypeError
        
    # 1) Calculate date of earliest possible cohort entry
    earliest_cohort_entry = (
        lazy_static
        .with_columns((pl.col('YEAR_OF_BIRTH') + pl.duration(days=int(age_at_entry_range[0]*365.25))).alias('EARLIEST_COHORT_ENTRY'))              # Calculate the earliest cohort entry date
        .select(["PRACTICE_ID", "PATIENT_ID", "EARLIEST_COHORT_ENTRY"])                                                         # take only the columns we need
        .filter(pl.col("EARLIEST_COHORT_ENTRY") <= pl.lit(study_period[1]).str.strptime(pl.Date, fmt="%F"))                     # Remove patients who are not old enough to join study
    )

    # 2) Remove any event which occurs before each patients first date of cohort entry
    lazy_combined_frame = (
        lazy_combined_frame
        .join(earliest_cohort_entry, on=["PRACTICE_ID", "PATIENT_ID"], how="inner")
        .filter(pl.col("DATE") >= pl.col("EARLIEST_COHORT_ENTRY"))                                 # keep only events which occur after lower age limit
        .drop("EARLIEST_COHORT_ENTRY")
    )

    # 3) Remove patients whose earliest event is after upper age limit
    patients_under_latest_limit_at_first_event = (
        lazy_combined_frame
        .join(lazy_static.select(["PRACTICE_ID", "PATIENT_ID", "YEAR_OF_BIRTH"]), 
              on=["PRACTICE_ID", "PATIENT_ID"], how="inner")
        .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])
        .unique(subset=["PRACTICE_ID", "PATIENT_ID"], keep="first")
        .with_columns((pl.col('YEAR_OF_BIRTH') + pl.duration(days=int(age_at_entry_range[1]*365.25))).alias('LATEST_COHORT_ENTRY')) 
        .filter(pl.col("DATE") <= pl.col("LATEST_COHORT_ENTRY"))
        .select(["PRACTICE_ID", "PATIENT_ID"])
    )
    lazy_static = patients_under_latest_limit_at_first_event.join(lazy_static, on=["PRACTICE_ID", "PATIENT_ID"], how="inner")
    lazy_combined_frame = patients_under_latest_limit_at_first_event.join(lazy_combined_frame, on=["PRACTICE_ID", "PATIENT_ID"], how="inner")

    return lazy_static, lazy_combined_frame