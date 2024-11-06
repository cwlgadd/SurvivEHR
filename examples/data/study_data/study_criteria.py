import polars as pl
import logging

def cvd_inclusion_method(index_on_event="TYPE2DIABETES", 
                         outcomes=["IHDINCLUDINGMI_OPTIMALV2", "ISCHAEMICSTROKE_V2", "MINFARCTION", "STROKEUNSPECIFIED_V2", "STROKE_HAEMRGIC"],
                         exclude_on_events=["TYPE1DM"],                         
                         exclude_on_events_prior_to_index=['Statins'],
                         study_period=["1998-01-01", "2019-12-31"],
                         age_at_entry_range=[25, 85],
                         min_registered_years=1,
                         min_events=None,
                         ):

    CVD_inclusion = index_inclusion_method(index_on_event=index_on_event, 
                                           exclude_on_events_prior_to_index=exclude_on_events_prior_to_index,
                                           exclude_on_events=exclude_on_events,
                                           outcomes=outcomes,
                                           study_period=study_period,
                                           age_at_entry_range=age_at_entry_range,
                                           min_registered_years=min_registered_years,
                                           min_events=min_events,
                                           )
    return CVD_inclusion.fit

class index_inclusion_method():
    
    def __init__(self,
                 index_on_event,
                 outcomes,
                 exclude_on_events_prior_to_index=None,
                 exclude_on_events=None,
                 study_period=["1998-01-01", "2019-12-31"],
                 age_at_entry_range=[25, 85],
                 min_registered_years=1,
                 min_events=None,
                 ):
        """
        ARGS:
            lazy_static
    
            lazy_combined_frame
    
        KWARGS:
            index_on_event
                The condition each patient must have.
            exclude_on_events_prior_to_index
                Remove patients based on whether they have experienced events prior to their index date for the study.
                For example, if the study goal is to screen for being placed on a medication, we may want to remove those already on the medication from the study
            exclude_on_events
                Remove patients on whether they've experienced an event at any time.
                For example, if we focus our study based on Type II diabetes, any patients with a diagnosis of Type I diabetes can automatically be excluded.
            study_period
                The start and end date of the study period, in the form ["yyyy-mm-dd", "yyyy-mm-dd"] in increasing order.
                The start of the study period DOES NOT mean the start of observations, but contributes to determining the start of the indexing period. 
                The study end is the end of observations.
            age_at_entry_range
                the minimum and maximum age at cohort entry, in the form [lower,upper] in years
            min_registered_years
                the minimum number of years a patient must be registered at the practice for at cohort entry
            min_events
                the minimum number of events that must occur up to and including the index event to be included in the study
    
        Note: because this is called on a per practice basis, we dont need to worry about overlapping PATIENT_ID between practices (only the combination is unique)
        """
        
        self._index_on_event = index_on_event
        self._outcomes = outcomes
        self._exclude_on_events_prior_to_index = exclude_on_events_prior_to_index
        self._exclude_on_events = exclude_on_events
        self._study_period = study_period
        self._age_at_entry_range = age_at_entry_range
        self._min_registered_years = min_registered_years
        self._min_events = min_events

    def fit(self,
            lazy_static,
            lazy_combined_frame):

        # patient_id_for_checking = 5922416221434
        # Reduce the frames by removing any patients who do not satisfy global criteria
        lazy_static, lazy_combined_frame = self._remove_on_global_criteria(lazy_static, lazy_combined_frame)
        # Force collection
        lazy_static = lazy_static.collect().lazy()
        lazy_combined_frame = lazy_combined_frame.collect().lazy()
        # print(lazy_static.filter(pl.col("PATIENT_ID")==patient_id_for_checking).collect()) 
        # print(lazy_combined_frame.filter(pl.col("PATIENT_ID")==patient_id_for_checking).sort(["PRACTICE_ID", "PATIENT_ID", "DATE"]).collect())

        # Set an index date
        lazy_static, lazy_combined_frame = self._set_index_date(lazy_static, lazy_combined_frame)
        # Force collection
        lazy_static = lazy_static.collect().lazy()
        lazy_combined_frame = lazy_combined_frame.collect().lazy()
        # print(lazy_static.filter(pl.col("PATIENT_ID")==patient_id_for_checking).collect())
        # print(lazy_combined_frame.filter(pl.col("PATIENT_ID")==patient_id_for_checking).sort(["PRACTICE_ID", "PATIENT_ID", "DATE"]).collect())
        
        # Given this index date, reduce events to those leading to and including the date, and the final observation (observed or last seen within study period)
        lazy_static, lazy_combined_frame = self._reduce_on_index_date(lazy_static, lazy_combined_frame)
        # print(lazy_static.collect())  # .filter(pl.col("PATIENT_ID")==patient_id_for_checking)
        # print(lazy_combined_frame.filter(pl.col("PATIENT_ID")==patient_id_for_checking).sort(["PRACTICE_ID", "PATIENT_ID", "DATE"]).collect())

        return lazy_static, lazy_combined_frame

    
    def _remove_on_global_criteria(self,
                                   lazy_static,
                                   lazy_combined_frame): 

        # Exclude patients who have an `exclude_on_events` event
        if self._exclude_on_events is not None:
            # Get the patients who have any of these events, regardless of when
            patients_without_excluded_events = (
                lazy_combined_frame
                .with_columns(pl.col("EVENT").is_in(self._exclude_on_events).alias("IS_EXC_EVENT"))
                .groupby("PATIENT_ID").agg(pl.col("IS_EXC_EVENT").sum())
                .filter(pl.col("IS_EXC_EVENT")==0)
                .unique("PATIENT_ID")
                .select(pl.col('PATIENT_ID'))
            )
            # and reduce original frames using this list
            lazy_combined_frame = patients_without_excluded_events.join(lazy_combined_frame, on=["PATIENT_ID"], how="inner")  
            lazy_static = patients_without_excluded_events.join(lazy_static, on=["PATIENT_ID"], how="inner")  
    
        # Remove all patients who are not registered at the practice for a period of at least `min_registered_years` 
        lazy_static = (
            lazy_static
            .select([
                (pl.col("END_DATE") - pl.col("START_DATE")).dt.days().alias("DAYS_REGISTERED"), "*"
                ])
            .filter(pl.col('DAYS_REGISTERED') >= self._min_registered_years*365.25)
            .drop("DAYS_REGISTERED")
        )
        patients_within_registration_length = lazy_static.select(pl.col('PATIENT_ID'))
        lazy_combined_frame = patients_within_registration_length.join(lazy_combined_frame, on=["PATIENT_ID"], how="inner")

        return lazy_static, lazy_combined_frame
        
    
    def _set_index_date(self,
                        lazy_static,
                        lazy_combined_frame):
        
        # Retain only patients with the required events occurring (at any time)
        patients_with_index_event = (
            lazy_combined_frame
            .filter(pl.col("EVENT") == self._index_on_event)                                                      # Include only patients who experienced the events                  
        )
    
        # Reduce this list to include only patients with the required events occuring during the study period
        patients_with_index_event = (
            patients_with_index_event
            .filter(pl.col('DATE') >= pl.lit(self._study_period[0]).str.strptime(pl.Date, fmt="%F"))               # after study start date
            .filter(pl.col('DATE') <= pl.lit(self._study_period[1]).str.strptime(pl.Date, fmt="%F"))               # and before study end date
        )

        # and within the specified age range
        patients_with_index_event = (
            patients_with_index_event
            .filter(pl.col('DAYS_SINCE_BIRTH') >= self._age_at_entry_range[0]*365.25)               # index event occurred after minimum age
            .filter(pl.col('DAYS_SINCE_BIRTH') <= self._age_at_entry_range[1]*365.25)               # index event occurred before maximum age
        )

        # Get the first valid index event if multiple exist (e.g. repeat diagnosis), and set the date as the index date
        patients_with_index_event = (
            patients_with_index_event
            .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])                   # Sort to ensure date order within patients
            .unique(subset=["PRACTICE_ID", "PATIENT_ID"], keep="first")    # Keep chronologically first required event experienced by patient
            .with_columns((pl.col('DATE').alias('INDEX_DATE')))      # This is the earliest possible index date. It would be later if this lies outside of study dates
        )
        
        # Get this patient list
        patients_with_index_event = (
            patients_with_index_event
            .unique("PATIENT_ID")
            .select(pl.col('PATIENT_ID', "INDEX_DATE"))
        )
    
        # and reduce original frames using this list, adding the new index date to both frames
        lazy_combined_frame = patients_with_index_event.join(lazy_combined_frame, on=["PATIENT_ID"], how="inner")  
        lazy_static = patients_with_index_event.join(lazy_static.drop("INDEX_DATE"), on=["PATIENT_ID"], how="inner")  

        return lazy_static, lazy_combined_frame

    
    def _reduce_on_index_date(self,
                              lazy_static,
                              lazy_combined_frame,):
    
        #############################################################
        # Remove patients where `exclude_on_events_prior_to_index`
        # occured before the index event
        #############################################################

        # TODO: this is a costly operation
        if self._exclude_on_events_prior_to_index is not None:
            
            # Get the patients who have any of these events occurring before the index date
            # For example, if we are interested in the possibility of prescribing a medicine then we may want to exclude those already on the medicine
            patients_without_excluded_prior_events = (
                lazy_combined_frame
                .filter(pl.col('DATE') < pl.col('INDEX_DATE'))
                .with_columns(pl.col("EVENT").is_in(self._exclude_on_events_prior_to_index).alias("IS_EXC_EVENT"))
                .groupby("PATIENT_ID")
                .agg(pl.col("IS_EXC_EVENT").sum())
                .filter(pl.col("IS_EXC_EVENT")==0)
                .unique("PATIENT_ID")
                .select(pl.col('PATIENT_ID'))
            )
            
            # and reduce original frames using this list
            lazy_combined_frame = patients_without_excluded_prior_events.join(lazy_combined_frame, on=["PATIENT_ID"], how="inner")  
            lazy_static = patients_without_excluded_prior_events.join(lazy_static, on=["PATIENT_ID"], how="inner")  
            
        #############################################################
        # Retain only patients with valid events following the index date 
        # ... as we want followup predictions, we need to ensure an 
        #     event occurs - even if this is not an outcome
        #############################################################

        # Keep only patients with an event occurring between index and study end
        #   (remove patients where index event is last event in study)
        patients_with_target = (
            lazy_combined_frame
            .filter(pl.col("DATE") > pl.col("INDEX_DATE"))
            .filter(pl.col("DATE") <= pl.lit(self._study_period[1]).str.strptime(pl.Date, fmt="%F"))
            .select(pl.col('PATIENT_ID'))                                                                    # get the patient list
            .unique()
        )
        patients_with_target_list = patients_with_target.collect().to_series().to_list()
        
        lazy_combined_frame = lazy_combined_frame.filter(pl.col("PATIENT_ID").is_in(patients_with_target_list))
        lazy_static = lazy_static.filter(pl.col("PATIENT_ID").is_in(patients_with_target_list))
    
            
        #############################################################
        # GET OUTCOMES 
        # or if no outcome, the last seen event within study period
        #############################################################
            
        # Get all possible outcomes occurring after index, and before study end
        # If multiple outcomes follow index, we take all for now, but later reduce this to the first seen.
        lazy_combined_frame_outcomes = (
            lazy_combined_frame
            .filter(pl.col("EVENT").is_in(self._outcomes))
            .filter(pl.col("DATE") > pl.col("INDEX_DATE"))
            .filter(pl.col("DATE") <= pl.lit(self._study_period[1]).str.strptime(pl.Date, fmt="%F"))  # Only look at outcomes within study period
        )
    
        # Get last observation within study period
        lazy_combined_frame_last_event_in_study = (
            lazy_combined_frame
            .filter(pl.col("DATE") <= pl.lit(self._study_period[1]).str.strptime(pl.Date, fmt="%F"))  # Only look at events within study period
            .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])                                        # Sort to ensure date order within patients
            .unique(subset=["PRACTICE_ID", "PATIENT_ID"], keep="last")                          # Keep chronologically last event experienced by patient
        )
    
        # Take first between these - this will the first outcome if one has occurred, otherwise the last event
        outcome = (
            pl.concat([lazy_combined_frame_outcomes, lazy_combined_frame_last_event_in_study])
            .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])                                        # Sort to ensure date order within patients
            .unique(subset=["PRACTICE_ID", "PATIENT_ID"], keep="first")                          # Keep chronologically last event experienced by patient
        )

        # If this event occurs beyond 10 years beyond the index date, then replace with UNK event at index date + 10 years
    
        #############################################################
        # MERGE EVENTS UP TO AND INCLUDING INDEX, AND OUTCOME
        #############################################################
    
        # Get events which occured before index (this is re-used later)
        lazy_combined_frame_before_index = (
            lazy_combined_frame
            .filter(pl.col('DATE') <= pl.col('INDEX_DATE'))
        )
        
        # and merge this with the events that occurred up to and including the index from the last section
                # NOTE: As this is a vertical contatenation between the events, if for example, no outcome for a patient exists, then
                #        the resulting combined frame will only contain one event for that patient which could lead to downstream problems.
                #        This is not a problem currently as we filter out all patients without multiple events later.
        new_combined_frame = (
                pl.concat([lazy_combined_frame_before_index, outcome])
                .sort(["PRACTICE_ID", "PATIENT_ID", "DATE"])                                        # Sort to ensure date order within patients
            )

        #############################################################
        # REMOVE PATIENTS WITH FEWER THAN MIN EVENTS (EXCLUDING OUTCOME)
        #############################################################
        if self._min_events is not None:
            patients_with_min_events = (
                lazy_combined_frame
                .filter(pl.col('DATE') <= pl.col('INDEX_DATE'))
                .with_columns(pl.col("EVENT"))
                .groupby("PATIENT_ID")
                .agg(pl.col("EVENT").count())
                .filter(pl.col("EVENT")>=self._min_events)
                .unique("PATIENT_ID")
                .select(pl.col('PATIENT_ID'))
            )
            
            # and reduce original frames using this list
            new_combined_frame = patients_with_min_events.join(new_combined_frame, on=["PATIENT_ID"], how="inner")  
            lazy_static = patients_with_min_events.join(lazy_static, on=["PATIENT_ID"], how="inner")  
        
    
        return lazy_static, new_combined_frame