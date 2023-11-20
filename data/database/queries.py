def query_static(conditions:list, cursor):
    r""" SQL query which will update the subset of patients of interest. Repeated calls reduces this subset further

    ARGS:
        conditions (list of strings): 
            inputs for an sql query string which returns a list of the PRACTICE_PATIENT_IDs which were diagnosed with a condition
            
    """
    raise NotImplementedError
    cursor.execute("""SELECT practice_patient_id FROM static_table
                            WHERE
                                XXXX
                            IN (""" + ",".join(len(conditions) * ["?"]) + ")",
                        conditions
                       )
    return [ppid[0] for ppid in cursor.fetchall()]

def query_measurement(measurements:list, cursor):
    r""" SQL query which will return the subset of patients of interest. 

    ARGS:
        measurements (list of strings): 
            inputs for an sql query string which returns a list of the PRACTICE_PATIENT_IDs which have measurements

    TODO: add table with all compiled measurements
    """

    cursor.execute("""SELECT practice_patient_id FROM measurement_table
                            WHERE
                                event
                            IN (""" + ",".join(len(measurements) * ["?"]) + ")",
                        measurements
                       )
    return [ppid[0] for ppid in cursor.fetchall()]

def query_diagnosis(conditions:list, cursor):
    r""" SQL query which will return the subset of patients of interest.

    ARGS:
        conditions (list of strings): 
            inputs for an sql query string which returns a list of the PRACTICE_PATIENT_IDs which were diagnosed with a condition
            
    TODO: add table with all diagnoses
    """
    cursor.execute("""SELECT practice_patient_id FROM diagnosis_table
                            WHERE
                                event
                            IN (""" + ",".join(len(conditions) * ["?"]) + ")",
                        conditions
                       )
    return [ppid[0] for ppid in cursor.fetchall()]

# # Check what measurements are available
# cursor.execute("SELECT DISTINCT * FROM measurement_table")
# measurements = cursor.fetchall()
# print(measurements)

# Check what diagnoses are available
# cursor.execute("SELECT DISTINCT * FROM diagnosis_table")
# diagnoses = cursor.fetchall()
# print(diagnoses)
