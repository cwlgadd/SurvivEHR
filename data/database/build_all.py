from build_static_db import build_static_table
from build_diagnosis_db import build_diagnosis_table
from build_measurements_and_tests_db import build_measurements_table
import sqlite3

def build_tables(connector, path_to_static, path_to_diagnoses, path_to_measurements):
    """
    """
    
    build_static_table(conn, path_to_static)
    build_diagnosis_table(conn, path_to_diagnoses)
    build_measurements_table(conn, path_to_measurements)


if __name__ == "__main__":
    
    PATH_TO_DB = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"
    conn = sqlite3.connect(PATH_TO_DB)
    # conn = sqlite3.connect(':memory:')              # For debugging
    
    # Static covariates 
    PATH_TO_STATIC = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/static.csv"
    # Diagnoses
    PATH_TO_DIAGNOSIS = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/diagnosis_history.csv"
    # Measurements
    PATH_TO_MEASUREMENTS = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/measurements.csv"


    build_tables(conn, PATH_TO_STATIC, PATH_TO_DIAGNOSIS, PATH_TO_MEASUREMENTS)
    
    #close the connection
    conn.close()