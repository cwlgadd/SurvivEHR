from build_static_db import build_static_table
from build_diagnosis_db import build_diagnosis_table
import sqlite3

def build_tables(connector, path_to_static, path_to_diagnoses):
    """
    """
    
    try:
        build_static_table(conn, path_to_static)
    except:
        print("Static table already built")
    
    try:
        build_diagnosis_table(conn, path_to_diagnoses)
    except:
        print("Diagnosis table already built")
    
    

if __name__ == "__main__":
    
    PATH_TO_DB = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"
    conn = sqlite3.connect(PATH_TO_DB)
    # conn = sqlite3.connect(':memory:')              # For debugging
        
    
    # Static covariates 
    PATH_TO_STATIC = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/static.csv"
    # Diagnoses
    PATH_TO_DIAGNOSIS = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/diagnosis_history.csv"

    build_tables(conn, PATH_TO_STATIC, PATH_TO_DIAGNOSIS)
    
    #close the connection
    conn.close()