import os
from pathlib import Path
import sys
node_type = os.getenv('BB_CPU')
venv_dir = f'/rds/homes/g/gaddcz/Projects/CPRD/virtual-env-{node_type}'
venv_site_pkgs = Path(venv_dir) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
if venv_site_pkgs.exists():
    sys.path.insert(0, str(venv_site_pkgs))
    print(f"Added path '{venv_site_pkgs}' at start of search paths.")
else:
    print(f"Path '{venv_site_pkgs}' not found. Check that it exists and/or that it exists for node-type '{node_type}'.")


# !echo $SQLITE_TMPDIR
# !echo $TMPDIR
# !echo $USERPROFILE

import pytorch_lightning
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import sqlite3
import logging
from CPRD.data.database.build_static_db import Static
from CPRD.data.database.build_diagnosis_db import Diagnoses
from CPRD.data.database.build_measurements_and_tests_db import Measurements

if __name__ == "__main__":

    torch.manual_seed(1337)
    logging.basicConfig(level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}.")

    PATH_TO_DB = "/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/FoundationalModel/cprd.db"
    PATH_TO_STATIC = "/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/baseline/masterDataOptimalWithIMD_v3.csv"
    PATH_TO_DIAGNOSIS = "/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/baseline/masterDataOptimalWithIMD_v3.csv"
    PATH_TO_MEASUREMENTS = "/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/timeseries/measurement_and_tests/lab_measurements/"
    PATH_TO_MEDICATIONS = "/rds/projects/g/gokhalkm-optimal/OPTIMAL_MASTER_DATASET/data/timeseries/medications/"
    
    load = True
    if load:
        logging.warning(f"Load is true, if you want to re-build database set to False")
    
    static = Static(PATH_TO_DB, PATH_TO_STATIC, load=load)
    diagnosis = Diagnoses(PATH_TO_DB, PATH_TO_DIAGNOSIS, load=load)
    measurements = Measurements(PATH_TO_DB, PATH_TO_MEASUREMENTS, load=load)
    medications = Measurements(PATH_TO_DB, PATH_TO_MEDICATIONS, load=load)

    for table in [static, diagnosis, measurements, medications]:
        print(table)
    
    