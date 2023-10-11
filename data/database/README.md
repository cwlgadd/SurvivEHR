# CPRD
Datasets for the Clinical Practice Research Datalink (CPRD). Designed for Optimal and Mum-Predict research packages

## Description

SQL backend. This sub-module builds an SQL database using SQLite from the .csv files produced by DEXTER. These .csv files were cleaned in R with manually set thresholds for outlier detection.

Long term, this could be replaced by pure SQL code. After this, more sophisticated outlier detection / pre-processing can be done in Python downstream after a DL friendly representation is built.

## Function

TODO: fill
Three SQL tables: 1) static information; 2) diagnosis table, containing multi-class features; 3) measurement table, containing continous features.

### Static table

### Diagnosis table

### Measurements table

