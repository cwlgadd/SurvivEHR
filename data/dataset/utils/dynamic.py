# 
from typing import Optional, Any, Union
import polars as pl
import numpy as np
import logging


def preprocess_measurements(pl_lazy_frame: pl.LazyFrame,
                            practice_patient_ids: Optional[list[str]] = None,
                            method: str = "normalise",
                            remove_outliers: bool = True
                            ):
    r"""
    Perform standardisation pre-processing

    ARGS:
        pl_lazy_frame: 
            A lazy polars dataframe loaded from the SQL tables in the form:

            ┌──────────────────────┬───────┬──────────────────┬──────────────┬───────────────────────┐
            │ PRACTICE_PATIENT_ID  ┆ VALUE ┆ EVENT            ┆ AGE_AT_EVENT ┆ EVENT_TYPE            │
            │ ---                  ┆ ---   ┆ ---              ┆ ---          ┆ ---                   │
            │ str                  ┆ f64   ┆ str              ┆ i64          ┆ str                   │
            ╞══════════════════════╪═══════╪══════════════════╪══════════════╪═══════════════════════╡
            │ xxxxxxx              ┆ 22.3  ┆ bmi              ┆ 10151        ┆ univariate_regression │

            where only columns PRACTICE_PATIENT_ID, VALUE, and EVENT are required

    KWARGS:
        practice_patient_id (optional list of strings)
            If provided, standardisation is only performed over these patients
        method:
            Approach to use (str, default = "normalise"). ``normalise``: Standard normalisation, transform to zero
            mean unit variance. ``standardise``: standardisation, scale to range between zero and one.
        transform (bool, default = True)
            Whether to perform the standardisation in place. F.e. if standardisation should be pre-processed, or 
            if we just calculate the statistics.
    
    RETURN:
        Dictionary in the form:
        {"measurement1_name": (bias term, scale term),
         "measurement2_name": (...,     , ...       ),
         }
    """

    if practice_patient_ids is not None:
        pl_lazy_frame = pl_lazy_frame.filter(pl.col("PRACTICE_PATIENT_ID").is_in(practice_patient_ids))

    logging.info(f"... ... ...  Using standardisation method: {method.lower()}")
    match method.lower():
        case "normalise":            
            bias = (
                pl_lazy_frame            
                .groupby("EVENT")
                .agg(pl.col("VALUE").mean())            
                )
            scale = (
                pl_lazy_frame            
                .groupby("EVENT")
                .agg(pl.col("VALUE").std())
                )
        case "standardise":
            raise NotImplementedError
        case _:
            raise NotImplementedError

    bias = bias.collect()
    scale = scale.collect()
    
    # Convert bias and scale statistics to a dictionary which can then be used as a lookup in future operations
    #    Complicated chain of operations for one line, but essentially:
    #       1)  transpose the frame so each column is a measurement/test and the row is 
    #           the statistic. Column names are default (column_0, column_1, etc)
    #       2) rename the column names with the measurement names from the first row
    #       3) convert to dictionary which can now be used as a look up later
    bias = (
        bias
        .select(pl.col("VALUE"))
        .transpose()
        .rename(bias.transpose().head(1).to_dicts().pop())
        .to_dicts()
        .pop()
    )

    scale = (
        scale
        .select(pl.col("VALUE"))
        .transpose()
        .rename(scale.transpose().head(1).to_dicts().pop())
        .to_dicts()
        .pop()
    )    
    logging.debug(f"measurement/test scaling: {method.lower()}: \n bias:{bias}, \n scale: {scale}")
        
    standardisation_dict = {str(key): (bias, scale[key]) for key, bias in bias.items()}
    
    # Perform standardisation on the lazy frame values
    ####            
    # Add bias to every measurement in the lazy frame
    remap_bias = lambda code: bias.get(code, 0)
    pl_lazy_frame = (
        pl_lazy_frame
        .with_columns(
            pl.col("EVENT")
            .apply(remap_bias) 
            .alias("bias")
            )
    )

    # Add scale to every measurement in the lazy frame
    remap_scale = lambda code: scale.get(code, 1)
    pl_lazy_frame = (
        pl_lazy_frame
        .with_columns(
            pl.col("EVENT")
            .apply(remap_scale)
            .alias("scale")
            )
    )
    
    pl_lazy_frame = (
        pl_lazy_frame
        .with_columns(
            (pl.col("VALUE") - pl.col("bias")) / pl.col("scale")
            .alias("VALUE")
            )            
        .drop("bias")
        .drop("scale")
    )

    if remove_outliers:
        logging.info("... ... ... Removing measurement and test outliers. Using three deviations from mean as cutoff")        
        pl_lazy_frame = pl_lazy_frame.filter((pl.col("VALUE") < 3) | (pl.col("VALUE") > 3))
    else:
        logging.info("... ... ... Not removing measurement and test outliers (beyond initial pre-processing)")
        
    return pl_lazy_frame, standardisation_dict
