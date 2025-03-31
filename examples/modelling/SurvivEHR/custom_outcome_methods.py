import polars as pl

def custom_mm_outcomes(dm):
    """
    Extracts a list of outcome event codes from the datamodule's tokenizer.

    This function filters the `_event_counts` DataFrame in the tokenizer to select
    events that meet both of the following criteria:
    - Have a count greater than 0
    - Match the regex pattern `^[A-Z0-9_]+$` (i.e., consist of uppercase letters, digits, or underscores)

    Args:
        dm: A datamodule object with a `tokenizer._event_counts` attribute (assumed to be a Polars DataFrame).

    Returns:
        List[str]: A list of event code strings satisfying the above conditions.
    """
    conditions = (
        dm.tokenizer._event_counts.filter((pl.col("COUNT") > 0) &
            (pl.col("EVENT").str.contains(r'^[A-Z0-9_]+$')))
          .select("EVENT")
          .to_series()
          .to_list()
    )
    return conditions