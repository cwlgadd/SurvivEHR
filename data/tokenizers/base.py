import sqlite3
import polars as pl
import logging

class TokenizerBase():
    r"""
    Base class for custom tokenizers
    """
    
    @property
    def vocab_size(self):
        assert self._event_counts is not None, "Must first fit Vocabulary"        
        # return self._event_counts.select(pl.count()).to_numpy()[0][0]
        return self._vocab_size
    
    @property
    def fit_description(self):
        assert self._event_counts is not None
        return str(self._event_counts)

    @staticmethod
    def event_frequency(meta_information, 
                        include_measurements=True, 
                        include_diagnoses=True,
                       ) -> pl.DataFrame:
        r"""
        Get polars dataframe with three columns: event, count and relative frequencies
        
        Returns 
        ┌──────────────────────────┬─────────┬───────────┐
        │ EVENT                    ┆ COUNT   ┆ FREQUENCY │
        │ ---                      ┆ ---     ┆ ---       │
        │ str                      ┆ u32     ┆ f64       │
        ╞══════════════════════════╪═════════╪═══════════╡
        │ <event name 1>           ┆ n1      ┆ p1        │
        │ <event name 2>           ┆ n2      ┆ p2        │
        │ …                        ┆ …       ┆ …         │
        └──────────────────────────┴─────────┴───────────┘
        """
        _num_table_categories = meta_information["measurement_tables"].shape[0] + meta_information["diagnosis_table"].shape[0]
        logging.debug(f"number of table categories: {_num_table_categories}")
        _non_empty_table_categories = sum(meta_information["measurement_tables"]["count"] > 0) + sum(meta_information["diagnosis_table"]["count"] > 0)
        logging.debug(f"none empty table categories: {_non_empty_table_categories}")
        
        # Stack all the tokens that will be used. This requires that the tokens used have been pre-processed in the meta_information
        schema={"EVENT": str, "COUNT": pl.UInt32}
        counts = pl.DataFrame(schema=schema)
        if include_measurements:
            assert "measurement_tables" in meta_information.keys(), meta_information.keys()
            counts_measurements = pl.DataFrame(meta_information["measurement_tables"][["event", "count"]], schema=schema)
            counts = counts.vstack(counts_measurements)
            logging.debug(f"counts of measurements {counts_measurements}")
            logging.debug(f"sum of counts of measurements {sum(counts_measurements['COUNT'])}")
        if include_diagnoses:
            assert "diagnosis_table" in meta_information.keys(), meta_information.keys()
            counts_diagnosis = pl.DataFrame(meta_information["diagnosis_table"][["event", "count"]], schema=schema)
            counts = counts.vstack(counts_diagnosis)
            logging.debug(f"counts of diagnoses {counts_diagnosis}")
            logging.debug(f"sum of counts of diagnoses {sum(counts_diagnosis['COUNT'])}")

        # the total number of event tokens (WITHOUT MISSING DATA)
        total_number_of_tokens = sum(counts["COUNT"])
        logging.info(f"Tokenzier created based on {total_number_of_tokens:,} tokens")

        # Get the frequency of each as a new column, which will be used for mapping low frequency tokens to the UNK token
        counts = (counts
                  .lazy()
                  .with_columns(
                      (pl.col("COUNT") / total_number_of_tokens).alias("FREQUENCY")
                  )
                  .sort("COUNT")
                  .collect()
                 )
        
        return counts

    def __init__(self):
        self._event_counts = None
        
    def fit(self,
            event_counts:pl.DataFrame,
            **kwargs
           ):
        r"""
        """
        raise NotImplementedError
        
    def _map_to_unk(self,
                    event_counts:pl.DataFrame,
                    freq_threshold:float = 0.00001,
                    ):
        r"""
        Remove low frequency tokens, replacing with unk token. 
        
        ARGS:
            event_counts: (polars.DataFrame)
            
        KWARGS:
            freq_threshold (float): 
            
        RETURNS:
            polars.DataFrame
            ┌──────────────────────────┬─────────┬───────────┐
            │ EVENT                    ┆ COUNT   ┆ FREQUENCY │
            │ ---                      ┆ ---     ┆ ---       │
            │ str                      ┆ u32     ┆ f64       │
            ╞══════════════════════════╪═════════╪═══════════╡
            │ "UNK"                    ┆ n1      ┆ p1        │
            │ <event name 1>           ┆ n2      ┆ p2        │
            │ …                        ┆ …       ┆ …         │
            └──────────────────────────┴─────────┴───────────┘
            
        """
        # The low-occurrence tokens which will be treated as UNK token
        unk = event_counts.filter(pl.col("FREQUENCY") <= freq_threshold)
        unk_counts = pl.DataFrame({"EVENT": "UNK", 
                                    "COUNT": unk.select(pl.sum("COUNT")).to_numpy()[0][0], 
                                    "FREQUENCY": unk.select(pl.sum("FREQUENCY")).to_numpy()[0][0]},
                                   schema={"EVENT":"str", "COUNT": pl.UInt32, "FREQUENCY": pl.Float64}
                                  )
        event_counts = unk_counts.vstack(
            event_counts.filter(pl.col("FREQUENCY") > freq_threshold)
        )
        return event_counts
    
    def encode(self, sequence:list[str]):
        r"""
        Take a <> of strings, output a list of integers
        """
        return [self._stoi[c] if c in self._stoi.keys() else self._stoi["UNK"] for c in sequence] 
    
    def decode(self, sequence:list[str]):
        return ' '.join([self._itos[i] for i in sequence])
    
