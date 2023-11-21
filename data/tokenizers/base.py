import sqlite3
import polars as plr


class TokenizerBase():
    r"""
    Base class for custom tokenizers
    """
    
    @property
    def vocab_size(self):
        assert self._event_counts is not None, "Must first fit Vocabulary"        
        # return self._event_counts.select(plr.count()).to_numpy()[0][0]
        return self._vocab_size
    
    @property
    def fit_description(self):
        assert self._event_counts is not None
        return str(self._event_counts)

    @staticmethod
    def event_frequency(event_stream) -> plr.DataFrame:
        r"""
        Get polars dataframe with three columns: event, count and relative frequencies
        
        Returns 
        ┌──────────────────────────┬─────────┬──────────┐
        │ EVENT                    ┆ counts  ┆ freq     │
        │ ---                      ┆ ---     ┆ ---      │
        │ str                      ┆ u32     ┆ f64      │
        ╞══════════════════════════╪═════════╪══════════╡
        │ <event name 1>           ┆ n1      ┆ p1       │
        │ <event name 2>           ┆ n2      ┆ p2       │
        │ …                        ┆ …       ┆ …        │
        └──────────────────────────┴─────────┴──────────┘
        """
        event_freq = (event_stream
                      .select(plr.col("EVENT").explode())
                      .to_series(index=0)
                      .value_counts(sort=True)
                     )                        
        event_freq = event_freq.with_columns((plr.col('counts') / event_freq.select(plr.sum("counts"))).alias('freq'))
        return event_freq
    
    def __init__(self):
        self._event_counts = None
        
    def fit(self,
            event_counts:plr.DataFrame,
            **kwargs
           ):
        r"""
        """
        raise NotImplementedError
        
    def _map_to_unk(self,
                event_counts:plr.DataFrame,
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
            ┌──────────────────────────┬─────────┬──────────┐
            │ EVENT                    ┆ counts  ┆ freq     │
            │ ---                      ┆ ---     ┆ ---      │
            │ str                      ┆ u32     ┆ f64      │
            ╞══════════════════════════╪═════════╪══════════╡
            │ "UNK"                    ┆ n1      ┆ p1       │
            │ <event name 1>           ┆ n2      ┆ p2       │
            │ …                        ┆ …       ┆ …        │
            └──────────────────────────┴─────────┴──────────┘
            
        """
        # The low-occurrence tokens which will be treated as UNK token
        unk = event_counts.filter(plr.col("freq") <= freq_threshold)
        unk_counts = plr.DataFrame({"EVENT": "UNK", 
                                    "counts": unk.select(plr.sum("counts")).to_numpy()[0][0], 
                                    "freq": unk.select(plr.sum("freq")).to_numpy()[0][0]},
                                   schema={"EVENT":"str", "counts": plr.UInt32, "freq": plr.Float64}
                                  )
        event_counts = unk_counts.vstack(
            event_counts.filter(plr.col("freq") > freq_threshold)
        )
        return event_counts
    
    def encode(self, sequence:list[str]):
        r"""
        Take a <> of strings, output a list of integers
        """
        return [self._stoi[c] if c in self._stoi.keys() else self._stoi["UNK"] for c in sequence] 
    
    def decode(self, sequence:list[str]):
        return ' '.join([self._itos[i] for i in sequence])
    
