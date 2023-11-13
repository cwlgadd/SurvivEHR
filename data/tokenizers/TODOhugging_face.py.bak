import sqlite3
import polars as plr
# from CPRD.data.utils.tokenizers import TokenizerBase


class DiscreteTokenizer():
    r"""
    """
    
    @property
    def vocab_size(self):
        assert self._event_counts is not None, "Must first fit Vocabulary"
        return self._event_counts.select(plr.count()).to_numpy()[0][0]
    
    def __init__(self):
        self._event_counts = None
    
    def fit_vocabulary(self,
            event_counts:plr.DataFrame,
            **kwargs
           ):
        """
        """
        self._event_counts = self._filter(event_counts, **kwargs)
        print(self._event_counts)
        
    def _filter(self,
                event_counts:plr.DataFrame,
                freq_threshold:float = 0.00001,
               ):
        """
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
    
    def str_to_token(self, string):
        pass
    
    def token_to_str(self, string):
        pass
    
    
    
    
    
    
if __name__ == "__main__":
    
    tokenizer_diag = EventTokenizer()
    
    print(tokenizer_diag.vocab_size)
    
    test_sequence = ["HF", "STROKEUNSPECIFIED", "BIPOLAR"]
    print(test_sequence)
    
    encoded_sequence = tokenizer_diag.encode(test_sequence)
    print(encoded_sequence)
    
    decoded_sequence = tokenizer_diag.decode(encoded_sequence)
    print(decoded_sequence)
    

