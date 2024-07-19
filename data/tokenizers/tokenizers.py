import sqlite3
import polars as plr
from CPRD.data.tokenizers.base import TokenizerBase
import logging

class NonTabular(TokenizerBase):
    r"""
    Tokenizer based on :footcite:t:`gruver2023large` (`link`_).

    The simplest, and most naive tokenizer which ignores tabular structure of the data. 
    Events and consequent measurements are treated as subsequent tokens. For example,
    the event and measurement pair ["bmi", 23.3] is tokenised as a sequence
    ["bmi", "2", "3", ".", "3"].

    .. _link: https://arxiv.org/abs/2310.07820

    .. footbibliography::
    """
    is_tabular = False

    def __init__(self):
        super().__init__()
    
    def fit(self,
            meta_information,
            freq_threshold:float = 0,
            include_measurements: bool = True,
            include_diagnoses: bool = True,
            **kwargs
           ):
        """
        Given a polars dataframe with a token, it's count, and frequency on each row, define:
        1) the token to integer map and 2) the integer to token map.
        """
        # From the meta information extract the tokens, their total counts, and their frequency
        event_counts = self.event_frequency(meta_information, include_measurements=include_measurements, include_diagnoses=include_diagnoses)
        logging.debug(f"_event_counts: {event_counts}")

        # Given some threshold, map low frequency tokens to unk token
        self._event_counts = self._map_to_unk(event_counts, freq_threshold=freq_threshold)
        logging.debug(f"_event_counts: {self._event_counts}")

        # Tokens for each event, excluding numeric related tokens
        event_tokens = self._event_counts.select('EVENT').to_series().to_list()
        logging.debug(f"num_event_tokens: {len(event_tokens)}")
        
        # Combine with special tokens (padding; unknown=low frequency, masked, or unobsered in training set; and numeric digits)
        all_tokens = ["PAD", "UNK"] + [str(i) for i in range(10)] + ["."] + event_tokens[1:]
        self._vocab_size = len(all_tokens)
                
        # Create a mapping from strings to integers, and vice versa
        self._stoi = { ch:i for i,ch in enumerate(all_tokens) }
        self._itos = { i:ch for i,ch in enumerate(all_tokens) }

class Tabular(TokenizerBase):
    r"""
    Tokenizer which will not tokenize values.
    """
    
    is_tabular = True
    
    def __init__(self):
        super().__init__()
        
    def fit(self,
            meta_information,
            freq_threshold:float = 0,
            include_measurements: bool = True,
            include_diagnoses: bool = True,
            **kwargs
           ):
        """
        Given a polars dataframe with a token, it's count, and frequency on each row, define:
        1) the token to integer map and 2) the integer to token map.
        """
        # 
        event_counts = self.event_frequency(meta_information, include_measurements=include_measurements, include_diagnoses=include_diagnoses)
        logging.debug(f"event counts: {event_counts}")

        # Given some threshold, map low frequency tokens to unk token
        self._event_counts = self._map_to_unk(event_counts, freq_threshold=freq_threshold)
        logging.debug(f"thresholded event counts: {self._event_counts}")

        # Tokens for each event, excluding numeric related tokens
        event_tokens = self._event_counts.select('EVENT').to_series().to_list()
        logging.debug(f"number of event tokens: {len(event_tokens)}")
        
        # Combine with special tokens (padding; unknown=low frequency, masked, or unobsered in training set)
        all_tokens = ["PAD", "UNK"] + event_tokens[1:]
        self._vocab_size = len(all_tokens)
        logging.debug(f"vocab size {self._vocab_size}")
                
        # Create a mapping from strings to integers, and vice versa
        self._stoi = { ch:i for i,ch in enumerate(all_tokens) }
        self._itos = { i:ch for i,ch in enumerate(all_tokens) }


