# Adapted from  https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import torch
import math
from typing import Optional
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import logging

class PositionalEncoding(torch.nn.Module):
    r"""
        A module for applying index-based position encodings

    .. math::
        P\left(k, 2i\right) = \sin \left(\frac{k}{n**{2i/d}}\right) 
        P\left(k, 2i + 1\right) = \cos \left(\frac{k}{n**{2i/d}}\right)

    ARGS:
        encoding_dim: (d) The desired size of the output embedding.
        
    KWARGS:
        n_scalar: (n) The scalar used to initialize the frequency space. Defaults to 10,000 following "Attention Is All You Need".
        max_length: The maximum sequence length, for precomputing positional encoding.
    """

    def __init__(self, encoding_dim: int,
                 n_scalar: float = 10000.0,
                 max_length: int = 5000, 
                ):
        """
        """
        assert encoding_dim % 2 == 0, "Positional encoding dimension must be even"
        
        super().__init__()
        self.encoding_dim = encoding_dim

        # pre-compute positional encoding matrix        
        position = torch.arange(max_length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, encoding_dim, 2, device=device) * (-math.log(n_scalar) / encoding_dim))
        div_term = torch.nn.Parameter(div_term, requires_grad=False)
        self.pe = torch.zeros(1, max_length, encoding_dim, device=device)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)
        logging.debug("Initialised PositionalEncoder")

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        
        ARGS: 
            positions: positions
                Tensor, shape ``[bsz, seq_len]`` 
            
        Returns
                Tensor, shape ``[1, seq_len, encoding_dim]``

        """
        seq_len = positions.size(1)
        return self.pe[:, :seq_len, :]

    
class TemporalPositionalEncoding(torch.nn.Module):
    """A module for applying time-based position encodings


    .. math::
        P\left(k, 2i\right) = \sin \left(\frac{k}{n**{2i/d}}\right) 
        P\left(k, 2i + 1\right) = \cos \left(\frac{k}{n**{2i/d}}\right)

    ARGS:
        embedding_dim: (d) The desired size of the output embedding.
        
    KWARGS:
        n_scalar: (n) The maximum observed timepoint, used to initialize the frequency space. Defaults to 10,000 following "Attention Is All You Need".
    """
    
    def __init__(self, 
                 encoding_dim: int,
                 n_scalar: float = 10000.0,
                ):
        """
        """
        assert encoding_dim % 2 == 0, "Temporal positional encoding dimension must be even"
        
        super().__init__()
        self.encoding_dim = encoding_dim

        # pre-compute positional encoding matrix        
        div_term = torch.exp(torch.arange(0, encoding_dim, 2) * (-math.log(n_scalar) / encoding_dim))
        self.div_term = torch.nn.Parameter(div_term, requires_grad=False)
        logging.debug("Initialised TemporalPositionalEncoding")

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            positions: Time points
                Tensor, shape ``[bsz, seq_len]``

        Returns:
                Tensor, shape ``[bsz, seq_len, encoding_dim]``
        """
        bsz, seq_len = positions.shape
        positions = positions.unsqueeze(-1)                    # Unsqueeze for broadcasting through the encoding dim
 
        temporal_encodings = torch.zeros(bsz, seq_len, self.encoding_dim, device=positions.device)
        temporal_encodings[:, :, 0::2] = torch.sin(positions * self.div_term.unsqueeze(0).unsqueeze(0))    # [bsz, seq_len, 1] * [1, 1, encoding_dim / 2]
        temporal_encodings[:, :, 1::2] = torch.cos(positions * self.div_term.unsqueeze(0).unsqueeze(0))
        
        logging.debug(f"TPE: {positions.shape} maps to -> {temporal_encodings.shape} ")
        return temporal_encodings
    

def test(bsz=1, seq_len=10, embed_dim=6):
    
    pe = PositionalEncoding(embed_dim)
    tpe = TemporalPositionalEncoding(embed_dim)
    
    we = torch.zeros(seq_len, bsz, embed_dim)        # Zeros so i can just check the positional encoding part
    
    order_context_we = pe(we)
    print(f"\nOrder context word_embeddings: \n {order_context_we.squeeze()}")
    
    days, _ = torch.randint(0, 100*365, (seq_len, bsz)).sort(dim=0)   # Randomly sample position integers between 0 days and ~100 years old (in days)
    
    time_order_context_we = tpe(we, days)
    print(f"\nOrder context word_embeddings: \n {days} \n{time_order_context_we.squeeze()}")
    

if __name__ == "__main__":
    
    test()
    
    # Testing modules
#     import sqlite3
#     from CPRD.data.database import queries
#     from CPRD.data.foundational_loader import FoundationalDataModule

#     # Report what is in the db
#     ###########################
#     PATH_TO_DB = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"
#     conn = sqlite3.connect(PATH_TO_DB)
#     cursor = conn.cursor()
    
#     # Get the list of patients which fit our criterion
#     identifiers1 = queries.query_measurement(["bmi", "hydroxyvitamin2", "hydroxyvitamin3"], cursor)
#     identifiers2 = queries.query_diagnosis(["HF", "FIBROMYALGIA"], cursor)
#     identifiers = list(set(identifiers1).intersection(identifiers2))    # Turn smaller list into the set

#     # Built dataset
#     foundational_dm = FoundationalDataModule(identifiers=identifiers, max_seq_length=256, batch_size=64)
    
#     # Make positional encoding
#     positional_encoding = PositionalEncoding(100)
#     temporal_positional_encoding = TemporalPositionalEncoding(100)
    
#     print(positional_encoding)
#     print(temporal_positional_encoding)
    
    
#     for idx, batch in enumerate(foundational_dm.train_dataloader()):
#         break
#     print(f"Batch {idx}")
#     print(batch["input_positions"])
#     print(batch["input_positions"].shape)
#     print(batch["attention_mask"])
#     model_inputs = batch["input_ids"].numpy()
#     print(model_inputs)
#     print(type(model_inputs))

#     decoded_sequences = [foundational_dm.decode([model_inputs[j, i] for j in range(model_inputs.shape[0])]) 
#                          for i in range(model_inputs.shape[1])]
#     print(decoded_sequences[0])