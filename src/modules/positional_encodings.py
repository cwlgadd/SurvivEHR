# Adapted from  https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import torch
import math
from typing import Optional
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PositionalEncoding(torch.nn.Module):
    r"""
        A module for applying index-based position encodings

    .. math::
        P\left(k, 2i\right) = \sin \left(\frac{k}{n**{2i/d}}\right) 
        P\left(k, 2i + 1\right) = \cos \left(\frac{k}{n**{2i/d}}\right)

    ARGS:
        embedding_dim: (d) The desired size of the output embedding.
        
    KWARGS:
        n_scalar: (n) The scalar used to initialize the frequency space. 
        max_length: The maximum sequence length, for precomputing positional encoding
        dropout (Optional): Amount of dropout to apply to the output of combined emebedding and positional embedding
    """

    def __init__(self, embedding_dim: int,
                 n_scalar: float = 10000.0,
                 max_length: int = 5000, 
                 dropout: Optional[float] = 0.1):
        """
        """
        assert embedding_dim % 2 == 0, "Positional encoding dimension must be even"
        
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = torch.nn.Dropout(p=dropout) if dropout is not None else None

        # pre-compute positional encoding matrix        
        position = torch.arange(max_length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, device=device) * (-math.log(n_scalar) / embedding_dim))
        div_term = torch.nn.Parameter(div_term, requires_grad=False)
        self.pe = torch.zeros(max_length, 1, embedding_dim, device=device)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """
        
        ARGS: 
            x: Token embeddings
                Tensor, shape ``[seq_len, bsz, embedding_dim]``
            
        Returns
            The combined token and index-based positional embeddings 
                Tensor, shape ``[seq_len, bsz]``

        """
        x += self.pe[:x.size(0)]
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x

    
class TemporalPositionalEncoding(torch.nn.Module):
    """A module for applying time-based position encodings


    .. math::
        P\left(k, 2i\right) = \sin \left(\frac{k}{n**{2i/d}}\right) 
        P\left(k, 2i + 1\right) = \cos \left(\frac{k}{n**{2i/d}}\right)

    ARGS:
        embedding_dim: (d) The desired size of the output embedding.
        
    KWARGS:
        n_scalar: (n) The maximum observed timepoint, used to initialize the frequency space. 
        dropout (Optional): Amount of dropout to apply to the output of combined emebedding and positional embedding
    """
    
    def __init__(self, embedding_dim: int,
                 n_scalar: float = 10000.0,
                 dropout: Optional[float] = 0.1):
        """
        """
        assert embedding_dim % 2 == 0, "Temporal positional encoding dimension must be even"
        
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = torch.nn.Dropout(p=dropout) if dropout is not None else None

        # pre-compute positional encoding matrix        
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(n_scalar) / embedding_dim))
        self.div_term = torch.nn.Parameter(div_term, requires_grad=False)
        

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Token embeddings
                Tensor, shape ``[seq_len, bsz, embedding_dim]``
            t: Time points (assumed in days)
                Tensor, shape ``[seq_len, bsz]``

        Returns:
            The combined token and temporal positional embeddings 
                Tensor, shape ``[seq_len, bsz]``
        """
        seq_len, bsz, _ = x.shape
        t = t.unsqueeze(-1)                    # Unsqueeze for broadcasting through the hidden dim.
 
        temporal_embeddings = torch.zeros(seq_len, bsz, self.embedding_dim)

        temporal_embeddings[:, :, 0::2] = torch.sin(t * self.div_term.unsqueeze(0).unsqueeze(0))
        temporal_embeddings[:, :, 1::2] = torch.cos(t * self.div_term.unsqueeze(0).unsqueeze(0))

        x += temporal_embeddings
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x    
    

def test(bsz=1, seq_len=10, embed_dim=6):
    
    pe = PositionalEncoding(embed_dim, dropout=None)
    tpe = TemporalPositionalEncoding(embed_dim, dropout=None)
    
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