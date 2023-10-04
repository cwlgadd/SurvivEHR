import sqlite3

#TODO:
# account for filters so we don't bother tokenizing conditions which arent included

class DiagnosisTokenizer:
    """
    Tokenizer for diagnoses
    """

    PATH_TO_DB = "/rds/projects/s/subramaa-mum-predict/CharlesGadd_Oxford/FoundationModel/preprocessing/processed/cprd.db"

    def __init__(self):
        
        self.conn = sqlite3.connect(self.PATH_TO_DB)
        self.cursor = self.conn.cursor()
        
        # Get vocabulary
        self.cursor.execute("""SELECT DISTINCT
                                    condition
                               FROM 
                                   diagnosis_table
                             """)
        all_conditions = self.cursor.fetchall()
        self.vocab_size = len(all_conditions)
        self.encode_dict = {condition[0]: code + 1 for code, condition in enumerate(all_conditions)}
        self.decode_dict = {code + 1: condition[0] for code, condition in enumerate(all_conditions)}

        
    def encode(self, list_of_conditions : list) -> list:
        return [self.encode_dict[c] for c in list_of_conditions] # encoder: take a string, output a list of integers

    def decode(self, list_of_tokens: list) -> list:
        return [self.decode_dict[t] for t in list_of_tokens] # encoder: take a string, output a list of integers

if __name__ == "__main__":
    
    tokenizer_diag = DiagnosisTokenizer()
    
    print(tokenizer_diag.vocab_size)
    
    test_sequence = ["HF", "STROKEUNSPECIFIED", "BIPOLAR"]
    print(test_sequence)
    
    encoded_sequence = tokenizer_diag.encode(test_sequence)
    print(encoded_sequence)
    
    decoded_sequence = tokenizer_diag.decode(encoded_sequence)
    print(decoded_sequence)
    

