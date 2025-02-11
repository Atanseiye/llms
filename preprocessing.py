import re
import torch
from torch.utils.data import DataLoader, Dataset
import os
import tiktoken

class DatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text..
        tokenizer = TokenizerV2(vocabs(txt))  
        token_ids = tokenizer.encode(txt)

        # Use the slidding window to chunck the data into overlapping sequencs of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length+1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


#            Data Loader instance generator
# ====================================================

def create_dataloader_v1(txt, batch_size=4, 
                         max_length=256, stride=128, shuffle=True, 
                         drop_last=True, num_workers=0):
    
    tokenizer = TokenizerV2(vocabs(txt))
    tokenizer = tokenizer.encode(txt)
    # tokenizer = tiktoken.get_encoding('gpt2')

    dataset = DatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
    


#            Version 1 Tokenizer
# ====================================================
class TokenizerV1:

    def __init__(self, vocabs):
        self.str_to_int = vocabs
        self.int_to_str = {i:s for s, i in vocabs.items()}

    def encode(self, text):
        splits = re.split(r'([.,:;\'"!_]|--|\s)', text)
        processed_tokens = [i.strip() for i in splits if i.strip()]
        ids = [self.str_to_int[s] for s in processed_tokens]
        return ids

    def decode(self, ids):
        text = ' '.join(self.int_to_str[s] for s in ids)
        text = re.sub(r'\s+([,.;:_!"\'()])', r'\1', text)
        return text



#            Version 2 Tokenizer
# =====================================================
class TokenizerV2:
    """
    This tokenizer uses bit-pair tokenization
    """

    def __init__(self, vocabs):
        self.str_to_int = vocabs
        self.int_to_str = {i:s for s,i in vocabs.items()}

    def encode(self, text):
        # text = vocabs(text)
        split = re.split(r'([,.\'"_!()]|--|\s)', text)
        processed_tokens = [i.strip() for i in split if i.strip()]
        processed_tokens = [
            item if item in self.str_to_int
            else '<|unk|>' for item in processed_tokens
        ]
        ids = [self.str_to_int[s] for s in processed_tokens]
        return ids

    def decode(self, ids):
        text = ' '.join([self.int_to_str[s] for s in ids])
        text = re.sub(r'\s+([,.;:_!"\'()])', r'\1', text)
        return text
    


#            Vocabullary Generator
# ====================================================
def vocabs(raw_text):
    tokens = re.split(r'([,.;:_\'!"()]|--|\s)', raw_text)

    processed_tokens = [i.strip() for i in tokens if i.strip()]
    # Creating token IDs
    unique_tokens = sorted(set(processed_tokens))
    unique_tokens.extend(['<|endoftext|>', '<|unk|>'])

    # the vocabulary of the token IDs
    vocabs = {token:integer for integer,token in enumerate(unique_tokens)}
    return vocabs


#            Embedding Generator
# ====================================================
class embedding:

    def __init__(self, vocab_size, output_dim):
        self.vocab_size = vocab_size
        self.output_dim = output_dim


    def token_embedding_layer(vocab_size, output_dim):

        torch.manual_seed(123)
        embed_layer = torch.nn.Embedding(vocab_size, output_dim)
        return embed_layer