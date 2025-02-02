import re
import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import TokenizerV2, vocabs
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


# ==================

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
    


# ==================
# print(torch.__version__)
# if __name__ == "__main__":
#     dataloaders = create_dataloader_v1(
#         raw_text, batch_size=1, max_length=10, stride=2, shuffle=False, num_workers=os.cpu_count()
#     )

#     data_iter = iter(dataloaders)
#     first_batch = next(data_iter)
#     print(first_batch)