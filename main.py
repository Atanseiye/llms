from preprocessing import TokenizerV2, vocabs
from preprocessing import DatasetV1, Dataset, DataLoader, create_dataloader_v1
from preprocessing import embedding
from config import YOR_GPT_CONFIG_124M
import torch
import re

with open('the-verdict.txt', 'r') as file:
    raw_text = file.read()


# ============= testing ==============
# parameter values
vocab = vocabs(raw_text=raw_text)
max_lenght = 512
vocab_size = len(vocab)
output_dim = 756

embedding_layer = embedding.token_embedding_layer(YOR_GPT_CONFIG_124M['vocab_size'], YOR_GPT_CONFIG_124M['output_dim'])
pos_embedding_layer = embedding.token_embedding_layer(max_lenght, YOR_GPT_CONFIG_124M['output_dim'])


if __name__ == '__main__':
    datas = create_dataloader_v1(
        raw_text, batch_size=12, max_length=max_lenght, stride=384,
        shuffle=False, drop_last=False, num_workers=0
        )
    
    reatl_data = iter(datas)
    inputs, outputs = next(reatl_data)
    token_embeddings = embedding_layer(inputs)
    pos_embedding = pos_embedding_layer(torch.arange(max_lenght))
    input_embedding = token_embeddings + pos_embedding

    print(input_embedding)

    # make a commit tomorrow, first thing first
    # changed tester.py to main.py