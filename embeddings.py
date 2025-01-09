import torch
import numpy
from data_prep import DatasetV1, Dataset, DataLoader, create_dataloader_v1
from tokenizers import vocabs

'''
this is a library that is trained on a yoruba dataset of about 80 million tokens 
that helps converts yoruba word to it's corresponding vectors which however helps to 
capture similarities in the text.
'''


with open('the-verdict.txt', 'r') as file:
    raw_text = file.read()


class embedding:

    def __init__(self, vocab_size, output_dim):
        self.vocab_size = vocab_size
        self.output_dim = output_dim


    def token_embedding_layer(vocab_size, output_dim):

        torch.manual_seed(123)
        embed_layer = torch.nn.Embedding(vocab_size, output_dim)
        return embed_layer
    
vocab = vocabs(raw_text=raw_text)
# testing
vocab_size = len(vocab)
output_dim = 756
embedding_layer = embedding.token_embedding_layer(vocab_size, output_dim)


if __name__ == '__main__':
    datas = create_dataloader_v1(
        raw_text, batch_size=12, max_length=512, stride=384,
        shuffle=False, drop_last=False, num_workers=0
        )
    reatl_data = iter(datas)
    inputs, outputs = next(reatl_data)
    token_embeddings = embedding_layer(inputs)
    print(token_embeddings.shape)