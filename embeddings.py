import torch
import numpy
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





