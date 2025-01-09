import torch

'''
this is a library that is trained on a yoruba dataset of about 80 million tokens 
that helps converts yoruba word to it's corresponding vectors which however helps to 
capture similarities in the text.
'''

class embedding:

    def __init__(self, vocab_size, output_dim):
        self.vocab_size = vocab_size
        self.output_dim = output_dim


    def train_nn(vocab_size, output_dim):

        torch.manual_seed(123)
        embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        return embedding_layer
    

# testing
vocab_size = 6
output_dim = 3
print(embedding.train_nn(vocab_size, output_dim))