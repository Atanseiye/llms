import torch


class Attention:

    def __init__(self, query, word_vectors):
        self.query = query
        self.word_vectors = word_vectors

    def simplified(word_vectors):
        attention_score = word_vectors @ word_vectors.T
        attention_weight = torch.softmax(attention_score, dim=0)
        context_vector = attention_weight @ word_vectors
        return context_vector

    def self():
        pass

    def causal():
        pass

    def multihead():
        pass

