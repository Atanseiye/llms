import torch
from torch import nn


class Attention(nn.Module):

    def __init__(self, d_in, d_out, context_lenght, dropout, qvk_bais=True):
        super().__init__()
        self.d_in = d_in
        self.W_query = nn.Linear(d_in, d_out, bias=qvk_bais)
        self.W_key = nn.Linear(d_in, d_out, bias=qvk_bais)
        self.W_value = nn.Linear(d_in, d_out, bias=qvk_bais)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_lenght, context_lenght), diagonal=1))


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

