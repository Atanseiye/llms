import torch
from torch import nn


class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_lenght, dropout, qvk_bais=True):
        super().__init__()
        self.d_out = d_out
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


    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        value = self.W_value(x)
        queries = self.W_query(x)

        atten_score = queries @ keys.transpose(1, 2)
        atten_score.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        atten_weight = torch.softmax(
            atten_score / keys.shape[-1]**0.5, dim=1
        )
        atten_weight = self.dropout(atten_weight)
        context_vec = atten_weight @ value
        return context_vec


class MultiHead(nn.Module):
    def __init__(self, d_in, d_out, context_lenght, num_head, dropout, qvk_bais=True):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_lenght, dropout, qvk_bais)
                for _ in range(num_head)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
