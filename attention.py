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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_lenght, num_head, dropout, qvk_bais=True):
        super().__init__()
        assert (d_out % num_head == 0), \
        "d_out must be divisible by the number of heads"

        self.d_out = d_out
        self.num_head = num_head
        self.head_dim = d_out // num_head

        self.W_query = nn.Linear(d_in, d_out, bias=qvk_bais)
        self.W_key = nn.Linear(d_in, d_out, bias=qvk_bais)
        self.W_value = nn.Linear(d_in, d_out, bias=qvk_bais)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', 
            torch.triu(torch.ones(context_lenght, context_lenght), 
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Transpose
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute Scaled dot-product attention score
        att_score = queries @ keys.transpose(2, 3) # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Using the mask to fill attention score
        att_score.masked_fill_(mask_bool, -torch.inf)

        # Computing the attention weight
        attn_weight = torch.softmax(att_score / keys.shape[0]**0.5, dim=-1)
        attn_weight = self.dropout(attn_weight)

        # Shape (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weight * values).transpose(1, 2)

        # Combine head where self.d_out = self.num_heads * self.heads_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj()[context_vec]

        return context_vec
    


class MultiHeadAttention2(nn.Module):
    def __init__(self, d_in, d_out, context_lenght, num_head, dropout, qvk_bais=True):
        super().__init__()
        assert d_out % num_head == 0, "d_out must be divisible by num_head"

        self.d_out = d_out
        self.num_head = num_head
        self.head_dim = d_out // num_head

        self.W_query = nn.Linear(d_in, d_out, bias=qvk_bais)
        self.W_key = nn.Linear(d_in, d_out, bias=qvk_bais)
        self.W_value = nn.Linear(d_in, d_out, bias=qvk_bais)
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        # Masking for causal attention
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_lenght, context_lenght), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x).view(b, num_tokens, self.num_head, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(b, num_tokens, self.num_head, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_head, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        attn_scores = (queries @ keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply causal mask
        mask = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill_(mask, float('-inf'))

        # Compute attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context_vec = attn_weights @ values  # Matrix multiplication, not element-wise

        # Reshape and project output
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # Correct function call

        return context_vec
