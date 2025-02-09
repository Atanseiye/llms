import torch
from torch import nn
from layerNorm import LayerNorm
from layers import Feedforward
from attention import MultiHeadAttention2

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention2(
            d_in=config['emb_dim'],
            d_out=config['emb_dim'],
            context_lenght=config['context_lenght'],
            num_head=config['n_heads'],
            dropout=config['drop_rate'],
            qvk_bais=config['qkv_bias']
        )
        self.ff = Feedforward(config)
        self.norm1 = LayerNorm(config['emb_dim'])
        self.norm2 = LayerNorm(config['emb_dim'])
        self.drop_shortcut = nn.Dropout(config['drop_rate'])

    def forward(self, x):
        # Shortcut connection for attention block
        print('at input', x.shape)
        shortcut = x
        x = self.norm1(x)
        print('After Normalization', x.shape)
        x = self.att(x)
        print('After attention', x.shape)
        x = self.drop_shortcut(x)
        print('After dropout shortcut', x.shape)
        x = x + shortcut # Add original input back 1

        # Shortcut connection for the Feedforward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # Add original input back 2

        return x