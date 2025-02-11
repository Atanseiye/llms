from util import GELU
from torch import nn
import torch

#           Feedforward Neural Network
# ====================================================
class Feedforward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(config['emb_dim'], 4 * config['emb_dim']),
            GELU(),
            nn.Linear(4 * config['emb_dim'], config['emb_dim'])
        )

    def forward(self, x):
        return self.layer(x)
    

#            Layer Normarlization Module
# ====================================================
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift