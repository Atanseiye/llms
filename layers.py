from config import YOR_GPT_CONFIG_124M
from activation import GELU
from torch import nn

# Feedforward Neural Network
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
    



