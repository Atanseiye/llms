from config import YOR_GPT_CONFIG_124M
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emd = nn.Embedding(cfg['vocab_size'], cfg['emd_size'])
        self.pos_emd = nn.Embedding(cfg['context_length'], cfg['emd_size'])
        self.drop_emd = nn.Dropout(cfg['drop_rate'])
        
         