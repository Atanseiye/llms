from config import YOR_GPT_CONFIG_124M
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emd = nn.Embedding(cfg['vocab_size'], cfg['emd_size'])
        self.pos_emd = nn.Embedding(cfg['context_length'], cfg['emd_size'])
        self.drop_emd = nn.Dropout(cfg['drop_rate'])

        # Use a placeholder fot transformer block
        self.trf_block = nn.Sequential(
            * [DummyTransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        # Use placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg['emd_dim'], cfg['vocab_size'], bias=False
        )

        def forward(seld, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emd(in_idx)
            pos_embeds = self.pos_emd(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds+pos_embeds
            x = self.drop_emd(x)
            x = self.trf_block(x)
            x = self.final_norm(x)
            logits = self.out_head(x)

            return logits
        

class DummyTransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__ ()
        # A simple PlaceHolder

    def forward(self, x):
        # this block does nothing and just returns it's inputs
        return x
    

class DummyLayerNorm(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        # this is a simple placeholder

    def forward(self, x):
        # this function does nothing, it just returns it's imputs
        return x
    