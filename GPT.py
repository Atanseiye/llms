import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import YOR_GPT_CONFIG_124M
from v2_preprocessing import TokenizerV2, vocabs, create_dataloader_v1
from transformer import TransformerBlock
from layers import LayerNorm
from tokenizers import Tokenizer
from tokenizers.models import BPE
import os

# ================================
# GPT Model Definition
# ================================
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emd = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emd = nn.Embedding(config['context_lenght'], config['emb_dim'])
        self.drop_emd = nn.Dropout(config['drop_rate'])

        # Transformer Blocks
        self.trf_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])]
        )

        # Layer Normalization & Output
        self.final_norm = LayerNorm(config['emb_dim'])
        self.out_head = nn.Linear(config['emb_dim'], config['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        print(f"Max token ID in input: {in_idx.max().item()}, Vocab Size: {self.tok_emd.num_embeddings}")
        tok_embeds = self.tok_emd(in_idx)
        pos_embeds = self.pos_emd(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emd(x)
        x = self.trf_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
