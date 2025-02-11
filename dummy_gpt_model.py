from config import YOR_GPT_CONFIG_124M
from attention import MultiHeadAttention
from layers import LayerNorm, Feedforward
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.tok_emd = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emd = nn.Embedding(config['context_lenght'], config['emb_dim'])
        self.drop_emd = nn.Dropout(config['drop_rate'])

        # Use a placeholder fot transformer block
        self.trf_block = nn.Sequential(
            * [DummyTransformerBlock(config) for _ in range(config['n_layers'])]
        )

        # Use placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(config['emb_dim'])
        self.out_head = nn.Linear(
            config['emb_dim'], config['vocab_size'], bias=False
        )

    def forward(self, in_idx):
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

    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
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
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # Add original input back 1

        # Shortcut connection for the Feedforward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # Add original input back 2

        return x
    

class DummyLayerNorm(nn.Module):

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
    


# ## Sample
import tiktoken
tokenizer = tiktoken.get_encoding('gpt2')
batch = []
txt1 = 'Every steps moves you'
txt2 = 'Every day holds a'
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(YOR_GPT_CONFIG_124M)
logits = model(batch)
print(f"Output shape: {logits.shape} \n")
print(logits)