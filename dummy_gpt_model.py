from config import YOR_GPT_CONFIG_124M
from preprocessing import TokenizerV2, vocabs
from preprocessing import embedding
from transformer import DummyTransformerBlock, TransformerBlock
from layers import LayerNorm
from postprocessing import generate_text_sample
import torch
import torch.nn as nn

class GPTModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.tok_emd = embedding.token_embedding_layer(config['vocab_size'], config['emb_dim'])
        self.pos_emd = embedding.token_embedding_layer(config['context_lenght'], config['emb_dim'])
        self.drop_emd = nn.Dropout(config['drop_rate'])

        # Use a placeholder fot transformer block
        self.trf_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['n_layers'])]
        )

        # Use placeholder for LayerNorm
        self.final_norm = LayerNorm(config['emb_dim'])
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
        


    

# ## Sample
import tiktoken
# tokenizer = tiktoken.get_encoding('gpt2')
txt1 = 'Every steps moves you'
txt2 = 'Every day holds a'
txt3 = 'It is time to'
txt = txt1 + txt2 + txt3

with open('the-verdict.txt', 'r') as file:
    text = file.read()

tokenizer = TokenizerV2(vocabs(text))
batch = []
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch.append(torch.tensor(tokenizer.encode(txt3)))
batch = torch.stack(batch, dim=0)


torch.manual_seed(123)
model = GPTModel(YOR_GPT_CONFIG_124M)
logits = model(batch)

start_context = 'Hello, I am'
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
model.eval()
out = generate_text_sample(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=YOR_GPT_CONFIG_124M['context_lenght']
)

print(f"Output is given as {out}")


# -------------------------------------------------------------------------
tot_params = sum(p.numel() for p in model.parameters())
print(f'The total number of parameters on thr model is {tot_params:,}')

print(f'Token Embedding Shape: {model.tok_emd.weight.shape}')
print(f'Output Layer Shape: {model.out_head.weight.shape}')