from attention import CausalAttention, MultiHead
from layers import Feedforward
from config import YOR_GPT_CONFIG_124M
import torch
from transformer import TransformerBlock

inputs = torch.tensor(
    [
        [.43, .15, .89],
        [.55, .87, .56],
        [.57, .85, .64],
        [.22, .58, .33],
        [.77, .25, .10],
        [.05, .80, .55]
    ]
)

# causal attention
torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0)
context_lenght = batch.shape[1]
d_in = batch.shape[2]
d_out = batch.shape[0]
# ca = CausalAttention(d_in, d_out, context_lenght, 0.0)
# context_vec = ca(batch)
# print(context_vec)

### MultiHead
mha = MultiHead(d_in, d_out, context_lenght, 2, 0.0)
context_vec = mha(batch)
# print(context_vec)


### Testing the Feedforward
ffn = Feedforward(YOR_GPT_CONFIG_124M)
x = torch.rand(2, 4, 768)
out = ffn(x)
# print(out, '\n', out.shape)


### Test Transformer
block =TransformerBlock(YOR_GPT_CONFIG_124M)
output = block(x)
print('Shape of the transformer block', output.shape, '\n')
print('Output of the transformer block', output)