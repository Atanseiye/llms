from attention import Attention
import torch

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

torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0)
context_lenght = batch.shape[1]
d_in = batch.shape[2]
d_out = batch.shape[0]
ca = Attention(d_in, d_out, context_lenght, 0.0)
context_vec = ca.causal(batch)
print(context_vec)