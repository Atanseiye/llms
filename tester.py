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
ca = Attention(3, 2, context_lenght, 0.0)
context_vec = ca.causal(batch)
print(context_vec)