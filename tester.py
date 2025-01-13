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

simplified_attention = Attention.simplified(inputs)
print(simplified_attention)
