import torch
import matplotlib.pyplot as plt
# import mlp_toolkits.mplot3D as Axes3D

def softmax_naive(x):
    return torch.exp(x)/torch.exp(x).sum(dim=0)

# this are vector representation of the text 'Your hourney starts with one step'
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

# Corresponding Words
words = ['Your', 'journey', 'starts', 'with', 'one', 'step']

query = inputs[1]

note = inputs @ inputs.T

attention_weigth  = torch.softmax(note, dim=0)

context_vector = attention_weigth @ inputs

print(context_vector)