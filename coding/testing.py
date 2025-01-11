import torch
import matplotlib.pyplot as plt
import mlp_toolkits.mplot3D as Axes3D

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

# Extract x, y, z coord
x_coord = inputs[:, 0].numpy()
y_coord = inputs[:, 1].numpy()
z_coord = inputs[:, 2].numpy()