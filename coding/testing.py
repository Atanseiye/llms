import torch
import matplotlib.pyplot as plt
# import mlp_toolkits.mplot3D as Axes3D

def softmax_naive(x):
    return torch.exp(x)/torch.exp(x).sum(dim=0)

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

query = inputs[1]

att_score_2 = torch.empty(6, 6)
for i, i_x in enumerate(inputs):
    for j, j_x in enumerate(inputs):
        att_score_2[i, j] = torch.dot(i_x, j_x)

attention_score = torch.softmax(att_score_2, dim=0)
    

# context_vec = torch.zeros(query.shape)
# for i, i_x in enumerate(inputs):
#     for j, j_x in enumerate(inputs):
#         context_vec += attention_score[2] * i_x

note = inputs @ inputs.T

attention_score_2  = torch.softmax(note, dim=-1)

print(attention_score)
print('\n \n')
print(attention_score_2)