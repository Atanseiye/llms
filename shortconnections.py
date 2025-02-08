import torch
from torch import nn
from activation import GELU

class ExamDeepNN(nn.Module):

    def __init__(self, layer_size, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_size[0], layer_size[1]), GELU()),
            nn.Sequential(nn.Linear(layer_size[1], layer_size[2]), GELU()),
            nn.Sequential(nn.Linear(layer_size[2], layer_size[3]), GELU()),
            nn.Sequential(nn.Linear(layer_size[3], layer_size[4]), GELU()),
            nn.Sequential(nn.Linear(layer_size[4], layer_size[5]), GELU()),
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)

            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
            
        return x
    

def print_grad(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has a grad mean of {param.grad.abs().mean().item()}")
    

# Testing
layer_size=[3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
without_shortcut = ExamDeepNN(
    layer_size, use_shortcut=False
)

with_shortcut = ExamDeepNN(
    layer_size, use_shortcut=True
)

print('When shortcut connection was not used: \n',)
print_grad(without_shortcut, sample_input)
print('\n')
print('When shortcut connection was used: \n',)
print_grad(with_shortcut, sample_input)
