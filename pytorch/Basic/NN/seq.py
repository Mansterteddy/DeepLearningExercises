import torch
from collections import OrderedDict
from torch.nn import Linear, ReLU, Sequential

mlp = torch.nn.Sequential(OrderedDict([
    ('layer1', Sequential(Linear(2, 20), ReLU())),
    ('layer2', Sequential(Linear(20, 20), ReLU())),
    ('layer3', Sequential(Linear(20, 2)))
]))

print(mlp)

for n, c in mlp.named_modules():
    print(f'{n or "The whole network"} is a {type(c).__name__}')

for name, param in mlp.named_parameters():
    print(f'{name} has shape {tuple(param.shape)}')