import os
import torch

net = torch.nn.Linear(3, 2)
print(net)

print(net(torch.tensor([1.0, 0.0, 0.0])))

x_batch = torch.tensor([
    [1.0, 0., 0.],
    [0., 1.0, 0.],
    [0., 0., 1.0],
    [0., 0., 0.],
])

print(net(x_batch))

print("weight is: ", net.weight)
print("bias is: ", net.bias)

for name, param in net.named_parameters():
    print(f'{name} = {param}\n')

for k, v in net.state_dict().items():
    print(f'{k}: {v.type()}{tuple(v.shape)}')

torch.save(net.state_dict(), "linear.pth")

net.load_state_dict(torch.load("linear.pth"))