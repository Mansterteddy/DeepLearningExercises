import torch

net = torch.nn.Linear(3, 2)

x_batch = torch.tensor([
    [1.0, 0., 0.],
    [0., 1.0, 0.],
    [0., 0., 1.0],
    [0., 0., 0.],
])

y_batch = net(x_batch)

loss = ((y_batch - torch.tensor([[1.0, 1.0]])) ** 2).sum(1).mean()
print(f"loss is {loss}")

loss.backward()
print(f'weight is {net.weight} and grad is:\n{net.weight.grad}\n')
print(f'bias is {net.bias} and grad is:\n{net.bias.grad}\n')

log = []
for _ in range(10000):
    y_batch = net(x_batch)
    loss = ((y_batch - torch.tensor([[1.0, 1.0]])) ** 2).sum(1).mean()
    log.append(loss.item())
    net.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in net.parameters():
            p[...] -= 0.01 * p.grad
print(f'weight is {net.weight}\n')
print(f'bias is {net.bias}\n')

import matplotlib.pyplot as plt

plt.ylabel('loss')
plt.xlabel('iteration')
plt.plot(log)
plt.show()