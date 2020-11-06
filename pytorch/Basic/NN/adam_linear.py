import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

net = torch.nn.Linear(3, 2)
optimizer = Adam(net.parameters(), lr=0.01)
# Move the network, target value, and training inputs to the GPU
net.cuda()
target = torch.tensor([[1.0, 1.0]], device='cuda') 
log = []
for _ in range(1000):
    y_batch = net(torch.randn(100, 3, device='cuda'))
    loss = ((y_batch - target) ** 2).sum(1).mean()
    log.append(loss.item())
    net.zero_grad()
    loss.backward()
    optimizer.step()

print(f'weight is {net.weight}\n')
print(f'bias is {net.bias}\n')

plt.ylabel('loss')
plt.xlabel('iteration')
plt.plot(log)
plt.show()