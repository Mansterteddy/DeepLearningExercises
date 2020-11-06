import torch
from matplotlib import pyplot as plt

x = torch.linspace(0, 5, 100, requires_grad=True)
print(x.grad)
y = (x**2).cos()
y.sum().backward()
print(x.grad)

plt.plot(x.detach(), y.detach(), label='y')
plt.plot(x.detach(), x.grad, label='dy/dx')
plt.legend()
plt.show()