import torch
from matplotlib import pyplot as plt

x = torch.linspace(0, 5, 100, requires_grad=True)
x2 = torch.linspace(0, 5, 100, requires_grad=True)

y = x2**3 - 6 * x**2 + 8 * x
y.sum().backward()

plt.plot(x.detach(), y.detach(), label='y')
plt.plot(x.detach(), x.grad, label='dy/dx')
plt.plot(x2.detach(), x2.grad, label='dy/dx2')
plt.legend()
plt.show()