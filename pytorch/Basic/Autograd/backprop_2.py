import torch
from matplotlib import pyplot as plt

x = torch.linspace(0, 5, 100, requires_grad=True)

y = x**3 - 6 * x**2 + 8 * x
y.sum().backward(create_graph=True)

dy = x.grad.clone()
x.grad.zero_()
dy.sum().backward()

plt.plot(x.detach(), y.detach(), label='y')
#plt.plot(x.detach(), x.grad.detach(), label='dy/dx')
plt.plot(x.detach(), x.grad.detach(), label='d2y/dx')
plt.legend()
plt.show()