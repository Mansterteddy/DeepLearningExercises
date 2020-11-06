import torch 
from matplotlib import pyplot as plt

x = torch.linspace(0, 5, 100, requires_grad=True)
y = (x**3 - 6 * x**2 + 8 * x)

print(x)
print(y)
print(y.sum())

dydx = torch.autograd.grad(y.sum(), [x])[0]

plt.plot(x.detach(), y.detach(), label='y')
plt.plot(x.detach(), dydx, label='dy/dx')
plt.legend()
plt.show()