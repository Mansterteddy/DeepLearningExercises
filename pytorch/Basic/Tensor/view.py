import torch

x = torch.randn((2, 3, 4, 5))
print(x.shape)
print(x)

y = x.permute(0, 1, 3, 2)
print(y.shape)
print(y)
print(x)

z = x.view(2, -1)
print(z.shape)
print(z)
print(x)