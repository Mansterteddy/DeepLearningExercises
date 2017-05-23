"""
Optim x^2 + xy + y^2 +4
"""

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

# Define my own loss function
class Optim_Function(nn.Module):

    def __init__(self):
        super(Optim_Function, self).__init__()

    def forward(self, x):
        self.loss = torch.add(torch.add(torch.add(torch.pow(x[0], 2), torch.mul(x[0], x[1])), torch.pow(x[1], 2)), Variable(torch.Tensor([4])))
        return self.loss

    def backward(self):
        self.loss.backward()

xy = Variable(torch.Tensor([300, 400]), requires_grad=True)

loss = torch.add(torch.add(torch.add(torch.pow(xy[0], 2), torch.mul(xy[0], xy[1])), torch.pow(xy[1], 2)), Variable(torch.Tensor([4])))

print "loss: ", loss
loss.backward()
print(xy.grad)

criterion = Optim_Function()
print "res: ", criterion(xy)

xy = nn.Parameter(xy.data)
optimizer = optim.SGD([xy], lr=0.1)
#optimizer = optim.LBFGS([xy])

run = [0]
while run[0] <= 100:

    def closure():
        optimizer.zero_grad()
        loss = torch.add(torch.add(torch.add(torch.pow(xy[0], 2), torch.mul(xy[0], xy[1])), torch.pow(xy[1], 2)),
                         Variable(torch.Tensor([4])))
        loss.backward()
        print "loss: ", loss.data[0]
        run[0] += 1
        return loss

    optimizer.step(closure)
