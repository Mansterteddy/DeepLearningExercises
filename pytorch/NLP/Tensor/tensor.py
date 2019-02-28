import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# torch.tensor(data) creates a torch.Tensor object with the given data
V_data = [1., 2., 3.]
V = torch.tensor(V_data)
print(V)

# Creates a matrix
M_data = [[1., 2., 3.], [4., 5., 6.]]
M = torch.tensor(M_data)
print(M)

# Create a 3D tensor of size 2x2x2
T_data = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
T = torch.tensor(T_data)
print(T)

# Index into V and get a scalar (0 dimensional tensor)
print(V[0])
# Get a Python number from it
print(V[0].item())

# Index into M and get a vector
print(M[0])

# Index into T and get a matrix
print(T[0])

x = torch.randn((3, 4, 5))
print(x)

# Operations with Tensors
x = torch.Tensor([1., 2., 3.])
y = torch.Tensor([4., 5., 6.])
z = x + y
print(z)

# By default, it concatenates along the first axis (concatenates rows)
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)

# Concatenate columns:
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
# second arg specifies which axis to concat along
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)

x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12)) # Reshape to 2 rows. 12 columns
# Same as above. If one of the dimensions is -1, its size can be inferred
print(x.view(2, -1))

# Tensor factory methods have a ``requires_grad`` flag
x = torch.tensor([1., 2., 3.], requires_grad=True)

# With requires_grad=True, you can still do all the operations you previsouly could
y = torch.tensor([4., 5., 6.], requires_grad=True)
z = x + y
print(z)

# But z knows something extra
print(z.grad_fn)

# Let sum up all the entries in z
s = z.sum()
print(s)
print(s.grad_fn)

# calling .backward() on any variable will run backprop, starting from it.
s.backward()
print(x.grad)

x = torch.randn(2, 2)
y = torch.randn(2, 2)
# By default, user created Tensors have ``require_grad=False``
print(x.requires_grad, y.requires_grad)
z = x + y
# So you can't backprop through z
print(z.grad_fn)

# ``.requires_grad_( ... )`` changes an existing Tensor's ``requires_grad``
# flag in-place. The input flag defaults to ``True`` if not given.
x = x.requires_grad_()
y = y.requires_grad_()
# z contains enough information to compute gradients, as we saw above
z = x + y
print(z.grad_fn)
# If any input to an operation has ``requires_grad=True``, so will the output
print(z.requires_grad)


# Now z has the computation history that relates itself to x and y
# Can we just take its values, and **detach** it from its history?
new_z = z.detach()

# ... does new_z have information to backprop to x and y?
# NO!
print(new_z.grad_fn)
# And how could it? ``z.detach()`` returns a tensor that shares the same storage
# as ``z``, but with the computation history forgotten. It doesn't know anything
# about how it was computed.
# In essence, we have broken the Tensor away from its past history

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)