import torch

x_init = torch.randn(2)
x = x_init.clone()
x.requires_grad = True

optimizer = torch.optim.Adam([x], lr=0.1)

bowl = torch.tensor([[0.4410, -1.0317], [-0.2844, -0.1035]])
track, losses = [], []

for iter in range(21):
    objective = torch.mm(bowl, x[:, None]).norm()
    optimizer.zero_grad()
    objective.backward()
    optimizer.step()
    track.append(x.detach().clone())
    losses.append(objective.detach())

print(track)
print(losses)