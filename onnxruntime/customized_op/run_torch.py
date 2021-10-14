import torch

class CustomInverse(torch.nn.Module):
    def forward(self, x):
        return torch.inverse(x) + x 

x = torch.Tensor([[-3.7806,  1.0857, -0.8645],
        [-0.0398,  0.3996, -0.7268],
        [ 0.3433,  0.6064,  0.0934]])
model = CustomInverse()
y = model(x)
print(y)
print(x.dtype)
#torch.onnx.export(model, (x), "export_torch.onnx")