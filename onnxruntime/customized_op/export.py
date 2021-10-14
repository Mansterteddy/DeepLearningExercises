import torch

class CustomInverse(torch.nn.Module):
    def forward(self, x):
        return torch.inverse(x) + x 

from torch.onnx import register_custom_op_symbolic

def my_inverse(g, input):
    return g.op("ai.onnx.contrib::Inverse", input)

register_custom_op_symbolic('::inverse', my_inverse, 1)

x0 = torch.Tensor([[-3.7806,  1.0857, -0.8645],
        [-0.0398,  0.3996, -0.7268],
        [ 0.3433,  0.6064,  0.0934]])
t_model = CustomInverse()
print(t_model(x0))

torch.onnx.export(t_model, 
                    (x0, ),
                    "export.onnx", 
                    opset_version=12, 
                    verbose=True,
                    input_names= ["input"],
                    output_names= ["output"]
                    )