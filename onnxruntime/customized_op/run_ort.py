import onnx
import torch
import numpy as np
import onnxruntime as ort
from onnxruntime_extensions import onnx_op, PyOp, PyOrtFunction

@onnx_op(op_type="Inverse")
def inverse(x):
        return np.linalg.inv(x)

x = [[-3.7806,  1.0857, -0.8645], [-0.0398,  0.3996, -0.7268], [ 0.3433,  0.6064,  0.0934]]
x = np.asarray(x, dtype=np.float32)

onnx_model = onnx.load("export.onnx")
onnx_fn = PyOrtFunction.from_model(onnx_model)
y = onnx_fn(x)
print(y)