import io
import onnx
import unittest
import platform
import torchvision
import numpy as np
from onnxruntime_extensions import PyOrtFunction, hook_model_op, PyOp
from onnxruntime_extensions.onnxprocess import torch_wrapper as torch
from onnxruntime_extensions.onnxprocess import trace_for_onnx, pyfunc_from_model

input_text = ['test sentence', 'sentence 2']
f = io.BytesIO()

with trace_for_onnx(input_text, names=['in_text']) as tc_sess:
    tc_inputs = tc_sess.get_inputs()[0]
    print(tc_inputs)
    batchsize = tc_inputs.size()[0]
    shape = [batchsize, 2]
    fuse_output = torch.zeros(*shape).size()
    tc_sess.save_as_onnx(f, fuse_output)

m = onnx.load_model_from_string(f.getvalue())
onnx.save_model(m, 'temp_test00.onnx')
fu_m = PyOrtFunction.from_model(m)
result = fu_m(input_text)
print(result)