import numpy as np
from onnxruntime_extensions import PyOrtFunction

encode = PyOrtFunction.from_model("gpt2_tok.onnx")

input_text = ["Hello, World!", "Bye, World!"]
res_1, res_2 = encode(input_text)
print(res_1)
print(res_2)