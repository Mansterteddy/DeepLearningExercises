import onnx
import onnxruntime as ort
from onnx import helper, onnx_pb as onnx_proto
from onnxruntime_extensions import make_onnx_model

nodes = []
nodes.append(helper.make_node("RankLMTokenizer", ["input"], ["output"], domain='ai.onnx.contrib'))

input = helper.make_tensor_value_info("input", onnx_proto.TensorProto.STRING, [None, None])
output = helper.make_tensor_value_info("output", onnx_proto.TensorProto.INT64, [None, None])
graph = helper.make_graph(nodes, "RankLM", [input], [output])
model = make_onnx_model(graph)
onnx.save(model, "RankLMToken.onnx")