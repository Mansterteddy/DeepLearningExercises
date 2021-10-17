import onnx
import onnxruntime as ort
from onnx import helper, onnx_pb as onnx_proto
from onnxruntime_extensions import make_onnx_model

nodes = []
nodes.append(helper.make_node("RankLMTokenizer", ["query", "title", "snippet", "url", "market"], ["input_ids", "segment_ids", "input_mask"], domain='ai.onnx.contrib'))

query = helper.make_tensor_value_info("query", onnx_proto.TensorProto.STRING, [None, None])
title = helper.make_tensor_value_info("title", onnx_proto.TensorProto.STRING, [None, None])
snippet = helper.make_tensor_value_info("snippet", onnx_proto.TensorProto.STRING, [None, None])
url = helper.make_tensor_value_info("url", onnx_proto.TensorProto.STRING, [None, None])
market = helper.make_tensor_value_info("market", onnx_proto.TensorProto.STRING, [None, None])

input_ids = helper.make_tensor_value_info("input_ids", onnx_proto.TensorProto.INT64, [None, None])
segment_ids = helper.make_tensor_value_info("segment_ids", onnx_proto.TensorProto.INT64, [None, None])
input_mask = helper.make_tensor_value_info("input_mask", onnx_proto.TensorProto.INT64, [None, None])

graph = helper.make_graph(nodes, "RankLM", [query, title, snippet, url, market], [input_ids, segment_ids, input_mask])
model = make_onnx_model(graph)
onnx.save(model, "ranklm_optim_tokenizer.onnx")

score = helper.make_tensor_value_info("score", onnx_proto.TensorProto.FLOAT, [None, None])
logits = helper.make_tensor_value_info("logits", onnx_proto.TensorProto.FLOAT, [None, None])

prev_model = onnx.load("ranklm.onnx").graph.node
nodes.extend(prev_model)

graph = helper.make_graph(nodes, "RankLM", [query, snippet, url, title, market], [score, logits])
model = make_onnx_model(graph)
onnx.save(model, "ranklm_whole_onnx.onnx")