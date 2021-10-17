import onnx
import onnxruntime as ort
from onnx import helper, onnx_pb as onnx_proto
from onnxruntime_extensions import make_onnx_model

nodes = []
nodes.append(helper.make_node("QueryNormalize", ["query"], ["NormalizedQuery"], domain='ai.onnx.contrib'))
nodes.append(helper.make_node("TitleNormalize", ["title"], ["NormalizedTitle"], domain='ai.onnx.contrib'))
nodes.append(helper.make_node("SnippetNormalize", ["snippet"], ["NormalizedSnippet"], domain='ai.onnx.contrib'))
nodes.append(helper.make_node("UrlNormalize", ["url"], ["NormalizedUrl"], domain='ai.onnx.contrib'))
nodes.append(helper.make_node("MarketNormalize", ["market"], ["NormalizedMarket"], domain='ai.onnx.contrib'))

nodes.append(helper.make_node("QueryTokenize", ["NormalizedQuery"], ["TokenizedQuery"], domain='ai.onnx.contrib'))
nodes.append(helper.make_node("TitleTokenize", ["NormalizedTitle"], ["TokenizedTitle"], domain='ai.onnx.contrib'))
nodes.append(helper.make_node("SnippetTokenize", ["NormalizedSnippet"], ["TokenizedSnippet"], domain='ai.onnx.contrib'))
nodes.append(helper.make_node("UrlTokenize", ["NormalizedUrl"], ["TokenizedUrl"], domain='ai.onnx.contrib'))
nodes.append(helper.make_node("MarketTokenize", ["NormalizedMarket"], ["TokenizedMarket"], domain='ai.onnx.contrib'))

nodes.append(helper.make_node("IdConcat", ["TokenizedQuery", "TokenizedTitle", "TokenizedSnippet", "TokenizedUrl", "TokenizedMarket"], 
                ["input_ids", "segment_ids", "input_mask"], domain='ai.onnx.contrib'))

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
onnx.save(model, "ranklm_raw_tokenizer.onnx")