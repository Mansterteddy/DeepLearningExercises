import onnx

onnx_model = onnx.load("meb_batch.onnx")
#print(onnx_model.graph.node)

for item in onnx_model.graph.node:
    if item.name == "Div_85":
        print(item.input)
        item.input[0] = "64"
        print(item.input)
    if item.name == "Div_87":
        print(item.input)
        item.input[0] = "102"
        print(item.input)
    if item.name == "Div_186":
        print(item.input)
        item.input[0] = "181"
        print(item.input)
    if item.name == "Div_188":
        print(item.input)
        item.input[0] = "219"
        print(item.input)

    if item.name in ["Loop_45", "Loop_75", "Loop_146", "Loop_176"]:
        onnx_model.graph.node.remove(item)


onnx.save_model(onnx_model, "meb_batch_removeLoop.onnx")