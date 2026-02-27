import onnxscript as onnxs
from onnxscript import opset17 as onnxs_op
import numpy as np

import onnx
from onnx import numpy_helper as onnx_np_helper

@onnxs.script(onnxs.opset17)
def MyModel(X: onnxs.FLOAT[1, 49], Y:onnxs.FLOAT[49, 5], Z:onnxs.FLOAT[1, 5]) -> onnxs.FLOAT[1, 5]:

    matmul1 = onnxs_op.MatMul(X, Y)
    relu1 = onnxs_op.Relu(matmul1)
    add1 = onnxs_op.Add(relu1, Z)

    Y = onnxs_op.Relu(add1)
    return Y

model_proto = MyModel.to_model_proto()
onnx.save(model_proto, "basic2.onnx")
