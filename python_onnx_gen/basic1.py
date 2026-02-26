import onnxscript as onnxs
from onnxscript import opset17 as onnxs_op
import numpy as np

import onnx
from onnx import numpy_helper as onnx_np_helper

@onnxs.script(onnxs.opset17)
def BasicOps(X: onnxs.FLOAT[1, 10]) -> onnxs.FLOAT[1, 5]:

    W = onnxs_op.Constant(value=onnx_np_helper.from_array(np.random.randn(10, 5).astype(np.float32)))
    B = onnxs_op.Constant(value=onnx_np_helper.from_array(np.random.randn(5).astype(np.float32)))

    matmul_res = onnxs_op.MatMul(X, W)
    add_res = onnxs_op.Add(matmul_res, B)
    Y = onnxs_op.Relu(add_res)
    return Y

model_proto = BasicOps.to_model_proto()
onnx.save(model_proto, "basic1.onnx")
