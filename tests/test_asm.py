from annc.builder import mlir
from annc.helper import kp
from annc.types import ElemType
from annc.ops import constant, matmul, add, relu
from annc.types import TensorType
from annc.types import TilingAttr

import numpy as np

@kp.jit
def fused_matmul(inputs, **kwargs):
    lhs = inputs[0]
    rhs = inputs[1]
    cm = inputs[2]
    bias = inputs[3]
    matmul(ElemType.FP32(), [1024,1024], lhs, rhs, cm, False)

module: mlir.Module = fused_matmul(
    name="demo",
    outputs=['out'],
    inputs=[
        np.random.randn(1024,1024).astype(np.float32),
        np.random.randn(1024,1024).astype(np.float32),
        np.zeros((1024,1024)).astype(np.float32),
        np.random.randn(1024).astype(np.float32)
    ]
)

module.operation.print(large_elements_limit=16)

from annc.passmanager import PassManager

module.context.__enter__()
pm = PassManager.parse("builtin.module()")

pm.add("atir-tiling")
# pm.add("atir-select-lowering-strategy")
# pm.add("convert-atir-to-linalg")
# pm.add("annc-one-shot-bufferize")
# pm.add("cache-parallel")

pm.run(module.operation)
module.operation.print(large_elements_limit=16)
module.context.__exit__(None, None, None)
