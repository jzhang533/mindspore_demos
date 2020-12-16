import mindspore as ms
import numpy as np
from mindspore import Tensor
from mindspore import context, Model
from mindspore import ops

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')

x = Tensor(np.random.randn(4, 10), ms.float32)
y = Tensor(np.random.randn(10, 10), ms.float32)
matmul = ops.MatMul()

for i in range(10):
    if np.random.rand() > 0.5:
        x = matmul(x, y)

print(y)
