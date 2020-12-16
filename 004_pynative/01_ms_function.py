import mindspore as ms
from mindspore import dtype as mstype
from mindspore.common.api import ms_function
import numpy as np
from mindspore import Tensor
from mindspore import context, Model
from mindspore import ops

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
#context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')

x = Tensor(np.random.randn(4, 10), ms.float32)
y = Tensor(np.random.randn(10, 10), ms.float32)
minval = Tensor(0, mstype.int32)
maxval = Tensor(100, mstype.int32)
threshold = Tensor(50, mstype.int32)

matmul = ops.MatMul()

@ms_function
def loop_fn(x, y):
    for i in range(10):
        x = matmul(x, y)
    return x

print(loop_fn(x, y))

# failed to make the following codes work
@ms_function
def loop_and_cond_fn(x, y):
    for i in range(10):
        t = ops.uniform((1,), minval, maxval, dtype=ms.int32)
        if t > threshold:
            x = matmul(x, y)
    return x

print(loop_and_cond_fn(x, y))
