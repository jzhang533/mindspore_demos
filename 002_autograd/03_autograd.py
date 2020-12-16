import mindspore as ms
from mindspore import context, Model
from mindspore import ops

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

x = ms.Tensor([5], ms.float32)
# forward
def ff(x):
    return (x + 3) * (x + 4) * 0.5
# backward
grad = ops.composite.GradOperation()(ff)(x)
print(grad)

# the following codes doesnt work, 
# the forward computation has to be 
# wrapted into a function, so mindspore 
# can do a source code transformation. 
if True:
    x = ms.Tensor([5], ms.float32)
    # forward
    y = (x + 3) * (x + 4) * 0.5
    # backward
    grad = ops.composite.GradOperation()(y)(x)
    print(grad)

