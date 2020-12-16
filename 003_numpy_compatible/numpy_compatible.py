import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore import dtype as mstype

ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='CPU')
#ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target='GPU')

t0 = ms.Tensor(2, ms.float32)
t1 = ms.Tensor([2], ms.float32)
t2 = ms.Tensor(np.random.randn(3, 4))
t3 = ms.Tensor(np.random.randn(2, 3, 4, 5, 6, 7, 8, 9))
print("tensors")
print("scalar tensor: ", t0)
print("1-D tensor: ", t1)
print("2-D tensor: ", t2)
print("8-D tensor: ", t3.shape)

np_type = mstype.dtype_to_nptype(mstype.int32)
ms_type = mstype.pytype_to_dtype(int)
py_type = mstype.dtype_to_pytype(mstype.float64)

print("tensor types")
print(np_type)
print(ms_type)
print(py_type)

x = ms.Tensor(np.array([[1, 2], [3, 4]]), mstype.int32)
y = ms.Tensor(1.0, mstype.int32)
z = ms.Tensor(2, mstype.int32)
m = ms.Tensor(True, mstype.bool_)
n = ms.Tensor((1, 2, 3), mstype.int16)
p = ms.Tensor([4.0, 5.0, 6.0], mstype.float64)

print(x, "\n", y, "\n", z, "\n", m, "\n", n, "\n", p)


print("numpy compat basic indexing, requires pynative mode")
t = ms.Tensor(np.random.randn(4, 6), mstype.float32)

print("Origin Tensor:", t)
print("First row:", t[0])
print("First row:", t[0, :])
print("First column:", t[:, 0])
print("Last column:", t[:, -1])
print("All element:", t[:])
print("First row and second column:", t[0, 1])

try: 
    print("broadcasting case1")
    t1 = ms.Tensor(np.random.randn(2, 3, 1, 5), mstype.float32)
    t2 = ms.Tensor(np.random.randn(3, 4), mstype.float32)
    print(t1 + t2)
except :
    print("failed")

try: 
    print("broadcasting case2")
    t1 = ms.Tensor(np.random.randn(2, 3), mstype.float32)
    t2 = ms.Tensor(np.random.randn(3), mstype.float32)
    print(t1 + t2)
except :
    print("failed")
