import mindspore as ms
from mindspore import context, Model
from mindspore import ops

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

grad_all = ops.composite.GradOperation()

def func(x): 
    return x * x * x

def df_func(x):
    return grad_all(func)(x)

def df2_func(x):
    return grad_all(df_func)(x)

def df3_func(x):
    return grad_all(df2_func)(x)


if __name__ == "__main__":
     
    t = ms.Tensor([5], ms.float32)
    print("tensor: ", t)
    print("first order grad: ", df_func(t))
    print("second order grad: ", df2_func(t))
    print("third order grad: ", df3_func(t))
