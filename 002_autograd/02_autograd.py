import mindspore as ms
from mindspore import context, Model
from mindspore import ops

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

grad_all = ops.composite.GradOperation(get_all=True)

def func(x, y): 
    return x*x*x + y*(x*x)

def df_func(x, y):
    return grad_all(func)(x, y)

def df2_func(x, y):
    return grad_all(df_func)(x, y)

def df3_func(x, y):
    return grad_all(df2_func)(x, y)


if __name__ == "__main__":
     
    t1 = ms.Tensor([5], ms.float32)
    t2 = ms.Tensor([6], ms.float32)
    print("tensors: ", t1, t2)
    print("first order grad: ", df_func(t1, t2))
    print("second order grad: ", df2_func(t1, t2))
    print("third order grad: ", df3_func(t1, t2))
