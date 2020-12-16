import numpy as np
import torch

x = torch.randn(4, 10)
y = torch.randn(10, 10)

@torch.jit.script
def loop_fn(x, y):
    for i in range(10):
        #if np.random.rand() > 0.5:
        if torch.rand(1) > 0.5:

            x = torch.matmul(x, y)
    return x

print(loop_fn(x,y))
print(loop_fn.code)
