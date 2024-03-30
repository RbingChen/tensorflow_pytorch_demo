# coding:utf-8

import torch
import matplotlib.pyplot as plt

x = torch.arange(-8.0,8.0,0.1,requires_grad=True)
y = torch.sigmoid(x)
# backward 标量时，不需要torch.ones_like(x)，如 y.sum().backward()
y.backward(torch.ones_like(x),retain_graph=True)
#https://zhuanlan.zhihu.com/p/83172023
#y.backward(retain_graph=True)

plt.figure(1)

plt.plot(x.detach(),x.grad,"x", "grad of sigmoid ")
plt.show()