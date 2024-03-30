# coding:utf-8

import torch

#a = torch.arange(3).reshape(3,1)
# b = torch.arange(2).reshape(1,2)
# print(a,b)
# print(a+b)

# a = torch.arange(3).reshape(3,1)
# A = a.numpy()
# B = torch.tensor(A)
# print(type(A),type(B))
#
# C = torch.tensor([3.5])
# print(C,C.item())

x = torch.arange(4.0)
x.requires_grad_(True)
print(x.grad)

y = 2 *torch.dot(x,x)
y.backward()
print(x.grad)
