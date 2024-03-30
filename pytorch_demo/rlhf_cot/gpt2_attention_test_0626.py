# coding:utf-8

import torch
import math

bias = torch.tril(torch.ones(12,12)).view(1,12,12)
q = torch.rand([2,12,8])

att = (q@q.transpose(-2,-1))*(1.0/math.sqrt(8))
att_mask = att.masked_fill(bias[:,:12,:12]==0,float('-inf'))
print(att_mask)
print(att)
print(bias)