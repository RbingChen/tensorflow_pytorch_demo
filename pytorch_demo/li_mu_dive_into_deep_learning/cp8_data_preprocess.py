# coding:utf-8

import torch
from torch import nn
from d2l import torch as d2l
import re
from  Vocab import  Vocab
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_machine(): #@save
    """将时间机器数据集加载到⽂本⾏的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
lines = read_time_machine()
print(f'# ⽂本总⾏数: {len(lines)}')
print(lines[0])
print(lines[10])


def tokenize(lines, token="word"):
    if token=="word":
        return [line.split() for line in lines]
    elif token=="char":
        return [list(line) for line in lines]
    else:
        print("error: unkown token "+ token)

tokens = tokenize(lines)
#for line in tokens:
#    print(line)

vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])