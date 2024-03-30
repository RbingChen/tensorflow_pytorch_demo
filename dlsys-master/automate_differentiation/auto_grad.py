# coding:utf-8

import collections
from collections import deque
from typing import List, Optional, NamedTuple, Tuple, Union
import numpy as np
from numpy._typing import NDArray


def compute_dependencies(root):
    # deps: {op: num}
    deps = {}
    q = deque()
    traversed = {root}
    q.append(root)
    while len(q) != 0:
        cur = q.pop()
        if len(cur.next_ops) == 0:
            continue
        for next in cur.next_ops:
            deps[next] = deps.get(next, 0) + 1
            if next not in traversed:
                q.append(next)
                traversed.add(next)
    return deps


class Tensor(object):
    """tensor"""

    def __init__(self, ndarray: NDArray, requires_grad=False, grad_fn=None):
        super(Tensor, self).__init__()
        self.ndarray = ndarray  # ①
        self.requires_grad = requires_grad  # ②
        self.grad_fn = grad_fn  # ③
        self.grad = None  # ④
        self._grad_accmulator = None

    def is_leaf(self) -> bool:
        return self.requires_grad and self.grad_fn is None

    def backward(self, output_grad):
        if self.grad_fn is None:
            raise "backward could not be called if grad_fn is None"
        execute_graph(self.grad_fn, output_grad)

    def __str__(self):
        grad_info = f' grad_fn={self.grad_fn}' if self.grad_fn is not None else ''
        return f'tensor({self.ndarray}{grad_info})'

    def __repr__(self):
        return self.__str__()


# 注意 Operator 里计算的都是 Tensor 内部的数据，即 NDArray
class Operator(object):
    def __init__(self):
        super(Operator, self).__init__()
        self.next_ops = []  # ①

    def forward(self, *args: Tuple[NDArray]) -> NDArray:
        raise NotImplementedError("Should be override by subclass")

    def backward(self, output_grad: Tuple[NDArray]) -> Union[NDArray, Tuple[NDArray]]:
        raise NotImplementedError("Should be override by subclass")

    def __call__(self, *args: Tuple[Tensor]) -> Tensor:
        grad_fn = None
        requires_grad = any((t.requires_grad for t in args))  # ①

        if requires_grad:
            # add edges
            for input in args:
                if input.is_leaf():  # ②
                    if input._grad_accmulator is None:
                        input._grad_accmulator = OpAccumulate(input)
                    self.next_ops.append(input._grad_accmulator)
                else:
                    self.next_ops.append(input.grad_fn)  # ③
            grad_fn = self

        inputs = [t.ndarray for t in args]
        output = self.forward(*inputs)  # ④
        return Tensor(output, requires_grad=requires_grad, grad_fn=grad_fn)  # ⑤


class OpAccumulate(Operator):
    def __init__(self, tensor):
        super(OpAccumulate, self).__init__()
        self.tensor = tensor

    def backward(self, grad):
        self.tensor.grad = Tensor(grad)
        return grad


def execute_graph(root, output_grad):
    deps = compute_dependencies(root)
    inputs = {root: output_grad}  # ①

    q = deque()
    q.append(root)
    while len(q) != 0:
        task = q.pop()
        input = inputs[task]
        outputs = task.backward(input)
        if not isinstance(outputs, collections.abc.Sequence):
            outputs = [outputs]

        for next_op, output in zip(task.next_ops, outputs):
            if next_op is None:
                continue

            # accumulate the "inputs" for next_op # ②
            op_input = inputs.get(next_op, 0)
            inputs[next_op] = op_input + output

            deps[next_op] -= 1
            if deps[next_op] == 0:  # ③
                q.append(next_op)


class OpEWiseAdd(Operator):
    # func: y = a + b
    # deri: y'/a' = 1
    # deri: y'/b' = 1
    def forward(self, a: NDArray, b: NDArray):
        return a + b

    def backward(self, grad: NDArray):
        ret = grad, grad
        return ret


def add(a, b):
    return OpEWiseAdd()(a, b)


class OpEWiseMul(Operator):
    # func: y = a * b
    # deri: y'/a' = b
    # deri: y'/b' = a
    def forward(self, a: NDArray, b: NDArray):
        self.a = a
        self.b = b
        return a * b

    def backward(self, grad: NDArray):
        return self.b * grad, self.a * grad


def mul(a, b):
    return OpEWiseMul()(a, b)


class OpSin(Operator):
    # func: y = sin(x)
    # deri: y'/x' = cos(x)
    def forward(self, x: NDArray):
        self.x = x
        return np.sin(x)

    def backward(self, grad: NDArray):
        ret = np.cos(self.x) * grad
        return ret


def sin(x):
    return OpSin()(x)


x1 = Tensor(np.array([0.5]), requires_grad=True)
x2 = Tensor(np.array([0.5]), requires_grad=True)
v3 = sin(x1)
v4 = mul(x1, x2)
v5 = add(v3, v4)
grad = np.array([1])
v5.backward(grad)
print(x1.grad)
# tensor([1.37758256]
print(x2.grad)
# tensor([0.5])
