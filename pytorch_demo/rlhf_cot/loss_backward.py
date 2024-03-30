# coding:utf-8

import numpy as np


def sigmoid(x):
    return 1 / (1.0 + np.exp(-x))


class NN:
    def __init__(self, dim=[]):
        self.w1 = np.random(10, 16)
        self.b1 = 0.0
        self.w2 = np.random(16, 1)
        self.b2 = 0.0

    def forward(self, X):
        """
         X: [B,10]
         H1: [B,16]
         H2: [B,1]
        """
        self.H1 = sigmoid(np.matmul(X, self.w1) + self.b1)
        self.H2 = sigmoid(np.matmul(self.H1, self.w2) + self.b2)
        return self.H2

    def loss(self,  Y):
        """
        Y:[B,1]
        """
        return np.sum(Y * np.log(self.H2) - (1 - Y) * np.log(1 - self.H2))

    def backward(self):
        t_g = self.Y/self.H2 +(1-self.Y)/(1-self.H2)
        h2_g = (self.H2) * (1 - self.H2)
        h1_g = (self.H1) * (1 - self.H1)

        b2_gradient = h2_g
        w2_gradient = h2_g * self.H1
        b1_gradient = h2_g * self.w2 * h1_g
        w1_gradient = h2_g * self.w2 * h1_g * self.X
