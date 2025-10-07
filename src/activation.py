import numpy as np


class ReLU:
    def forward(self, x):
        self.input = x
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0
        return self.dinputs


class LReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.input = x
        self.output = np.where(x > 0, x, self.alpha * x)
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.where(self.input > 0, dvalues, dvalues * self.alpha)
        return self.dinputs


class Sigmoid:
    def forward(self, x):
        self.input = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * (self.output * (1 - self.output))
        return self.dinputs


class Tanh:
    def forward(self, x):
        self.input = x
        self.output = np.tanh(x)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output**2)
        return self.dinputs


class Softmax:
    def forward(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        return self.dinputs


class Linear:
    def forward(self, x):
        self.input = x
        self.output = x
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        return self.dinputs
