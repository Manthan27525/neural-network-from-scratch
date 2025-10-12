import numpy as np


class InputLayer:
    def forward(self, X):
        self.output = X
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues
        return self.dinputs


class Dense:
    def __init__(
        self,
        input_size,
        output_size,
        activation=None,
        learning_rate=0.01,
        weight_init="random",
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        if weight_init == "random":
            self.weights = np.random.randn(input_size, output_size) * 0.01
        elif weight_init == "xavier":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(
                2 / (input_size + output_size)
            )
        else:
            raise ValueError("Unknown weight initialization type")

        self.biases = np.zeros((1, output_size))

        self.activation = activation

    def forward(self, X):
        self.input = X

        self.linear_output = np.dot(X, self.weights) + self.biases

        if self.activation is not None:
            self.output = self.activation.forward(self.linear_output)
        else:
            self.output = self.linear_output

        return self.output

    def backward(self, dvalues):
        if self.activation is not None:
            dactivation = self.activation.backward(dvalues)
        else:
            dactivation = dvalues

        self.dweights = np.dot(self.input.T, dactivation)
        self.dbiases = np.sum(dactivation, axis=0, keepdims=True)
        self.dinputs = np.dot(dactivation, self.weights.T)

        self.weights -= self.learning_rate * self.dweights
        self.biases -= self.learning_rate * self.dbiases

        return self.dinputs

    class Dropout:
        def __init__(self, rate):
            self.rate = 1 - rate

        def forward(self, inputs):
            self.mask = (np.random.rand(*inputs.shape) < self.rate) / self.rate
            self.output = inputs * self.mask

        def backward(self, dvalues):
            self.dinputs = dvalues * self.mask
