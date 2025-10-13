import numpy as np


class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError


class MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        self.y_true = y_true
        self.y_pred = y_pred
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_pred, y_true):
        samples = y_true.shape[0]
        return (2 * (y_pred - y_true)) / samples


class BinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        self.y_true = y_true
        self.y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        return -np.mean(
            y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred)
        )

    def backward(self, y_pred, y_true):
        samples = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * samples)


class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = y_pred.shape[0]
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred[np.arange(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred * y_true, axis=1)

        return -np.mean(np.log(correct_confidences))

    def backward(self, y_pred, y_true):
        samples = y_pred.shape[0]
        labels = y_true.shape[1] if len(y_true.shape) > 1 else np.max(y_true) + 1
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        return -y_true / (y_pred * samples)
